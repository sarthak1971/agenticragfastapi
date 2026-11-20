"""Agentic RAG backend: your LangGraph notebook served over FastAPI + SSE.
Run:  uvicorn server:app --reload     then open http://localhost:8000
"""
import io, json, os

import httpx
import numpy as np
from typing import List, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b",
               temperature=0, max_tokens=2048)
# ---- Lightweight vector store: Gemini embeddings (API) + numpy. No torch, no FAISS, ~150 MB RAM ----
EMB_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
DIM = 768


def embed(texts, task):
    """task: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    out = []
    with httpx.Client(timeout=60) as c:
        for i in range(0, len(texts), 100):  # API accepts up to 100 per batch
            reqs = [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]},
                     "taskType": task, "outputDimensionality": DIM} for t in texts[i:i + 100]]
            r = c.post(EMB_URL, headers={"x-goog-api-key": os.getenv("GEMINI_API_KEY", "")}, json={"requests": reqs})
            r.raise_for_status()
            out += [e["values"] for e in r.json()["embeddings"]]
    v = np.array(out, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # truncated dims must be re-normalised


class Store:
    def __init__(self):
        self.docs, self.vecs = [], np.zeros((0, DIM), dtype="float32")

    def add(self, docs):
        v = embed([d.page_content for d in docs], "RETRIEVAL_DOCUMENT")
        self.vecs = np.vstack([self.vecs, v])
        self.docs += docs

    def search(self, q, k=3):
        if not self.docs:
            return []
        scores = self.vecs @ embed([q], "RETRIEVAL_QUERY")[0]
        return [self.docs[i] for i in np.argsort(-scores)[:k]]


SAMPLE = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps of computation in a cyclic manner.",
    "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It retrieves relevant documents and uses them to provide context for generating more accurate responses.",
    "Vector databases store high-dimensional vectors and enable efficient similarity search. They are commonly used in RAG systems to find relevant documents based on semantic similarity.",
    "Agentic systems are AI systems that can take actions, make decisions, and interact with their environment autonomously. They often use planning and reasoning capabilities.",
]
store = Store()
store.add([Document(page_content=t, metadata={"source": "sample-knowledge"}) for t in SAMPLE])
DOCS = {"sample-knowledge": len(SAMPLE)}  # filename -> chunk count


# ---- Your graph (same nodes/edges), plus chat history in the prompt ----
class AgentState(TypedDict):
    question: str
    history: List[dict]
    documents: List[Document]
    answer: str
    needs_retrieval: bool


def decide_retrieval(state: AgentState):
    kws = ["what", "how", "explain", "describe", "tell me"]
    return {"needs_retrieval": any(k in state["question"].lower() for k in kws)}


def retrieve_documents(state: AgentState):
    return {"documents": store.search(state["question"], k=3)}


async def generate_answer(state: AgentState):
    docs = state.get("documents", [])
    if docs:
        ctx = "\n\n".join(d.page_content for d in docs)
        prompt = f"Based on the following context, answer the question:\n\nContext:\n{ctx}\n\nQuestion: {state['question']}\n\nAnswer:"
    else:
        prompt = f"Answer the following question: {state['question']}"
    msgs = [(h["role"], h["content"]) for h in state.get("history", [])[-6:]]
    msgs.append(("user", prompt))
    res = await llm.ainvoke(msgs)
    return {"answer": res.content}


wf = StateGraph(AgentState)
wf.add_node("decide", decide_retrieval)
wf.add_node("retrieve", retrieve_documents)
wf.add_node("generate", generate_answer)
wf.set_entry_point("decide")
wf.add_conditional_edges("decide", lambda s: "retrieve" if s["needs_retrieval"] else "generate",
                         {"retrieve": "retrieve", "generate": "generate"})
wf.add_edge("retrieve", "generate")
wf.add_edge("generate", END)
graph = wf.compile()
NODES = {"decide", "retrieve", "generate"}


# ---- API ----
app = FastAPI()


class ChatIn(BaseModel):
    question: str
    history: List[dict] = []


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatIn):
    state = {"question": body.question, "history": body.history, "documents": [],
             "answer": "", "needs_retrieval": False}

    async def stream():
        try:
            async for ev in graph.astream_events(state, version="v2"):
                kind, name = ev["event"], ev["name"]
                node = ev.get("metadata", {}).get("langgraph_node")
                if name in NODES and kind == "on_chain_start":
                    yield sse({"type": "step", "node": name, "status": "running"})
                elif name in NODES and kind == "on_chain_end":
                    out = ev["data"].get("output") or {}
                    yield sse({"type": "step", "node": name, "status": "done"})
                    if name == "decide" and not out.get("needs_retrieval"):
                        yield sse({"type": "step", "node": "retrieve", "status": "skipped"})
                    if name == "retrieve":
                        yield sse({"type": "sources", "items": [
                            {"text": d.page_content, "source": d.metadata.get("source", "document")}
                            for d in out.get("documents", [])]})
                elif kind == "on_chat_model_stream" and node == "generate":
                    text = ev["data"]["chunk"].content
                    if text:
                        yield sse({"type": "token", "text": text})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), chunk_size: int = Form(800), chunk_overlap: int = Form(100)):
    raw = await file.read()
    if file.filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
    else:
        text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        raise HTTPException(400, "No readable text found in this file.")
    size = max(100, chunk_size)
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=max(0, min(chunk_overlap, size // 2)))
    chunks = sp.create_documents([text], metadatas=[{"source": file.filename}])
    try:
        store.add(chunks)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Embedding API error: {e}")
    DOCS[file.filename] = DOCS.get(file.filename, 0) + len(chunks)
    return {"chunks": len(chunks), "total": len(store.docs)}


@app.get("/api/stats")
async def stats():
    return {"total": len(store.docs), "docs": [{"name": k, "chunks": v} for k, v in DOCS.items()]}


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
