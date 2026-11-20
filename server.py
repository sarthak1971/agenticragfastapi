"""Agentic RAG backend: your LangGraph notebook served over FastAPI + SSE.
Run:  uvicorn server:app --reload     then open http://localhost:8000
"""
import io, json, os
from typing import List, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b",
               temperature=0, max_tokens=2048)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = FastEmbedEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir="./models")

SAMPLE = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps of computation in a cyclic manner.",
    "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It retrieves relevant documents and uses them to provide context for generating more accurate responses.",
    "Vector databases store high-dimensional vectors and enable efficient similarity search. They are commonly used in RAG systems to find relevant documents based on semantic similarity.",
    "Agentic systems are AI systems that can take actions, make decisions, and interact with their environment autonomously. They often use planning and reasoning capabilities.",
]
vectorstore = FAISS.from_documents(
    [Document(page_content=t, metadata={"source": "sample-knowledge"}) for t in SAMPLE], embeddings)
# Fix: as_retriever(k=3) is silently ignored; k belongs in search_kwargs
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
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
    return {"documents": retriever.invoke(state["question"])}


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
    vectorstore.add_documents(chunks)
    DOCS[file.filename] = DOCS.get(file.filename, 0) + len(chunks)
    return {"chunks": len(chunks), "total": vectorstore.index.ntotal}


@app.get("/api/stats")
async def stats():
    return {"total": vectorstore.index.ntotal, "docs": [{"name": k, "chunks": v} for k, v in DOCS.items()]}


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
