import os, json, time
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from src.embedder import Qwen3Embedder
from src.vector_store import TurboVecStore
from src.generator import Qwen3Generator
from src.pipeline import RAGPipeline
from src.reranker import Qwen3Reranker
from src.evaluator import RAGEvaluator

INDEX_DIR = "./index"
DOCS_DIR  = "./data/docs"

embedder     = Qwen3Embedder(device="cuda")
vector_store = TurboVecStore()
generator    = Qwen3Generator(model_path="./models/Qwen3-8B-Q4_K_M.gguf")
reranker     = Qwen3Reranker(device="cuda")
rag          = RAGPipeline(embedder, vector_store, generator, reranker=reranker, top_k=5)
evaluator    = RAGEvaluator(generator, embedder)

_index_file = os.path.join(INDEX_DIR, "turbovec.tvim")
if os.path.exists(_index_file):
    try:
        rag.vector_store = TurboVecStore.load(INDEX_DIR)
    except Exception:
        pass
elif os.path.exists(DOCS_DIR):
    doc_files = (
        list(Path(DOCS_DIR).rglob("*.pdf"))
        + list(Path(DOCS_DIR).rglob("*.txt"))
        + list(Path(DOCS_DIR).rglob("*.md"))
    )
    if doc_files:
        try:
            n = rag.index_documents(DOCS_DIR)
            if n > 0:
                rag.vector_store.save(INDEX_DIR)
        except Exception:
            pass

app = FastAPI(title="TurboQwenRAG")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/stats")
async def stats():
    docs = []
    if os.path.exists(DOCS_DIR):
        docs = sorted(
            f.name for f in Path(DOCS_DIR).iterdir()
            if f.suffix.lower() in {".pdf", ".txt", ".md"}
        )
    return {"vectors": len(rag.vector_store), "docs": docs}


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    os.makedirs(DOCS_DIR, exist_ok=True)
    saved = []
    for file in files:
        dest = os.path.join(DOCS_DIR, file.filename)
        with open(dest, "wb") as f:
            f.write(await file.read())
        saved.append(file.filename)

    t0      = time.time()
    n       = rag.index_documents(DOCS_DIR)
    elapsed = round(time.time() - t0, 1)
    rag.vector_store.save(INDEX_DIR)

    return {"chunks": n, "files": saved, "elapsed": elapsed}


class ChatRequest(BaseModel):
    message: str
    history: List[List[str]] = []
    confidence_threshold: float = 0.0


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    if not req.message.strip() or rag.vector_store.is_empty:
        def empty():
            if rag.vector_store.is_empty:
                yield f"data: {json.dumps({'text': 'No documents indexed yet. Upload a document first.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    history   = [(h[0], h[1]) for h in req.history if len(h) == 2]
    q_emb     = rag.embedder.embed_query(req.message)
    retrieved = rag.vector_store.search(q_emb, k=rag.top_k)

    if req.confidence_threshold > 0.0:
        retrieved = [r for r in retrieved if r["score"] >= req.confidence_threshold]
    if rag.reranker and retrieved:
        retrieved = rag.reranker.rerank(req.message, retrieved, top_k=rag.top_k)

    context = rag._build_context(retrieved)
    sources = list(dict.fromkeys(
        r["metadata"].get("filename", "Unknown") for r in retrieved
    ))

    def stream():
        for chunk in rag.generator.generate_stream(req.message, context, history=history):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


class EvalRequest(BaseModel):
    question: str
    answer: str
    context: str = ""


@app.post("/api/evaluate")
async def evaluate(req: EvalRequest):
    if not req.question.strip() or not req.answer.strip():
        return {"error": "Provide both a question and an answer."}

    if req.context.strip():
        contexts = [c.strip() for c in req.context.strip().split("---") if c.strip()]
    else:
        if rag.vector_store.is_empty:
            return {"error": "No context provided and index is empty."}
        q_result = rag.query(req.question)
        contexts = [r["content"] for r in q_result["retrieved"]]
        if not contexts:
            return {"error": "No relevant context found in index."}

    try:
        return evaluator.evaluate(req.question, req.answer, contexts)
    except Exception as e:
        return {"error": str(e)}


def _start_tunnel(port: int):
    import threading

    def _try_cloudflared():
        import re, stat, subprocess, urllib.request
        bin_path = "/usr/local/bin/cloudflared"
        if not os.path.exists(bin_path):
            print("Downloading cloudflared...")
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                bin_path,
            )
            os.chmod(bin_path, os.stat(bin_path).st_mode | stat.S_IEXEC)

        proc = subprocess.Popen(
            [bin_path, "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for raw in proc.stdout:
            line = raw.decode()
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                print(f"\n  Public URL: {m.group()}\n")
                break
            if "failed" in line.lower() or "error" in line.lower():
                print(f"  Tunnel error: {line.strip()}")

    def _run():
        _try_cloudflared()

    threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Expose via Cloudflare tunnel")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.share:
        import threading
        threading.Timer(2.0, _start_tunnel, args=(args.port,)).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port)