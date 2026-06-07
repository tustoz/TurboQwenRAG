import os
import shutil
import time
import gradio as gr

from src.embedder import Qwen3Embedder
from src.vector_store import TurboVecStore
from src.generator import Qwen3Generator
from src.pipeline import RAGPipeline

INDEX_DIR = "./index"
DOCS_DIR  = "./data/docs"

# Pipeline (loaded once at startup)
print("-" * 60)
print("TurboQwenRAG - Initializing pipeline...")
print("-" * 60)

embedder     = Qwen3Embedder(device="cuda")
vector_store = TurboVecStore()
generator    = Qwen3Generator(model_path="./models/Qwen3-8B-Q4_K_M.gguf")
rag          = RAGPipeline(embedder, vector_store, generator, top_k=5)

# Auto-load saved index if it exists from a previous session
_index_file = os.path.join(INDEX_DIR, "turbovec.tvim")
if os.path.exists(_index_file):
    try:
        rag.vector_store = TurboVecStore.load(INDEX_DIR)
        print(f"Auto-loaded saved index: {len(rag.vector_store)} vectors")
    except Exception as e:
        print(f"Could not load saved index: {e}")

print("-" * 60)
print("Pipeline ready. Launching Gradio UI...")
print("-" * 60)


# Handlers
def upload_and_index(files):
    """Upload file and rebuild index."""
    if not files:
        return "No files selected."

    os.makedirs(DOCS_DIR, exist_ok=True)
    saved = []

    for file in files:
        filename = os.path.basename(file.name)
        dest     = f"{DOCS_DIR}/{filename}"
        shutil.copy(file.name, dest)
        saved.append(filename)

    # Re-index
    t0      = time.time()
    n       = rag.index_documents(DOCS_DIR)
    elapsed = time.time() - t0

    # Save index to disk so it survives restarts
    rag.vector_store.save(INDEX_DIR)

    return (
        f"Indexed {n} chunks from {len(saved)} file in {elapsed:.1f}s\n"
        f"Files: {', '.join(saved)}"
    )


def chat(message, history):
    """Handle chat query and return updated history."""
    if not message.strip():
        return history, ""

    if rag.vector_store.is_empty:
        reply = "There are no documents yet. Upload a PDF/TXT/MD document first."
        history.append((message, reply))
        return history, ""

    result  = rag.query(message, verbose=False)
    sources = result["sources"]

    reply = result["answer"]
    if sources:
        src_str = ", ".join(sources)
        reply  += f"\n\n*Sources: {src_str}*"

    history.append((message, reply))
    return history, ""


def clear_chat():
    return [], ""


def show_index_stats():
    n = len(rag.vector_store)
    if n == 0:
        return "Index is empty. Upload a document to get started."
    return f"Active index: {n} indexed chunks"


# UI
with gr.Blocks(
    title="TurboQwenRAG",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 900px; margin: auto; }"
) as demo:

    gr.Markdown(
        "# TurboQwenRAG\n"
        "> *Qwen3-8B - Qwen3-Embedding-0.6B - turbovec TurboQuant*"
    )

    with gr.Row():
        # ---- Left column: Upload ----
        with gr.Column(scale=1):
            gr.Markdown("### Upload Documents")
            file_upload = gr.File(
                file_count = "multiple",
                file_types = [".pdf", ".txt", ".md"],
                label      = "PDF / TXT / Markdown",
            )
            upload_btn    = gr.Button("Index Documents", variant="primary")
            upload_status = gr.Textbox(
                label       = "Status",
                interactive = False,
                lines       = 3,
            )
            index_stats = gr.Textbox(
                label       = "Index Info",
                interactive = False,
                value       = show_index_stats(),
            )
            refresh_btn = gr.Button("Refresh Stats")

        # ---- Right column: Chat ----
        with gr.Column(scale=2):
            gr.Markdown("### Chat")
            chatbot   = gr.Chatbot(height=400, label="Conversation")
            msg_input = gr.Textbox(
                placeholder = "Type your question about the documents...",
                label       = "Question",
                lines       = 2,
            )
            with gr.Row():
                send_btn  = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")

    # ---- Event wiring ----
    upload_btn.click(
        upload_and_index,
        inputs  = [file_upload],
        outputs = [upload_status],
    )
    upload_btn.click(
        show_index_stats,
        outputs = [index_stats],
    )
    refresh_btn.click(show_index_stats, outputs=[index_stats])
    send_btn.click(
        chat,
        inputs  = [msg_input, chatbot],
        outputs = [chatbot, msg_input],
    )
    msg_input.submit(
        chat,
        inputs  = [msg_input, chatbot],
        outputs = [chatbot, msg_input],
    )
    clear_btn.click(clear_chat, outputs=[chatbot, msg_input])


if __name__ == "__main__":
    demo.launch(share=True, debug=False, quiet=True)