import os
import shutil
import time
from pathlib import Path
import gradio as gr

from src.embedder import Qwen3Embedder
from src.vector_store import TurboVecStore
from src.generator import Qwen3Generator
from src.pipeline import RAGPipeline
from src.reranker import Qwen3Reranker
from src.evaluator import RAGEvaluator

INDEX_DIR = "./index"
DOCS_DIR  = "./data/docs"

print("-" * 60)
print("TurboQwenRAG - Initializing pipeline...")
print("-" * 60)

embedder     = Qwen3Embedder(device="cuda")
vector_store = TurboVecStore()
generator    = Qwen3Generator(model_path="./models/Qwen3-8B-Q4_K_M.gguf")
reranker     = Qwen3Reranker(device="cuda")
rag          = RAGPipeline(
    embedder,
    vector_store,
    generator,
    reranker=reranker,
    top_k=5,
    confidence_threshold=0.0,
)
evaluator = RAGEvaluator(generator, embedder)

_index_file = os.path.join(INDEX_DIR, "turbovec.tvim")
if os.path.exists(_index_file):
    try:
        rag.vector_store = TurboVecStore.load(INDEX_DIR)
        print(f"Auto-loaded saved index: {len(rag.vector_store)} vectors")
    except Exception as e:
        print(f"Could not load saved index: {e}")
elif os.path.exists(DOCS_DIR):
    doc_files = (
        list(Path(DOCS_DIR).rglob("*.pdf"))
        + list(Path(DOCS_DIR).rglob("*.txt"))
        + list(Path(DOCS_DIR).rglob("*.md"))
    )
    if doc_files:
        print(f"Found {len(doc_files)} document(s) without index. Auto-building...")
        try:
            n = rag.index_documents(DOCS_DIR)
            if n > 0:
                rag.vector_store.save(INDEX_DIR)
                print(f"Auto-built index: {n} chunks")
        except Exception as e:
            print(f"Could not auto-build index: {e}")

print("-" * 60)
print("Pipeline ready. Launching Gradio UI...")
print("-" * 60)


def upload_and_index(files):
    if not files:
        return "No files selected."

    os.makedirs(DOCS_DIR, exist_ok=True)
    saved = []

    for file in files:
        filename = os.path.basename(file.name)
        dest     = f"{DOCS_DIR}/{filename}"
        shutil.copy(file.name, dest)
        saved.append(filename)

    t0      = time.time()
    n       = rag.index_documents(DOCS_DIR)
    elapsed = time.time() - t0
    rag.vector_store.save(INDEX_DIR)

    return (
        f"Indexed {n} chunks from {len(saved)} file(s) in {elapsed:.1f}s\n"
        f"Files: {', '.join(saved)}"
    )


def chat(message, history, clean_hist, confidence_threshold):
    if not message.strip():
        return history, "", clean_hist

    if rag.vector_store.is_empty:
        reply = "There are no documents yet. Upload a PDF/TXT/MD document first."
        history.append((message, reply))
        return history, "", clean_hist

    result  = rag.query(
        message,
        confidence_threshold=confidence_threshold,
        history=clean_hist,
        verbose=False,
    )
    sources = result["sources"]
    reply   = result["answer"]

    if sources:
        src_str = ", ".join(sources)
        reply  += f"\n\n*Sources: {src_str}*"

    new_clean_hist = list(clean_hist) + [(message, result["answer"])]
    history.append((message, reply))
    return history, "", new_clean_hist


def clear_chat():
    return [], "", []


def show_index_stats():
    n = len(rag.vector_store)
    if n == 0:
        return "Index is empty. Upload a document to get started."
    return f"Active index: {n} indexed chunks"


def run_evaluation(question, answer, context_text):
    if not question.strip() or not answer.strip():
        return "Provide both a question and an answer to evaluate."

    if context_text.strip():
        contexts = [
            c.strip() for c in context_text.strip().split("---") if c.strip()
        ]
    else:
        if rag.vector_store.is_empty:
            return "No context provided and index is empty. Upload documents first."
        q_result = rag.query(question, verbose=False)
        contexts = [r["content"] for r in q_result["retrieved"]]
        if not contexts:
            return "No relevant context found in the index for this question."

    try:
        metrics = evaluator.evaluate(question, answer, contexts)
    except Exception as e:
        return f"Evaluation failed: {e}"

    lines = [
        f"Context Relevance  : {metrics['context_relevance']:.4f}",
        f"Faithfulness       : {metrics['faithfulness']:.4f}",
        f"Answer Relevance   : {metrics['answer_relevance']:.4f}",
        f"Overall Score      : {metrics['overall_score']:.4f}",
    ]
    return "\n".join(lines)


with gr.Blocks(
    title="TurboQwenRAG",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 1000px; margin: auto; }",
) as demo:

    gr.Markdown(
        "# TurboQwenRAG\n"
        "> *Qwen3-8B - Qwen3-Embedding-0.6B - Qwen3-Reranker-0.6B - turbovec TurboQuant*"
    )

    with gr.Tabs():

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Documents")
                    file_upload = gr.File(
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md"],
                        label="PDF / TXT / Markdown",
                    )
                    upload_btn    = gr.Button("Index Documents", variant="primary")
                    upload_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3,
                    )
                    index_stats = gr.Textbox(
                        label="Index Info",
                        interactive=False,
                        value=show_index_stats(),
                    )
                    refresh_btn = gr.Button("Refresh Stats")
                    confidence_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Filter chunks below this score (0 = no filter)",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Chat")
                    chatbot            = gr.Chatbot(height=420, label="Conversation")
                    clean_history_state = gr.State([])
                    msg_input = gr.Textbox(
                        placeholder="Type your question about the documents...",
                        label="Question",
                        lines=2,
                    )
                    with gr.Row():
                        send_btn  = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear Chat")

            upload_btn.click(
                upload_and_index,
                inputs=[file_upload],
                outputs=[upload_status],
            )
            upload_btn.click(show_index_stats, outputs=[index_stats])
            refresh_btn.click(show_index_stats, outputs=[index_stats])
            send_btn.click(
                chat,
                inputs=[msg_input, chatbot, clean_history_state, confidence_slider],
                outputs=[chatbot, msg_input, clean_history_state],
            )
            msg_input.submit(
                chat,
                inputs=[msg_input, chatbot, clean_history_state, confidence_slider],
                outputs=[chatbot, msg_input, clean_history_state],
            )
            clear_btn.click(
                clear_chat,
                outputs=[chatbot, msg_input, clean_history_state],
            )

        with gr.Tab("Evaluation"):
            gr.Markdown(
                "### RAGAS-style Evaluation\n"
                "Evaluate RAG quality using local models. No paid API required.\n\n"
                "**Context Relevance** measures embedding similarity between the question and retrieved chunks. "
                "**Faithfulness** verifies whether the answer's claims are grounded in the context. "
                "**Answer Relevance** checks if the answer addresses the question by reverse-generating questions."
            )
            with gr.Row():
                with gr.Column():
                    eval_question = gr.Textbox(
                        label="Question",
                        placeholder="Enter the question asked...",
                        lines=2,
                    )
                    eval_answer = gr.Textbox(
                        label="Answer",
                        placeholder="Enter the AI answer to evaluate...",
                        lines=4,
                    )
                    eval_context = gr.Textbox(
                        label="Context (optional)",
                        placeholder=(
                            "Paste context passages separated by ---\n"
                            "Leave blank to auto-retrieve from the current index."
                        ),
                        lines=6,
                    )
                    eval_btn = gr.Button("Run Evaluation", variant="primary")
                with gr.Column():
                    eval_results = gr.Textbox(
                        label="Evaluation Results",
                        interactive=False,
                        lines=8,
                    )

            eval_btn.click(
                run_evaluation,
                inputs=[eval_question, eval_answer, eval_context],
                outputs=[eval_results],
            )


if __name__ == "__main__":
    demo.launch(share=True, debug=False, quiet=True)
