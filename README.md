# TurboQwenRAG

> **Lightweight, fast, secure, and free document chat system powered by Qwen AI and TurboVec search.**
> Upload your documents, ask questions, and get answers from your own files - no paid API required.

---

## Tech Stack

| Komponen | Model / Library | Detail |
|---|---|---|
| **Embedding** | `Qwen/Qwen3-Embedding-0.6B` | Instruction-aware, 8K context, 1024-dim, Apache 2.0 |
| **Vector Store** | `turbovec IdMapIndex` | TurboQuant 4-bit, Rust+SIMD, zero training, `.tvim` persistence |
| **LLM** | `Qwen3-8B-Q4_K_M` | ~5GB VRAM, llama.cpp CUDA, chatml format |
| **Framework** | `LangChain` | Document loading, text splitting |
| **UI** | `Gradio` | Web interface, shareable link |

---

## Architecture

User Query
↓
[Qwen3-Embedding-0.6B]  instruction-aware query embedding (1024-dim)
↓
[turbovec IdMapIndex]    TurboQuant 4-bit ANN search (Rust + SIMD)
↓
[Context Assembly]       format top-k chunks + source attribution
↓
[Qwen3-8B Q4_K_M]       answer generation (llama.cpp + CUDA)
↓
Answer + Sources

---

## Getting Started

### Local Setup
```bash
git clone https://github.com/tustoz/TurboQwenRAQ
cd TurboQwenRAQ

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python with CUDA (adjust for your CUDA version)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Download Qwen3-8B GGUF
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3-8B-GGUF', 'Qwen3-8B-Q4_K_M.gguf', local_dir='./models')
"

# Run
python app.py
```

---

## Supported File Formats

- **PDF** - via PyMuPDF (text extraction per page)
- **TXT** - plain text files
- **Markdown** - .md files

---

## VRAM Usage

| Component | VRAM |
|---|---|
| Qwen3-Embedding-0.6B | ~1.2 GB |
| Qwen3-8B Q4_K_M (model) | ~5.0 GB |
| Qwen3-8B (KV cache, ctx=4096) | ~0.5 GB |
| turbovec index (in RAM, not VRAM) | — |
| **Total** | **~6.7 GB** |

---

## AI Engineering Skills Demonstrated

| Skill | Implementation |
|---|---|
| **RAG Pipeline Design** | End-to-end: ingestion → embedding → retrieval → generation |
| **Instruction-Aware Embedding** | Separate prompts for query vs document in Qwen3-Embedding |
| **Quantized Vector Search** | turbovec TurboQuant 4-bit, Rust+SIMD, zero training overhead |
| **LLM Quantization** | Q4_K_M GGUF via llama.cpp, full GPU offload on T4 |
| **Prompt Engineering** | System prompt, context injection, /no_think mode for Qwen3 |
| **Persistent Storage** | turbovec `.tvim` binary format, incremental add/remove |
| **Modular Architecture** | Separated embedder/ingest/vector_store/generator/pipeline |
| **Production Thinking** | Zero external API dependency, cost-conscious hardware usage |

---

## Project Structure

rag-knowledge-base/
├── RAG_KnowledgeBase.ipynb  ← Main notebook
├── src/
│   ├── embedder.py          ← Qwen3-Embedding-0.6B
│   ├── ingest.py            ← Document loader + chunker
│   ├── vector_store.py      ← turbovec IdMapIndex wrapper
│   ├── generator.py         ← Qwen3-8B via llama.cpp
│   └── pipeline.py          ← RAG chain assembly
├── data/docs/               ← Your documents here
├── index/                   ← Saved turbovec index
├── models/                  ← GGUF model files
├── app.py                   ← Standalone Gradio app
└── requirements.txt

---

## License

MIT - free for personal and commercial use.