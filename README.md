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
| Qwen3-Embedding-0.6B | ~1.3 GB |
| Qwen3-8B Q4_K_M (model) | ~5.0 GB |
| Qwen3-8B (KV cache, ctx=8192) | ~1.0 GB |
| turbovec index (in RAM, not VRAM) | — |
| **Total** | **~7.3 GB** |

---

## License

MIT - free for personal and commercial use.