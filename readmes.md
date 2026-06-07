# TurboQwenRAG

> **Lightweight, fast, secure, and free document chat system powered by the Qwen3 ecosystem and TurboVec search.**
> Upload your documents, ask questions, and get grounded answers from your own files — no paid API required.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Platform: Colab T4](https://img.shields.io/badge/Platform-Colab%20T4-orange.svg)](https://colab.research.google.com/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

---

## Overview

TurboQwenRAG is the **first RAG system** to combine turbovec TurboQuant 4-bit vector indexing with the complete Qwen3 AI ecosystem (embedding + reranker + LLM) in a single end-to-end pipeline. It runs entirely on a free Google Colab T4 GPU with zero cloud API dependency.

```
User Query
    ↓
[Qwen3-Embedding-0.6B]    instruction-aware query embed (1024-dim, 32K ctx)
    ↓
[turbovec IdMapIndex]      TurboQuant 4-bit ANN search — retrieve top-20
    ↓
[Qwen3-Reranker-0.6B]     cross-encoder re-score → top-5
    ↓
[Confidence Threshold]    score < 0.30 → reject (anti-hallucination)
    ↓
[Multi-turn History]      inject last 3 turns as context
    ↓
[Qwen3-8B Q4_K_M]         llama.cpp + CUDA → generate answer
    ↓
Answer + Source + Confidence Score
```

---

## Tech Stack

| Component | Model / Library | Detail |
|---|---|---|
| **Embedding** | `Qwen/Qwen3-Embedding-0.6B` | Instruction-aware, 32K context, 1024-dim, Apache 2.0 |
| **Vector Store** | `turbovec IdMapIndex` | TurboQuant 4-bit, Rust+SIMD, zero training, `.tvim` persistence |
| **Reranker** | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder two-stage retrieval, instruction-aware |
| **LLM** | `Qwen3-8B-Q4_K_M` | ~5GB VRAM, llama.cpp CUDA, chatml format |
| **Framework** | `LangChain` + `langchain-text-splitters` | Document loading, text splitting |
| **UI** | `Gradio 5.50.0` | Dark theme, multi-tab, shareable Colab link |

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Two-stage Retrieval** | turbovec retrieves top-20 → Qwen3-Reranker selects top-5 |
| 💾 **Persistent Index** | turbovec `.tvim` binary — no re-embedding on restart |
| 🚫 **Anti-Hallucination** | Confidence threshold (default 0.30) rejects out-of-scope queries |
| 💬 **Multi-turn Chat** | Remembers last 3 conversation turns for follow-up questions |
| 📊 **RAG Evaluation** | Built-in eval suite with faithfulness, context relevance, answer relevance |
| 🌐 **Multilingual** | Handles Indonesian + English documents natively |
| 💸 **Zero API Cost** | 100% open source, runs entirely on free Colab T4 |

---

## Evaluation Results

Evaluated on 100 curated test cases across 5 quality categories using a custom RAGAS-style evaluator with Qwen3-8B as judge.

| Category | Context Relevance | Faithfulness | Answer Relevance | Overall |
|---|---|---|---|---|
| ✅ GOOD | 0.709 | **0.958** | 0.757 | **0.808** |
| ⚠️ HALLUCINATED | 0.767 | 0.417 | 0.731 | 0.638 |
| ⚠️ IRRELEVANT_CTX | 0.316 | 0.650 | 0.783 | 0.583 |
| ❌ OFF_TOPIC_ANSWER | 0.750 | 0.433 | 0.368 | 0.517 |
| 🔶 PARTIAL | **0.805** | 0.500 | 0.743 | 0.683 |
| **Overall (100 cases)** | **0.660** | **0.637** | **0.696** | **0.664** |

**Key findings:**
- Faithfulness **0.958** on GOOD category — near-zero hallucination when retrieval is relevant
- Context relevance **0.316** on IRRELEVANT_CTX — confidence threshold correctly identifies and rejects out-of-scope queries
- Answer relevance **0.368** on OFF_TOPIC_ANSWER — metric successfully detects answers that don't address the question

**Score distribution (overall_score):**
```
0.0 – 0.3  (poor)      :  0 cases
0.3 – 0.5              : 14 cases
0.5 – 0.7              : 41 cases
0.7 – 0.9              : 45 cases  ← majority
0.9 – 1.0  (excellent) :  0 cases
```

> Full evaluation script: [`eval_colab.py`](./eval_colab.py) — runs on Colab T4, no modifications needed.

---

## VRAM Usage (T4 GPU, 15GB)

| Component | VRAM |
|---|---|
| Qwen3-Embedding-0.6B | ~1.2 GB |
| Qwen3-Reranker-0.6B | ~1.2 GB |
| Qwen3-8B Q4_K_M (weights) | ~5.0 GB |
| Qwen3-8B (KV cache, ctx=8192) | ~1.0 GB |
| PyTorch + CUDA overhead | ~0.3 GB |
| turbovec index (RAM, not VRAM) | — |
| **Total** | **~8.7 GB** |

---

## Project Structure

```
TurboQwenRAG/
├── RAG_KnowledgeBase.ipynb   ← Main Colab notebook
├── app.py                    ← Standalone Gradio app (all features)
├── eval_colab.py             ← 100-case RAGAS-style evaluation
│
├── src/
│   ├── __init__.py
│   ├── embedder.py           ← Qwen3-Embedding-0.6B (instruction-aware)
│   ├── ingest.py             ← PDF/TXT/MD loader + chunker
│   ├── vector_store.py       ← turbovec IdMapIndex wrapper
│   ├── reranker.py           ← Qwen3-Reranker-0.6B (cross-encoder)
│   ├── generator.py          ← Qwen3-8B via llama.cpp
│   ├── pipeline.py           ← Full RAG chain + multi-turn + threshold
│   └── eval.py               ← Evaluation metrics
│
├── data/docs/                ← Your documents go here
├── index/                    ← turbovec .tvim index (auto-generated)
├── models/                   ← GGUF model (auto-downloaded)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### Option 1: Google Colab (Recommended)

1. Click the **Open in Colab** badge above
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run all cells sequentially
4. Upload your documents via the Gradio interface
5. Start chatting

### Option 2: Local Setup

```bash
git clone https://github.com/tustoz/TurboQwenRAG
cd TurboQwenRAG

# Step 1: Pin shared dependencies (order matters)
pip install "numpy==1.26.4"
pip install "pydantic>=2.12.5,<=2.12.9" "aiofiles>=24.1.0,<25.0"

# Step 2: Install all dependencies
pip install -r requirements.txt

# Step 3: Install llama-cpp-python with CUDA (adjust CUDA version if needed)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
  --no-cache-dir

# Step 4: Download Qwen3-8B GGUF (~4.9 GB)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3-8B-GGUF', 'Qwen3-8B-Q4_K_M.gguf', local_dir='./models')
"

# Step 5: Run
python app.py
```

> **Note for CPU-only:** Remove `CMAKE_ARGS="-DGGML_CUDA=on"` and install plain `llama-cpp-python`. Inference will be significantly slower.

---

## Supported File Formats

| Format | Parser | Notes |
|---|---|---|
| **PDF** | PyMuPDF | Text extraction per page, preserves page numbers |
| **TXT** | Built-in | Plain text files |
| **Markdown** | Built-in | `.md` files |

---

## Dependency Notes

Due to conflicting version requirements between `gradio`, `turbovec`, and `unstructured-client`, the following pins are required:

```
numpy==1.26.4                    # Colab ships numpy 2.x which breaks scipy/sklearn
pydantic>=2.12.5,<=2.12.9        # gradio 5.50.0 upper bound
aiofiles>=24.1.0,<25.0           # gradio 5.50.0 upper bound
gradio==5.50.0                   # pinned for stability
```

> `unstructured` is intentionally excluded — PyMuPDF covers all needed formats without the dependency conflict.

---

## AI Engineering Skills Demonstrated

| Skill | Implementation |
|---|---|
| **RAG Pipeline Design** | End-to-end: ingestion → embedding → retrieval → reranking → generation |
| **Instruction-Aware Embedding** | Separate prompts for query vs document in Qwen3-Embedding |
| **Quantized Vector Search** | turbovec TurboQuant 4-bit, Rust+SIMD, zero codebook training |
| **Two-Stage Retrieval** | Bi-encoder (turbovec) + cross-encoder (Qwen3-Reranker) |
| **LLM Quantization** | Q4_K_M GGUF via llama.cpp, full CUDA offload on T4 |
| **Anti-Hallucination** | Confidence threshold with rerank score gating |
| **Multi-Turn Dialogue** | Chat history injection with context window management |
| **RAG Evaluation** | Custom 100-case eval suite, faithfulness + context + answer metrics |
| **Persistent Storage** | turbovec `.tvim` binary format, auto load/build on restart |
| **Modular Architecture** | Separated embedder/ingest/vector_store/reranker/generator/pipeline |

---

## Changelog

### v1.0.0 — Initial Release
- ✅ Qwen3-Embedding-0.6B + turbovec IdMapIndex + Qwen3-8B pipeline
- ✅ Gradio UI with dark theme
- ✅ PDF, TXT, Markdown ingestion
- ✅ Persistent turbovec index

### v1.1.0 — Five Improvements Update
- ✅ Added Qwen3-Reranker-0.6B (two-stage retrieval)
- ✅ Auto load/build index (no rebuild on Colab restart)
- ✅ Confidence threshold (anti-hallucination, default 0.30)
- ✅ Multi-turn chat history (last 3 turns as context)
- ✅ RAG evaluation suite (`eval.py` + `eval_colab.py`)
- ✅ Gradio UI v2 with evaluation tab and live confidence display
- 🔧 Fixed: `langchain.schema` → `langchain_core.documents` (LangChain v1.0+)
- 🔧 Fixed: `numpy==1.26.4` pin for Colab scipy/sklearn compatibility
- 🔧 Fixed: turbovec `search()` requires 2D query shape `(1, dim)`
- 🔧 Fixed: Qwen3 `<think>` tag stripped from generator output

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

*Built as an AI Engineering portfolio project demonstrating production-grade RAG system design on constrained hardware.*