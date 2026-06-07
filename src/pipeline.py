import numpy as np
from src.embedder import Qwen3Embedder
from src.vector_store import TurboVecStore
from src.generator import Qwen3Generator
from src.ingest import load_documents, chunk_documents
from turbovec import IdMapIndex


class RAGPipeline:

    def __init__(
        self,
        embedder,
        vector_store,
        generator,
        reranker=None,
        top_k: int = 5,
        confidence_threshold: float = 0.0,
    ):
        self.embedder             = embedder
        self.vector_store         = vector_store
        self.generator            = generator
        self.reranker             = reranker
        self.top_k                = top_k
        self.confidence_threshold = confidence_threshold

    def index_documents(self, data_dir="./data/docs"):
        docs = load_documents(data_dir)
        if not docs:
            print("No documents found in", data_dir)
            return 0

        chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)

        print("\nGenerating embeddings...")
        texts  = [c.page_content for c in chunks]
        embeds = self.embedder.embed_documents(texts, batch_size=16)

        self.vector_store.index    = IdMapIndex(
            dim       = TurboVecStore.EMBEDDING_DIM,
            bit_width = TurboVecStore.BIT_WIDTH,
        )
        self.vector_store.metadata = {}
        self.vector_store._next_id = 0
        self.vector_store.build(chunks, embeds)

        print(f"Indexed {len(chunks)} chunks from {len(docs)} pages")
        return len(chunks)

    def add_documents(self, data_dir):
        docs   = load_documents(data_dir)
        chunks = chunk_documents(docs)
        texts  = [c.page_content for c in chunks]
        embeds = self.embedder.embed_documents(texts)
        self.vector_store.add(chunks, embeds)
        return len(chunks)

    def _build_context(self, retrieved):
        parts = []
        for i, doc in enumerate(retrieved, 1):
            filename = doc["metadata"].get("filename", "Unknown")
            page     = doc["metadata"].get("page", "")
            page_str = f", page. {page}" if page else ""
            content  = doc["content"]
            parts.append(f"[Document {i} - {filename}{page_str}]\n{content}")
        return "\n\n---\n\n".join(parts)

    def query(
        self,
        question: str,
        top_k: int = None,
        confidence_threshold: float = None,
        history: list = None,
        verbose: bool = False,
    ):
        if self.vector_store.is_empty:
            return {
                "question"  : question,
                "answer"    : "No documents have been indexed yet. Please upload the documents first.",
                "sources"   : [],
                "retrieved" : [],
                "num_chunks": 0,
            }

        k         = top_k or self.top_k
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        q_emb     = self.embedder.embed_query(question)
        retrieved = self.vector_store.search(q_emb, k=k)

        if threshold > 0.0:
            retrieved = [r for r in retrieved if r["score"] >= threshold]

        if self.reranker and retrieved:
            retrieved = self.reranker.rerank(question, retrieved, top_k=k)

        if verbose:
            print(f"\nRetrieved {len(retrieved)} chunks:")
            for r in retrieved:
                src        = r["metadata"].get("filename", "?")
                score_key  = "rerank_score" if "rerank_score" in r else "score"
                print(f"  [{r[score_key]:.4f}] ({src}) {r['content'][:80]}...")

        context = self._build_context(retrieved)
        answer  = self.generator.generate(question, context, history=history)

        sources = list(dict.fromkeys(
            r["metadata"].get("filename", "Unknown") for r in retrieved
        ))

        return {
            "question"  : question,
            "answer"    : answer,
            "sources"   : sources,
            "retrieved" : retrieved,
            "num_chunks": len(retrieved),
        }
