import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class Qwen3Embedder:
    """
    Instruction-aware text embedder using Qwen3-Embedding-0.6B.

    Embedding dim: 1024
    Context window: 32K tokens
    Languages ​​: 100+
    License: Apache 2.0
    """

    # Different instructions for queries vs documents
    QUERY_INSTRUCTION = "Represent this query for retrieving relevant documents: "
    DOC_INSTRUCTION   = "Represent this document for retrieval: "

    def __init__(self, device: str = "cuda"):
        print("Loading Qwen3-Embedding-0.6B...")
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device=device,
        )
        self.device = device
        self.embedding_dim = 1024
        print(f"Embedder ready on {device} with dim={self.embedding_dim}")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query string to float32 numpy array shape (1024,)"""
        vec = self.model.encode(
            query,
            prompt=self.QUERY_INSTRUCTION,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed list of document strings to float32 numpy array shape (N, 1024)"""
        vecs = self.model.encode(
            texts,
            prompt=self.DOC_INSTRUCTION,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        return vecs.astype(np.float32)

    def embed_query_lc(self, text: str) -> list[float]:
        return self.embed_query(text).tolist()

    def embed_documents_lc(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts).tolist()