import numpy as np
from sentence_transformers import SentenceTransformer

class Qwen3Embedder:
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
        vec = self.model.encode(
            query,
            prompt=self.QUERY_INSTRUCTION,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
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