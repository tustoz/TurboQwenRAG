import os, json, pickle
import numpy as np
from turbovec import IdMapIndex
from langchain_core.documents import Document


class TurboVecStore:
    EMBEDDING_DIM = 1024
    BIT_WIDTH     = 4

    def __init__(self):
        self.index     = IdMapIndex(dim=self.EMBEDDING_DIM, bit_width=self.BIT_WIDTH)
        self.metadata  = {}
        self._next_id  = 0
        self._is_built = False

    def build(self, chunks: list, embeddings: np.ndarray) -> "TurboVecStore":
        assert len(chunks) == len(embeddings)
        assert embeddings.dtype == np.float32

        n   = len(chunks)
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.uint64)
        self.index.add_with_ids(embeddings, ids)

        for chunk, uid in zip(chunks, ids):
            self.metadata[int(uid)] = {
                "content" : chunk.page_content,
                "metadata": chunk.metadata,
            }

        self._next_id += n
        self._is_built = True
        print(f"turbovec index built: {n} vectors with dim={self.EMBEDDING_DIM} and bit_width={self.BIT_WIDTH}")
        return self

    def add(self, chunks: list, embeddings: np.ndarray) -> None:
        n   = len(chunks)
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.uint64)
        self.index.add_with_ids(embeddings, ids)
        for chunk, uid in zip(chunks, ids):
            self.metadata[int(uid)] = {
                "content" : chunk.page_content,
                "metadata": chunk.metadata,
            }
        self._next_id += n
        print(f"Added {n} vectors (total: {self._next_id})")

    def remove(self, chunk_id: int) -> None:
        self.index.remove(np.uint64(chunk_id))
        self.metadata.pop(chunk_id, None)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list:
        """
        turbovec search() need query shape (1, dim) not (dim,)
        returns: scores shape (1, k), ids shape (1, k),
        so wee need to get [0] for first query result.
        """
        # make sure it's float32
        q = query_embedding.astype(np.float32)

        # Reshape: (dim,) to (1, dim)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        scores_2d, ids_2d = self.index.search(q, k)

        # Get the first line (single query)
        scores = scores_2d[0]
        ids    = ids_2d[0]

        results = []
        for score, uid in zip(scores, ids):
            uid_int = int(uid)
            if uid_int in self.metadata:
                entry = self.metadata[uid_int]
                results.append({
                    "content" : entry["content"],
                    "metadata": entry["metadata"],
                    "score"   : float(score),
                    "id"      : uid_int,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def save(self, index_dir: str = "./index") -> None:
        os.makedirs(index_dir, exist_ok=True)

        index_path  = os.path.join(index_dir, "turbovec.tvim")
        meta_path   = os.path.join(index_dir, "metadata.pkl")
        config_path = os.path.join(index_dir, "config.json")

        self.index.write(index_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        config = {
            "next_id"      : self._next_id,
            "embedding_dim": self.EMBEDDING_DIM,
            "bit_width"    : self.BIT_WIDTH,
            "total_vectors": len(self.metadata),
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Index saved to {index_dir}/")
        print(f"turbovec.tvim : {os.path.getsize(index_path) / 1e6:.1f} MB")
        print(f"metadata.pkl  : {os.path.getsize(meta_path)  / 1e3:.1f} KB")
        print(f"Total vectors : {len(self.metadata)}")

    @classmethod
    def load(cls, index_dir: str = "./index") -> "TurboVecStore":
        index_path  = os.path.join(index_dir, "turbovec.tvim")
        meta_path   = os.path.join(index_dir, "metadata.pkl")
        config_path = os.path.join(index_dir, "config.json")

        assert os.path.exists(index_path), f"Index not found: {index_path}"

        store           = cls.__new__(cls)
        store.EMBEDDING_DIM = 1024
        store.BIT_WIDTH     = 4
        store.index         = IdMapIndex.load(index_path)

        with open(meta_path, "rb") as f:
            store.metadata = pickle.load(f)

        with open(config_path, "r") as f:
            config = json.load(f)

        store._next_id  = config["next_id"]
        store._is_built = True

        print(f"Index loaded from {index_dir}/")
        print(f"Vectors : {config['total_vectors']}")
        return store

    @property
    def is_empty(self) -> bool:
        return len(self.metadata) == 0

    def __len__(self) -> int:
        return len(self.metadata)

    def __repr__(self) -> str:
        return (f"TurboVecStore(vectors={len(self.metadata)}, "
                f"dim={self.EMBEDDING_DIM}, bit_width={self.BIT_WIDTH})")