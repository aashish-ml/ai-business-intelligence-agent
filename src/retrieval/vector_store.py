from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class VectorStore:
    """
    Lightweight local vector store using NumPy.

    Embeddings are normalized, so cosine similarity
    is equivalent to the dot product.
    """

    def __init__(
        self,
        storage_path: str = "vector_store/rag_index.npz",
    ):
        self.storage_path = Path(
            storage_path
        )

        self.embeddings: np.ndarray | None = None
        self.chunks: list[dict[str, Any]] = []

    def build(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match "
                "number of embeddings."
            )

        if len(chunks) == 0:
            raise ValueError(
                "Cannot build an empty vector store."
            )

        self.chunks = chunks

        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            self.storage_path,
            embeddings=self.embeddings,
        )

        metadata_path = (
            self.storage_path.with_suffix(
                ".metadata.npy"
            )
        )

        np.save(
            metadata_path,
            np.array(
                self.chunks,
                dtype=object,
            ),
            allow_pickle=True,
        )

    def load(self) -> None:

        if not self.storage_path.exists():
            raise FileNotFoundError(
                f"Vector index not found: "
                f"{self.storage_path}"
            )

        metadata_path = (
            self.storage_path.with_suffix(
                ".metadata.npy"
            )
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector metadata not found: "
                f"{metadata_path}"
            )

        data = np.load(
            self.storage_path,
        )

        self.embeddings = data[
            "embeddings"
        ]

        self.chunks = np.load(
            metadata_path,
            allow_pickle=True,
        ).tolist()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if self.embeddings is None:
            raise RuntimeError(
                "Vector store is not loaded."
            )

        if not self.chunks:
            return []

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        scores = self.embeddings @ query

        top_k = min(
            top_k,
            len(scores),
        )

        indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in indices:
            chunk = dict(
                self.chunks[index]
            )

            chunk.pop(
                "embedding",
                None,
            )

            chunk["score"] = float(
                scores[index]
            )

            results.append(chunk)

        return results

    def count(self) -> int:
        return len(self.chunks)