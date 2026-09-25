from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import get_settings


class EmbeddingService:
    """
    Generates dense semantic embeddings for RAG documents
    and user queries.
    """

    def __init__(
        self,
        model_name: str | None = None,
    ):
        settings = get_settings()

        self.model_name = (
            model_name
            or settings.embedding_model
        )

        self.model = SentenceTransformer(
            self.model_name
        )

    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray:

        if not texts:
            return np.empty(
                (0, 384),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype(
            np.float32
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding[0].astype(
            np.float32
        )

    def embed_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], np.ndarray]:

        if not chunks:
            return [], np.empty(
                (0, 384),
                dtype=np.float32,
            )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embed_texts(texts)

        enriched_chunks = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            enriched_chunks.append(
                {
                    **chunk,
                    "embedding": embedding,
                }
            )

        return (
            enriched_chunks,
            embeddings,
        )