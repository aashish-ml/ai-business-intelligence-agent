from __future__ import annotations

from typing import Any

from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.utils.config import get_settings


class SemanticRetriever:
    """
    Performs semantic search over the business knowledge base.
    """

    def __init__(
        self,
        vector_store_path: str = "vector_store/rag_index.npz",
    ):
        settings = get_settings()

        self.top_k = settings.rag_top_k
        self.min_score = settings.rag_min_score

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore(
            storage_path=vector_store_path
        )

        self.vector_store.load()

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        query_embedding = (
            self.embedding_service.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
        )

        filtered_results = [
            result
            for result in results
            if result["score"] >= self.min_score
        ]

        return filtered_results


if __name__ == "__main__":
    retriever = SemanticRetriever()

    query = (
        "What should we do with high-risk customers?"
    )

    results = retriever.search(query)

    print("=" * 60)
    print("SEMANTIC SEARCH TEST")
    print("=" * 60)

    print("Query:", query)
    print("Results:", len(results))

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"RESULT {index}")
        print("Document:", result["filename"])
        print("Section:", result["section"])
        print(
            "Score:",
            round(result["score"], 4),
        )
        print("Text:", result["text"])

    print("=" * 60)