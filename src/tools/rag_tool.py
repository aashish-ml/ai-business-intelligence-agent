"""
Agent-facing Retrieval-Augmented Generation (RAG) tool.

Provides semantic search over the business knowledge base and exposes
a module-level interface for the central ToolRouter.
"""

from __future__ import annotations

from typing import Any

from src.retrieval.retriever import SemanticRetriever


class RAGToolError(RuntimeError):
    """Raised when a RAG tool operation fails."""


class RAGTool:
    """
    Agent-facing RAG search tool.

    Retrieves relevant business-policy knowledge and returns
    traceable evidence for the agent.
    """

    def __init__(
        self,
        retriever: SemanticRetriever | None = None,
    ) -> None:
        self.retriever = (
            retriever
            if retriever is not None
            else SemanticRetriever()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Search the business knowledge base.
        """

        if not query or not query.strip():
            raise RAGToolError(
                "RAG search query cannot be empty."
            )

        if top_k <= 0:
            raise RAGToolError(
                "top_k must be greater than zero."
            )

        try:
            results = self.retriever.search(
                query=query,
                top_k=top_k,
            )
        except Exception as exc:
            raise RAGToolError(
                f"RAG search failed: {exc}"
            ) from exc

        evidence: list[dict[str, Any]] = []

        for result in results:
            evidence.append(
                {
                    "document_id": result["document_id"],
                    "filename": result["filename"],
                    "section": result["section"],
                    "score": round(
                        result["score"],
                        4,
                    ),
                    "text": result["text"],
                }
            )

        return {
            "success": True,
            "query": query,
            "result_count": len(evidence),
            "results": evidence,
        }


# ============================================================
# Module-Level Tool Interface
# ============================================================

_default_rag_tool = RAGTool()


def rag_search(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search the business knowledge base.

    This is the module-level interface used by ToolRouter.
    """

    return _default_rag_tool.search(
        query=query,
        top_k=top_k,
    )


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    result = rag_search(
        "What should we do with high-risk customers?"
    )

    print(result)