from __future__ import annotations

from src.retrieval.document_loader import DocumentLoader
from src.retrieval.chunker import DocumentChunker
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore


def main() -> None:
    print("=" * 60)
    print("RAG INDEX BUILD")
    print("=" * 60)

    # 1. Load documents
    loader = DocumentLoader()

    documents = loader.load_documents()

    print(f"Documents loaded: {len(documents)}")

    # 2. Chunk documents
    chunker = DocumentChunker(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = chunker.chunk_documents(
        documents
    )

    print(f"Chunks created: {len(chunks)}")

    # 3. Generate embeddings
    print("Loading embedding model...")

    embedding_service = EmbeddingService()

    enriched_chunks, embeddings = (
        embedding_service.embed_chunks(
            chunks
        )
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    # 4. Build vector store
    vector_store = VectorStore()

    vector_store.build(
        enriched_chunks,
        embeddings,
    )

    print(
        f"Vector index created: "
        f"{vector_store.count()} chunks"
    )

    print(
        f"Index path: "
        f"{vector_store.storage_path}"
    )

    print("=" * 60)
    print("RAG INDEX BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()