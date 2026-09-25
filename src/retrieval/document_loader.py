from __future__ import annotations

from pathlib import Path
from typing import Any


class DocumentLoader:
    """
    Loads knowledge-base documents from disk.

    Currently supports Markdown and plain-text files.
    """

    SUPPORTED_EXTENSIONS = {".md", ".txt"}

    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)

    def load_documents(self) -> list[dict[str, Any]]:
        if not self.knowledge_dir.exists():
            raise FileNotFoundError(
                f"Knowledge directory not found: {self.knowledge_dir}"
            )

        documents: list[dict[str, Any]] = []

        for file_path in sorted(self.knowledge_dir.iterdir()):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            content = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if not content:
                continue

            documents.append(
                {
                    "document_id": file_path.stem,
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "content": content,
                }
            )

        return documents


if __name__ == "__main__":
    loader = DocumentLoader()

    documents = loader.load_documents()

    print(f"Documents loaded: {len(documents)}")

    for document in documents:
        print(
            f"- {document['document_id']}: "
            f"{len(document['content'])} characters"
        )