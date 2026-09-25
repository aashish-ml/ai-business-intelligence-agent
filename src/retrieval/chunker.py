from __future__ import annotations

import re
from typing import Any


class DocumentChunker:
    """
    Splits business documents into retrieval-friendly chunks.

    The chunker preserves document identity and section information
    so retrieved evidence can later be traced back to its source.
    """

    def __init__(
        self,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_sections(
        self,
        content: str,
    ) -> list[tuple[str, str]]:
        """
        Split Markdown content using heading boundaries.
        """

        pattern = r"(?m)^(#{1,6})\s+(.+?)\s*$"

        matches = list(re.finditer(pattern, content))

        if not matches:
            return [("General", content.strip())]

        sections: list[tuple[str, str]] = []

        for index, match in enumerate(matches):
            heading = match.group(2).strip()

            start = match.end()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(content)
            )

            section_content = content[start:end].strip()

            if section_content:
                sections.append(
                    (
                        heading,
                        section_content,
                    )
                )

        return sections

    def _chunk_text(self, text: str) -> list[str]:
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []

        start = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = end - self.chunk_overlap

        return chunks

    def chunk_document(
        self,
        document: dict[str, Any],
    ) -> list[dict[str, Any]]:
        content = document["content"]

        sections = self._split_sections(content)

        chunks: list[dict[str, Any]] = []

        chunk_index = 0

        for section_name, section_content in sections:
            section_chunks = self._chunk_text(
                section_content
            )

            for chunk_text in section_chunks:
                chunks.append(
                    {
                        "chunk_id": (
                            f"{document['document_id']}"
                            f"_chunk_{chunk_index}"
                        ),
                        "document_id": document[
                            "document_id"
                        ],
                        "filename": document[
                            "filename"
                        ],
                        "section": section_name,
                        "text": chunk_text,
                        "metadata": {
                            "document_id": document[
                                "document_id"
                            ],
                            "filename": document[
                                "filename"
                            ],
                            "section": section_name,
                            "chunk_index": chunk_index,
                        },
                    }
                )

                chunk_index += 1

        return chunks

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        all_chunks: list[dict[str, Any]] = []

        for document in documents:
            all_chunks.extend(
                self.chunk_document(document)
            )

        return all_chunks