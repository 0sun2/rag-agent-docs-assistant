"""Common interface for chunking strategies.

각 전략은 `Chunker` 프로토콜을 만족하는 callable: `chunk(docs) -> list[Document]`.
교체 가능한 인터페이스 원칙을 따른다.
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class Chunker(Protocol):
    """Splits a list of Documents into a list of chunked Documents."""

    name: str

    def chunk(self, docs: list[Document]) -> list[Document]:  # pragma: no cover - protocol
        ...
