"""Four chunking strategies: fixed, recursive, markdown, semantic.

- **fixed**: 고정 길이 캐릭터 분할 (`CharacterTextSplitter`). 가장 단순한 베이스라인.
- **recursive**: 단락/문장 경계를 우선 보존하는 분할 (`RecursiveCharacterTextSplitter`).
- **markdown**: 헤더(`#`, `##`, ...) 기준 구조 인식 분할 (`MarkdownHeaderTextSplitter`)
  → 헤더로 자른 뒤 너무 긴 섹션은 `RecursiveCharacterTextSplitter`로 2차 분할.
- **semantic**: 임베딩 기반 의미 경계 분할 (`SemanticChunker`).
  비용이 크므로 **샘플 모드**(소수 파일만)에서만 사용 — Phase 2 점검용.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed
# ---------------------------------------------------------------------------
@dataclass
class FixedChunker:
    """Fixed-size character splitter with a hard upper bound.

    `CharacterTextSplitter`는 단일 구분자(`\\n\\n`)만 사용하므로, 빈 줄이 없는 거대 블록
    (긴 코드/표/JSON)이 그대로 한 청크가 될 수 있다. 임베딩 토큰 한도(텍스트-임베딩-3-small
    기준 8,191 토큰)를 넘기지 않도록, 1차 분할 후 `hard_max_chars` 를 초과하는 청크는
    `RecursiveCharacterTextSplitter` 로 강제 재분할한다.
    """

    name: str = "fixed"
    chunk_size: int = field(default_factory=lambda: settings.chunk_size)
    chunk_overlap: int = field(default_factory=lambda: settings.chunk_overlap)
    hard_max_chars: int = 8000  # ~2k tokens 미만 보장

    def chunk(self, docs: list[Document]) -> list[Document]:
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        primary = splitter.split_documents(docs)

        fallback = RecursiveCharacterTextSplitter(
            chunk_size=self.hard_max_chars,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        out: list[Document] = []
        oversized = 0
        for c in primary:
            if len(c.page_content) <= self.hard_max_chars:
                c.metadata["chunk_strategy"] = self.name
                out.append(c)
                continue
            oversized += 1
            for piece in fallback.split_text(c.page_content):
                out.append(
                    Document(
                        page_content=piece,
                        metadata={**c.metadata, "chunk_strategy": self.name},
                    )
                )
        if oversized:
            logger.info(
                "FixedChunker: hard-split %d oversized chunks (> %d chars)",
                oversized,
                self.hard_max_chars,
            )
        return out


# ---------------------------------------------------------------------------
# Recursive
# ---------------------------------------------------------------------------
@dataclass
class RecursiveChunker:
    name: str = "recursive"
    chunk_size: int = field(default_factory=lambda: settings.chunk_size)
    chunk_overlap: int = field(default_factory=lambda: settings.chunk_overlap)

    def chunk(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        out = splitter.split_documents(docs)
        for c in out:
            c.metadata["chunk_strategy"] = self.name
        return out


# ---------------------------------------------------------------------------
# Markdown (header-aware + recursive secondary split)
# ---------------------------------------------------------------------------
@dataclass
class MarkdownChunker:
    name: str = "markdown"
    chunk_size: int = field(default_factory=lambda: settings.chunk_size)
    chunk_overlap: int = field(default_factory=lambda: settings.chunk_overlap)

    def chunk(self, docs: list[Document]) -> list[Document]:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,
        )
        secondary = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        out: list[Document] = []
        for doc in docs:
            try:
                header_chunks = header_splitter.split_text(doc.page_content)
            except Exception as e:  # noqa: BLE001 - markdown 파서가 비정형 입력에 약함
                logger.warning(
                    "MarkdownHeaderTextSplitter failed for %s: %s",
                    doc.metadata.get("source_path"),
                    e,
                )
                header_chunks = [Document(page_content=doc.page_content, metadata={})]

            for hc in header_chunks:
                merged_meta = {**doc.metadata, **hc.metadata}
                refined = secondary.split_text(hc.page_content)
                for piece in refined:
                    out.append(
                        Document(
                            page_content=piece,
                            metadata={**merged_meta, "chunk_strategy": self.name},
                        )
                    )
        return out


# ---------------------------------------------------------------------------
# Semantic (embedding-based — sample only)
# ---------------------------------------------------------------------------
@dataclass
class SemanticChunker:
    """Wraps `langchain_experimental.text_splitter.SemanticChunker`.

    임베딩 호출 비용이 있으므로 Phase 2에서는 **소수 파일** 입력만 권장.
    """

    name: str = "semantic"
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 95.0

    def chunk(self, docs: list[Document]) -> list[Document]:
        # 무거운 의존성은 lazy import
        from langchain_experimental.text_splitter import SemanticChunker as _LCSemantic

        from src.rag.embedding.embedder import get_embeddings

        embeddings = get_embeddings()  # 기본: BGE-M3 (오픈소스)
        splitter = _LCSemantic(
            embeddings=embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
        )
        out = splitter.split_documents(docs)
        for c in out:
            c.metadata["chunk_strategy"] = self.name
        return out


def build_default_chunkers() -> dict[str, object]:
    """Return all four strategies keyed by name."""
    return {
        "fixed": FixedChunker(),
        "recursive": RecursiveChunker(),
        "markdown": MarkdownChunker(),
        "semantic": SemanticChunker(),
    }
