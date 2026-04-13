"""Phase 2 (로딩 + 청킹) CLI.

수행 순서:
    1) `data/raw/langchain/` 의 모든 .md/.mdx 를 Document 로 로드
    2) fixed / recursive / markdown 3종은 **전체 문서**에 청킹 적용
    3) semantic 은 비용 때문에 **샘플 5개 파일**에만 적용 (Phase 2 점검용)
    4) 각 전략 결과를 `data/processed/chunks/{strategy}.jsonl` 로 저장
    5) 동일 샘플 파일 1~2개를 골라 4종 청킹 결과를 stdout + experiments/chunking_samples.md 에 출력
    6) 전략별 청크 수 / 평균·중앙값 길이 통계를 함께 출력

사용:
    uv run python -m src.rag.chunking.run
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document

from src.config import settings
from src.rag.chunking.strategies import (
    FixedChunker,
    MarkdownChunker,
    RecursiveChunker,
    SemanticChunker,
)
from src.rag.ingest.load import load_documents

logger = logging.getLogger(__name__)

CHUNKS_DIR = Path("./data/processed/chunks")
SAMPLES_MD = Path("./experiments/chunking_samples.md")
SEMANTIC_SAMPLE_FILES = 5  # semantic 은 비용 때문에 N개만
PER_STRATEGY_SAMPLES = 4  # 비교 출력용 청크 수


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def write_jsonl(chunks: Iterable[Document], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {"page_content": c.page_content, "metadata": c.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def chunk_stats(chunks: list[Document]) -> dict[str, float]:
    if not chunks:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0}
    lens = [len(c.page_content) for c in chunks]
    return {
        "count": len(chunks),
        "mean": statistics.mean(lens),
        "median": statistics.median(lens),
        "min": min(lens),
        "max": max(lens),
    }


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------
def pick_comparison_files(docs: list[Document], k: int = 2) -> list[Document]:
    """길이/구조가 적당한 파일을 비교용으로 선정.

    조건: 3KB ~ 15KB, '##' 헤더 3개 이상. 우선 oss/langchain 또는 oss/langgraph.
    """

    def score(d: Document) -> tuple[int, int]:
        path = d.metadata.get("source_path", "")
        priority = 0
        if path.startswith("oss/langchain/") or path.startswith("oss/langgraph/"):
            priority = 1
        return (priority, d.page_content.count("\n## "))

    candidates = [
        d
        for d in docs
        if 3000 <= len(d.page_content) <= 15000 and d.page_content.count("\n## ") >= 3
    ]
    if not candidates:
        candidates = [d for d in docs if 2000 <= len(d.page_content) <= 20000]
    candidates.sort(key=score, reverse=True)
    return candidates[:k]


def chunks_for_source(chunks: list[Document], source_path: str) -> list[Document]:
    return [c for c in chunks if c.metadata.get("source_path") == source_path]


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def _truncate(text: str, n: int = 400) -> str:
    text = text.strip().replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + " …"


def render_samples(
    sample_docs: list[Document],
    results: dict[str, list[Document]],
    semantic_chunks: list[Document],
) -> str:
    """Markdown 리포트 본문 생성. stdout 출력에도 같은 내용 사용."""
    lines: list[str] = []
    lines.append("# Chunking samples (Phase 2)")
    lines.append("")
    lines.append(f"- chunk_size = {settings.chunk_size}, chunk_overlap = {settings.chunk_overlap}")
    lines.append(f"- semantic 적용 파일 수: {SEMANTIC_SAMPLE_FILES}")
    lines.append("")

    # 전략별 전체 통계
    lines.append("## Strategy stats (전체 코퍼스)")
    lines.append("")
    lines.append("| strategy | count | mean | median | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ("fixed", "recursive", "markdown"):
        s = chunk_stats(results[name])
        lines.append(
            f"| {name} | {s['count']} | {s['mean']:.0f} | {s['median']:.0f} | "
            f"{s['min']} | {s['max']} |"
        )
    s = chunk_stats(semantic_chunks)
    lines.append(
        f"| semantic* | {s['count']} | {s['mean']:.0f} | {s['median']:.0f} | "
        f"{s['min']} | {s['max']} |"
    )
    lines.append("")
    lines.append(f"\\* semantic 은 샘플 {SEMANTIC_SAMPLE_FILES}개 파일에만 적용한 결과입니다.")
    lines.append("")

    # 비교: 같은 파일을 4전략으로
    for sd in sample_docs:
        sp = sd.metadata["source_path"]
        lines.append(f"## File: `{sp}`")
        lines.append(f"- length: {len(sd.page_content)} chars")
        lines.append("")
        for name in ("fixed", "recursive", "markdown", "semantic"):
            source_chunks = (
                chunks_for_source(semantic_chunks, sp)
                if name == "semantic"
                else chunks_for_source(results[name], sp)
            )
            lines.append(f"### [{name}] — {len(source_chunks)} chunks")
            if not source_chunks:
                lines.append("_(no chunks for this file — semantic may have skipped it)_")
                lines.append("")
                continue
            for i, c in enumerate(source_chunks[:PER_STRATEGY_SAMPLES]):
                lines.append(f"**chunk {i}** ({len(c.page_content)} chars)")
                lines.append("")
                lines.append("```")
                lines.append(_truncate(c.page_content, 500))
                lines.append("```")
                lines.append("")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    docs = load_documents()

    # 비교용 파일 먼저 선정 (semantic 샘플에도 반드시 포함)
    sample_docs = pick_comparison_files(docs, k=2)
    sample_paths = {d.metadata["source_path"] for d in sample_docs}
    logger.info("Comparison sample files: %s", sorted(sample_paths))

    # 1) 전체 코퍼스 청킹 — fixed/recursive/markdown
    chunkers = {
        "fixed": FixedChunker(),
        "recursive": RecursiveChunker(),
        "markdown": MarkdownChunker(),
    }
    results: dict[str, list[Document]] = {}
    for name, ch in chunkers.items():
        logger.info("Chunking [%s] over %d docs ...", name, len(docs))
        chunks = ch.chunk(docs)
        results[name] = chunks
        out_path = CHUNKS_DIR / f"{name}.jsonl"
        n = write_jsonl(chunks, out_path)
        logger.info("[%s] %d chunks → %s", name, n, out_path)

    # 2) Semantic — 샘플 N개 파일에만
    semantic_pool: list[Document] = list(sample_docs)
    if len(semantic_pool) < SEMANTIC_SAMPLE_FILES:
        # 샘플 비교 파일 + 추가 파일을 합쳐 N개로 채움
        existing = {d.metadata["source_path"] for d in semantic_pool}
        for d in docs:
            if d.metadata["source_path"] in existing:
                continue
            if 2000 <= len(d.page_content) <= 12000:
                semantic_pool.append(d)
            if len(semantic_pool) >= SEMANTIC_SAMPLE_FILES:
                break

    logger.info("Chunking [semantic] over %d sample docs ...", len(semantic_pool))
    try:
        semantic_chunks = SemanticChunker().chunk(semantic_pool)
    except Exception as e:  # noqa: BLE001 - 임베딩 호출 실패는 sample 단계에서 치명적이지 않음
        logger.error("Semantic chunking failed: %s", e)
        semantic_chunks = []
    n = write_jsonl(semantic_chunks, CHUNKS_DIR / "semantic.jsonl")
    logger.info("[semantic] %d chunks → %s", n, CHUNKS_DIR / "semantic.jsonl")

    # 3) 리포트
    report = render_samples(sample_docs, results, semantic_chunks)
    SAMPLES_MD.parent.mkdir(parents=True, exist_ok=True)
    SAMPLES_MD.write_text(report, encoding="utf-8")
    print("\n" + report)
    logger.info("Sample report written to %s", SAMPLES_MD)


if __name__ == "__main__":
    main()
