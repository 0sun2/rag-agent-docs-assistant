"""Load LangChain docs from `data/raw/langchain/` into LangChain `Document` objects.

규칙:
    - 파일 1개 = `Document` 1개 (청킹은 별도 단계).
    - YAML frontmatter (`---` ... `---`)는 파싱해서 `metadata`로 분리.
    - `metadata['source_path']` 에 `crawl_output_dir` 기준 상대경로를 기록.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)

DOC_SUFFIXES = {".md", ".mdx"}


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter_dict, body). Empty dict if no frontmatter."""
    if not text.startswith("---"):
        return {}, text
    # 두 번째 '---' 위치 탐색
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_raw = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    try:
        data = yaml.safe_load(fm_raw) or {}
        if not isinstance(data, dict):
            return {}, text
        return data, body
    except yaml.YAMLError as e:
        logger.warning("Failed to parse frontmatter: %s", e)
        return {}, text


def _flatten_metadata(fm: dict[str, Any]) -> dict[str, Any]:
    """Chroma 등은 메타데이터 값으로 scalar만 허용. list/dict는 문자열로 직렬화."""
    out: dict[str, Any] = {}
    for k, v in fm.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = ", ".join(str(x) for x in v)
        else:
            out[k] = str(v)
    return out


def load_file(path: Path, root: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)
    metadata: dict[str, Any] = _flatten_metadata(fm)
    metadata["source_path"] = str(path.relative_to(root))
    metadata["file_name"] = path.name
    return Document(page_content=body, metadata=metadata)


def load_documents(root: Path | None = None) -> list[Document]:
    """Load every .md/.mdx file under `root` (default: settings.crawl_output_dir)."""
    root = Path(root or settings.crawl_output_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Doc root does not exist: {root}. Run `python -m src.rag.ingest.crawl` first."
        )

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in DOC_SUFFIXES)
    logger.info("Loading %d files from %s", len(files), root)
    docs = [load_file(p, root) for p in files]
    logger.info("Loaded %d documents", len(docs))
    return docs


def main() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    docs = load_documents()
    # 간단 통계
    total_chars = sum(len(d.page_content) for d in docs)
    avg = total_chars / max(len(docs), 1)
    logger.info("Total chars: %d, avg chars/doc: %.1f", total_chars, avg)
    # 첫 문서 미리보기
    if docs:
        d = docs[0]
        logger.info("Sample doc[0] metadata: %s", d.metadata)
        logger.info("Sample doc[0] head:\n%s", d.page_content[:300])


if __name__ == "__main__":
    main()
