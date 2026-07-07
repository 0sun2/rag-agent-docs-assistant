"""Embed chunked documents and persist them to ChromaDB.

입력: `data/processed/chunks/{strategy}.jsonl` (Phase 2 산출물)
출력: `settings.chroma_persist_dir` 아래에 (전략 × 모델) 별도 컬렉션
       컬렉션명 규칙: f"{settings.chroma_collection_name}__{strategy}__{model_slug}"
       예) langchain_docs__recursive__bge-m3

특징:
    - 전략 × 임베딩 모델 조합마다 별도 컬렉션 → Phase 3 비교 실험을 깔끔하게 분리
    - 동일 컬렉션이 이미 같은 청크 수로 인덱싱돼 있으면 skip → 재실행 안전
    - 청크 ID 는 `{source_path}::{strategy}::{idx}` 로 결정 (재현성)
    - 임베딩은 기본 HuggingFace BGE-M3 (오픈소스). `--provider/--model` 로 다른 모델 가능.

사용:
    uv run python -m src.rag.embedding.run                           # 4전략 × 기본 모델
    uv run python -m src.rag.embedding.run recursive markdown        # 일부 전략만
    uv run python -m src.rag.embedding.run --model BAAI/bge-large-en-v1.5
    uv run python -m src.rag.embedding.run --provider openai --model text-embedding-3-small
    uv run python -m src.rag.embedding.run --reset
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from src.config import settings
from src.rag.embedding.embedder import get_embeddings, model_slug

logger = logging.getLogger(__name__)

CHUNKS_DIR = Path("./data/processed/chunks")
STRATEGIES = ("fixed", "recursive", "markdown", "semantic")
EMBED_BATCH = 256  # 로컬 임베딩은 rate limit 없음 — 메모리/속도만 고려


def load_chunks_jsonl(path: Path) -> list[Document]:
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(page_content=obj["page_content"], metadata=obj["metadata"]))
    return docs


def collection_name_for(strategy: str, model_name: str) -> str:
    return f"{settings.chroma_collection_name}__{strategy}__{model_slug(model_name)}"


def make_chunk_ids(chunks: list[Document], strategy: str) -> list[str]:
    counters: dict[str, int] = {}
    ids: list[str] = []
    for c in chunks:
        src = c.metadata.get("source_path", "unknown")
        i = counters.get(src, 0)
        counters[src] = i + 1
        ids.append(f"{src}::{strategy}::{i}")
    return ids


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma 는 metadata 값으로 str/int/float/bool/None 만 허용."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def index_strategy(
    strategy: str,
    *,
    embeddings,
    model_name: str,
    reset: bool,
) -> int:
    jsonl_path = CHUNKS_DIR / f"{strategy}.jsonl"
    if not jsonl_path.exists():
        logger.warning("Skip [%s]: %s not found", strategy, jsonl_path)
        return 0

    chunks = load_chunks_jsonl(jsonl_path)
    if not chunks:
        logger.warning("Skip [%s]: no chunks in %s", strategy, jsonl_path)
        return 0
    for c in chunks:
        c.metadata = _sanitize_metadata(c.metadata)

    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    coll_name = collection_name_for(strategy, model_name)

    client = chromadb.PersistentClient(path=str(persist_dir))

    existing_count = 0
    try:
        existing = client.get_collection(coll_name)
        existing_count = existing.count()
    except Exception:  # noqa: BLE001 - 컬렉션 없음
        pass

    if existing_count == len(chunks) and not reset:
        logger.info(
            "[%s] '%s' already has %d items — skipping (use --reset to rebuild)",
            strategy,
            coll_name,
            existing_count,
        )
        return existing_count

    if existing_count > 0:
        logger.info("[%s] deleting existing collection '%s' (%d items)", strategy, coll_name, existing_count)
        client.delete_collection(coll_name)

    vectorstore = Chroma(
        client=client,
        collection_name=coll_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    ids = make_chunk_ids(chunks, strategy)
    logger.info(
        "[%s] embedding %d chunks → '%s' (batch=%d)",
        strategy,
        len(chunks),
        coll_name,
        EMBED_BATCH,
    )

    for start in tqdm(range(0, len(chunks), EMBED_BATCH), desc=f"embed[{strategy}]"):
        end = min(start + EMBED_BATCH, len(chunks))
        vectorstore.add_documents(documents=chunks[start:end], ids=ids[start:end])

    final_count = client.get_collection(coll_name).count()
    logger.info("[%s] done. '%s' now has %d items", strategy, coll_name, final_count)
    return final_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks → ChromaDB")
    parser.add_argument(
        "strategies",
        nargs="*",
        choices=STRATEGIES,
        help="대상 전략. 비우면 4개 전부.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="huggingface | openai | bedrock (기본: settings.embedding_provider)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="모델 이름 (기본: settings.embedding_model)",
    )
    parser.add_argument("--reset", action="store_true", help="기존 컬렉션 강제 삭제 후 재구축")
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    provider = args.provider or settings.embedding_provider
    model_name = args.model or settings.embedding_model
    targets = args.strategies or list(STRATEGIES)

    logger.info("Provider=%s, Model=%s, Strategies=%s", provider, model_name, targets)
    embeddings = get_embeddings(provider=provider, model=model_name)

    summary: dict[str, int] = {}
    for s in targets:
        summary[s] = index_strategy(
            s, embeddings=embeddings, model_name=model_name, reset=args.reset
        )

    print("\n=== Indexing summary ===")
    print(f"  provider : {provider}")
    print(f"  model    : {model_name}  (slug: {model_slug(model_name)})")
    print(f"  device   : {settings.embedding_device}")
    for s, n in summary.items():
        print(f"  {s:10s} {n:>6d} items")
    print(f"persist dir: {settings.chroma_persist_dir}")


if __name__ == "__main__":
    main()
