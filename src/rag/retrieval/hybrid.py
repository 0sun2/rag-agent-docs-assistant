"""Hybrid retrieval — BM25(sparse) + Chroma(dense) EnsembleRetriever.

BM25 는 인덱스 파일이 없고 청크 전체를 in-memory 로 올려 돌린다.
→ 우리 규모(~35k 청크, 평균 1k 문자)에서 메모리 수백 MB 수준이라 OK.
LangChain `EnsembleRetriever` 가 RRF(Reciprocal Rank Fusion)로 두 랭킹을 결합.

주의:
    - BM25 는 **같은 청킹 전략의 jsonl** 에서 로드해야 dense 와 문서집합이 정확히 일치.
    - 가중치 기본값 0.5/0.5. Phase 3 실험에서 조정 가능하도록 파라미터화.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import settings
from src.rag.retrieval.retriever import get_vectorstore

logger = logging.getLogger(__name__)

CHUNKS_DIR = Path("./data/processed/chunks")


def _load_chunks(strategy: str) -> list[Document]:
    path = CHUNKS_DIR / f"{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks jsonl not found: {path}. Run chunking first."
        )
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(page_content=obj["page_content"], metadata=obj["metadata"])
            )
    return docs


def get_bm25_retriever(strategy: str, k: int) -> BM25Retriever:
    """청크 jsonl 에서 BM25 in-memory retriever 생성."""
    docs = _load_chunks(strategy)
    logger.info("Building BM25 index for [%s] (%d docs)", strategy, len(docs))
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


def get_hybrid_retriever(
    *,
    strategy: str = "recursive",
    provider: str | None = None,
    model: str | None = None,
    k: int | None = None,
    dense_weight: float = 0.5,
) -> EnsembleRetriever:
    """BM25 + dense 를 `EnsembleRetriever` 로 결합.

    Args:
        strategy: 청킹 전략. dense/bm25 양쪽 동일해야 함.
        model: 임베딩 모델 (dense 측).
        k: 각 retriever 가 뽑을 top-k. 최종 ensemble 도 같은 k 로 자름.
        dense_weight: dense 가중치 (0~1). bm25 는 `1 - dense_weight`.
    """
    k = k or settings.top_k

    vs = get_vectorstore(strategy=strategy, provider=provider, model=model)
    dense = vs.as_retriever(search_kwargs={"k": k})

    bm25 = get_bm25_retriever(strategy, k)

    return EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[dense_weight, 1.0 - dense_weight],
    )
