"""Retriever factory — Chroma 컬렉션을 LangChain Retriever 로 감싼다.

전략 × 임베딩 모델 조합은 `embedding/run.py` 의 `collection_name_for()` 규칙을 그대로 재사용.
인덱싱 때 쓴 임베딩 모델과 동일한 모델을 쿼리에도 써야 차원/의미가 맞는다 —
그래서 `get_retriever()` 는 한 번에 embeddings + vectorstore 를 같이 만든다.

Phase 2 의 검색은 단순 cosine top-k. MMR/hybrid/reranker 는 Phase 3 에서 추가.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import settings
from src.rag.embedding.embedder import get_embeddings
from src.rag.embedding.run import collection_name_for

logger = logging.getLogger(__name__)


def get_vectorstore(
    *,
    strategy: str = "recursive",
    provider: str | None = None,
    model: str | None = None,
) -> Chroma:
    """해당 (strategy, model) 조합으로 적재된 Chroma 컬렉션을 연다.

    Args:
        strategy: 청킹 전략명. `fixed|recursive|markdown|semantic` 중 하나.
        provider: 임베딩 프로바이더. 기본은 `settings.embedding_provider`.
        model: 임베딩 모델 이름. 기본은 `settings.embedding_model`.

    Raises:
        ValueError: 컬렉션이 존재하지 않거나 비어 있을 때.
    """
    provider = provider or settings.embedding_provider
    model = model or settings.embedding_model
    coll_name = collection_name_for(strategy, model)

    persist_dir = Path(settings.chroma_persist_dir)
    # 서버 모드(HttpClient) 우선 — Docker Compose 에서 chromadb 컨테이너 사용 시
    if settings.chroma_host:
        logger.info(
            "Using Chroma HttpClient at %s:%d", settings.chroma_host, settings.chroma_port
        )
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    else:
        if not persist_dir.exists():
            raise ValueError(
                f"Chroma persist dir not found: {persist_dir}. "
                "Run `python -m src.rag.embedding.run` first."
            )
        client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        existing = client.get_collection(coll_name)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"Collection '{coll_name}' not found. "
            f"Run `python -m src.rag.embedding.run {strategy} --model {model}` first."
        ) from e

    count = existing.count()
    if count == 0:
        raise ValueError(f"Collection '{coll_name}' is empty.")

    embeddings = get_embeddings(provider=provider, model=model)
    logger.info("Opened collection '%s' (%d items)", coll_name, count)

    return Chroma(
        client=client,
        collection_name=coll_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def get_retriever(
    *,
    strategy: str = "recursive",
    provider: str | None = None,
    model: str | None = None,
    top_k: int | None = None,
) -> VectorStoreRetriever:
    """단순 cosine top-k Retriever 반환.

    Phase 3 에서 MMR/hybrid/reranker 로 대체 가능하도록 호출부는 인터페이스만 의존할 것.
    """
    vs = get_vectorstore(strategy=strategy, provider=provider, model=model)
    k = top_k or settings.top_k
    return vs.as_retriever(search_kwargs={"k": k})
