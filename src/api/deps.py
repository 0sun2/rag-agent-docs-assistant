"""싱글톤 빌더 — retriever / agent graph 캐싱.

retriever 는 (strategy, model, method) 키별로 lru_cache.
agent graph 는 단일 인스턴스 (Phase 4 프로덕션 구성 그대로).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.retrievers import BaseRetriever

from src.rag.retrieval.hybrid import get_hybrid_retriever
from src.rag.retrieval.rerank import wrap_with_reranker
from src.rag.retrieval.retriever import get_vectorstore

logger = logging.getLogger(__name__)

EMBED_MODEL_MAP = {
    "bge-m3": ("huggingface", "BAAI/bge-m3"),
    "bge-large-en-v1.5": ("huggingface", "BAAI/bge-large-en-v1.5"),
}


@lru_cache(maxsize=16)
def get_cached_retriever(
    strategy: str, embedding_model: str, method: str, top_k: int
) -> BaseRetriever:
    provider, model = EMBED_MODEL_MAP[embedding_model]
    if method == "dense":
        vs = get_vectorstore(strategy=strategy, provider=provider, model=model)
        return vs.as_retriever(search_kwargs={"k": top_k})
    hybrid = get_hybrid_retriever(
        strategy=strategy, provider=provider, model=model, k=top_k
    )
    if method == "hybrid":
        return hybrid
    return wrap_with_reranker(hybrid, top_n=top_k)


@lru_cache(maxsize=1)
def get_cached_agent_graph():
    from src.agent.graph.agent import build_agent_graph

    return build_agent_graph()


@lru_cache(maxsize=1)
def get_cached_thread_agent_graph():
    """thread_id 기반 영속 대화용 그래프 — SqliteSaver checkpointer 포함.

    stateless 그래프와 분리해 기존 /agent/chat 하위호환을 유지한다.
    FastAPI sync 엔드포인트는 스레드풀에서 돌므로 check_same_thread=False
    (SqliteSaver가 내부 lock으로 직렬화).
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from src.agent.graph.agent import build_agent_graph
    from src.config import settings

    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_db_path), check_same_thread=False)
    return build_agent_graph(checkpointer=SqliteSaver(conn))
