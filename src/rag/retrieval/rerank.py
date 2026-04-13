"""Cross-encoder reranker — hybrid 결과를 재정렬.

기본 모델: `BAAI/bge-reranker-v2-m3` — BGE-M3 계열과 호환되는 최신 multilingual reranker.
1차 retriever 에서 `fetch_k` 개를 가져와 cross-encoder 로 점수를 다시 매기고 상위 `top_n` 개만 반환.

LangChain `ContextualCompressionRetriever` + `CrossEncoderReranker` 사용 —
base retriever 를 그대로 감쌀 수 있어 dense/hybrid 어느 쪽에도 적용 가능.
"""

from __future__ import annotations

import logging

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever

from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


def get_cross_encoder(model_name: str = DEFAULT_RERANKER) -> HuggingFaceCrossEncoder:
    """로컬 GPU/CPU cross-encoder 로드."""
    logger.info("Loading cross-encoder reranker: %s (device=%s)", model_name, settings.embedding_device)
    return HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs={"device": settings.embedding_device},
    )


def wrap_with_reranker(
    base_retriever: BaseRetriever,
    *,
    reranker_model: str = DEFAULT_RERANKER,
    top_n: int = 5,
) -> ContextualCompressionRetriever:
    """base retriever 위에 cross-encoder reranker 를 씌운다.

    Args:
        base_retriever: dense 또는 hybrid retriever. fetch_k 는 base 설정 그대로.
        reranker_model: cross-encoder 모델 이름.
        top_n: rerank 후 최종 반환 개수.
    """
    ce = get_cross_encoder(reranker_model)
    compressor = CrossEncoderReranker(model=ce, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
