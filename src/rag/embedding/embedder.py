"""Embedding model factory.

LLM/임베딩 교체 가능 원칙에 따라 구체 구현은 이 모듈 뒤에 숨긴다.
호출부는 모델/프로바이더를 모르고 `Embeddings` 인터페이스만 사용한다.

기본: HuggingFace `BAAI/bge-m3` (오픈소스, 다국어, dense 1024-dim, max 8k tokens)
비교군: `BAAI/bge-large-en-v1.5` (영어 특화, dense 1024-dim, max 512 tokens)
선택: OpenAI `text-embedding-3-small` — Phase 3 비교 실험에서만 사용

BGE 계열은 query/passage 비대칭이 있다 → langchain-huggingface 의 `HuggingFaceBgeEmbeddings`
가 알아서 query 측에 instruction prefix 를 붙여준다. 다만 BGE-M3 는 instruction 이 필요
없으므로 `query_instruction=""` 로 비활성.

모델 슬러그(파일/컬렉션명용)는 `model_slug()` 로 얻는다.
"""

from __future__ import annotations

import logging
import re

from langchain_core.embeddings import Embeddings

from src.config import settings

logger = logging.getLogger(__name__)


def model_slug(model_name: str | None = None) -> str:
    """Return a filesystem/collection-safe slug for the configured model.

    Examples:
        BAAI/bge-m3              -> bge-m3
        BAAI/bge-large-en-v1.5   -> bge-large-en-v1.5
        text-embedding-3-small   -> text-embedding-3-small
    """
    name = model_name or settings.embedding_model
    tail = name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", tail).strip("-").lower()


def _build_huggingface(model_name: str) -> Embeddings:
    # lazy import — torch/sentence-transformers 는 무거움
    from langchain_huggingface import HuggingFaceEmbeddings

    model_kwargs = {"device": settings.embedding_device}
    encode_kwargs = {
        "batch_size": settings.embedding_batch_size,
        "normalize_embeddings": settings.embedding_normalize,
    }

    # BGE-M3 는 query instruction 불필요. 그 외 BGE 계열은 권장 instruction 사용.
    is_bge = "bge" in model_name.lower()
    is_m3 = "bge-m3" in model_name.lower()

    if is_bge and not is_m3:
        # langchain-huggingface 의 BGE 전용 래퍼가 있으면 사용 (query instruction 자동)
        try:
            from langchain_huggingface import HuggingFaceBgeEmbeddings  # type: ignore

            return HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        except ImportError:
            logger.info("HuggingFaceBgeEmbeddings unavailable, falling back to base wrapper")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def _build_openai(model_name: str) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model_name,
        api_key=settings.openai_api_key,
        max_retries=8,
        request_timeout=60.0,
    )


def _build_bedrock(model_name: str) -> Embeddings:
    """Bedrock 관리형 임베딩 (Titan V2 등) — GPU 불필요, 호출당 과금.

    AWS 배포 구성용 (구축계획서 Phase A-3). 임베딩 모델을 바꾸면 색인·검색
    임베딩 동일 원칙에 따라 전체 재색인 필수:
        uv run python -m src.rag.embedding.run --provider bedrock --model amazon.titan-embed-text-v2:0
    """
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(model_id=model_name, region_name=settings.aws_region)


def get_embeddings(
    provider: str | None = None,
    model: str | None = None,
) -> Embeddings:
    """Return a LangChain `Embeddings` instance for the requested provider/model.

    기본은 `settings.embedding_provider` / `settings.embedding_model`.
    """
    provider = (provider or settings.embedding_provider).lower()
    model = model or settings.embedding_model

    logger.info("Loading embeddings: provider=%s, model=%s", provider, model)
    if provider == "huggingface":
        return _build_huggingface(model)
    if provider == "openai":
        return _build_openai(model)
    if provider == "bedrock":
        return _build_bedrock(model)
    raise ValueError(f"Unknown embedding provider: {provider}")
