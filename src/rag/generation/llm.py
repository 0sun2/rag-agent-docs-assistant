"""LLM factory — 임베딩 팩토리와 동일한 추상화 패턴.

provider는 `.env`의 `LLM_PROVIDER`로 전환:
    - "openai"  : ChatOpenAI (`OPENAI_LLM_MODEL`, 기본 gpt-4o-mini)
    - "bedrock" : ChatBedrockConverse (`BEDROCK_MODEL_ID` 추론 프로파일 ID).
      자격증명은 ~/.aws/credentials 또는 IAM 역할에서 해석 — .env에 키를 넣지 않는다.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from src.config import settings

logger = logging.getLogger(__name__)


def _build_openai(model_name: str, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_retries=4,
        timeout=60.0,
    )


def _build_bedrock(model_name: str, temperature: float) -> BaseChatModel:
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model=model_name,
        region_name=settings.aws_region,
        temperature=temperature,
    )


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """LangChain `BaseChatModel` 인스턴스를 반환.

    Args:
        provider: "openai" | "bedrock". 생략 시 `settings.llm_provider`.
        model: 모델 ID. 생략 시 provider별 settings 기본값.
        temperature: 샘플링 온도.
    """
    provider = (provider or settings.llm_provider).lower()
    if provider == "openai":
        model = model or settings.openai_llm_model
        logger.info("Loading LLM: provider=%s, model=%s", provider, model)
        return _build_openai(model, temperature)
    if provider == "bedrock":
        model = model or settings.bedrock_model_id
        logger.info("Loading LLM: provider=%s, model=%s", provider, model)
        return _build_bedrock(model, temperature)
    raise ValueError(f"Unknown LLM provider: {provider}")
