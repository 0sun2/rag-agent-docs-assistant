"""LLM factory — 임베딩 팩토리와 동일한 추상화 패턴.

기본: OpenAI `gpt-4o-mini` (`.env` `OPENAI_LLM_MODEL`).
추후 로컬 LLM(ollama 등)으로 교체 가능하도록 provider 분기만 열어 둠.
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


def get_llm(
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """LangChain `BaseChatModel` 인스턴스를 반환."""
    provider = provider.lower()
    model = model or settings.openai_llm_model
    logger.info("Loading LLM: provider=%s, model=%s", provider, model)
    if provider == "openai":
        return _build_openai(model, temperature)
    raise ValueError(f"Unknown LLM provider: {provider}")
