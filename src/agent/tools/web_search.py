"""web_search tool — Tavily 기반 웹 검색.

설계:
    - `langchain-tavily` 의 `TavilySearch` 를 얇게 래핑. Tavily 는 LLM 에이전트 용도로
      설계된 검색 API 라서 결과가 이미 "요약 + URL" 형태로 정제돼 있어 RAG 파이프라인과
      궁합이 좋음.
    - API 키는 `settings.tavily_api_key` 에서 읽고, 없으면 호출 시점에 명확한 에러.
    - **docs_search 와의 역할 분리**: docs_search 는 우리가 인덱싱한 LangChain/LangGraph
      공식 문서 전용, web_search 는 그 외 일반 웹 (최신 블로그, GitHub issue, 릴리즈
      노트, Stack Overflow 등). 에이전트 프롬프트에서 우선순위 명시.
    - `lru_cache` 로 클라이언트 재사용 — 매 호출마다 TavilySearch 인스턴스를 다시
      만들 필요 없음.

반환 포맷: docs_search 와 유사하게 `[i] <url>\n<title>\n<content>` 로 통일.
LLM 이 동일한 인용 규칙 ("path 자리에 URL") 으로 처리하게 함.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.tools import tool

from src.agent.security.sanitize import sanitize_tool_output
from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_tavily_client():
    """TavilySearch 클라이언트 싱글톤. 첫 호출에 모듈 로드 + 키 검증."""
    if not settings.tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY is not set. Add it to .env — see .env.example. "
            "Get a free key at https://tavily.com"
        )
    # 임포트는 여기 안에서 — 의존성 미설치 환경에서도 다른 tool 은 동작하도록 lazy import
    from langchain_tavily import TavilySearch

    return TavilySearch(
        max_results=settings.tavily_max_results,
        tavily_api_key=settings.tavily_api_key,
        topic="general",
    )


def _format_results(raw: dict | list) -> str:
    """TavilySearch 반환(dict with 'results' key, or plain list)을 프롬프트용 문자열로.

    langchain-tavily 버전에 따라 반환 타입이 다를 수 있어 방어적으로 처리.
    """
    if isinstance(raw, dict):
        results = raw.get("results", [])
    elif isinstance(raw, list):
        results = raw
    else:
        return str(raw)

    if not results:
        return "No web results found."

    # docs_search 와 동일 규칙: 첫 줄 = 인용할 URL, 나머지는 title + content.
    parts: list[str] = []
    for r in results:
        url = r.get("url", "unknown")
        title = r.get("title", "")
        content = r.get("content", "")
        parts.append(f"{url}\n{title}\n{content}")
    return "\n\n---\n\n".join(parts)


@tool
def web_search(query: str) -> str:
    """Search the general web for up-to-date information via Tavily.

    Use this tool for questions that are NOT about LangChain/LangGraph official docs
    (use docs_search for those). Good use cases: recent releases, third-party library
    comparisons, GitHub issues, Stack Overflow discussions, blog posts, general
    programming questions, current events.

    Args:
        query: A focused search query. Prefer specific phrases over vague questions.

    Returns:
        A formatted string of the top results, each prefixed with its URL. Cite URLs
        the same way you cite doc paths: verbatim, as `- <url>`, without modification.
    """
    logger.info("web_search(%r)", query)
    client = _get_tavily_client()
    raw = client.invoke({"query": query})
    # 외부 웹 텍스트는 인젝션 위험이 가장 큰 입력 — 데이터 경계로 래핑해 반환
    return sanitize_tool_output("web_search", _format_results(raw))
