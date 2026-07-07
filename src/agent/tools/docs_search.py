"""docs_search tool — Phase 3 최종 프로덕션 구성을 그대로 LangChain tool 로 감싼다.

구성: `recursive × bge-large-en-v1.5 × hybrid_rerank`
- chunking: recursive
- embedding: BAAI/bge-large-en-v1.5
- retrieval: BM25 + dense EnsembleRetriever (RRF) → BAAI/bge-reranker-v2-m3

RAGAS 기준 answer_relevancy 0.965 / context_recall 0.903 / context_precision 0.882 로
4 조합 중 최상. 자세한 비교 근거는 `experiments/ragas_eval.md` + `docs/portfolio/problem_solving.md` #10.

반환 포맷은 LLM 이 읽고 인용까지 생성할 수 있도록 문자열 — 각 청크를
`[i] source: <path>\n<content>` 형태로 나열한다. 구조화 결과(원본 Document)는
`AgentState.tool_results` 에 별도 저장할 수 있도록 `search_docs()` 함수도 같이 제공.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from langchain_core.documents import Document

# 청크 본문의 절대 URL 을 플레이스홀더로 치환 — LLM 이 인용 target 으로 오해해 합성
# 하는 것을 원천 차단. 인용할 수 있는 유일한 source 는 청크 첫 줄의 파일 경로.
_URL_RE = re.compile(r"https?://\S+")
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool

from src.agent.security.sanitize import sanitize_tool_output
from src.rag.retrieval.hybrid import get_hybrid_retriever
from src.rag.retrieval.rerank import wrap_with_reranker

logger = logging.getLogger(__name__)

# Phase 3 에서 확정된 프로덕션 구성이 settings 기본값. AWS 배포 시 .env 로 스왑
# (PROD_EMBEDDING_PROVIDER=bedrock + PROD_USE_RERANKER=false — GPU 제거).
from src.config import settings  # noqa: E402


@lru_cache(maxsize=1)
def _get_prod_retriever() -> BaseRetriever:
    """프로덕션 retriever 를 첫 호출 시 한 번만 로드하고 캐시한다.

    BM25 인덱스 빌드(수십초) + cross-encoder 모델 로드(수백MB) 비용이 있어 캐싱 필수.
    프로세스 생존 기간 동안 유지된다.
    """
    logger.info(
        "Building production docs_search retriever: %s x %s x %s",
        settings.prod_strategy,
        settings.prod_embedding_model,
        "hybrid_rerank" if settings.prod_use_reranker else "hybrid",
    )
    hybrid = get_hybrid_retriever(
        strategy=settings.prod_strategy,
        provider=settings.prod_embedding_provider,
        model=settings.prod_embedding_model,
        # reranker 가 있으면 넓게 fetch 후 재정렬, 없으면 top_k 만
        k=settings.prod_fetch_k if settings.prod_use_reranker else settings.prod_top_k,
    )
    if not settings.prod_use_reranker:
        return hybrid
    return wrap_with_reranker(hybrid, top_n=settings.prod_top_k)


def search_docs(query: str) -> list[Document]:
    """원본 `Document` 리스트 그대로 반환 — 호출부가 메타데이터를 활용할 때 사용."""
    retriever = _get_prod_retriever()
    return retriever.invoke(query)


def _format_docs(docs: list[Document]) -> str:
    """LLM 프롬프트용 문자열 포맷팅. source 경로를 앞에 붙여 인용 강제."""
    if not docs:
        return "No relevant documents found."
    # 각 청크는 첫 줄에 경로만 단독으로, 그 다음에 본문. 구분자로 분리.
    # LLM 은 "첫 줄 = 인용할 경로" 규칙으로 해석 (프롬프트에서 지정).
    parts: list[str] = []
    for d in docs:
        src = d.metadata.get("source_path") or d.metadata.get("file_name") or "unknown"
        content = _URL_RE.sub("[url-removed]", d.page_content)
        parts.append(f"{src}\n{content}")
    return "\n\n---\n\n".join(parts)


@tool
def docs_search(query: str) -> str:
    """Search the LangChain/LangGraph official documentation for relevant passages.

    Use this tool whenever the user asks about LangChain or LangGraph concepts, APIs,
    classes, methods, configuration, migration, or usage patterns. Prefer this over
    general web search for framework-specific questions.

    Args:
        query: A focused natural-language question or keyword phrase. Good queries
            are specific (e.g. "how to create a custom tool with @tool decorator"),
            not vague ("tell me about langchain").

    Returns:
        A formatted string of the top 5 most relevant documentation passages, each
        prefixed with its source file path. Cite the source path in your final answer.
    """
    docs = search_docs(query)
    logger.info("docs_search(%r) → %d passages", query, len(docs))
    # 인덱싱된 문서도 외부 유래 텍스트 — 동일한 데이터 경계로 래핑
    return sanitize_tool_output("docs_search", _format_docs(docs))
