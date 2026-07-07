"""MCP 서버 — docs_search / error_analyze 를 MCP 표준 도구로 노출 (개발계획서 4-2).

AgentCore Gateway에서 배운 "도구 표준화" 개념의 셀프호스팅 실습:
이 프로젝트의 검색엔진(hybrid + rerank)을 MCP stdio 서버로 노출하면
Claude Code 등 모든 MCP 클라이언트에서 LangChain/LangGraph 문서 검색을
실용 도구로 쓸 수 있다.

Claude Code 등록:
    claude mcp add langchain-docs -- \
        uv --directory /home/lys/workspace/llm-docs-assistant run --extra mcp \
        python -m src.mcp_server.server

주의: 첫 docs_search 호출 시 retriever(임베딩 + BM25 + 리랭커) 로드로
수십 초~수 분 걸릴 수 있다 (이후 프로세스 캐시).
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.WARNING)  # stdio 서버 — stdout 오염 방지

mcp = FastMCP("llm-docs-assistant")


@mcp.tool()
def docs_search(query: str) -> str:
    """Search the LangChain/LangGraph official documentation for relevant passages.

    Uses a hybrid (BM25 + dense) retriever with cross-encoder reranking over
    1,500+ indexed official docs. Each result chunk starts with its source
    file path, followed by the passage body.

    Args:
        query: A focused natural-language question or keyword phrase, e.g.
            "how to create a custom tool with @tool decorator".
    """
    # lazy import — 서버 기동은 즉시, 무거운 retriever는 첫 호출 시 로드
    from src.agent.tools.docs_search import _format_docs, search_docs

    return _format_docs(search_docs(query))


@mcp.tool()
def error_analyze(error_text: str) -> str:
    """Parse a Python traceback or error log and suggest a likely cause + next action.

    Args:
        error_text: The raw error message, stack trace, or log snippet.
    """
    from src.agent.tools.error_analyze import error_analyze as _tool

    return _tool.func(error_text)  # LangChain tool의 원본 함수 호출


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
