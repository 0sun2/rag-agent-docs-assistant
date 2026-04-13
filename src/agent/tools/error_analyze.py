"""error_analyze tool — 자유 텍스트 에러/스택트레이스 파싱 + 원인 추정.

설계:
    - 입력: Python 트레이스백, 일반 로그 조각, 에러 메시지 등 자유 텍스트.
    - 파싱: 정규식으로 "마지막 예외 타입 + 메시지 + 가장 안쪽 프레임(파일/라인)" 추출.
      파싱 실패 시 raw text 를 그대로 LLM 에 넘김 (graceful degradation).
    - 분석: `get_llm()` 을 불러 (a) 원인 추정, (b) 다음 액션 제안 (예: "docs_search
      로 `ImportError: cannot import X` 확인", "code_generate 로 최소 재현 예제 작성")
      까지만. **이 tool 은 다른 tool 을 직접 호출하지 않음** — 에이전트가 제안을 받아
      ReAct 루프의 다음 step 에서 판단.
    - 출력은 구조화된 문자열: Parsed / Likely cause / Suggested next action 3 섹션.
      LLM 이 읽고 다음 tool 호출을 결정하는 데 쓰기 쉬운 형태.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.rag.generation.llm import get_llm

logger = logging.getLogger(__name__)


# Python traceback 마지막 라인: `ExceptionType: message` — 멀티라인 메시지도 처리.
# 사용자가 IDE 에서 복붙할 때 흔히 leading whitespace 가 붙으므로 `^\s*` 로 허용.
_EXC_LINE_RE = re.compile(
    r"^\s*(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Interrupt))"
    r"(?::\s*(?P<msg>.*))?$",
    re.MULTILINE,
)
# `  File "path", line N, in func`
_FRAME_RE = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)'
)


def _parse_traceback(text: str) -> dict[str, str | None]:
    """Python 트레이스백에서 핵심 정보 추출. 파싱 불가 필드는 None."""
    exc_type: str | None = None
    exc_msg: str | None = None
    # 여러 매치 중 마지막 (실제 raise 된 예외가 맨 아래)
    for m in _EXC_LINE_RE.finditer(text):
        exc_type = m.group("type")
        exc_msg = (m.group("msg") or "").strip() or None

    frames = _FRAME_RE.findall(text)
    innermost = frames[-1] if frames else (None, None, None)
    file_, line_, func_ = innermost if frames else (None, None, None)

    return {
        "exception_type": exc_type,
        "exception_message": exc_msg,
        "innermost_file": file_,
        "innermost_line": line_,
        "innermost_func": func_,
        "frame_count": str(len(frames)) if frames else "0",
    }


ANALYZE_SYSTEM = """You are an expert Python/LangChain error analyst. Given a parsed
error and the raw text, produce a brief analysis with two sections:

**Likely cause**: One or two sentences identifying the most probable root cause.
If the error is a LangChain/LangGraph import error, mention the `langchain_classic`
reorganization as a candidate hypothesis.

**Suggested next action**: A concrete next step framed for an agent with access to
these tools: docs_search (LangChain/LangGraph docs), web_search (general web),
code_generate (write+lint Python code). Recommend ONE tool and give the exact query
or task description to pass. Do not call tools yourself — just recommend.

Be concise. No preamble. No bullet lists beyond the two headers."""


def _format_parsed(parsed: dict[str, str | None]) -> str:
    lines = ["### Parsed"]
    if parsed["exception_type"]:
        exc_line = parsed["exception_type"]
        if parsed["exception_message"]:
            exc_line += f": {parsed['exception_message']}"
        lines.append(f"- exception: {exc_line}")
    else:
        lines.append("- exception: (could not parse — not a standard Python traceback)")
    if parsed["innermost_file"]:
        lines.append(
            f"- innermost frame: {parsed['innermost_file']}:{parsed['innermost_line']} "
            f"in {parsed['innermost_func']}"
        )
    lines.append(f"- total frames: {parsed['frame_count']}")
    return "\n".join(lines)


@tool
def error_analyze(error_text: str) -> str:
    """Parse a free-form error message or Python traceback and propose a root cause.

    Use this tool when the user shares an error, stack trace, or unexpected log output
    and wants to understand what went wrong. This tool does NOT search docs or the
    web — it returns a parsed summary, a likely root cause hypothesis, and a
    recommended next tool to call. You (the agent) decide whether to follow the
    recommendation based on the rest of the conversation.

    Args:
        error_text: Raw error text. Can be a full Python traceback, a single-line
            error message, or a log snippet. Longer context generally yields better
            analysis.

    Returns:
        A three-section string: `### Parsed`, `### Likely cause`, and
        `### Suggested next action`.
    """
    logger.info("error_analyze: %d chars", len(error_text))
    parsed = _parse_traceback(error_text)
    parsed_block = _format_parsed(parsed)

    llm = get_llm(temperature=0.0)
    user_content = (
        f"Raw error text:\n```\n{error_text.strip()}\n```\n\n"
        f"Parsed summary:\n{parsed_block}"
    )
    response = llm.invoke([
        SystemMessage(content=ANALYZE_SYSTEM),
        HumanMessage(content=user_content),
    ])
    analysis = response.content if isinstance(response.content, str) else str(response.content)

    return f"{parsed_block}\n\n{analysis.strip()}\n"
