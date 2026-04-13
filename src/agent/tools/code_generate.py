"""code_generate tool — 독립적인 코드 생성 + 로컬 ruff 린팅.

설계 포인트:
    - **docs_search 에 의존하지 않음**. 에이전트가 두 tool 을 필요에 따라 조합할 뿐,
      이 tool 내부에서는 문서 검색을 하지 않는다 (사용자 지시).
    - LLM 은 `get_llm()` 팩토리 재사용 — 에이전트 본체와 동일 모델.
    - 생성 후 **ruff check/format 을 subprocess 로 실제 실행**해서 린트 결과를
      tool 출력에 포함. LLM 이 "린트 결과를 보고 수정" 하는 다음 iteration 을 유도할 수
      있도록 실제 진단을 그대로 전달.
    - 언어 범위는 Python 한정 (ruff 대상). 다른 언어 요청이 오면 린팅 없이 생성만 하고
      `lint_status: skipped (non-python)` 로 표시.

반환 포맷 (문자열):
    ```
    ### Generated Code
    ```python
    ...
    ```

    ### Lint (ruff)
    status: passed | failed | skipped
    <lint output or empty>
    ```
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.rag.generation.llm import get_llm

logger = logging.getLogger(__name__)


CODE_GEN_SYSTEM = """You are a Python code generator. Given a task description, produce
a single self-contained Python code block that accomplishes it.

Requirements:
- Output ONLY a single fenced code block. No prose before or after.
- Use `from __future__ import annotations` at the top.
- Add type hints on all function signatures.
- Keep imports minimal and at the top.
- If the task is ambiguous, make a reasonable assumption and add a brief comment
  explaining the assumption at the top of the code.
- Do NOT invent API calls you are not sure exist. Prefer standard library or
  widely-known packages.
"""

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """LLM 응답에서 첫 번째 코드 펜스를 추출. 펜스가 없으면 전체를 코드로 간주."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _run_ruff(code: str) -> tuple[str, str]:
    """임시 파일에 코드를 쓰고 `ruff check` + `ruff format --check` 실행.

    Returns:
        (status, combined_output)
        status: "passed" | "failed" | "unavailable"
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = Path(f.name)

    try:
        try:
            check = subprocess.run(
                ["ruff", "check", "--no-cache", "--output-format=concise", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            return "unavailable", "ruff not installed in this environment"

        fmt = subprocess.run(
            ["ruff", "format", "--check", "--no-cache", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        parts: list[str] = []
        if check.stdout.strip():
            parts.append(f"[check]\n{check.stdout.strip()}")
        if check.stderr.strip():
            parts.append(f"[check-err]\n{check.stderr.strip()}")
        if fmt.stdout.strip():
            parts.append(f"[format]\n{fmt.stdout.strip()}")
        if fmt.stderr.strip():
            parts.append(f"[format-err]\n{fmt.stderr.strip()}")

        status = "passed" if (check.returncode == 0 and fmt.returncode == 0) else "failed"
        combined = "\n\n".join(parts) if parts else "(no issues reported)"
        return status, combined
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _is_python_task(task: str) -> bool:
    """요청 문자열을 간단히 스캔해 파이썬 작업인지 판정.

    휴리스틱: 다른 언어가 명시되지 않았으면 Python 으로 간주 (우리 도메인 기본값).
    """
    lowered = task.lower()
    non_python_markers = [
        "javascript", "typescript", " js ", " ts ", "node.js",
        "golang", " go ", "rust", "java ", "c++", "c#", "ruby", "php",
    ]
    return not any(m in f" {lowered} " for m in non_python_markers)


@tool
def code_generate(task: str) -> str:
    """Generate a self-contained Python code snippet for a given task, then lint it with ruff.

    **YOU MUST call this tool for any request that asks you to write, implement,
    scaffold, or produce Python code.** Do NOT write code blocks inline in your final
    answer — always route code generation through this tool so the output goes through
    ruff linting. The only exception is when you are explaining an existing snippet
    that was already returned by another tool.

    This tool does NOT search documentation — if the task requires knowledge of a
    specific library API, call docs_search first and include the relevant findings
    (exact import paths, class names, decorators) in the task description you pass here.

    Args:
        task: A clear description of what the code should do. Include any constraints,
            required libraries, function signatures, or API details the generator needs.
            The more specific, the better — e.g. "Write a function `load_jsonl(path:
            Path) -> list[dict]` that streams a JSONL file line by line".

    Returns:
        A string containing the generated code block and the ruff lint report. If lint
        fails, the diagnostics are included so you can decide whether to call this tool
        again with a refined task description.
    """
    logger.info("code_generate: %s", task[:120])
    llm = get_llm(temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=CODE_GEN_SYSTEM),
        HumanMessage(content=task),
    ])
    raw = response.content if isinstance(response.content, str) else str(response.content)
    code = _extract_code(raw)

    if not _is_python_task(task):
        return (
            "### Generated Code\n"
            f"```\n{code}\n```\n\n"
            "### Lint (ruff)\n"
            "status: skipped (non-python)\n"
        )

    status, lint_output = _run_ruff(code)
    return (
        "### Generated Code\n"
        f"```python\n{code}\n```\n\n"
        "### Lint (ruff)\n"
        f"status: {status}\n"
        f"{lint_output}\n"
    )
