"""에이전트 수준 평가 — 테스트 4분면 자동 채점 (개발계획서 4-1).

RAGAS가 retrieval+생성 품질을 재는 것과 별개로, **에이전트 자체**의 행동을 잰다:
    - 툴 선택 정확도: 의도한 도구를 호출했는가 / 금지 도구를 안 불렀는가 (trace 기계 판정)
    - 멀티스텝 성공률: docs_search → code_generate 순서 연쇄가 이뤄졌는가
    - 인용 형식 준수율: `Sources:` 섹션 규칙 (정규식 판정)
    - 스텝 수 / 토큰 / 비용 분포

4분면 (각 12문항, data/eval/agent_eval_dataset.jsonl):
    ① docs_only   docs_search 단독      ② code_only   code_generate 단독(순수 파이썬)
    ③ multi_step  검색→코드생성 연쇄     ④ no_tool     도구 불필요

사용:
    uv run python -m src.agent.evaluation.run                 # 전체 48문항
    uv run python -m src.agent.evaluation.run --limit 2       # 4분면 × 2문항 스모크
    uv run python -m src.agent.evaluation.run --quadrant code_only

결과: experiments/agent_eval.md + agent_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from src.rag.generation.usage import usage_from_messages

logger = logging.getLogger(__name__)

DATASET_PATH = Path("data/eval/agent_eval_dataset.jsonl")
OUTPUT_DIR = Path("experiments")

QUADRANTS = ["docs_only", "code_only", "multi_step", "no_tool"]

# Sources 섹션: "Sources:" 라인 뒤에 "- <target>" 불릿이 1개 이상.
# markdown 링크([..](..))나 번호 prefix가 섞이면 규칙 위반.
_SOURCES_RE = re.compile(r"^Sources:\s*\n(?:- \S.*\n?)+", re.MULTILINE)
_BAD_CITATION_RE = re.compile(r"^Sources:\s*\n(?:.*(?:\[.+\]\(.+\)|^\d+\.).*\n?)+", re.MULTILINE)


@dataclass
class ItemResult:
    id: str
    quadrant: str
    question: str
    called_tools: list[str] = field(default_factory=list)
    tool_selection_ok: bool = False
    multistep_ok: bool | None = None  # multi_step 분면만
    citation_ok: bool | None = None  # citation_required 문항만
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    answer_preview: str = ""
    error: str | None = None


def _load_dataset(limit: int | None, quadrant: str | None) -> list[dict]:
    items = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    if quadrant:
        items = [it for it in items if it["quadrant"] == quadrant]
    if limit:
        per_q: dict[str, int] = {}
        picked = []
        for it in items:
            if per_q.get(it["quadrant"], 0) < limit:
                picked.append(it)
                per_q[it["quadrant"]] = per_q.get(it["quadrant"], 0) + 1
        items = picked
    return items


def _tool_call_sequence(messages: list) -> list[str]:
    seq: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            seq.extend(tc["name"] for tc in m.tool_calls)
    return seq


def _judge(item: dict, called: list[str], answer: str) -> tuple[bool, bool | None, bool | None]:
    expected = set(item["expected_tools"])
    forbidden = set(item["forbidden_tools"])
    called_set = set(called)

    tool_ok = expected.issubset(called_set) and not (forbidden & called_set)

    multistep_ok: bool | None = None
    if item["quadrant"] == "multi_step":
        try:
            multistep_ok = called.index("docs_search") < called.index("code_generate")
        except ValueError:
            multistep_ok = False

    citation_ok: bool | None = None
    if item["citation_required"]:
        citation_ok = bool(_SOURCES_RE.search(answer)) and not _BAD_CITATION_RE.search(answer)

    return tool_ok, multistep_ok, citation_ok


def run_item(graph, item: dict) -> ItemResult:
    res = ItemResult(id=item["id"], quadrant=item["quadrant"], question=item["question"])
    state = {
        "messages": [HumanMessage(content=item["question"])],
        "tool_results": [],
        "current_query": item["question"],
        "iteration_count": 0,
    }
    t0 = time.monotonic()
    try:
        final = graph.invoke(state)
    except Exception as e:  # noqa: BLE001
        res.error = str(e)[:300]
        res.latency_s = round(time.monotonic() - t0, 2)
        logger.exception("[%s] agent invoke failed", item["id"])
        return res
    res.latency_s = round(time.monotonic() - t0, 2)

    messages = final["messages"]
    answer = next(
        (str(m.content) for m in reversed(messages)
         if isinstance(m, AIMessage) and not m.tool_calls and m.content),
        "",
    )
    res.called_tools = _tool_call_sequence(messages)
    res.tool_selection_ok, res.multistep_ok, res.citation_ok = _judge(
        item, res.called_tools, answer
    )
    res.iterations = final.get("iteration_count", 0)
    usage = usage_from_messages(messages)
    res.input_tokens = usage.input_tokens
    res.output_tokens = usage.output_tokens
    res.cost_usd = usage.cost_usd
    res.answer_preview = answer[:200]
    return res


def _rate(values: list[bool]) -> str:
    if not values:
        return "—"
    return f"{sum(values) / len(values):.1%} ({sum(values)}/{len(values)})"


def _write_report(results: list[ItemResult]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "agent_eval.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2)
    )

    lines: list[str] = ["# 에이전트 수준 평가 — 테스트 4분면\n"]
    lines.append(f"- N questions: **{len(results)}** (오류 {sum(1 for r in results if r.error)}건 제외 판정)")
    lines.append("- 판정: trace 기계 판정 (LLM 심판 없음) — `src/agent/evaluation/run.py`\n")

    lines.append("## 분면별 결과\n")
    lines.append("| 분면 | N | 툴 선택 정확도 | 멀티스텝 성공률 | 인용 준수율 | 평균 스텝 | 평균 토큰(in/out) | 평균 비용($) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    ok_results = [r for r in results if not r.error]
    for q in QUADRANTS:
        rs = [r for r in ok_results if r.quadrant == q]
        if not rs:
            continue
        tool_rate = _rate([r.tool_selection_ok for r in rs])
        multi = _rate([r.multistep_ok for r in rs if r.multistep_ok is not None])
        cite = _rate([r.citation_ok for r in rs if r.citation_ok is not None])
        steps = f"{statistics.mean(r.iterations for r in rs):.1f}"
        toks = (
            f"{statistics.mean(r.input_tokens for r in rs):,.0f}"
            f" / {statistics.mean(r.output_tokens for r in rs):,.0f}"
        )
        costs = [r.cost_usd for r in rs if r.cost_usd is not None]
        cost = f"{statistics.mean(costs):.5f}" if costs else "—"
        lines.append(f"| {q} | {len(rs)} | {tool_rate} | {multi} | {cite} | {steps} | {toks} | {cost} |")

    lines.append("\n## 전체\n")
    lines.append(f"- 툴 선택 정확도: **{_rate([r.tool_selection_ok for r in ok_results])}**")
    multi_all = [r.multistep_ok for r in ok_results if r.multistep_ok is not None]
    cite_all = [r.citation_ok for r in ok_results if r.citation_ok is not None]
    lines.append(f"- 멀티스텝 성공률: **{_rate(multi_all)}**")
    lines.append(f"- 인용 형식 준수율: **{_rate(cite_all)}**")

    lines.append("\n## 실패 케이스\n")
    fails = [
        r for r in ok_results
        if not r.tool_selection_ok or r.multistep_ok is False or r.citation_ok is False
    ]
    if not fails:
        lines.append("(없음)")
    for r in fails:
        lines.append(
            f"- `{r.id}` called={r.called_tools} tool_ok={r.tool_selection_ok}"
            f" multistep={r.multistep_ok} citation={r.citation_ok} — {r.question[:80]}"
        )
    errors = [r for r in results if r.error]
    if errors:
        lines.append("\n## 오류\n")
        for r in errors:
            lines.append(f"- `{r.id}`: {r.error}")

    (OUTPUT_DIR / "agent_eval.md").write_text("\n".join(lines) + "\n")
    logger.info("Report written: %s", OUTPUT_DIR / "agent_eval.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="분면당 문항 수 제한")
    parser.add_argument("--quadrant", choices=QUADRANTS, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    from src.agent.graph.agent import build_agent_graph

    items = _load_dataset(args.limit, args.quadrant)
    logger.info("Running agent eval on %d items", len(items))
    graph = build_agent_graph()

    results: list[ItemResult] = []
    for i, item in enumerate(items, 1):
        logger.info("[%d/%d] %s: %s", i, len(items), item["id"], item["question"][:60])
        r = run_item(graph, item)
        logger.info(
            "  → tools=%s tool_ok=%s multistep=%s citation=%s (%.1fs)",
            r.called_tools, r.tool_selection_ok, r.multistep_ok, r.citation_ok, r.latency_s,
        )
        results.append(r)

    _write_report(results)

    ok = [r for r in results if not r.error]
    print(f"\n툴 선택 정확도: {_rate([r.tool_selection_ok for r in ok])}")
    print(f"결과: {OUTPUT_DIR / 'agent_eval.md'}")


if __name__ == "__main__":
    main()
