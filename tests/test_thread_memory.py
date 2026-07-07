"""thread 기반 서버측 대화 메모리 테스트 — SqliteSaver 영속화 검증 (LLM 호출 없음)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.graph.agent import build_agent_graph


class _FakeToolModel(GenericFakeChatModel):
    """bind_tools를 no-op으로 — 그래프 조립 테스트용."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


@tool
def dummy_tool(query: str) -> str:
    """A dummy tool for graph wiring tests."""
    return "ok"


def _graph(db_path: Path, answers: list[str]):
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    llm = _FakeToolModel(messages=iter([AIMessage(content=a) for a in answers]))
    return build_agent_graph(
        llm=llm, tools=[dummy_tool], checkpointer=SqliteSaver(conn)
    )


def _state(msg: str) -> dict:
    return {
        "messages": [HumanMessage(content=msg)],
        "tool_results": [],
        "current_query": msg,
        "iteration_count": 0,
    }


def test_thread_accumulates_history(tmp_path: Path):
    graph = _graph(tmp_path / "ckpt.sqlite", ["first answer", "second answer"])
    cfg = {"configurable": {"thread_id": "t1"}}

    final1 = graph.invoke(_state("hello"), cfg)
    assert len(final1["messages"]) == 2  # user + ai

    final2 = graph.invoke(_state("again"), cfg)
    contents = [m.content for m in final2["messages"]]
    assert contents == ["hello", "first answer", "again", "second answer"]


def test_threads_are_isolated(tmp_path: Path):
    graph = _graph(tmp_path / "ckpt.sqlite", ["a1", "a2"])
    graph.invoke(_state("thread one msg"), {"configurable": {"thread_id": "t1"}})
    final = graph.invoke(_state("thread two msg"), {"configurable": {"thread_id": "t2"}})
    contents = [m.content for m in final["messages"]]
    assert contents == ["thread two msg", "a2"]


def test_history_survives_restart(tmp_path: Path):
    """새 커넥션(서버 재시작 시뮬레이션)으로도 thread state가 복원돼야 한다."""
    db = tmp_path / "ckpt.sqlite"
    cfg = {"configurable": {"thread_id": "persist"}}

    graph1 = _graph(db, ["turn1 answer"])
    graph1.invoke(_state("remember me"), cfg)

    graph2 = _graph(db, ["turn2 answer"])  # 완전히 새로운 그래프/커넥션
    state = graph2.get_state(cfg)
    contents = [m.content for m in state.values["messages"]]
    assert contents == ["remember me", "turn1 answer"]

    final = graph2.invoke(_state("and continue"), cfg)
    assert [m.content for m in final["messages"]] == [
        "remember me", "turn1 answer", "and continue", "turn2 answer",
    ]
