"""SSE 이벤트 변환 테스트 — fake LLM으로 token/done 이벤트 흐름 검증."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.graph.agent import build_agent_graph
from src.api.main import agent_stream_events


class _FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


@tool
def dummy_tool(query: str) -> str:
    """A dummy tool for stream tests."""
    return "ok"


def test_stream_yields_tokens_and_done(tmp_path: Path):
    conn = sqlite3.connect(str(tmp_path / "ckpt.sqlite"), check_same_thread=False)
    llm = _FakeToolModel(messages=iter([AIMessage(content="hello streamed world")]))
    graph = build_agent_graph(
        llm=llm, tools=[dummy_tool], checkpointer=SqliteSaver(conn)
    )

    initial = {
        "messages": [HumanMessage(content="hi")],
        "tool_results": [],
        "current_query": "hi",
        "iteration_count": 0,
    }
    events = list(
        agent_stream_events(graph, initial, {"configurable": {"thread_id": "s1"}})
    )

    types = [e["type"] for e in events]
    assert types[-1] == "done"
    assert "token" in types

    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == "hello streamed world"
    assert events[-1]["answer"] == "hello streamed world"
    assert events[-1]["iterations"] == 1
