"""LangGraph 에이전트 상태 스키마.

ReAct 루프가 읽고 쓰는 전체 상태를 하나의 TypedDict 로 정의.
`messages` 는 LangGraph 의 `add_messages` reducer 로 누적되고, 나머지는 각 노드가 덮어쓰거나
리스트에 append 하는 형태로 관리된다.

필드:
    messages: 대화/툴 호출 이력 (HumanMessage, AIMessage, ToolMessage 혼재). LLM 의 바인딩된
        tool 호출 + ToolNode 의 실행 결과가 여기로 들어온다.
    tool_results: 각 툴 실행의 구조화된 결과 (디버깅/추적/Phase 5 데모용). messages 의
        ToolMessage 와 중복 정보지만, source 인용 메타데이터를 보존하기 위해 별도로 둠.
    current_query: 유저의 최근 질문 (멀티턴 확장 시 루프 내에서 참조).
    iteration_count: ReAct 루프 횟수. max 5 에서 강제 종료 (무한 tool 호출 방지).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ToolResult(TypedDict):
    """툴 한 번 실행의 구조화 결과."""

    tool: str
    input: dict[str, Any]
    output: Any


class AgentState(TypedDict):
    """ReAct 에이전트 전체 상태."""

    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: list[ToolResult]
    current_query: str
    iteration_count: int


MAX_ITERATIONS = 5
"""ReAct 루프 상한. 이 횟수에 도달하면 tool 호출 여부와 무관하게 종료."""
