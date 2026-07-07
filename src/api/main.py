"""FastAPI 서버 — RAG QA + ReAct 에이전트 엔드포인트.

엔드포인트:
    GET  /health        — 헬스체크
    POST /rag/qa        — 1-turn RAG QA. retrieval 설정 전환 가능.
    POST /agent/chat    — stateless 멀티턴. 클라이언트가 message history 를 보낸다.

실행:
    uv run uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.agent.security.input_guard import check_user_input
from src.api.deps import (
    get_cached_agent_graph,
    get_cached_retriever,
    get_cached_thread_agent_graph,
)
from src.api.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentThreadChatRequest,
    AgentThreadChatResponse,
    AgentThreadHistoryResponse,
    AgentTraceStep,
    ChatMessage,
    RAGQARequest,
    RAGQAResponse,
    RAGSource,
    UsageInfo,
)
from src.rag.generation.chain import RAGChain
from src.rag.generation.llm import get_llm
from src.rag.generation.usage import UsageReport, log_usage, usage_from_messages

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Docs Assistant API",
    description="LangChain/LangGraph 문서 RAG QA + ReAct 에이전트",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _to_usage_info(report: UsageReport | None) -> UsageInfo | None:
    if report is None or report.llm_calls == 0:
        return None
    return UsageInfo(**report.to_dict())


# ─────────── RAG QA ───────────

@app.post("/rag/qa", response_model=RAGQAResponse)
def rag_qa(req: RAGQARequest) -> RAGQAResponse:
    guard = check_user_input(req.question)
    if guard.blocked:
        return RAGQAResponse(answer=guard.message, sources=[])

    try:
        retriever = get_cached_retriever(
            req.strategy, req.embedding_model, req.method, req.top_k
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Retriever build failed")
        raise HTTPException(status_code=500, detail=f"retriever: {e}") from e

    try:
        chain = RAGChain(retriever=retriever, llm=get_llm())
        result = chain.invoke(req.question)
    except Exception as e:  # noqa: BLE001
        logger.exception("RAG chain failed")
        raise HTTPException(status_code=500, detail=f"chain: {e}") from e

    sources = [
        RAGSource(
            source_path=str(d.metadata.get("source_path", "unknown")),
            snippet=d.page_content[:600],
        )
        for d in result.sources
    ]
    return RAGQAResponse(
        answer=result.answer, sources=sources, usage=_to_usage_info(result.usage)
    )


# ─────────── Agent ───────────

def _to_lc_messages(msgs: list[ChatMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in msgs:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(
                AIMessage(content=m.content, tool_calls=m.tool_calls or [])
            )
        elif m.role == "tool":
            out.append(
                ToolMessage(content=m.content, name=m.name or "tool", tool_call_id=m.name or "tc")
            )
        elif m.role == "system":
            out.append(SystemMessage(content=m.content))
    return out


def _from_lc_messages(msgs: list[BaseMessage]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in msgs:
        if isinstance(m, HumanMessage):
            out.append(ChatMessage(role="user", content=str(m.content)))
        elif isinstance(m, AIMessage):
            out.append(
                ChatMessage(
                    role="assistant",
                    content=str(m.content or ""),
                    tool_calls=list(m.tool_calls) if m.tool_calls else None,
                )
            )
        elif isinstance(m, ToolMessage):
            out.append(
                ChatMessage(role="tool", content=str(m.content), name=m.name)
            )
        elif isinstance(m, SystemMessage):
            out.append(ChatMessage(role="system", content=str(m.content)))
    return out


def _build_trace(msgs: list[BaseMessage]) -> list[AgentTraceStep]:
    """이번 턴에 새로 생긴 메시지에서 tool 호출/결과 trace를 추출."""
    trace: list[AgentTraceStep] = []
    for m in msgs:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                trace.append(
                    AgentTraceStep(
                        kind="tool_call",
                        name=tc["name"],
                        payload=json.dumps(tc.get("args", {}), ensure_ascii=False),
                    )
                )
        elif isinstance(m, ToolMessage):
            trace.append(
                AgentTraceStep(
                    kind="tool_result",
                    name=m.name or "tool",
                    payload=str(m.content)[:800],
                )
            )
    return trace


def _final_answer(msgs: list[BaseMessage]) -> str:
    return next(
        (
            str(m.content)
            for m in reversed(msgs)
            if isinstance(m, AIMessage) and not m.tool_calls and m.content
        ),
        "(no answer)",
    )


@app.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(req: AgentChatRequest) -> AgentChatResponse:
    graph = get_cached_agent_graph()
    lc_msgs = _to_lc_messages(req.messages)

    # 가장 마지막 user message 를 current_query 로
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        "",
    )

    guard = check_user_input(last_user)
    if guard.blocked:
        return AgentChatResponse(
            answer=guard.message, trace=[], messages=req.messages, iterations=0
        )

    initial_state = {
        "messages": lc_msgs,
        "tool_results": [],
        "current_query": last_user,
        "iteration_count": 0,
    }

    try:
        final = graph.invoke(initial_state)
    except Exception as e:  # noqa: BLE001
        logger.exception("Agent invoke failed")
        raise HTTPException(status_code=500, detail=f"agent: {e}") from e

    new_msgs = final["messages"][len(lc_msgs):]

    trace = _build_trace(new_msgs)
    answer = _final_answer(final["messages"])

    usage = usage_from_messages(new_msgs)
    log_usage("agent_chat", usage)

    return AgentChatResponse(
        answer=answer,
        trace=trace,
        messages=_from_lc_messages(final["messages"]),
        iterations=final.get("iteration_count", 0),
        usage=_to_usage_info(usage),
    )


# ─────────── Agent (thread 기반 영속 대화) ───────────

@app.post("/agent/thread/chat", response_model=AgentThreadChatResponse)
def agent_thread_chat(req: AgentThreadChatRequest) -> AgentThreadChatResponse:
    """서버측 메모리 버전 — 히스토리는 checkpointer(SqliteSaver)가 관리한다.

    클라이언트는 `thread_id` + 새 메시지 1건만 보낸다. 같은 thread_id로 다시
    호출하면 서버가 저장된 대화에 이어서 응답한다 (서버 재시작에도 유지).
    """
    guard = check_user_input(req.message)
    if guard.blocked:
        return AgentThreadChatResponse(
            thread_id=req.thread_id, answer=guard.message, trace=[], iterations=0
        )

    graph = get_cached_thread_agent_graph()
    config = {"configurable": {"thread_id": req.thread_id}}

    # 이번 턴에 새로 생긴 메시지만 slice 하기 위해 이전 길이를 기록
    prev_state = graph.get_state(config)
    prev_len = len(prev_state.values.get("messages", [])) if prev_state.values else 0

    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "tool_results": [],
        "current_query": req.message,
        "iteration_count": 0,  # 턴마다 tool 루프 예산 리셋
    }

    try:
        final = graph.invoke(initial_state, config)
    except Exception as e:  # noqa: BLE001
        logger.exception("Agent thread invoke failed")
        raise HTTPException(status_code=500, detail=f"agent: {e}") from e

    new_msgs = final["messages"][prev_len + 1:]  # +1 = 이번 턴 user message

    usage = usage_from_messages(new_msgs)
    log_usage("agent_thread_chat", usage)

    return AgentThreadChatResponse(
        thread_id=req.thread_id,
        answer=_final_answer(final["messages"]),
        trace=_build_trace(new_msgs),
        iterations=final.get("iteration_count", 0),
        usage=_to_usage_info(usage),
    )


def _chunk_text(chunk: AIMessageChunk) -> str:
    """토큰 델타에서 텍스트만 추출 (Bedrock은 content가 블록 리스트일 수 있음)."""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return ""


def agent_stream_events(graph, initial_state: dict, config: dict) -> Iterator[dict]:
    """그래프 실행을 SSE 이벤트(dict)로 변환하는 제너레이터.

    이벤트 타입:
        token       — LLM 토큰 델타 {text}
        tool_call   — 도구 호출 시작 {name, args}
        tool_result — 도구 실행 결과 {name, preview}
        done        — 종료 {answer, iterations, usage}
    """
    new_msgs: list[BaseMessage] = []
    iterations = 0
    for mode, payload in graph.stream(
        initial_state, config, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            chunk, meta = payload
            if (
                isinstance(chunk, AIMessageChunk)
                and meta.get("langgraph_node") == "agent"
            ):
                text = _chunk_text(chunk)
                if text:
                    yield {"type": "token", "text": text}
        else:  # updates — 노드 단위 산출물 (완성된 메시지)
            for update in payload.values():
                if not update:
                    continue
                iterations = update.get("iteration_count", iterations)
                for m in update.get("messages", []):
                    new_msgs.append(m)
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            yield {
                                "type": "tool_call",
                                "name": tc["name"],
                                "args": tc.get("args", {}),
                            }
                    elif isinstance(m, ToolMessage):
                        yield {
                            "type": "tool_result",
                            "name": m.name or "tool",
                            "preview": str(m.content)[:500],
                        }

    usage = usage_from_messages(new_msgs)
    log_usage("agent_thread_stream", usage)
    yield {
        "type": "done",
        "answer": _final_answer(new_msgs),
        "iterations": iterations,
        "usage": usage.to_dict() if usage.llm_calls else None,
    }


@app.post("/agent/thread/stream")
def agent_thread_stream(req: AgentThreadChatRequest) -> StreamingResponse:
    """SSE 스트리밍 버전 — 토큰 델타 + 도구 호출 이벤트를 실시간 전송.

    이벤트는 `data: {json}\\n\\n` 라인으로 흐르고 마지막에 type=done 이 온다.
    """
    guard = check_user_input(req.message)

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    if guard.blocked:
        def blocked_stream() -> Iterator[str]:
            yield sse({"type": "token", "text": guard.message})
            yield sse({"type": "done", "answer": guard.message, "iterations": 0, "usage": None})

        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    graph = get_cached_thread_agent_graph()
    config = {"configurable": {"thread_id": req.thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "tool_results": [],
        "current_query": req.message,
        "iteration_count": 0,
    }

    def event_stream() -> Iterator[str]:
        try:
            for event in agent_stream_events(graph, initial_state, config):
                yield sse(event)
        except Exception as e:  # noqa: BLE001
            logger.exception("Agent stream failed")
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/agent/thread/{thread_id}/history", response_model=AgentThreadHistoryResponse)
def agent_thread_history(thread_id: str) -> AgentThreadHistoryResponse:
    """저장된 thread의 대화 히스토리 조회 (UI 이어하기용)."""
    graph = get_cached_thread_agent_graph()
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state.values else []
    return AgentThreadHistoryResponse(
        thread_id=thread_id, messages=_from_lc_messages(messages)
    )
