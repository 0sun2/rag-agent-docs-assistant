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

from fastapi import FastAPI, HTTPException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.api.deps import get_cached_agent_graph, get_cached_retriever
from src.api.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentTraceStep,
    ChatMessage,
    RAGQARequest,
    RAGQAResponse,
    RAGSource,
)
from src.rag.generation.chain import RAGChain
from src.rag.generation.llm import get_llm

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Docs Assistant API",
    description="LangChain/LangGraph 문서 RAG QA + ReAct 에이전트",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ─────────── RAG QA ───────────

@app.post("/rag/qa", response_model=RAGQAResponse)
def rag_qa(req: RAGQARequest) -> RAGQAResponse:
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
    return RAGQAResponse(answer=result.answer, sources=sources)


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


@app.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(req: AgentChatRequest) -> AgentChatResponse:
    graph = get_cached_agent_graph()
    lc_msgs = _to_lc_messages(req.messages)

    # 가장 마지막 user message 를 current_query 로
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        "",
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

    trace: list[AgentTraceStep] = []
    for m in new_msgs:
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

    answer = next(
        (
            str(m.content)
            for m in reversed(final["messages"])
            if isinstance(m, AIMessage) and not m.tool_calls and m.content
        ),
        "(no answer)",
    )

    return AgentChatResponse(
        answer=answer,
        trace=trace,
        messages=_from_lc_messages(final["messages"]),
        iterations=final.get("iteration_count", 0),
    )
