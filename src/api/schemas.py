"""FastAPI request / response 스키마."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ─────────── Usage ───────────

class UsageInfo(BaseModel):
    """요청 1건의 토큰 사용량 + 추정 비용 (usage.UsageReport 미러)."""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    cost_usd: float | None = None
    cost_krw: float | None = None


# ─────────── RAG QA ───────────

class RAGQARequest(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: Literal["recursive", "markdown", "fixed"] = "recursive"
    embedding_model: Literal["bge-m3", "bge-large-en-v1.5"] = "bge-large-en-v1.5"
    method: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(5, ge=1, le=20)


class RAGSource(BaseModel):
    source_path: str
    snippet: str


class RAGQAResponse(BaseModel):
    answer: str
    sources: list[RAGSource]
    usage: UsageInfo | None = None


# ─────────── Agent ───────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    name: str | None = None  # tool name (role=tool 일 때)
    tool_calls: list[dict] | None = None  # assistant 가 호출한 tool


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


class AgentTraceStep(BaseModel):
    kind: Literal["tool_call", "tool_result"]
    name: str
    payload: str  # JSON-encoded args (tool_call) 또는 result preview (tool_result)


class AgentChatResponse(BaseModel):
    answer: str
    trace: list[AgentTraceStep]
    messages: list[ChatMessage]  # full updated history (클라이언트가 다음 턴에 그대로 보냄)
    iterations: int
    usage: UsageInfo | None = None
