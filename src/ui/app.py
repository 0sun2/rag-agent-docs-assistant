"""Streamlit 데모 UI — Phase 5.

FastAPI 백엔드(`src/api/main.py`)를 호출하는 순수 프론트엔드.
RAG/에이전트 로직을 직접 import 하지 않아 컨테이너 분리가 깔끔하다.

두 탭:
    1. RAG QA — retrieval 설정 + 답변 + 검색 청크 + 토큰/비용
    2. Agent — thread 기반 영속 대화(서버측 checkpointer) + SSE 실시간 스트리밍
       (토큰 점진 렌더링, 도구 호출은 status 배지로 표시)

환경 변수:
    API_BASE_URL  — 기본 http://localhost:8000

실행:
    uv run streamlit run src/ui/app.py
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
HTTP_TIMEOUT = 120.0

st.set_page_config(page_title="LLM Docs Assistant", layout="wide")

EMBED_MODELS = ["bge-large-en-v1.5", "bge-m3"]
STRATEGIES = ["recursive", "markdown", "fixed"]
METHODS = ["dense", "hybrid", "hybrid_rerank"]


# ─────────── session state ───────────

if "rag_result" not in st.session_state:
    st.session_state.rag_result = None
if "thread_id" not in st.session_state:
    # 서버측 메모리(checkpointer)의 대화 키 — 같은 ID로 다시 접속하면 이어진다
    st.session_state.thread_id = uuid.uuid4().hex[:12]
if "agent_messages" not in st.session_state:
    # 화면 표시용 history (원본은 서버 checkpointer가 보관)
    st.session_state.agent_messages = []  # list[dict] {role, content}
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = []
if "agent_usage" not in st.session_state:
    st.session_state.agent_usage = None


def _api_health_ok() -> bool:
    try:
        r = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


# ─────────── header ───────────

st.title("🦜 LLM Docs Assistant")
st.caption(f"LangChain / LangGraph 공식 문서 RAG QA + ReAct 에이전트 · API: `{API_BASE_URL}`")

if not _api_health_ok():
    st.error(
        f"⚠️ API 서버에 연결할 수 없습니다 ({API_BASE_URL}). "
        "`uv run uvicorn src.api.main:app --port 8000` 또는 `docker compose up` 으로 실행하세요."
    )

tab_rag, tab_agent = st.tabs(["📚 RAG QA", "🤖 Agent"])


# =============== RAG QA TAB ===============
with tab_rag:
    with st.sidebar:
        st.header("RAG 설정")
        strategy = st.selectbox("Chunking", STRATEGIES, index=0, key="rag_strategy")
        embedding_model = st.selectbox("Embedding model", EMBED_MODELS, index=0, key="rag_model")
        method = st.selectbox(
            "Retrieval method", METHODS, index=2, key="rag_method",
            help="dense / BM25+dense hybrid / hybrid + cross-encoder rerank",
        )
        top_k = st.slider("top-k", 1, 10, 5, key="rag_topk")

    st.subheader("Ask a question about LangChain / LangGraph")
    question = st.text_input(
        "Question",
        placeholder="How do I create a custom tool in LangChain?",
        key="rag_question",
    )
    if st.button("🔎 Search & Answer", type="primary", key="rag_submit") and question:
        payload = {
            "question": question,
            "strategy": st.session_state.rag_strategy,
            "embedding_model": st.session_state.rag_model,
            "method": st.session_state.rag_method,
            "top_k": st.session_state.rag_topk,
        }
        try:
            with st.spinner("Calling /rag/qa…"):
                r = httpx.post(
                    f"{API_BASE_URL}/rag/qa", json=payload, timeout=HTTP_TIMEOUT
                )
                r.raise_for_status()
                st.session_state.rag_result = r.json()
        except httpx.HTTPError as e:
            st.error(f"API 호출 실패: {e}")
            st.session_state.rag_result = None

    if st.session_state.rag_result:
        st.markdown("### 답변")
        st.markdown(st.session_state.rag_result["answer"])
        usage = st.session_state.rag_result.get("usage")
        if usage:
            st.caption(
                f"🧮 {usage['model']} · in {usage['input_tokens']:,} / out {usage['output_tokens']:,} tok"
                f" · ${usage['cost_usd']} (~₩{usage['cost_krw']})"
                if usage.get("cost_usd") is not None
                else f"🧮 in {usage['input_tokens']:,} / out {usage['output_tokens']:,} tok"
            )
        with st.sidebar:
            st.divider()
            st.header("📎 Retrieved chunks")
            for i, src in enumerate(st.session_state.rag_result["sources"], 1):
                with st.expander(f"[{i}] {src['source_path']}"):
                    st.text(src["snippet"])


# =============== AGENT TAB ===============

def _load_thread_history(thread_id: str) -> list[dict]:
    """서버 checkpointer에 저장된 대화를 화면 표시용으로 변환."""
    r = httpx.get(
        f"{API_BASE_URL}/agent/thread/{thread_id}/history", timeout=HTTP_TIMEOUT
    )
    r.raise_for_status()
    out = []
    for m in r.json()["messages"]:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
            out.append({"role": "assistant", "content": m["content"]})
    return out


def _stream_agent_events(thread_id: str, message: str) -> Iterator[dict]:
    """SSE 라인을 파싱해 이벤트 dict를 순서대로 낸다."""
    with httpx.stream(
        "POST",
        f"{API_BASE_URL}/agent/thread/stream",
        json={"thread_id": thread_id, "message": message},
        timeout=HTTP_TIMEOUT,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: "):
                yield json.loads(line[len("data: "):])


with tab_agent:
    with st.sidebar:
        st.header("Agent")
        st.caption("4 tools: docs_search · code_generate · web_search · error_analyze")
        st.text_input("Thread ID", key="thread_id", help="같은 ID로 다시 접속하면 대화가 이어집니다 (서버측 메모리)")
        col_a, col_b = st.columns(2)
        if col_a.button("📂 불러오기", key="agent_load", use_container_width=True):
            try:
                st.session_state.agent_messages = _load_thread_history(
                    st.session_state.thread_id
                )
                st.session_state.agent_trace = []
            except httpx.HTTPError as e:
                st.error(f"히스토리 로드 실패: {e}")
            st.rerun()
        if col_b.button("✨ 새 세션", key="agent_new", use_container_width=True):
            st.session_state.pop("thread_id")
            st.session_state.agent_messages = []
            st.session_state.agent_trace = []
            st.session_state.agent_usage = None
            st.rerun()

    # render history (user + final assistant 메시지만)
    for m in st.session_state.agent_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("에이전트에게 물어보세요…")
    if user_input:
        st.session_state.agent_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        trace: list[dict] = []
        result: dict = {}
        status = st.status("🤖 Agent working…", expanded=True)

        def _token_gen() -> Iterator[str]:
            try:
                for evt in _stream_agent_events(st.session_state.thread_id, user_input):
                    if evt["type"] == "token":
                        yield evt["text"]
                    elif evt["type"] == "tool_call":
                        args = json.dumps(evt["args"], ensure_ascii=False)
                        status.write(f"🛠 **{evt['name']}** `{args[:200]}`")
                        trace.append({"kind": "tool_call", "name": evt["name"], "payload": args})
                    elif evt["type"] == "tool_result":
                        status.write(f"📥 **{evt['name']}** → {evt['preview'][:200]}…")
                        trace.append({"kind": "tool_result", "name": evt["name"], "payload": evt["preview"]})
                    elif evt["type"] == "done":
                        result.update(evt)
                    elif evt["type"] == "error":
                        st.error(f"Agent 오류: {evt['message']}")
            except httpx.HTTPError as e:
                st.error(f"Agent API 호출 실패: {e}")

        with st.chat_message("assistant"):
            streamed = st.write_stream(_token_gen())

        status.update(state="complete", label="✅ 완료", expanded=False)

        answer = result.get("answer") or (streamed if isinstance(streamed, str) else "")
        if answer:
            st.session_state.agent_messages.append({"role": "assistant", "content": answer})
        st.session_state.agent_trace = trace
        st.session_state.agent_usage = result.get("usage")

    if st.session_state.agent_usage:
        u = st.session_state.agent_usage
        cost = f" · ${u['cost_usd']} (~₩{u['cost_krw']})" if u.get("cost_usd") is not None else ""
        st.caption(
            f"🧮 {u['model']} · {u['llm_calls']} calls · in {u['input_tokens']:,} / out {u['output_tokens']:,} tok{cost}"
        )

    if st.session_state.agent_trace:
        with st.sidebar:
            st.divider()
            st.header("🔍 Last run trace")
            for step in st.session_state.agent_trace:
                if step["kind"] == "tool_call":
                    st.markdown(f"🛠 **{step['name']}**")
                    st.code(step["payload"], language="json")
                else:
                    st.markdown(f"📥 **{step['name']}** →")
                    st.code(step["payload"][:600])
