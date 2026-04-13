"""Streamlit 데모 UI — Phase 5.

FastAPI 백엔드(`src/api/main.py`)의 `/rag/qa` 와 `/agent/chat` 을 호출하는
순수 프론트엔드. RAG/에이전트 로직을 직접 import 하지 않아 컨테이너 분리가 깔끔하다.

두 탭:
    1. RAG QA — retrieval 설정 + 답변 + 검색 청크
    2. Agent — ReAct 에이전트 멀티턴 채팅 + tool trace

환경 변수:
    API_BASE_URL  — 기본 http://localhost:8000

실행:
    uv run streamlit run src/ui/app.py
"""

from __future__ import annotations

import os

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
if "agent_messages" not in st.session_state:
    # 클라이언트 측 대화 history (API 가 stateless 이므로 매번 전송)
    st.session_state.agent_messages = []  # list[dict] {role, content, name?, tool_calls?}
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = []


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
        with st.sidebar:
            st.divider()
            st.header("📎 Retrieved chunks")
            for i, src in enumerate(st.session_state.rag_result["sources"], 1):
                with st.expander(f"[{i}] {src['source_path']}"):
                    st.text(src["snippet"])


# =============== AGENT TAB ===============
with tab_agent:
    with st.sidebar:
        st.header("Agent")
        st.caption("4 tools: docs_search · code_generate · web_search · error_analyze")
        if st.button("🧹 Clear chat", key="agent_clear"):
            st.session_state.agent_messages = []
            st.session_state.agent_trace = []
            st.rerun()

    # render history (user + final assistant 메시지만)
    for m in st.session_state.agent_messages:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        elif m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
            with st.chat_message("assistant"):
                st.markdown(m["content"])

    user_input = st.chat_input("에이전트에게 물어보세요…")
    if user_input:
        st.session_state.agent_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.spinner("Agent thinking…"):
                r = httpx.post(
                    f"{API_BASE_URL}/agent/chat",
                    json={"messages": st.session_state.agent_messages},
                    timeout=HTTP_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            st.error(f"Agent API 호출 실패: {e}")
            data = None

        if data:
            # 서버가 돌려준 full history 로 교체 (tool 메시지 포함)
            st.session_state.agent_messages = data["messages"]
            st.session_state.agent_trace = data["trace"]
            with st.chat_message("assistant"):
                st.markdown(data["answer"])

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
