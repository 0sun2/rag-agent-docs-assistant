"""Retrieval → Prompt → Generation 체인.

LCEL 로 얇게 구성. 답변과 함께 사용한 컨텍스트 청크(인용)를 같이 반환한다.
프롬프트에서 출처 명시를 강제 — 답변 마지막에 `[source_path]` 형식으로 인용하도록 지시.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from src.rag.generation.llm import get_llm
from src.rag.generation.usage import UsageReport, log_usage, usage_from_response
from src.rag.retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a precise technical assistant for LangChain / LangGraph documentation.
Answer the user's question using ONLY the provided context snippets.
If the context is insufficient, say so explicitly — do not invent APIs.
Include short code examples when the context contains them.
At the end of your answer, list the sources you used as bullet points in the form:
  - [source_path]
Use only sources that actually appeared in the context below."""

USER_PROMPT = """Question:
{question}

Context snippets (each prefixed with its source path):
{context}

Answer:"""


@dataclass
class QAResult:
    question: str
    answer: str
    sources: list[Document]
    usage: UsageReport | None = None


def _format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_path", "unknown")
        blocks.append(f"[{i}] source: {src}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )


class RAGChain:
    """Thin wrapper — retrieve → format → LLM → (answer, sources)."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt or build_prompt()

    def invoke(self, question: str) -> QAResult:
        docs = self.retriever.invoke(question)
        context = _format_context(docs)
        messages = self.prompt.format_messages(question=question, context=context)
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        usage = usage_from_response(response)
        log_usage("rag_qa", usage)
        return QAResult(question=question, answer=str(answer), sources=docs, usage=usage)


def build_rag_chain(
    *,
    strategy: str = "recursive",
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    top_k: int | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    temperature: float = 0.0,
) -> RAGChain:
    """편의 팩토리 — 기본 컬렉션은 `recursive × bge-m3`."""
    retriever = get_retriever(
        strategy=strategy,
        provider=embedding_provider,
        model=embedding_model,
        top_k=top_k,
    )
    llm = get_llm(provider=llm_provider, model=llm_model, temperature=temperature)
    return RAGChain(retriever=retriever, llm=llm)
