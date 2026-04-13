"""CLI QA 데모 — 기본 컬렉션(`recursive × bge-m3`)으로 질문에 답한다.

사용:
    uv run python -m src.rag.generation.cli "How do I create a custom tool in LangChain?"
    uv run python -m src.rag.generation.cli --strategy markdown --top-k 8 "..."
    uv run python -m src.rag.generation.cli --model BAAI/bge-large-en-v1.5 "..."
"""

from __future__ import annotations

import argparse
import logging

from src.config import settings
from src.rag.embedding.run import STRATEGIES
from src.rag.generation.chain import build_rag_chain

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG QA CLI")
    parser.add_argument("question", help="질문")
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    parser.add_argument(
        "--provider", default=None, help="임베딩 프로바이더 (기본: settings)"
    )
    parser.add_argument(
        "--model", default=None, help="임베딩 모델 (기본: settings.embedding_model)"
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    chain = build_rag_chain(
        strategy=args.strategy,
        embedding_provider=args.provider,
        embedding_model=args.model,
        top_k=args.top_k,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature,
    )

    result = chain.invoke(args.question)

    print("\n=== Question ===")
    print(result.question)
    print("\n=== Answer ===")
    print(result.answer)
    print("\n=== Retrieved sources ===")
    for i, d in enumerate(result.sources, 1):
        src = d.metadata.get("source_path", "unknown")
        preview = d.page_content.replace("\n", " ")[:140]
        print(f"[{i}] {src}")
        print(f"    {preview}...")


if __name__ == "__main__":
    main()
