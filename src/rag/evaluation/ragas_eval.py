"""RAGAS 기반 generation 품질 평가 — Phase 3 최종 단계.

비교군: `recursive × {bge-m3, bge-large-en-v1.5}` × `{dense, hybrid_rerank}` = 4 구성

지표 (ragas 0.4):
    - faithfulness        : 답변이 context 로 뒷받침되는가 (환각 방지)
    - answer_relevancy    : 답변이 질문에 대답하는가
    - context_recall      : ground-truth 답변을 도출할 수 있을 만큼 context 가 충분한가
    - context_precision   : 검색된 context 중 실제로 필요한 것의 비율

판정자(LLM/Embeddings): OpenAI `gpt-4o-mini` + `text-embedding-3-small`
    - 로컬 LLM/임베더(generation 측) 와 분리해서 독립성 확보
    - 판정 비용은 작음 (전체 수백 건 호출)

결과 저장: `experiments/ragas_eval.{json,md}`
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import settings
from src.rag.embedding.embedder import model_slug
from src.rag.evaluation.retrieval_eval import DEFAULT_QA_FILE, load_qa
from src.rag.generation.chain import RAGChain
from src.rag.generation.llm import get_llm
from src.rag.retrieval.hybrid import get_hybrid_retriever
from src.rag.retrieval.rerank import DEFAULT_RERANKER, wrap_with_reranker
from src.rag.retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)

EXP_DIR = Path("./experiments")
JSON_OUT = EXP_DIR / "ragas_eval.json"
MD_OUT = EXP_DIR / "ragas_eval.md"

DEFAULT_COMBOS = [
    ("recursive", "BAAI/bge-m3"),
    ("recursive", "BAAI/bge-large-en-v1.5"),
]
DEFAULT_METHODS = ("dense", "hybrid_rerank")
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


@dataclass
class ConfigResult:
    strategy: str
    model: str
    method: str
    n_questions: int
    scores: dict[str, float]  # metric_name -> mean score

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "model": self.model,
            "method": self.method,
            "n_questions": self.n_questions,
            "scores": self.scores,
        }


def build_chain(strategy: str, model: str, method: str, top_k: int) -> RAGChain:
    """dense / hybrid_rerank 용 RAGChain 생성."""
    llm = get_llm(provider="openai", temperature=0.0)
    if method == "dense":
        retriever = get_retriever(strategy=strategy, model=model, top_k=top_k)
    elif method == "hybrid_rerank":
        base = get_hybrid_retriever(strategy=strategy, model=model, k=20, dense_weight=0.5)
        retriever = wrap_with_reranker(base, reranker_model=DEFAULT_RERANKER, top_n=top_k)
    else:
        raise ValueError(f"Unknown method: {method}")
    return RAGChain(retriever=retriever, llm=llm)


def generate_samples(chain: RAGChain, qa_records: list[dict], label: str) -> list[dict]:
    """RAG 체인으로 답변 생성 → RAGAS 입력 포맷."""
    samples: list[dict] = []
    for qa in tqdm(qa_records, desc=f"gen[{label}]", leave=False):
        result = chain.invoke(qa["question"])
        samples.append(
            {
                "user_input": qa["question"],
                "retrieved_contexts": [d.page_content for d in result.sources],
                "response": result.answer,
                "reference": qa["ground_truth_answer"],
            }
        )
    return samples


def run_ragas(samples: list[dict]) -> dict[str, float]:
    """RAGAS 4지표 평균을 반환."""
    # Lazy import — ragas 은 무거움
    import warnings

    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # 주의: ragas 0.4 의 `metrics.collections` 는 InstructorLLM 만 지원.
    # LangChain 래퍼와 호환되는 legacy `ragas.metrics` 경로를 의도적으로 사용.
    # deprecation 경고는 무시 (0.5 로 업그레이드할 때 collections 로 이사 예정).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0.0,
            max_retries=4,
            timeout=60.0,
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
            max_retries=4,
            request_timeout=60.0,
        )
    )

    # legacy 모듈식 메트릭 — LLM/embeddings 는 evaluate() 에서 주입
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_emb)

    # result 는 pandas-like — 각 지표 컬럼의 평균을 뽑는다
    df = result.to_pandas()
    scores: dict[str, float] = {}
    for name in METRIC_NAMES:
        # ragas 가 컬럼명을 일부 변형할 수 있어 fuzzy match
        col = next((c for c in df.columns if c.lower().replace("-", "_") == name), None)
        if col is None:
            logger.warning("Metric column '%s' not found in %s", name, list(df.columns))
            scores[name] = float("nan")
        else:
            scores[name] = float(df[col].mean())
    return scores


def fmt_table(results: list[ConfigResult]) -> list[str]:
    header = ["combo", "method"] + list(METRIC_NAMES)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in results:
        combo = f"{r.strategy} × {model_slug(r.model)}"
        cells = [combo, r.method]
        cells += [f"{r.scores.get(m, float('nan')):.3f}" for m in METRIC_NAMES]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def compute_deltas(results: list[ConfigResult]) -> list[str]:
    """combo 별 dense → hybrid_rerank 증분."""
    by_combo: dict[tuple[str, str], dict[str, ConfigResult]] = {}
    for r in results:
        by_combo.setdefault((r.strategy, r.model), {})[r.method] = r

    lines = ["## 증분 분석 (hybrid_rerank vs dense)\n"]
    header = ["combo"] + [f"Δ{m}" for m in METRIC_NAMES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for (strategy, model), methods in by_combo.items():
        base = methods.get("dense")
        opt = methods.get("hybrid_rerank")
        if base is None or opt is None:
            continue
        combo = f"{strategy} × {model_slug(model)}"
        cells = [combo]
        for m in METRIC_NAMES:
            d = opt.scores.get(m, float("nan")) - base.scores.get(m, float("nan"))
            cells.append(f"{d:+.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_markdown(results: list[ConfigResult], *, n_questions: int, path: Path) -> None:
    lines: list[str] = []
    lines.append("# RAGAS Generation 품질 평가 — Phase 3\n")
    lines.append(f"- N questions: **{n_questions}**")
    lines.append("- 비교군: `recursive × {bge-m3, bge-large-en-v1.5}` × `{dense, hybrid_rerank}`")
    lines.append("- 판정자: `gpt-4o-mini` + `text-embedding-3-small`")
    lines.append(f"- 지표: {', '.join(METRIC_NAMES)}\n")

    lines.append("## 전체 결과\n")
    lines += fmt_table(results)
    lines.append("")
    lines += compute_deltas(results)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_console(results: list[ConfigResult]) -> None:
    print("\n=== RAGAS Evaluation ===")
    header = f"{'combo':<32s} {'method':<15s}"
    for m in METRIC_NAMES:
        header += f" {m[:14]:<14s}"
    print(header)
    print("-" * len(header))
    for r in results:
        combo = f"{r.strategy} × {model_slug(r.model)}"
        row = f"{combo:<32s} {r.method:<15s}"
        for m in METRIC_NAMES:
            row += f" {r.scores.get(m, float('nan')):<14.3f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS generation quality evaluation")
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument("--limit", type=int, default=None, help="앞 N개 QA 만 사용 (시험용)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--methods", nargs="*", choices=list(DEFAULT_METHODS), default=list(DEFAULT_METHODS)
    )
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=MD_OUT)
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    qa_records = load_qa(args.qa_file)
    if args.limit:
        qa_records = qa_records[: args.limit]
    logger.info("Using %d QA records", len(qa_records))

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for RAGAS judge + generation")

    results: list[ConfigResult] = []
    for strategy, model in DEFAULT_COMBOS:
        for method in args.methods:
            label = f"{strategy}×{model_slug(model)}/{method}"
            logger.info("=== Running [%s] ===", label)
            try:
                chain = build_chain(strategy, model, method, args.top_k)
                samples = generate_samples(chain, qa_records, label)
                logger.info("[%s] generated %d samples, running RAGAS...", label, len(samples))
                scores = run_ragas(samples)
                results.append(
                    ConfigResult(
                        strategy=strategy,
                        model=model,
                        method=method,
                        n_questions=len(qa_records),
                        scores=scores,
                    )
                )
                logger.info("[%s] scores=%s", label, scores)
            except Exception as e:  # noqa: BLE001
                logger.exception("FAILED [%s]: %s", label, e)

    if not results:
        logger.error("No RAGAS results")
        return

    print_console(results)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_questions": results[0].n_questions,
                "metrics": list(METRIC_NAMES),
                "results": [r.to_dict() for r in results],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_markdown(results, n_questions=results[0].n_questions, path=args.md_out)
    print(f"\nSaved: {args.json_out}")
    print(f"Saved: {args.md_out}")


if __name__ == "__main__":
    main()
