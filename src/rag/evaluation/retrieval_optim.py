"""Retrieval 최적화 비교 — dense vs hybrid vs hybrid+rerank.

Phase 3 후반. baseline retrieval_eval.py 가 4×2 매트릭스를 훑었다면,
이 스크립트는 **메인 비교군 (`recursive × {bge-m3, bge-large-en-v1.5}`)** 에
각 최적화 기법을 순차 적용하여 **전후 지표 변화** 를 측정한다.

비교 대상 (method):
    1. dense          : baseline — Chroma cosine top-k (retrieval_eval.py 와 동일)
    2. hybrid         : BM25(sparse) + dense ensemble (RRF, 0.5/0.5)
    3. hybrid+rerank  : hybrid 결과 위에 cross-encoder(`bge-reranker-v2-m3`) 적용

공정성:
    - 지표는 retrieval_eval.py 와 동일 (hit@k, MRR@k), 같은 QA 데이터셋(`data/eval/qa_dataset.jsonl`)
    - `fetch_k=20` 로 1차 후보를 넉넉히 뽑고 reranker 는 top_n=max_k 로 정렬 → baseline 의 max_k=10 과 직접 비교 가능
    - semantic 전략 제외 (grain/coverage 이슈는 problem_solving.md #6, #7 참조)

출력:
    - 콘솔 표
    - `experiments/retrieval_optim.json`
    - `experiments/retrieval_optim.md`

사용:
    uv run python -m src.rag.evaluation.retrieval_optim
    uv run python -m src.rag.evaluation.retrieval_optim --methods dense hybrid
    uv run python -m src.rag.evaluation.retrieval_optim --fetch-k 30 --dense-weight 0.6
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from langchain_core.documents import Document
from tqdm import tqdm

from src.config import settings
from src.rag.embedding.embedder import model_slug
from src.rag.evaluation.retrieval_eval import (
    DEFAULT_QA_FILE,
    _first_hit_rank,
    load_qa,
)
from src.rag.retrieval.hybrid import get_hybrid_retriever
from src.rag.retrieval.rerank import DEFAULT_RERANKER, wrap_with_reranker
from src.rag.retrieval.retriever import get_vectorstore

logger = logging.getLogger(__name__)

EXP_DIR = Path("./experiments")
JSON_OUTPUT = EXP_DIR / "retrieval_optim.json"
MD_OUTPUT = EXP_DIR / "retrieval_optim.md"

DEFAULT_KS = (3, 5, 10)
DEFAULT_COMBOS = [
    ("recursive", "BAAI/bge-m3"),
    ("recursive", "BAAI/bge-large-en-v1.5"),
]
DEFAULT_METHODS = ("dense", "hybrid", "hybrid_rerank")

Method = Literal["dense", "hybrid", "hybrid_rerank"]

RetrieveFn = Callable[[str], list[Document]]


@dataclass
class OptimResult:
    strategy: str
    model: str
    method: str
    n_questions: int
    hit_rate: dict[int, float] = field(default_factory=dict)
    mrr: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "model": self.model,
            "method": self.method,
            "n_questions": self.n_questions,
            "hit_rate": {str(k): v for k, v in self.hit_rate.items()},
            "mrr": {str(k): v for k, v in self.mrr.items()},
        }


def build_retrieve_fn(
    method: Method,
    *,
    strategy: str,
    model: str,
    max_k: int,
    fetch_k: int,
    dense_weight: float,
    reranker_model: str,
) -> RetrieveFn:
    """주어진 method 에 맞는 (question) -> list[Document] 함수 생성.

    method 별 fetch 전략:
        dense          : vs.similarity_search(q, k=max_k)
        hybrid         : ensemble 각 k=max_k, 결합 결과 [:max_k]
        hybrid_rerank  : ensemble 각 k=fetch_k, rerank top_n=max_k
    """
    if method == "dense":
        vs = get_vectorstore(strategy=strategy, model=model)

        def _fn(q: str) -> list[Document]:
            return vs.similarity_search(q, k=max_k)

        return _fn

    if method == "hybrid":
        retr = get_hybrid_retriever(
            strategy=strategy, model=model, k=max_k, dense_weight=dense_weight
        )

        def _fn(q: str) -> list[Document]:
            return retr.invoke(q)[:max_k]

        return _fn

    if method == "hybrid_rerank":
        base = get_hybrid_retriever(
            strategy=strategy, model=model, k=fetch_k, dense_weight=dense_weight
        )
        wrapped = wrap_with_reranker(base, reranker_model=reranker_model, top_n=max_k)

        def _fn(q: str) -> list[Document]:
            return wrapped.invoke(q)

        return _fn

    raise ValueError(f"Unknown method: {method}")


def evaluate(
    retrieve_fn: RetrieveFn,
    qa_records: list[dict],
    ks: tuple[int, ...],
    label: str,
) -> tuple[dict[int, float], dict[int, float]]:
    hits_at_k = {k: 0 for k in ks}
    rr_at_k = {k: 0.0 for k in ks}

    for qa in tqdm(qa_records, desc=label, leave=False):
        gt: set[str] = set(qa.get("ground_truth_contexts") or [])
        if not gt:
            continue
        docs = retrieve_fn(qa["question"])
        retrieved_sources = [d.metadata.get("source_path", "") for d in docs]
        rank = _first_hit_rank(retrieved_sources, gt)
        for k in ks:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
                rr_at_k[k] += 1.0 / rank

    n = len(qa_records)
    return (
        {k: hits_at_k[k] / n for k in ks},
        {k: rr_at_k[k] / n for k in ks},
    )


def fmt_table(results: list[OptimResult], ks: tuple[int, ...]) -> list[str]:
    """Markdown 표 라인 리스트."""
    header = ["combo", "method"]
    header += [f"hit@{k}" for k in ks]
    header += [f"MRR@{k}" for k in ks]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in results:
        combo = f"{r.strategy} × {model_slug(r.model)}"
        cells = [combo, r.method]
        cells += [f"{r.hit_rate[k]:.3f}" for k in ks]
        cells += [f"{r.mrr[k]:.3f}" for k in ks]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def compute_deltas(results: list[OptimResult], ks: tuple[int, ...]) -> list[str]:
    """combo 별 dense → hybrid → hybrid_rerank 증분 표."""
    by_combo: dict[tuple[str, str], dict[str, OptimResult]] = {}
    for r in results:
        by_combo.setdefault((r.strategy, r.model), {})[r.method] = r

    lines = ["## 증분 분석 (vs dense baseline)\n"]
    header = ["combo", "method"]
    for k in ks:
        header.append(f"Δhit@{k}")
    for k in ks:
        header.append(f"ΔMRR@{k}")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for (strategy, model), methods in by_combo.items():
        base = methods.get("dense")
        if base is None:
            continue
        for m_name in ("hybrid", "hybrid_rerank"):
            r = methods.get(m_name)
            if r is None:
                continue
            combo = f"{strategy} × {model_slug(model)}"
            cells = [combo, m_name]
            cells += [f"{r.hit_rate[k] - base.hit_rate[k]:+.3f}" for k in ks]
            cells += [f"{r.mrr[k] - base.mrr[k]:+.3f}" for k in ks]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_markdown(
    results: list[OptimResult],
    ks: tuple[int, ...],
    *,
    n_questions: int,
    fetch_k: int,
    dense_weight: float,
    reranker_model: str,
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Retrieval 최적화 비교 — Phase 3\n")
    lines.append(f"- N questions: **{n_questions}**")
    lines.append(f"- 메인 비교군: `recursive × {{bge-m3, bge-large-en-v1.5}}`")
    lines.append(f"- k values: {list(ks)}")
    lines.append(f"- Hybrid: BM25 + Chroma dense (EnsembleRetriever, dense_weight={dense_weight})")
    lines.append(f"- Reranker: `{reranker_model}` (fetch_k={fetch_k} → top_n=max_k)")
    lines.append(
        "- `fixed`, `markdown`, `semantic` 제외 — 사유는 `docs/portfolio/problem_solving.md` #6, #7\n"
    )

    lines.append("## 전체 결과\n")
    lines += fmt_table(results, ks)
    lines.append("")
    lines += compute_deltas(results, ks)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_table(results: list[OptimResult], ks: tuple[int, ...]) -> None:
    print("\n=== Retrieval Optimization ===")
    header = f"{'combo':<32s} {'method':<15s}"
    for k in ks:
        header += f" hit@{k:<3d}"
    for k in ks:
        header += f" MRR@{k:<3d}"
    print(header)
    print("-" * len(header))
    for r in results:
        combo = f"{r.strategy} × {model_slug(r.model)}"
        row = f"{combo:<32s} {r.method:<15s}"
        for k in ks:
            row += f" {r.hit_rate[k]:<6.3f}"
        for k in ks:
            row += f" {r.mrr[k]:<6.3f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval optimization comparison")
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument(
        "--methods", nargs="*", choices=list(DEFAULT_METHODS), default=list(DEFAULT_METHODS)
    )
    parser.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS))
    parser.add_argument(
        "--fetch-k", type=int, default=20, help="rerank 1차 후보 개수 (기본 20)"
    )
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--json-out", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--md-out", type=Path, default=MD_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    qa_records = load_qa(args.qa_file)
    logger.info("Loaded %d QA records", len(qa_records))

    ks = tuple(sorted(set(args.ks)))
    max_k = max(ks)

    results: list[OptimResult] = []
    for strategy, model in DEFAULT_COMBOS:
        for method in args.methods:
            label = f"{strategy}×{model_slug(model)}/{method}"
            logger.info("Running [%s]", label)
            try:
                fn = build_retrieve_fn(
                    method,  # type: ignore[arg-type]
                    strategy=strategy,
                    model=model,
                    max_k=max_k,
                    fetch_k=args.fetch_k,
                    dense_weight=args.dense_weight,
                    reranker_model=args.reranker_model,
                )
                hr, mrr = evaluate(fn, qa_records, ks, label)
                results.append(
                    OptimResult(
                        strategy=strategy,
                        model=model,
                        method=method,
                        n_questions=len(qa_records),
                        hit_rate=hr,
                        mrr=mrr,
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("FAILED [%s]: %s", label, e)

    if not results:
        logger.error("No results")
        return

    print_console_table(results, ks)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_questions": results[0].n_questions,
                "ks": list(ks),
                "fetch_k": args.fetch_k,
                "dense_weight": args.dense_weight,
                "reranker_model": args.reranker_model,
                "results": [r.to_dict() for r in results],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_markdown(
        results,
        ks,
        n_questions=results[0].n_questions,
        fetch_k=args.fetch_k,
        dense_weight=args.dense_weight,
        reranker_model=args.reranker_model,
        path=args.md_out,
    )
    print(f"\nSaved: {args.json_out}")
    print(f"Saved: {args.md_out}")


if __name__ == "__main__":
    main()
