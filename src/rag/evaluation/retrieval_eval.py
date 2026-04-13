"""Retrieval 성능 평가 — hit_rate@k, MRR@k.

평가 대상: (chunking_strategy × embedding_model) 의 8개 조합.
입력: `data/eval/qa_dataset.jsonl` (Phase 3 에서 만든 QA 초안)
출력: 콘솔 표 + `experiments/retrieval_eval.json` + `experiments/retrieval_eval.md`

정의:
    - hit@k : retrieved top-k 내에 ground-truth source_path 가 하나라도 있으면 1
    - MRR@k : 첫 hit 의 역순위(1/rank). top-k 안에 없으면 0
    - grain : 파일 경로(`source_path`) 단위. 청킹 전략이 달라도 공정하게 비교 가능

성능 최적화:
    - max_k(=기본 10)로 한 번만 retrieval → k=3,5,10 계산을 같은 결과에서 파생
    - 임베딩 모델은 조합당 한 번만 로드 (조합별로 새 커넥션 생성)

사용:
    uv run python -m src.rag.evaluation.retrieval_eval
    uv run python -m src.rag.evaluation.retrieval_eval --strategies recursive markdown
    uv run python -m src.rag.evaluation.retrieval_eval --models BAAI/bge-m3
    uv run python -m src.rag.evaluation.retrieval_eval --ks 1 3 5 10 20
    uv run python -m src.rag.evaluation.retrieval_eval --qa-file data/eval/qa_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import settings
from src.rag.embedding.embedder import model_slug
from src.rag.embedding.run import STRATEGIES
from src.rag.retrieval.retriever import get_vectorstore

logger = logging.getLogger(__name__)

DEFAULT_QA_FILE = Path("./data/eval/qa_dataset.jsonl")
EXP_DIR = Path("./experiments")
JSON_OUTPUT = EXP_DIR / "retrieval_eval.json"
MD_OUTPUT = EXP_DIR / "retrieval_eval.md"

DEFAULT_MODELS = ["BAAI/bge-m3", "BAAI/bge-large-en-v1.5"]
DEFAULT_KS = (3, 5, 10)


@dataclass
class ComboResult:
    strategy: str
    model: str
    n_questions: int
    hit_rate: dict[int, float] = field(default_factory=dict)
    mrr: dict[int, float] = field(default_factory=dict)
    # 난이도별 hit_rate@top_k (top_k = max(ks))
    hit_rate_by_difficulty: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "model": self.model,
            "n_questions": self.n_questions,
            "hit_rate": {str(k): v for k, v in self.hit_rate.items()},
            "mrr": {str(k): v for k, v in self.mrr.items()},
            "hit_rate_by_difficulty": self.hit_rate_by_difficulty,
        }


def load_qa(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"QA dataset not found: {path}. "
            "Run `python -m src.rag.evaluation.build_dataset` first."
        )
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"QA dataset empty: {path}")
    return records


def _first_hit_rank(retrieved_sources: list[str], gt_contexts: set[str]) -> int | None:
    """1-based rank of the first retrieved doc whose source_path ∈ gt_contexts.

    None if no hit.
    """
    for i, src in enumerate(retrieved_sources, 1):
        if src in gt_contexts:
            return i
    return None


def evaluate_combo(
    strategy: str,
    model: str,
    qa_records: list[dict],
    ks: tuple[int, ...],
    provider: str | None = None,
) -> ComboResult:
    """단일 (strategy, model) 조합 평가."""
    max_k = max(ks)

    vs = get_vectorstore(strategy=strategy, provider=provider, model=model)

    hits_at_k = {k: 0 for k in ks}
    reciprocal_ranks = {k: 0.0 for k in ks}
    # difficulty × hit@max_k 집계
    diff_counts: dict[str, int] = {}
    diff_hits: dict[str, int] = {}

    for qa in tqdm(qa_records, desc=f"{strategy} × {model_slug(model)}", leave=False):
        gt: set[str] = set(qa.get("ground_truth_contexts") or [])
        if not gt:
            logger.warning("QA %s has no ground_truth_contexts — skipping", qa.get("id"))
            continue

        docs = vs.similarity_search(qa["question"], k=max_k)
        retrieved_sources = [d.metadata.get("source_path", "") for d in docs]
        rank = _first_hit_rank(retrieved_sources, gt)

        for k in ks:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
                reciprocal_ranks[k] += 1.0 / rank

        diff = qa.get("difficulty", "unknown")
        diff_counts[diff] = diff_counts.get(diff, 0) + 1
        if rank is not None and rank <= max_k:
            diff_hits[diff] = diff_hits.get(diff, 0) + 1

    n = len(qa_records)
    result = ComboResult(
        strategy=strategy,
        model=model,
        n_questions=n,
        hit_rate={k: hits_at_k[k] / n for k in ks},
        mrr={k: reciprocal_ranks[k] / n for k in ks},
        hit_rate_by_difficulty={
            d: diff_hits.get(d, 0) / diff_counts[d] for d in diff_counts
        },
    )
    return result


def _fmt_row(label: str, r: ComboResult, ks: tuple[int, ...]) -> str:
    parts = [label]
    for k in ks:
        parts.append(f"{r.hit_rate[k]:.3f}")
    for k in ks:
        parts.append(f"{r.mrr[k]:.3f}")
    return " | ".join(parts)


def write_markdown(results: list[ComboResult], ks: tuple[int, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Retrieval Evaluation — Phase 3 Baseline\n")
    lines.append(f"- N questions: **{results[0].n_questions if results else 0}**")
    lines.append(f"- Metric grain: source file path (`source_path`)")
    lines.append(f"- k values: {list(ks)}")
    lines.append(f"- Retrieval method: cosine top-k (baseline, no MMR/hybrid/reranker)\n")

    # main table
    header = ["strategy × model"]
    header += [f"hit@{k}" for k in ks]
    header += [f"MRR@{k}" for k in ks]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in results:
        label = f"{r.strategy} × {model_slug(r.model)}"
        lines.append("| " + _fmt_row(label, r, ks) + " |")
    lines.append("")

    # difficulty breakdown at max_k
    max_k = max(ks)
    lines.append(f"## Hit-rate @ {max_k} by difficulty\n")
    # collect difficulty keys across results
    diff_keys: list[str] = []
    for r in results:
        for d in r.hit_rate_by_difficulty:
            if d not in diff_keys:
                diff_keys.append(d)
    diff_keys.sort()
    lines.append("| strategy × model | " + " | ".join(diff_keys) + " |")
    lines.append("|" + "|".join(["---"] * (len(diff_keys) + 1)) + "|")
    for r in results:
        label = f"{r.strategy} × {model_slug(r.model)}"
        cells = [f"{r.hit_rate_by_difficulty.get(d, 0.0):.3f}" for d in diff_keys]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_table(results: list[ComboResult], ks: tuple[int, ...]) -> None:
    print("\n=== Retrieval Evaluation ===")
    header = f"{'strategy × model':<40s}"
    for k in ks:
        header += f" hit@{k:<4d}"
    for k in ks:
        header += f" MRR@{k:<4d}"
    print(header)
    print("-" * len(header))
    for r in results:
        label = f"{r.strategy} × {model_slug(r.model)}"
        row = f"{label:<40s}"
        for k in ks:
            row += f" {r.hit_rate[k]:<7.3f}"
        for k in ks:
            row += f" {r.mrr[k]:<7.3f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation — hit_rate@k, MRR@k")
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument(
        "--strategies", nargs="*", choices=STRATEGIES, default=None,
        help="평가할 chunking 전략 (기본: 4개 전부)",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"평가할 임베딩 모델 (기본: {DEFAULT_MODELS})",
    )
    parser.add_argument("--provider", default=None, help="임베딩 provider (기본: settings)")
    parser.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--json-out", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--md-out", type=Path, default=MD_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    qa_records = load_qa(args.qa_file)
    logger.info("Loaded %d QA records from %s", len(qa_records), args.qa_file)

    strategies = args.strategies or list(STRATEGIES)
    models = args.models or list(DEFAULT_MODELS)
    ks = tuple(sorted(set(args.ks)))

    results: list[ComboResult] = []
    for model in models:
        for strategy in strategies:
            try:
                logger.info("Evaluating [%s × %s]", strategy, model)
                r = evaluate_combo(strategy, model, qa_records, ks, provider=args.provider)
                results.append(r)
            except Exception as e:  # noqa: BLE001
                logger.error("FAILED [%s × %s]: %s", strategy, model, e)

    if not results:
        logger.error("No results produced — check errors above")
        return

    print_console_table(results, ks)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_questions": results[0].n_questions,
                "ks": list(ks),
                "results": [r.to_dict() for r in results],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_markdown(results, ks, args.md_out)
    print(f"\nSaved: {args.json_out}")
    print(f"Saved: {args.md_out}")


if __name__ == "__main__":
    main()
