"""RAGAS 결과를 그룹 바차트로 시각화.

입력: experiments/ragas_eval.json
출력: docs/images/ragas.png

4 config × 4 metric 그룹 바. config 별 색 구분, metric 별 x 축 그룹.
텍스트 테이블보다 상대 비교가 한눈에 보이도록 히어로 구성 선호 강조.

실행:
    uv run python -m src.rag.evaluation.plot_ragas
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path("experiments/ragas_eval.json")
OUT = Path("docs/images/ragas.png")

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def _label(r: dict) -> str:
    model = "bge-m3" if "bge-m3" in r["model"] else "bge-large-en"
    return f"{model}\n{r['method']}"


def main() -> None:
    data = json.loads(SRC.read_text())
    results = data["results"]

    labels = [_label(r) for r in results]
    n_cfg = len(results)
    n_met = len(METRICS)

    x = np.arange(n_met)
    width = 0.8 / n_cfg

    fig, ax = plt.subplots(figsize=(11, 6))

    # 프로덕션 winner 강조 (bge-large-en × hybrid_rerank)
    def _is_winner(r: dict) -> bool:
        return "bge-large-en" in r["model"] and r["method"] == "hybrid_rerank"

    palette = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500"]
    for i, r in enumerate(results):
        vals = [r["scores"][m] for m in METRICS]
        offset = (i - (n_cfg - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=labels[i],
            color=palette[i % len(palette)],
            edgecolor="black" if _is_winner(r) else "none",
            linewidth=2 if _is_winner(r) else 0,
        )
        for b, v in zip(bars, vals, strict=False):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(
        f"RAGAS evaluation — {data['n_questions']} QA × 4 configs\n"
        "(recursive chunking, judge: gpt-4o-mini)",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
