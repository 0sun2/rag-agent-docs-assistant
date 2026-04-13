"""평가용 QA 데이터셋 초안 생성기 (Phase 3).

접근:
    1) 큐레이션된 핵심 개념 문서 목록에서 파일을 읽음
    2) 파일당 structured-output LLM 호출로 3개 QA 생성 (factual/howto/compare)
    3) `data/eval/qa_dataset.jsonl` 에 누적 저장 (idempotent: 이미 있으면 덮어씀)
    4) 사용자는 수동 검증으로 불량/중복 QA 제거

설계 결정:
    - `ground_truth_contexts` 는 source 파일 경로 리스트로 저장한다.
      청크 단위가 아닌 파일 단위 → 4개 chunking 전략 비교에서 공정한 grain.
    - LLM 은 `gpt-4o-mini` 로 초안만 만들고, 품질 통제는 사람이 한다.
    - 큐레이션 파일 목록은 `CURATED_FILES` 로 상수화 — 재현 가능.
    - `--limit N` 으로 일부만 먼저 돌려 품질 점검 후 전체 실행.

사용:
    uv run python -m src.rag.evaluation.build_dataset --limit 3        # 시험
    uv run python -m src.rag.evaluation.build_dataset                  # 전체
    uv run python -m src.rag.evaluation.build_dataset --qas-per-file 2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.config import settings
from src.rag.generation.llm import get_llm

logger = logging.getLogger(__name__)

DOC_ROOT = Path(settings.crawl_output_dir)  # data/raw/langchain
EVAL_DIR = Path("./data/eval")
OUTPUT_FILE = EVAL_DIR / "qa_dataset.jsonl"

Difficulty = Literal["factual", "howto", "compare"]

# 큐레이션: LangChain/LangGraph/DeepAgents 의 핵심 개념 문서.
# integrations/reference 는 세부 벤더별이라 제외. langchain/python 역시 통합 문서 위주라 제외.
CURATED_FILES: list[str] = [
    # --- langchain core concepts ---
    "oss/langchain/overview.mdx",
    "oss/langchain/agents.mdx",
    "oss/langchain/tools.mdx",
    "oss/langchain/models.mdx",
    "oss/langchain/messages.mdx",
    "oss/langchain/structured-output.mdx",
    "oss/langchain/streaming.mdx",
    "oss/langchain/rag.mdx",
    "oss/langchain/retrieval.mdx",
    "oss/langchain/short-term-memory.mdx",
    "oss/langchain/long-term-memory.mdx",
    "oss/langchain/human-in-the-loop.mdx",
    "oss/langchain/context-engineering.mdx",
    "oss/langchain/mcp.mdx",
    # --- langgraph core concepts ---
    "oss/langgraph/overview.mdx",
    "oss/langgraph/graph-api.mdx",
    "oss/langgraph/functional-api.mdx",
    "oss/langgraph/persistence.mdx",
    "oss/langgraph/memory.mdx",
    "oss/langgraph/streaming.mdx",
    "oss/langgraph/interrupts.mdx",
    "oss/langgraph/durable-execution.mdx",
    "oss/langgraph/workflows-agents.mdx",
    "oss/langgraph/agentic-rag.mdx",
    "oss/langgraph/thinking-in-langgraph.mdx",
    # --- deepagents ---
    "oss/deepagents/overview.mdx",
    "oss/deepagents/subagents.mdx",
    "oss/deepagents/context-engineering.mdx",
]


class QAPair(BaseModel):
    """단일 QA 스펙."""

    question: str = Field(
        description="사용자가 묻는 구체적인 질문. 이 문서를 읽지 않았어도 자연스러운 질문 형태."
    )
    ground_truth_answer: str = Field(
        description="문서 내용만 근거로 한 간결하고 정확한 정답. 1~4문장."
    )
    difficulty: Difficulty = Field(
        description="factual | howto | compare"
    )


class QABatch(BaseModel):
    """파일당 생성되는 QA 묶음."""

    qa_pairs: list[QAPair]


SYSTEM_PROMPT = """You create evaluation QA pairs for a RAG benchmark over the LangChain / LangGraph / DeepAgents documentation.

Rules:
- Ground ALL questions and answers STRICTLY in the provided document content. Do not invent APIs, flags, or behaviors that are not in the text.
- Each question must be self-contained (do not reference "this document" or "above").
- Answers must be concise (1–4 sentences) and directly supported by the document.
- Produce exactly the requested number of QA pairs with the requested difficulty mix.

Difficulty definitions:
- factual: Asks for a specific fact, definition, or single API element stated in the doc.
- howto: Asks how to accomplish a task or follow a procedure described in the doc (may include a short code snippet in the answer if the doc shows one).
- compare: Asks about differences, trade-offs, or when to choose between two concepts/APIs/options that appear in the doc.

Write the questions from the perspective of a developer using the library, not the author."""

USER_PROMPT = """Source file: {source_path}

Generate {n} QA pairs with difficulty mix:
{mix}

Document content:
---
{content}
---"""


def _read_doc(rel_path: str, max_chars: int = 12000) -> str:
    path = DOC_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Curated file missing: {path}")
    text = path.read_text(encoding="utf-8")
    # 매우 긴 문서는 앞 12k 문자만 (대부분 langchain 문서는 이 안에 들어옴)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated ...]"
    return text


def _build_mix(qas_per_file: int) -> tuple[str, dict[Difficulty, int]]:
    """파일당 QA 수를 3가지 난이도로 분배."""
    if qas_per_file == 3:
        dist: dict[Difficulty, int] = {"factual": 1, "howto": 1, "compare": 1}
    elif qas_per_file == 2:
        dist = {"factual": 1, "howto": 1, "compare": 0}
    elif qas_per_file == 1:
        dist = {"factual": 0, "howto": 1, "compare": 0}
    else:
        # 4 이상: 비율 유지 (factual 30 / howto 40 / compare 30)
        f = max(1, round(qas_per_file * 0.3))
        h = max(1, round(qas_per_file * 0.4))
        c = max(0, qas_per_file - f - h)
        dist = {"factual": f, "howto": h, "compare": c}
    mix_str = ", ".join(f"{k}: {v}" for k, v in dist.items() if v > 0)
    return mix_str, dist


def generate_for_file(llm_structured, rel_path: str, qas_per_file: int) -> list[dict]:
    """파일 1개에 대해 LLM 에게 QA 초안을 요청."""
    content = _read_doc(rel_path)
    mix_str, dist = _build_mix(qas_per_file)

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_PROMPT.format(
                source_path=rel_path,
                n=qas_per_file,
                mix=mix_str,
                content=content,
            )
        ),
    ]
    batch: QABatch = llm_structured.invoke(messages)

    # difficulty 분포 검증 (다르면 경고만)
    got: dict[str, int] = {}
    for qa in batch.qa_pairs:
        got[qa.difficulty] = got.get(qa.difficulty, 0) + 1
    if got != {k: v for k, v in dist.items() if v > 0}:
        logger.warning("[%s] requested mix=%s, got=%s", rel_path, dist, got)

    records: list[dict] = []
    for qa in batch.qa_pairs:
        records.append(
            {
                "question": qa.question.strip(),
                "ground_truth_answer": qa.ground_truth_answer.strip(),
                "ground_truth_contexts": [rel_path],
                "difficulty": qa.difficulty,
                "source_file": rel_path,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QA evaluation dataset (draft)")
    parser.add_argument(
        "--limit", type=int, default=None, help="처음 N개 파일만 처리 (시험용)"
    )
    parser.add_argument(
        "--qas-per-file", type=int, default=3, help="파일당 QA 개수 (기본 3)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="출력 jsonl 경로 (기본: data/eval/qa_dataset.jsonl)",
    )
    parser.add_argument("--llm-model", default=None, help="LLM 모델 (기본: settings)")
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    files = CURATED_FILES[: args.limit] if args.limit else CURATED_FILES
    logger.info(
        "Building QA dataset: %d files × %d QAs = %d target",
        len(files),
        args.qas_per_file,
        len(files) * args.qas_per_file,
    )

    llm = get_llm(provider="openai", model=args.llm_model, temperature=0.3)
    llm_structured = llm.with_structured_output(QABatch)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for i, rel in enumerate(files, 1):
        try:
            recs = generate_for_file(llm_structured, rel, args.qas_per_file)
            all_records.extend(recs)
            logger.info("[%d/%d] %s → %d QAs", i, len(files), rel, len(recs))
        except Exception as e:  # noqa: BLE001
            logger.error("[%d/%d] %s FAILED: %s", i, len(files), rel, e)

    # id 부여
    for idx, rec in enumerate(all_records, 1):
        rec["id"] = f"qa_{idx:04d}"
        # id 가 앞에 오도록 재정렬
        rec_sorted = {"id": rec["id"], **{k: v for k, v in rec.items() if k != "id"}}
        all_records[idx - 1] = rec_sorted

    with args.output.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 요약
    by_diff: dict[str, int] = {}
    for rec in all_records:
        by_diff[rec["difficulty"]] = by_diff.get(rec["difficulty"], 0) + 1
    print("\n=== QA dataset built ===")
    print(f"  output     : {args.output}")
    print(f"  total QAs  : {len(all_records)}")
    print(f"  files used : {len(files)}")
    for d, n in sorted(by_diff.items()):
        pct = n / len(all_records) * 100 if all_records else 0
        print(f"  {d:8s} : {n:3d} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
