# experiments/

Phase 2 ~ Phase 3 의 실험 산출물 모음. 각 실험은 **구조화 결과(`*.json`) + 사람용 리포트(`*.md`)**
두 짝으로 관리된다. 재현 스크립트는 `src/rag/evaluation/` 아래.

## 파일 목록

| 파일 | 내용 | 생성 스크립트 |
|---|---|---|
| `chunking_samples.md` | 4 전략(fixed/recursive/markdown/semantic) 청킹 샘플 — 경계 품질 육안 확인용 | `src/rag/chunking/run.py` |
| `retrieval_eval.{md,json}` | Phase 3 베이스라인 — 4 청킹 × 2 임베딩 = 8 조합, cosine top-k 만 | `src/rag/evaluation/retrieval_eval.py` |
| `retrieval_optim.{md,json}` | dense → hybrid(BM25+dense) → hybrid+rerank 단계별 비교 | `src/rag/evaluation/retrieval_optim.py` |
| `ragas_eval.{md,json}` | 4 config × RAGAS 4지표 (faithfulness / answer_relevancy / context_recall / context_precision) | `src/rag/evaluation/ragas_eval.py` |

## 재현

```bash
# QA 데이터셋 (28 파일 × 3 QA = 84개)
uv run python -m src.rag.evaluation.build_dataset

# 1) 청킹 + 샘플 리포트
uv run python -m src.rag.chunking.run

# 2) Baseline retrieval 벤치 (4×2 매트릭스)
uv run python -m src.rag.evaluation.retrieval_eval

# 3) 최적화 비교 (recursive × 2모델 × 3방식)
uv run python -m src.rag.evaluation.retrieval_optim

# 4) RAGAS 평가 (OPENAI_API_KEY 필요)
uv run python -m src.rag.evaluation.ragas_eval
```

> 2~4 는 `data/processed/chroma` 에 해당 컬렉션이 먼저 적재돼 있어야 한다
> (`python -m src.rag.embedding.run ...` 참고).

## 실험 스토리 아크

각 단계의 **가설 → 결과 → 결정** 흐름.

### 1. 청킹 전략 비교 (`retrieval_eval`)
- **가설**: 문서 구조를 존중하는 markdown/recursive 가 fixed 보다 정확도 높을 것.
- **결과**: `fixed × bge-m3` 가 hit@10 0.893 으로 1위. recursive/markdown 은 그보다 낮음.
- **결정**: grain artifact 의심 (source 파일 단위 평가에서 fixed 의 긴 청크가 "아무 내용이나 걸리면 hit" 로 인정될 가능성). `semantic × *` 는 Phase 2 샘플 5파일만 인덱싱해 0.000 → 비교 제외.
- **→ problem_solving #6, #7**

### 2. Hybrid + Reranker (`retrieval_optim`)
- **가설**: BM25 가 exact keyword 를 보강하고, cross-encoder reranker 가 상위 랭킹을 교정해 MRR 이 크게 오를 것.
- **결과** (recursive 기준):
  - bge-m3: dense → rerank = hit@10 **0.821 → 0.940 (+0.119)**, MRR@10 **0.634 → 0.792 (+0.158)**
  - bge-large-en: hit@10 **0.857 → 0.940 (+0.083)**, MRR@10 **0.693 → 0.821 (+0.128)**
- **관찰**: reranker 가 hit 보다 **MRR 을 더 크게 개선** — "정답이 top-k 안에 있었지만 낮은 순위였던 것을 위로 올려주는" 역할. 두 임베딩 모두 rerank 후 hit@10 0.940 에 수렴 → 남은 6% 는 QA 데이터셋 결함 의심.
- **→ problem_solving #8**

### 3. 생성 품질 평가 (`ragas_eval`)
- **가설**: retrieval MRR 개선이 answer quality 로 transfer 될 것.
- **결과**: `recursive × bge-large-en × hybrid_rerank` 가 3개 지표(answer_relevancy 0.965 / context_recall 0.903 / context_precision 0.882) 1위. faithfulness 만 bge-m3 rerank(0.935) 가 1위.
- **이상 현상**: bge-large-en 은 rerank 시 faithfulness 가 **역행** (0.924 → 0.913). precision 은 올랐는데 faithfulness 가 내려감 → 넓은 컨텍스트에서 LLM 이 근거 밖 서술을 확장했을 가능성.
- **결정**: 프로덕션 구성 = `recursive × bge-large-en-v1.5 × hybrid_rerank`. faithfulness 열세는 Phase 4 에이전트의 **엄격한 인용 규칙 + `[url-removed]` 치환** 으로 보완.
- **→ Phase 4 docs_search tool 구성**

## 주의사항

- RAGAS 는 비결정적 — `gpt-4o-mini` 샘플링으로 ±0.01 정도 변동 가능.
- BM25 는 in-memory — 청크 jsonl 재생성 시 반드시 같은 strategy 로 재실행 필요
  (dense 와 corpus 일치 보장).
- Hit-rate grain 은 `source_path` 고정. chunk-level grain 으로 보려면 평가
  스크립트의 `_is_hit` 함수 수정 필요.
