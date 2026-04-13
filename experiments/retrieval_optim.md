# Retrieval 최적화 비교 — Phase 3

- N questions: **84**
- 메인 비교군: `recursive × {bge-m3, bge-large-en-v1.5}`
- k values: [3, 5, 10]
- Hybrid: BM25 + Chroma dense (EnsembleRetriever, dense_weight=0.5)
- Reranker: `BAAI/bge-reranker-v2-m3` (fetch_k=20 → top_n=max_k)
- `fixed`, `markdown`, `semantic` 제외 — 사유는 `docs/portfolio/problem_solving.md` #6, #7

## 전체 결과

| combo | method | hit@3 | hit@5 | hit@10 | MRR@3 | MRR@5 | MRR@10 |
|---|---|---|---|---|---|---|---|
| recursive × bge-m3 | dense | 0.667 | 0.750 | 0.821 | 0.605 | 0.624 | 0.634 |
| recursive × bge-m3 | hybrid | 0.762 | 0.798 | 0.881 | 0.659 | 0.668 | 0.679 |
| recursive × bge-m3 | hybrid_rerank | 0.869 | 0.905 | 0.940 | 0.778 | 0.787 | 0.792 |
| recursive × bge-large-en-v1.5 | dense | 0.786 | 0.810 | 0.857 | 0.681 | 0.687 | 0.693 |
| recursive × bge-large-en-v1.5 | hybrid | 0.833 | 0.869 | 0.905 | 0.718 | 0.727 | 0.733 |
| recursive × bge-large-en-v1.5 | hybrid_rerank | 0.905 | 0.917 | 0.940 | 0.815 | 0.818 | 0.821 |

## 증분 분석 (vs dense baseline)

| combo | method | Δhit@3 | Δhit@5 | Δhit@10 | ΔMRR@3 | ΔMRR@5 | ΔMRR@10 |
|---|---|---|---|---|---|---|---|
| recursive × bge-m3 | hybrid | +0.095 | +0.048 | +0.060 | +0.054 | +0.044 | +0.045 |
| recursive × bge-m3 | hybrid_rerank | +0.202 | +0.155 | +0.119 | +0.173 | +0.163 | +0.157 |
| recursive × bge-large-en-v1.5 | hybrid | +0.048 | +0.060 | +0.048 | +0.038 | +0.041 | +0.040 |
| recursive × bge-large-en-v1.5 | hybrid_rerank | +0.119 | +0.107 | +0.083 | +0.135 | +0.132 | +0.129 |
