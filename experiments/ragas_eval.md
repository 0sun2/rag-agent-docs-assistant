# RAGAS Generation 품질 평가 — Phase 3

- N questions: **84**
- 비교군: `recursive × {bge-m3, bge-large-en-v1.5}` × `{dense, hybrid_rerank}`
- 판정자: `gpt-4o-mini` + `text-embedding-3-small`
- 지표: faithfulness, answer_relevancy, context_recall, context_precision

## 전체 결과

| combo | method | faithfulness | answer_relevancy | context_recall | context_precision |
|---|---|---|---|---|---|
| recursive × bge-m3 | dense | 0.881 | 0.908 | 0.771 | 0.683 |
| recursive × bge-m3 | hybrid_rerank | 0.935 | 0.954 | 0.874 | 0.866 |
| recursive × bge-large-en-v1.5 | dense | 0.924 | 0.951 | 0.834 | 0.760 |
| recursive × bge-large-en-v1.5 | hybrid_rerank | 0.913 | 0.965 | 0.903 | 0.882 |

## 증분 분석 (hybrid_rerank vs dense)

| combo | Δfaithfulness | Δanswer_relevancy | Δcontext_recall | Δcontext_precision |
|---|---|---|---|---|
| recursive × bge-m3 | +0.055 | +0.047 | +0.103 | +0.183 |
| recursive × bge-large-en-v1.5 | -0.011 | +0.014 | +0.068 | +0.122 |
