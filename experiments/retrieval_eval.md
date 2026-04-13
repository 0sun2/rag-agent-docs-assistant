# Retrieval Evaluation — Phase 3 Baseline

- N questions: **84**
- Metric grain: source file path (`source_path`)
- k values: [3, 5, 10]
- Retrieval method: cosine top-k (baseline, no MMR/hybrid/reranker)

| strategy × model | hit@3 | hit@5 | hit@10 | MRR@3 | MRR@5 | MRR@10 |
|---|---|---|---|---|---|---|
| fixed × bge-m3 | 0.738 | 0.810 | 0.893 | 0.655 | 0.670 | 0.682 |
| recursive × bge-m3 | 0.667 | 0.750 | 0.821 | 0.605 | 0.624 | 0.634 |
| markdown × bge-m3 | 0.679 | 0.702 | 0.798 | 0.569 | 0.575 | 0.588 |
| semantic × bge-m3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed × bge-large-en-v1.5 | 0.774 | 0.821 | 0.857 | 0.665 | 0.677 | 0.682 |
| recursive × bge-large-en-v1.5 | 0.786 | 0.810 | 0.857 | 0.681 | 0.687 | 0.693 |
| markdown × bge-large-en-v1.5 | 0.786 | 0.798 | 0.869 | 0.655 | 0.657 | 0.667 |
| semantic × bge-large-en-v1.5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Hit-rate @ 10 by difficulty

| strategy × model | compare | factual | howto |
|---|---|---|---|
| fixed × bge-m3 | 0.929 | 0.929 | 0.821 |
| recursive × bge-m3 | 0.857 | 0.786 | 0.821 |
| markdown × bge-m3 | 0.750 | 0.821 | 0.821 |
| semantic × bge-m3 | 0.000 | 0.000 | 0.000 |
| fixed × bge-large-en-v1.5 | 0.821 | 0.893 | 0.857 |
| recursive × bge-large-en-v1.5 | 0.821 | 0.893 | 0.857 |
| markdown × bge-large-en-v1.5 | 0.893 | 0.857 | 0.857 |
| semantic × bge-large-en-v1.5 | 0.000 | 0.000 | 0.000 |
