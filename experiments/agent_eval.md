# 에이전트 수준 평가 — 테스트 4분면

- N questions: **48** (오류 0건 제외 판정)
- 판정: trace 기계 판정 (LLM 심판 없음) — `src/agent/evaluation/run.py`

## 분면별 결과

| 분면 | N | 툴 선택 정확도 | 멀티스텝 성공률 | 인용 준수율 | 평균 스텝 | 평균 토큰(in/out) | 평균 비용($) |
|---|---|---|---|---|---|---|---|
| docs_only | 12 | 100.0% (12/12) | — | 91.7% (11/12) | 2.0 | 4,254 / 408 | 0.00088 |
| code_only | 12 | 100.0% (12/12) | — | — | 5.0 | 11,477 / 385 | 0.00195 |
| multi_step | 12 | 83.3% (10/12) | 83.3% (10/12) | — | 5.0 | 14,753 / 283 | 0.00238 |
| no_tool | 12 | 100.0% (12/12) | — | — | 1.0 | 1,614 / 195 | 0.00036 |

## 전체

- 툴 선택 정확도: **95.8% (46/48)**
- 멀티스텝 성공률: **83.3% (10/12)**
- 인용 형식 준수율: **91.7% (11/12)**

## 실패 케이스

- `docs_only-02` called=['docs_search'] tool_ok=True multistep=None citation=False — How do I create a StateGraph in LangGraph? Explain the required arguments.
- `multi_step-08` called=['code_generate', 'code_generate', 'code_generate', 'code_generate', 'code_generate'] tool_ok=False multistep=False citation=None — Create a LangChain prompt template with a system and human message and format it
- `multi_step-10` called=['code_generate', 'code_generate', 'code_generate', 'code_generate', 'code_generate'] tool_ok=False multistep=False citation=None — Implement a custom LangChain retriever class that returns fixed documents. Write
