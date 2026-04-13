"""README 의 mermaid 다이어그램을 PNG 로 저장.

GitHub 외 플랫폼(노션/PDF 포트폴리오)에서는 mermaid 가 렌더링되지 않으므로
`mermaid.ink` 공개 API 로 PNG 를 받아 `docs/images/` 에 저장해둔다.

출력:
    docs/images/arch_rag.png     — RAG 파이프라인
    docs/images/arch_agent.png   — ReAct 에이전트 그래프

실행:
    uv run python scripts/render_mermaid.py
"""

from __future__ import annotations

import base64
import zlib
from pathlib import Path

import httpx

OUT_DIR = Path("docs/images")


RAG = """flowchart LR
    A[langchain-ai/docs<br/>GitHub raw markdown] -->|httpx async<br/>Semaphore 16| B[data/raw]
    B --> C[load<br/>frontmatter→metadata]
    C --> D{Chunking}
    D -->|fixed| E1[jsonl]
    D -->|recursive| E2[jsonl]
    D -->|markdown| E3[jsonl]
    D -->|semantic| E4[jsonl]
    E1 & E2 & E3 & E4 --> F[Embedding<br/>bge-m3 / bge-large-en]
    F --> G[(Chroma<br/>8 collections)]
    G --> H[Dense retriever]
    E2 --> I[BM25 in-memory]
    H & I --> J[EnsembleRetriever<br/>RRF]
    J --> K[Cross-encoder<br/>bge-reranker-v2-m3]
    K --> L[RAGChain<br/>gpt-4o-mini]
    L --> M[Answer + 인용]
"""

AGENT = """flowchart LR
    START([START]) --> AG[agent node<br/>LLM + bound tools]
    AG -->|tool_calls| TN[ToolNode]
    TN --> AG
    AG -->|no tool_calls<br/>or iter ≥ 5| END([END])

    subgraph Tools
      T1[docs_search<br/>recursive × bge-large × hybrid_rerank]
      T2[code_generate<br/>ruff check + format]
      T3[web_search<br/>Tavily]
      T4[error_analyze<br/>traceback 파서 + 추천]
    end
    TN -.- Tools
"""


def _encode(diagram: str) -> str:
    """mermaid.ink pako 인코딩 — 긴 다이어그램도 URL 길이 제한 안 걸림."""
    payload = ('{"code":' + _json_str(diagram) + ',"mermaid":{"theme":"default"}}').encode()
    compressed = zlib.compress(payload, level=9)
    return base64.urlsafe_b64encode(compressed).decode()


def _json_str(s: str) -> str:
    import json as _json
    return _json.dumps(s)


def _render(name: str, diagram: str) -> None:
    encoded = _encode(diagram)
    url = f"https://mermaid.ink/img/pako:{encoded}?type=png&bgColor=white"
    r = httpx.get(url, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    out = OUT_DIR / name
    out.write_bytes(r.content)
    print(f"saved: {out} ({len(r.content):,} bytes)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _render("arch_rag.png", RAG)
    _render("arch_agent.png", AGENT)


if __name__ == "__main__":
    main()
