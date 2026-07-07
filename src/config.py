from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider: "openai" | "bedrock"
    llm_provider: str = "openai"

    # OpenAI (LLM 전용. 임베딩은 기본적으로 huggingface 사용)
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"  # Phase 3 비교 실험용

    # AWS Bedrock (자격증명은 ~/.aws/credentials 또는 IAM 역할 — .env에 키 넣지 않음)
    # 모델 ID는 온디맨드 미지원 모델이 있어 추론 프로파일 ID 사용
    bedrock_model_id: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    aws_region: str = "ap-southeast-1"
    # 선택: Bedrock Guardrail 입력 필터 (비우면 비활성)
    bedrock_guardrail_id: str = ""
    bedrock_guardrail_version: str = "DRAFT"

    # Embedding (오픈소스 기본)
    # provider: "huggingface" | "openai"
    embedding_provider: str = "huggingface"
    # 메인: BAAI/bge-m3, 비교군: BAAI/bge-large-en-v1.5
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"  # "cpu" | "cuda" | "mps"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True  # BGE 권장

    # Vector DB
    chroma_persist_dir: Path = Path("./data/processed/chroma")
    chroma_collection_name: str = "langchain_docs"
    # 서버 모드 — 비워두면 PersistentClient(로컬 파일), 설정 시 HttpClient 사용
    chroma_host: str = ""
    chroma_port: int = 8000

    # Retrieval
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Crawler
    crawl_output_dir: Path = Path("./data/raw/langchain")
    crawl_max_pages: int = 500
    # 동시 다운로드 슬롯 수. 429/503이 보이면 낮춰 대응. 16이 raw.githubusercontent.com 기준 안전선.
    crawl_concurrency: int = 16
    # 쉼표 구분. DOC_ROOT 기준 상대경로 prefix. 비워두면 전체.
    crawl_include_prefixes: str = "oss/"
    # 쉼표 구분. DOC_ROOT 기준 상대경로 substring. 해당 문자열이 포함된 경로는 제외.
    crawl_exclude_substrings: str = "oss/javascript/"

    # Web search (Tavily)
    tavily_api_key: str = ""
    tavily_max_results: int = 5

    # Agent memory (LangGraph checkpointer)
    checkpoint_db_path: Path = Path("./data/checkpoints.sqlite")
    # 모델에 보내는 히스토리 절삭 상한 (근사 토큰). 저장 state는 자르지 않음.
    history_max_tokens: int = 8000

    # Usage / cost
    usd_krw_rate: float = 1400.0  # 비용 추정용 환율 (KRW per USD)

    # Logging
    log_level: str = "INFO"


settings = Settings()
