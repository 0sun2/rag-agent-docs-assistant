# 단일 이미지 — FastAPI 와 Streamlit 이 같은 코드베이스를 공유.
# Compose 에서 command override 로 역할 분기.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# ── Dependency layer ──
# pyproject + lockfile + README(=project readme 참조) 만 먼저 COPY 해서
# 소스 변경 시에도 의존성 레이어 캐시 유지.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Source layer ──
COPY src/ ./src/
COPY data/processed/chunks/ ./data/processed/chunks/

# 프로젝트 자체 설치 (hatchling: packages=["src"])
RUN uv sync --frozen --no-dev

EXPOSE 8000 8501

# 기본 entrypoint 는 FastAPI (compose 에서 override)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
