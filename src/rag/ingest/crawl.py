"""Crawl LangChain official documentation into local Markdown files.

전략: LangChain 공식 문서는 별도 리포 `langchain-ai/docs` (브랜치 `main`)의 `src/` 트리에
`.md` / `.mdx` 형태로 관리됩니다 (LangChain / LangGraph / LangSmith 문서 통합).
GitHub REST API의 git tree 엔드포인트로 파일 목록을 가져온 뒤, raw.githubusercontent.com
에서 본문을 병렬(`asyncio` + `httpx.AsyncClient`)로 내려받아 `data/raw/langchain/` 아래에
동일한 디렉토리 구조로 저장합니다.

사이트(html.langchain.com)를 직접 크롤링하지 않는 이유:
    - 공식 문서가 마크다운 원본으로 공개되어 있으므로 노이즈(네비게이션, 사이드바 등) 없이
      깔끔한 텍스트를 그대로 얻을 수 있고, 청킹 품질이 더 좋아집니다.
    - robots.txt / 요청 정책 부담이 적습니다.

동시성:
    - `settings.crawl_concurrency` (기본 16) 슬롯의 `asyncio.Semaphore`로 상한 제어.
    - 캐시 히트(`out_path.exists()`)는 세마포어 슬롯을 점유하지 않음.
    - 429/503이 잦다면 동시성 값을 낮춰 대응.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm as atqdm

from src.config import settings

logger = logging.getLogger(__name__)

GITHUB_REPO = "langchain-ai/docs"
GITHUB_BRANCH = "main"
DOC_ROOT = "src"  # 리포 내 문서 루트 경로
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/"
TREE_API = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"

ALLOWED_SUFFIXES = {".md", ".mdx"}

_HEADERS = {"User-Agent": "llm-docs-assistant/0.1 (+https://github.com)"}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _aget(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """GET with bounded exponential backoff. Only retries on httpx.HTTPError."""
    resp = await client.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp


async def list_doc_files(client: httpx.AsyncClient) -> list[str]:
    """Return repo-relative paths for all doc files under DOC_ROOT, applying filters."""
    logger.info("Fetching git tree from %s", TREE_API)
    resp = await _aget(client, TREE_API)
    tree = resp.json().get("tree", [])

    include_prefixes = _parse_csv(settings.crawl_include_prefixes)
    exclude_substrings = _parse_csv(settings.crawl_exclude_substrings)

    paths: list[str] = []
    for node in tree:
        if node.get("type") != "blob":
            continue
        path = node["path"]
        if not path.startswith(f"{DOC_ROOT}/"):
            continue
        if Path(path).suffix.lower() not in ALLOWED_SUFFIXES:
            continue

        rel = path[len(DOC_ROOT) + 1 :]  # DOC_ROOT 기준 상대경로
        if include_prefixes and not any(rel.startswith(p) for p in include_prefixes):
            continue
        if any(sub in rel for sub in exclude_substrings):
            continue
        paths.append(path)

    logger.info(
        "Found %d doc files under %s (include=%s, exclude=%s)",
        len(paths),
        DOC_ROOT,
        include_prefixes or "*",
        exclude_substrings or "-",
    )
    return paths


async def _download_one(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    repo_path: str,
    out_dir: Path,
) -> Path | None:
    """Download a single doc file if not already present.

    Returns the output path on success (including cache hits), or None on handled failure.
    Unhandled exceptions (e.g. OSError on disk) propagate.
    """
    rel = Path(repo_path).relative_to(DOC_ROOT)
    out_path = out_dir / rel
    if out_path.exists():
        return out_path  # 캐시 히트 — 세마포어 슬롯 점유하지 않음

    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with sem:
        try:
            resp = await _aget(client, RAW_BASE + repo_path)
        except httpx.HTTPError as e:
            logger.warning("Failed to download %s: %s", repo_path, e)
            return None

    # 디스크 쓰기는 세마포어 바깥 — 네트워크 슬롯을 I/O로 점유하지 않음
    out_path.write_bytes(resp.content)
    return out_path


async def crawl_async(
    out_dir: Path | None = None,
    max_pages: int | None = None,
    concurrency: int | None = None,
) -> int:
    """Crawl LangChain docs concurrently. Returns number of files that now exist locally."""
    out_dir = Path(out_dir or settings.crawl_output_dir)
    max_pages = max_pages or settings.crawl_max_pages
    concurrency = concurrency or settings.crawl_concurrency
    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True, http2=False) as client:
        files = await list_doc_files(client)
        if max_pages:
            files = files[:max_pages]

        tasks = [
            asyncio.create_task(_download_one(sem, client, repo_path, out_dir))
            for repo_path in files
        ]

        saved = 0
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Downloading docs"):
            result = await coro
            if result is not None:
                saved += 1

    logger.info("Saved/verified %d files in %s (concurrency=%d)", saved, out_dir, concurrency)
    return saved


def crawl(
    out_dir: Path | None = None,
    max_pages: int | None = None,
    concurrency: int | None = None,
) -> int:
    """Synchronous entrypoint preserved for CLI and imports."""
    return asyncio.run(
        crawl_async(out_dir=out_dir, max_pages=max_pages, concurrency=concurrency)
    )


def main() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    crawl()


if __name__ == "__main__":
    main()
