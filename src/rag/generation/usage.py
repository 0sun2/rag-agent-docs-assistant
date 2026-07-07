"""토큰 usage 집계 + 비용 추정 유틸.

LangChain `AIMessage.usage_metadata`(input/output tokens)를 요청 단위로 모아
모델별 단가표로 USD/KRW 추정 비용을 계산한다. RAG 체인은 응답 1건, 에이전트는
루프 내 모든 AIMessage를 합산한다.

단가는 1M 토큰 기준 USD. 모델 ID prefix 매칭이라 추론 프로파일 ID
(`global.anthropic....`)도 잡힌다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

from langchain_core.messages import AIMessage, BaseMessage

from src.config import settings

logger = logging.getLogger(__name__)

# (input USD / 1M tokens, output USD / 1M tokens)
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _lookup_pricing(model: str) -> tuple[float, float] | None:
    """모델 ID에서 단가를 찾는다. 추론 프로파일 prefix가 붙어도 substring으로 매칭."""
    for key, price in PRICING_PER_1M.items():
        if key in model:
            return price
    return None


@dataclass
class UsageReport:
    """요청 1건의 토큰 사용량 + 추정 비용."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float | None = None
    cost_krw: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Accumulator:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    _models: set[str] = field(default_factory=set)

    def add_message(self, msg: AIMessage) -> None:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            return
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.llm_calls += 1
        model = (msg.response_metadata or {}).get("model_name") or (
            msg.response_metadata or {}
        ).get("model_id", "")
        if model:
            self._models.add(model)

    def report(self) -> UsageReport:
        model = next(iter(self._models)) if len(self._models) == 1 else ",".join(
            sorted(self._models)
        )
        cost_usd: float | None = None
        cost_krw: float | None = None
        pricing = _lookup_pricing(model) if model else None
        if pricing:
            in_price, out_price = pricing
            cost_usd = round(
                self.input_tokens / 1e6 * in_price + self.output_tokens / 1e6 * out_price,
                6,
            )
            cost_krw = round(cost_usd * settings.usd_krw_rate, 2)
        return UsageReport(
            model=model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            llm_calls=self.llm_calls,
            cost_usd=cost_usd,
            cost_krw=cost_krw,
        )


def usage_from_messages(messages: list[BaseMessage]) -> UsageReport:
    """메시지 리스트(에이전트 루프 산출물 등)에서 AIMessage usage를 합산."""
    acc = _Accumulator()
    for m in messages:
        if isinstance(m, AIMessage):
            acc.add_message(m)
    return acc.report()


def usage_from_response(response: BaseMessage) -> UsageReport:
    """단일 LLM 응답(RAG 체인)에서 usage 추출."""
    return usage_from_messages([response])


def log_usage(endpoint: str, report: UsageReport) -> None:
    """요청 단위 구조화(JSON) usage 로그."""
    logger.info(
        "usage %s",
        json.dumps({"endpoint": endpoint, **report.to_dict()}, ensure_ascii=False),
    )
