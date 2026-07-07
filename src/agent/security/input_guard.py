"""사용자 입력 필터 — Bedrock Guardrails 선택적 연동 (개발계획서 2-2).

`BEDROCK_GUARDRAIL_ID` 가 설정된 경우에만 활성화되는 옵션 계층.
LLM 호출 **전에** `apply_guardrail(source="INPUT")` 로 사용자 입력만 선검사해,
차단 시 LLM을 아예 호출하지 않는다 (토큰 비용 0).

Guardrail 리소스(PROMPT_ATTACK 필터 + PII ANONYMIZE)는 AWS 콘솔/CLI로 별도 생성:
    aws bedrock create-guardrail ... → ID/버전을 .env 에 설정
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """입력 검사 결과."""

    blocked: bool
    message: str = ""  # 차단 시 사용자에게 보여줄 안내


def check_user_input(text: str) -> GuardResult:
    """사용자 입력을 Guardrail로 선검사. 미설정 시 즉시 통과."""
    if not settings.bedrock_guardrail_id:
        return GuardResult(blocked=False)

    import boto3

    runtime = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    resp = runtime.apply_guardrail(
        guardrailIdentifier=settings.bedrock_guardrail_id,
        guardrailVersion=settings.bedrock_guardrail_version,
        source="INPUT",
        content=[{"text": {"text": text}}],
    )
    if resp.get("action") == "GUARDRAIL_INTERVENED":
        outputs = resp.get("outputs") or [{}]
        message = outputs[0].get("text", "요청이 안전 정책에 의해 차단되었습니다.")
        logger.warning("Guardrail blocked user input (assessments=%s)", resp.get("assessments"))
        return GuardResult(blocked=True, message=message)
    return GuardResult(blocked=False)
