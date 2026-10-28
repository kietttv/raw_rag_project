"""Output Guardrail module for PII redaction and empty hallucination detection.

Protects against:
  1. Sensitive PII exposure (Vietnamese/international phone numbers, CCCD/CMND, and email addresses).
  2. Empty / degenerate answers, template echo stubs, phantom citations ([k] > len(sources)), and contradictory refusals.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Any

from pydantic import BaseModel, Field

from src.config import settings
from src.models.schemas import ChatResponse, SourceDocument
from src.utils.logger import logger

FALLBACK_NO_INFO_MESSAGE = (
    "Tài liệu được cung cấp không có đủ thông tin để trả lời câu hỏi này."
)


class OutputViolationType(str, Enum):
    """Categorized violation reasons for output guardrail checks."""

    EMPTY_OR_DEGENERATE = "empty_or_degenerate"
    PROMPT_ECHO_STUB = "prompt_echo_stub"
    PHANTOM_CITATION = "phantom_citation"
    CONTRADICTORY_REFUSAL = "contradictory_refusal"


class OutputGuardResult(BaseModel):
    """Result payload from output guardrail inspection."""

    sanitized_answer: str = Field(..., description="Cleaned, redacted, and validated answer")
    is_hallucination_detected: bool = Field(
        default=False, description="Whether an empty or ungrounded hallucination was detected"
    )
    violation_type: OutputViolationType | None = Field(
        default=None, description="Category of hallucination violation if detected"
    )
    reason: str | None = Field(default=None, description="Explanation of violation if any")
    pii_redacted_count: int = Field(default=0, description="Total number of PII entities redacted")
    redacted_types: list[str] = Field(
        default_factory=list, description="List of PII types masked (email, cccd, phone)"
    )


# ---------------------------------------------------------------------------
# Regex Patterns for PII Redaction
# ---------------------------------------------------------------------------

# 1. Email pattern
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# 2. CCCD 12 digits (starting with 0, optional space/dot/dash separation)
_CCCD_12_PATTERN = re.compile(
    r"(?<!\d)0\d{2}[\s.-]?\d[\s.-]?\d{2}[\s.-]?\d{6}(?!\d)"
)

# 2b. CMND 9 digits or ID with explicit keyword context (e.g. CMND số: 123456789, CCCD: 001...)
_CMND_CONTEXT_PATTERN = re.compile(
    r"(?i)\b((?:cccd|cmnd|căn cước(?:\s+công dân)?|chứng minh(?:\s+nhân dân|\s+thư)?|số định danh)(?:\s*(?:số|:|-)+)*\s*)(?<!\d)\d{9,12}(?!\d)"
)

# 3. Vietnamese Phone Numbers (Mobile: 03x, 05x, 07x, 08x, 09x; Landline: 02x; with +84/0084/84 or 0)
_VN_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:(?:\+84|0084|84)[\s.-]?(?:3[2-9]|5[25689]|7[06-9]|8[1-9]|9[0-46-9]|2\d)|0(?:3[2-9]|5[25689]|7[06-9]|8[1-9]|9[0-46-9]|2\d))(?:[\s.-]?\d){7,8}(?!\d)"
)

# 3b. Parenthesized landline format: (024) 3823 4567 or (028) 3812 3456
_VN_LANDLINE_PAREN_PATTERN = re.compile(
    r"(?<!\d)\(02\d\)[\s.-]?\d{3,4}[\s.-]?\d{4}(?!\d)"
)

# Citation extraction pattern: [1], [2], [12]
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# Prompt prefix stripped patterns for echo checks
_PROMPT_PREFIX_CLEANER = re.compile(
    r"^(?:câu trả lời:|ngữ cảnh(?:\s*\(context\))?:|answer:|hướng dẫn trả lời:)\s*",
    re.IGNORECASE,
)

_STUB_TEXT_PATTERNS = {
    "",
    "n/a",
    "none",
    "null",
    "unknown",
    "không có",
    "không rõ",
    "chưa rõ",
    "...",
    "---",
}


def mask_pii(text: str) -> tuple[str, int, list[str]]:
    """Scan and redact sensitive PII (Email, CCCD/CMND, Phone) in text using Regex.

    Order of redaction:
      1. Email
      2. CCCD / CMND (12-digit & contextual 9-digit)
      3. Vietnamese Phone numbers

    Args:
        text: Input string potentially containing personal data.

    Returns:
        tuple[str, int, list[str]]: (sanitized_text, total_redacted, types_redacted)
    """
    if not text:
        return "", 0, []

    redacted_types: list[str] = []
    total_count = 0
    result = text

    # 1. Mask Emails
    email_matches = _EMAIL_PATTERN.findall(result)
    if email_matches:
        result = _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", result)
        total_count += len(email_matches)
        redacted_types.append("email")

    # 2. Mask Contextual CMND / CCCD
    cmnd_matches = _CMND_CONTEXT_PATTERN.findall(result)
    if cmnd_matches:
        result = _CMND_CONTEXT_PATTERN.sub(r"\1[CCCD_REDACTED]", result)
        total_count += len(cmnd_matches)
        if "cccd" not in redacted_types:
            redacted_types.append("cccd")

    # 2b. Mask 12-digit CCCD
    cccd_matches = _CCCD_12_PATTERN.findall(result)
    if cccd_matches:
        result = _CCCD_12_PATTERN.sub("[CCCD_REDACTED]", result)
        total_count += len(cccd_matches)
        if "cccd" not in redacted_types:
            redacted_types.append("cccd")

    # 3. Mask Vietnamese Phone numbers (parenthesized landline + general mobile/landline)
    paren_matches = _VN_LANDLINE_PAREN_PATTERN.findall(result)
    if paren_matches:
        result = _VN_LANDLINE_PAREN_PATTERN.sub("[PHONE_REDACTED]", result)
        total_count += len(paren_matches)
        if "phone" not in redacted_types:
            redacted_types.append("phone")

    phone_matches = _VN_PHONE_PATTERN.findall(result)
    if phone_matches:
        result = _VN_PHONE_PATTERN.sub("[PHONE_REDACTED]", result)
        total_count += len(phone_matches)
        if "phone" not in redacted_types:
            redacted_types.append("phone")

    return result, total_count, redacted_types


def check_empty_hallucination(
    answer: str,
    sources: list[SourceDocument] | None = None,
) -> tuple[bool, OutputViolationType | None, str | None, str]:
    """Check whether generated answer is an empty, stub, or phantom hallucination.

    Args:
        answer: Raw generated answer text from LLM.
        sources: Retrieved source references available for this query.

    Returns:
        tuple[bool, OutputViolationType | None, str | None, str]:
          (is_hallucination, violation_type, reason, cleaned_or_fallback_answer)
    """
    raw_stripped = (answer or "").strip()

    # 1. Empty or Degenerate check
    if not raw_stripped:
        return (
            True,
            OutputViolationType.EMPTY_OR_DEGENERATE,
            "Answer is empty or whitespace-only",
            FALLBACK_NO_INFO_MESSAGE,
        )

    alnum_count = sum(1 for c in raw_stripped if c.isalnum())
    if alnum_count < settings.MIN_ANSWER_LENGTH:
        return (
            True,
            OutputViolationType.EMPTY_OR_DEGENERATE,
            f"Answer contains fewer than {settings.MIN_ANSWER_LENGTH} alphanumeric characters",
            FALLBACK_NO_INFO_MESSAGE,
        )

    # 2. Prompt Echo Stub check
    without_prefix = _PROMPT_PREFIX_CLEANER.sub("", raw_stripped).strip()
    if without_prefix.lower() in _STUB_TEXT_PATTERNS or without_prefix == "":
        return (
            True,
            OutputViolationType.PROMPT_ECHO_STUB,
            "Answer is a template echo stub with no meaningful content",
            FALLBACK_NO_INFO_MESSAGE,
        )

    # 3. Phantom Citation check
    num_sources = len(sources) if sources is not None else 0
    citations = [int(m) for m in _CITATION_PATTERN.findall(raw_stripped)]

    if num_sources == 0 and citations:
        # Cited documents when zero source chunks were provided
        return (
            True,
            OutputViolationType.PHANTOM_CITATION,
            f"Answer fabricated citations {citations} when no source documents were provided",
            FALLBACK_NO_INFO_MESSAGE,
        )

    cleaned_answer = raw_stripped
    if num_sources > 0 and citations:
        # Check if any cited index exceeds available sources count
        invalid_citations = [idx for idx in citations if idx > num_sources or idx < 1]
        if invalid_citations:
            logger.bind(
                invalid_citations=invalid_citations,
                available_sources=num_sources,
            ).warning("Stripping phantom citations exceeding available source count")
            for inv_idx in set(invalid_citations):
                cleaned_answer = cleaned_answer.replace(f"[{inv_idx}]", "")
            cleaned_answer = " ".join(cleaned_answer.split())

    # 4. Contradictory Refusal check
    lower_ans = cleaned_answer.lower()
    refusal_anchor = "tài liệu được cung cấp không có đủ thông tin để trả lời"
    if refusal_anchor in lower_ans:
        # If model admitted no info but then rambled on for > 180 chars of speculation
        if len(cleaned_answer) > 180:
            return (
                True,
                OutputViolationType.CONTRADICTORY_REFUSAL,
                "Model contradicted refusal by appending ungrounded speculative text",
                FALLBACK_NO_INFO_MESSAGE,
            )

    return False, None, None, cleaned_answer


def inspect_output(
    answer: str,
    sources: list[SourceDocument] | None = None,
) -> OutputGuardResult:
    """Inspect and sanitize LLM answer against hallucination and PII leakage.

    Args:
        answer: Text answer generated by model.
        sources: Optional context source documents.

    Returns:
        OutputGuardResult: Cleaned answer with safety flags and PII metrics.
    """
    if not settings.ENABLE_OUTPUT_GUARD:
        return OutputGuardResult(sanitized_answer=answer or "")

    # 1. Check for empty or malformed hallucination
    is_hallu, v_type, reason, cleaned_text = check_empty_hallucination(
        answer=answer, sources=sources
    )

    if is_hallu:
        logger.bind(violation=v_type, reason=reason).warning(
            "Empty or ungrounded hallucination detected by output guard; replaced with safe fallback"
        )

    # 2. Redact sensitive PII in answer
    masked_answer, pii_count, redacted_types = mask_pii(cleaned_text)
    if pii_count > 0:
        logger.bind(count=pii_count, types=redacted_types).info(
            "Redacted sensitive PII entities from generated response"
        )

    return OutputGuardResult(
        sanitized_answer=masked_answer,
        is_hallucination_detected=is_hallu,
        violation_type=v_type,
        reason=reason,
        pii_redacted_count=pii_count,
        redacted_types=redacted_types,
    )


def sanitize_sources(sources: list[SourceDocument] | None) -> list[SourceDocument]:
    """Redact sensitive PII inside retrieved source document content snippets.

    Args:
        sources: List of SourceDocument objects retrieved by search.

    Returns:
        list[SourceDocument]: New list of SourceDocument objects with masked content.
    """
    if not sources or not settings.ENABLE_OUTPUT_GUARD or not settings.MASK_PII_IN_SOURCES:
        return sources or []

    sanitized: list[SourceDocument] = []
    for doc in sources:
        masked_content, count, _ = mask_pii(doc.content)
        sanitized.append(
            SourceDocument(
                content=masked_content,
                score=doc.score,
                metadata=dict(doc.metadata or {}),
            )
        )
    return sanitized


def apply_output_guard(response: ChatResponse | Any) -> ChatResponse | Any:
    """Apply output guardrail (hallucination check + PII redaction) to ChatResponse.

    Args:
        response: ChatResponse object (or duck-typed equivalent).

    Returns:
        ChatResponse: Updated response with safe answer and sanitized source contents.
    """
    if not settings.ENABLE_OUTPUT_GUARD or response is None:
        return response

    answer = getattr(response, "answer", "")
    sources = getattr(response, "sources", [])

    guard_res = inspect_output(answer=answer, sources=sources)
    cleaned_sources = sanitize_sources(sources)

    # Return updated instance
    if isinstance(response, ChatResponse):
        return ChatResponse(
            answer=guard_res.sanitized_answer,
            sources=cleaned_sources,
        )

    # Duck-typed fallback (e.g. for mock objects in tests)
    try:
        response.answer = guard_res.sanitized_answer
        response.sources = cleaned_sources
    except Exception:
        pass
    return response
