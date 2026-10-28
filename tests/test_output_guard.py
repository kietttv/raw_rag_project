"""Unit and integration tests for Output Guardrail (src/guardrails/output_guard.py)."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.guardrails.output_guard import (
    FALLBACK_NO_INFO_MESSAGE,
    OutputViolationType,
    apply_output_guard,
    check_empty_hallucination,
    inspect_output,
    mask_pii,
    sanitize_sources,
)
from src.main import app
from src.models.schemas import ChatResponse, SourceDocument


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Test PII Masking (Email, CCCD/CMND, Phone)
# ---------------------------------------------------------------------------

def test_mask_email():
    """Test redacting email addresses in text."""
    raw = "Vui lòng liên hệ support@company.vn hoặc john.doe+hr@gmail.com để được trợ giúp."
    masked, count, types = mask_pii(raw)
    assert count == 2
    assert "email" in types
    assert "support@company.vn" not in masked
    assert "john.doe+hr@gmail.com" not in masked
    assert "[EMAIL_REDACTED]" in masked


def test_mask_cccd_12_digits():
    """Test redacting 12-digit Vietnamese Citizen Identity Cards (CCCD)."""
    raw = "Số CCCD của công dân Nguyễn Văn A là 001099123456 và của chị B là 079 201 001234."
    masked, count, types = mask_pii(raw)
    assert count == 2
    assert "cccd" in types
    assert "001099123456" not in masked
    assert "079 201 001234" not in masked
    assert "[CCCD_REDACTED]" in masked


def test_mask_cmnd_contextual_9_digits():
    """Test redacting 9-digit ID cards when prefixed with contextual keywords."""
    raw = "Khách hàng xuất trình CMND số: 123456789 khi làm thủ tục."
    masked, count, types = mask_pii(raw)
    assert count >= 1
    assert "cccd" in types
    assert "123456789" not in masked
    assert "[CCCD_REDACTED]" in masked


def test_mask_vietnamese_phone_numbers():
    """Test redacting various Vietnamese mobile and landline phone number formats."""
    raw = (
        "Liên hệ hotline: 0912345678 hoặc 0987.654.321, "
        "số quốc tế +84 905 111 222, số bàn (024) 3823 4567."
    )
    masked, count, types = mask_pii(raw)
    assert count == 4
    assert "phone" in types
    assert "0912345678" not in masked
    assert "0987.654.321" not in masked
    assert "+84 905 111 222" not in masked
    assert "(024) 3823 4567" not in masked
    assert "[PHONE_REDACTED]" in masked


def test_preserve_non_pii_numbers():
    """Test that years, prices, and software versions are not accidentally masked."""
    raw = "Năm 2026, gói dịch vụ có giá 500.000 VNĐ trên phiên bản Python 3.11 tại Trang 12."
    masked, count, _ = mask_pii(raw)
    assert count == 0
    assert masked == raw


# ---------------------------------------------------------------------------
# Test Empty / Stub / Phantom Hallucination Detection
# ---------------------------------------------------------------------------

def test_hallucination_empty_or_whitespace():
    """Test detection of empty or whitespace-only answers."""
    is_hallu, v_type, reason, cleaned = check_empty_hallucination("   \n\t  ")
    assert is_hallu
    assert v_type == OutputViolationType.EMPTY_OR_DEGENERATE
    assert cleaned == FALLBACK_NO_INFO_MESSAGE


def test_hallucination_punctuation_only():
    """Test detection of degenerate punctuation-only answers."""
    is_hallu, v_type, _, cleaned = check_empty_hallucination("... --- ???")
    assert is_hallu
    assert v_type == OutputViolationType.EMPTY_OR_DEGENERATE
    assert cleaned == FALLBACK_NO_INFO_MESSAGE


def test_hallucination_prompt_echo_stub():
    """Test detection when LLM echoes template prefix with no real content."""
    is_hallu, v_type, _, cleaned = check_empty_hallucination("CÂU TRẢ LỜI: N/A")
    assert is_hallu
    assert v_type == OutputViolationType.PROMPT_ECHO_STUB
    assert cleaned == FALLBACK_NO_INFO_MESSAGE

    is_hallu2, v_type2, _, cleaned2 = check_empty_hallucination("Answer: None")
    assert is_hallu2
    assert v_type2 == OutputViolationType.PROMPT_ECHO_STUB
    assert cleaned2 == FALLBACK_NO_INFO_MESSAGE


def test_hallucination_phantom_citation_with_empty_sources():
    """Test phantom citation detection when sources list is empty."""
    raw_answer = "Theo tài liệu [1], doanh thu quý 3 đạt mức kỷ lục."
    is_hallu, v_type, reason, cleaned = check_empty_hallucination(raw_answer, sources=[])
    assert is_hallu
    assert v_type == OutputViolationType.PHANTOM_CITATION
    assert "fabricated citations" in reason
    assert cleaned == FALLBACK_NO_INFO_MESSAGE


def test_hallucination_phantom_citation_index_exceeded():
    """Test that citations exceeding available sources count are cleaned up."""
    sources = [SourceDocument(content="Nguồn 1 hợp lệ", metadata={"file_name": "doc1.txt"})]
    raw_answer = "Theo tài liệu [1] và tài liệu [5], thông tin này chính xác."

    is_hallu, v_type, _, cleaned = check_empty_hallucination(raw_answer, sources=sources)
    assert not is_hallu
    # [5] should be stripped while [1] is preserved
    assert "[1]" in cleaned
    assert "[5]" not in cleaned


def test_hallucination_contradictory_refusal():
    """Test detection when LLM claims no info but contradicts itself with speculation."""
    contradictory_text = (
        "Tài liệu được cung cấp không có đủ thông tin để trả lời câu hỏi này. "
        "Tuy nhiên theo tôi suy đoán thì hệ thống này rất có thể được xây dựng dựa trên công nghệ blockchain "
        "kết hợp với mạng nơ-ron sâu đa tầng và các giao thức bảo mật cấp quân sự tối tân chưa từng công bố."
    )
    is_hallu, v_type, _, cleaned = check_empty_hallucination(contradictory_text, sources=[])
    assert is_hallu
    assert v_type == OutputViolationType.CONTRADICTORY_REFUSAL
    assert cleaned == FALLBACK_NO_INFO_MESSAGE


# ---------------------------------------------------------------------------
# Test Source Document PII Masking
# ---------------------------------------------------------------------------

def test_sanitize_sources_masks_pii():
    """Test that PII in retrieved chunks is sanitized when MASK_PII_IN_SOURCES is True."""
    docs = [
        SourceDocument(
            content="Liên hệ kỹ thuật: dev@rag.com hoặc 0912345678",
            score=0.9,
            metadata={"file_name": "contact.txt"},
        )
    ]
    cleaned = sanitize_sources(docs)
    assert len(cleaned) == 1
    assert "dev@rag.com" not in cleaned[0].content
    assert "0912345678" not in cleaned[0].content
    assert "[EMAIL_REDACTED]" in cleaned[0].content
    assert "[PHONE_REDACTED]" in cleaned[0].content


# ---------------------------------------------------------------------------
# Test Full inspect_output & apply_output_guard
# ---------------------------------------------------------------------------

def test_inspect_output_combined_hallucination_and_pii():
    """Test inspect_output sanitizes PII on valid responses."""
    valid_with_pii = (
        "Để biết thêm chi tiết, hãy gửi email tới admin@domain.com "
        "hoặc gọi đến số 0988776655 để gặp chuyên viên."
    )
    docs = [SourceDocument(content="Dummy chunk")]
    res = inspect_output(valid_with_pii, sources=docs)
    assert not res.is_hallucination_detected
    assert res.pii_redacted_count == 2
    assert "[EMAIL_REDACTED]" in res.sanitized_answer
    assert "[PHONE_REDACTED]" in res.sanitized_answer


def test_apply_output_guard_on_chat_response():
    """Test apply_output_guard updates both answer and sources in ChatResponse."""
    resp = ChatResponse(
        answer="Liên hệ hotline: 0912345678",
        sources=[SourceDocument(content="Email: test@gmail.com")],
    )
    guarded = apply_output_guard(resp)
    assert "[PHONE_REDACTED]" in guarded.answer
    assert "0912345678" not in guarded.answer
    assert "[EMAIL_REDACTED]" in guarded.sources[0].content
    assert "test@gmail.com" not in guarded.sources[0].content


# ---------------------------------------------------------------------------
# Test API Endpoint Integration (POST /api/v1/chat)
# ---------------------------------------------------------------------------

def test_api_chat_redacts_pii_in_response(client):
    """Test that POST /api/v1/chat returns PII-redacted answer and sources."""
    mock_sources = [
        SourceDocument(
            content="Tác giả: admin@rag.vn, SĐT: 0912345678",
            metadata={"file_name": "contact.txt"},
        )
    ]
    mock_resp = ChatResponse(
        answer="Liên hệ tác giả qua email admin@rag.vn hoặc SĐT 0912345678.",
        sources=mock_sources,
    )

    with patch("src.api.v1.endpoints.chat.retrieve", return_value=mock_sources), \
         patch("src.api.v1.endpoints.chat.generate_answer", return_value=mock_resp):

        payload = {"query": "Cách liên hệ với tác giả?", "stream": False}
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "admin@rag.vn" not in data["answer"]
        assert "0912345678" not in data["answer"]
        assert "[EMAIL_REDACTED]" in data["answer"]
        assert "[PHONE_REDACTED]" in data["answer"]

        # Sources should also have PII redacted
        assert "admin@rag.vn" not in data["sources"][0]["content"]
        assert "[EMAIL_REDACTED]" in data["sources"][0]["content"]


def test_api_chat_empty_hallucination_fallback(client):
    """Test that POST /api/v1/chat falls back to standard message on empty model output."""
    mock_resp = ChatResponse(answer="   ", sources=[])

    with patch("src.api.v1.endpoints.chat.retrieve", return_value=[]), \
         patch("src.api.v1.endpoints.chat.generate_answer", return_value=mock_resp):

        payload = {"query": "Thông tin về dự án X?", "stream": False}
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["answer"] == FALLBACK_NO_INFO_MESSAGE

