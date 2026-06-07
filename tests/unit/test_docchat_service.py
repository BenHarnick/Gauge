"""Unit tests for the DocumentChatService."""

from __future__ import annotations

import pytest

from gauge.docchat.service import DocumentChatService

pytestmark = pytest.mark.unit


def test_upload_then_ask_round_trip(sample_plan_pdf_bytes: bytes) -> None:
    service = DocumentChatService()
    meta = service.upload_pdf("plan.pdf", sample_plan_pdf_bytes)
    assert meta.filename == "plan.pdf"
    assert meta.n_pages == 3
    assert meta.n_chunks >= 1

    response = service.ask(
        meta.document_id, "what is the office visit copay?"
    )
    # The copay info lives on page 2; the top citation should reflect that.
    assert response.citations, "expected citations"
    assert any(2 in c.page_numbers for c in response.citations)
    assert response.llm_used == "echo"
    assert response.question == "what is the office visit copay?"


def test_upload_rejects_invalid_pdf() -> None:
    service = DocumentChatService()
    with pytest.raises(ValueError):
        service.upload_pdf("bogus.pdf", b"not a pdf")


def test_ask_unknown_document_raises() -> None:
    service = DocumentChatService()
    with pytest.raises(KeyError):
        service.ask("missing-id", "any question")


def test_list_and_delete(sample_plan_pdf_bytes: bytes) -> None:
    service = DocumentChatService()
    meta = service.upload_pdf("plan.pdf", sample_plan_pdf_bytes)
    assert [d.document_id for d in service.store.list_meta()] == [
        meta.document_id
    ]
    assert service.store.delete(meta.document_id) is True
    assert service.store.list_meta() == []
    assert service.store.delete(meta.document_id) is False
