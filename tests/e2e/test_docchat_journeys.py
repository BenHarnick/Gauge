"""End-to-end document chat journeys."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gauge.api import create_app
from gauge.benefits.seed import build_default_repository
from gauge.docchat.service import DocumentChatService
from gauge.predictor.model import CostPredictor

pytestmark = pytest.mark.e2e


@pytest.fixture
def client(trained_predictor: CostPredictor) -> TestClient:
    return TestClient(
        create_app(
            build_default_repository(),
            trained_predictor,
            DocumentChatService(),
        )
    )


def test_journey_upload_and_ask(
    client: TestClient, sample_plan_pdf_bytes: bytes
) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("plan.pdf", sample_plan_pdf_bytes, "application/pdf")},
    ).json()
    doc_id = upload["document"]["document_id"]

    questions_and_expected_pages = [
        ("what is the annual deductible?", 1),
        ("what is the specialist copay?", 2),
        ("what does a generic drug cost?", 3),
    ]
    for question, expected_page in questions_and_expected_pages:
        response = client.post(
            "/chat",
            json={"document_id": doc_id, "question": question},
        ).json()
        cited_pages = {
            p for c in response["citations"] for p in c["page_numbers"]
        }
        assert expected_page in cited_pages, (
            f"question {question!r} did not cite page {expected_page}; "
            f"got pages {cited_pages}"
        )


def test_journey_upload_then_delete_blocks_chat(
    client: TestClient, sample_plan_pdf_bytes: bytes
) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("plan.pdf", sample_plan_pdf_bytes, "application/pdf")},
    ).json()
    doc_id = upload["document"]["document_id"]
    client.delete(f"/documents/{doc_id}")
    follow_up = client.post(
        "/chat", json={"document_id": doc_id, "question": "anything?"}
    )
    assert follow_up.status_code == 404
