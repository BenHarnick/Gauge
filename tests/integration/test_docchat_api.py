"""Integration tests for the docchat API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from health_app.api import create_app
from health_app.benefits.repository import InMemoryRepository
from health_app.docchat.service import DocumentChatService
from health_app.predictor.model import CostPredictor

pytestmark = pytest.mark.integration


@pytest.fixture
def client(
    seeded_repository: InMemoryRepository,
    trained_predictor: CostPredictor,
) -> TestClient:
    return TestClient(
        create_app(
            seeded_repository,
            trained_predictor,
            DocumentChatService(),
        )
    )


class TestUpload:
    def test_upload_pdf(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        response = client.post(
            "/documents",
            files={"file": ("plan.pdf", sample_plan_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["document"]["filename"] == "plan.pdf"
        assert body["document"]["n_pages"] == 3
        assert body["document"]["n_chunks"] >= 1

    def test_upload_empty_file_400(self, client: TestClient) -> None:
        response = client.post(
            "/documents",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 400

    def test_upload_non_pdf_415(self, client: TestClient) -> None:
        response = client.post(
            "/documents",
            files={"file": ("readme.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 415

    def test_upload_oversized_file_413(self, client: TestClient) -> None:
        from health_app.api import MAX_PDF_BYTES

        oversized = b"%" * (MAX_PDF_BYTES + 1)
        response = client.post(
            "/documents",
            files={"file": ("big.pdf", oversized, "application/pdf")},
        )
        assert response.status_code == 413

    def test_upload_unparseable_pdf_400(self, client: TestClient) -> None:
        """Bytes that claim to be PDF but fail parsing return 400."""
        response = client.post(
            "/documents",
            files={"file": ("bad.pdf", b"not-a-pdf-at-all", "application/pdf")},
        )
        assert response.status_code == 400


class TestListAndDelete:
    def test_list_starts_empty(self, client: TestClient) -> None:
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json() == []

    def test_delete_unknown_404(self, client: TestClient) -> None:
        response = client.delete("/documents/missing")
        assert response.status_code == 404

    def test_list_then_delete(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        upload = client.post(
            "/documents",
            files={"file": ("plan.pdf", sample_plan_pdf_bytes, "application/pdf")},
        ).json()
        doc_id = upload["document"]["document_id"]

        listing = client.get("/documents").json()
        assert [d["document_id"] for d in listing] == [doc_id]

        deleted = client.delete(f"/documents/{doc_id}")
        assert deleted.status_code == 204
        assert client.get("/documents").json() == []


class TestChat:
    def test_chat_returns_citations(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        upload = client.post(
            "/documents",
            files={"file": ("plan.pdf", sample_plan_pdf_bytes, "application/pdf")},
        ).json()
        doc_id = upload["document"]["document_id"]

        response = client.post(
            "/chat",
            json={
                "document_id": doc_id,
                "question": "what is the office visit copay?",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["document_id"] == doc_id
        assert body["llm_used"] == "echo"
        assert body["citations"]

    def test_chat_unknown_document_404(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={"document_id": "missing", "question": "anything?"},
        )
        assert response.status_code == 404

    def test_chat_validation_empty_question(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        upload = client.post(
            "/documents",
            files={"file": ("plan.pdf", sample_plan_pdf_bytes, "application/pdf")},
        ).json()
        doc_id = upload["document"]["document_id"]
        response = client.post(
            "/chat", json={"document_id": doc_id, "question": ""}
        )
        assert response.status_code == 422
