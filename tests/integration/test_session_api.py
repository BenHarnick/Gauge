"""Integration tests for the guided-session API endpoints.

Exercises all seven session routes via FastAPI's ``TestClient``:

  POST   /sessions
  POST   /sessions/{id}/document
  GET    /sessions/{id}/plan-draft
  POST   /sessions/{id}/plan
  GET    /sessions/{id}/estimate
  POST   /sessions/{id}/whatif
  POST   /sessions/{id}/chat

Each test class is scoped to one endpoint family and covers the happy path,
error cases, and validation constraints.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from health_app.api import create_app
from health_app.benefits.repository import InMemoryRepository
from health_app.docchat.llm import EchoLLM
from health_app.docchat.service import DocumentChatService
from health_app.plan_extract.extractor import PlanExtractor
from health_app.predictor.model import CostPredictor
from health_app.session.store import InMemorySessionStore

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(
    seeded_repository: InMemoryRepository,
    trained_predictor: CostPredictor,
) -> TestClient:
    """Fresh app with isolated session store and EchoLLM for every test."""
    llm = EchoLLM()
    chat_service = DocumentChatService(llm=llm)
    return TestClient(
        create_app(
            repository=seeded_repository,
            predictor=trained_predictor,
            chat_service=chat_service,
            session_store=InMemorySessionStore(),
            plan_extractor=PlanExtractor(llm=llm),
        )
    )


def _features_payload() -> dict:
    return {
        "age": 35,
        "sex": "female",
        "bmi": 27.5,
        "children": 1,
        "smoker": "no",
        "region": "northeast",
    }


def _confirm_plan_payload(**overrides) -> dict:
    base = {
        "deductible_cents": 150_000,
        "out_of_pocket_max_cents": 600_000,
        "coinsurance_rate": 0.20,
        "copays_cents": {},
        "plan_name": "Test Plan",
    }
    base.update(overrides)
    return base


def _create_session(client: TestClient) -> str:
    """Create a session and return its ID."""
    resp = client.post("/sessions", json={"features": _features_payload()})
    assert resp.status_code == 200
    return resp.json()["session_id"]


def _upload_pdf(client: TestClient, session_id: str, pdf_bytes: bytes) -> dict:
    """Upload a plan PDF to a session and return the response body."""
    resp = client.post(
        f"/sessions/{session_id}/document",
        files={"file": ("plan.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_returns_session_id_and_prediction(self, client: TestClient) -> None:
        resp = client.post("/sessions", json={"features": _features_payload()})
        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert body["session_id"]  # non-empty
        pred = body["prediction"]
        assert pred["lower_bound_cents"] <= pred["median_charges_cents"]
        assert pred["median_charges_cents"] <= pred["upper_bound_cents"]
        assert pred["mean_charges_cents"] >= 0

    def test_each_call_produces_unique_session_id(self, client: TestClient) -> None:
        ids = {
            client.post("/sessions", json={"features": _features_payload()}).json()["session_id"]
            for _ in range(5)
        }
        assert len(ids) == 5

    def test_missing_features_returns_422(self, client: TestClient) -> None:
        resp = client.post("/sessions", json={})
        assert resp.status_code == 422

    def test_invalid_age_returns_422(self, client: TestClient) -> None:
        bad = _features_payload() | {"age": -1}
        resp = client.post("/sessions", json={"features": bad})
        assert resp.status_code == 422

    def test_invalid_region_returns_422(self, client: TestClient) -> None:
        bad = _features_payload() | {"region": "narnia"}
        resp = client.post("/sessions", json={"features": bad})
        assert resp.status_code == 422

    def test_smoker_yes_gives_higher_prediction(self, client: TestClient) -> None:
        non_smoker = _features_payload() | {"smoker": "no"}
        smoker = _features_payload() | {"smoker": "yes"}
        resp_ns = client.post("/sessions", json={"features": non_smoker}).json()
        resp_s = client.post("/sessions", json={"features": smoker}).json()
        assert (
            resp_s["prediction"]["median_charges_cents"]
            >= resp_ns["prediction"]["median_charges_cents"]
        )


# ---------------------------------------------------------------------------
# POST /sessions/{id}/document
# ---------------------------------------------------------------------------


class TestAttachDocument:
    def test_upload_pdf_returns_document_id_and_draft(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        sid = _create_session(client)
        body = _upload_pdf(client, sid, sample_plan_pdf_bytes)
        assert body["document_id"]
        assert "plan_draft" in body

    def test_draft_contains_expected_fields(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        sid = _create_session(client)
        body = _upload_pdf(client, sid, sample_plan_pdf_bytes)
        draft = body["plan_draft"]
        # EchoLLM returns retrieved text; deductible appears in sample PDF.
        # We just assert the draft schema is present and correct.
        assert "deductible_cents" in draft
        assert "out_of_pocket_max_cents" in draft
        assert "coinsurance_rate" in draft
        assert "copays_cents" in draft
        assert isinstance(draft["unresolved_fields"], list)

    def test_unknown_session_returns_404(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        resp = client.post(
            "/sessions/ghost/document",
            files={"file": ("plan.pdf", io.BytesIO(sample_plan_pdf_bytes), "application/pdf")},
        )
        assert resp.status_code == 404

    def test_empty_file_returns_400(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/document",
            files={"file": ("plan.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert resp.status_code == 400

    def test_non_pdf_content_type_returns_415(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/document",
            files={"file": ("doc.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 415

    def test_oversized_pdf_returns_413(self, client: TestClient) -> None:
        from health_app.api import MAX_PDF_BYTES

        sid = _create_session(client)
        oversized = b"%" * (MAX_PDF_BYTES + 1)
        resp = client.post(
            f"/sessions/{sid}/document",
            files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")},
        )
        assert resp.status_code == 413

    def test_unparseable_pdf_returns_400(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/document",
            files={"file": ("bad.pdf", io.BytesIO(b"not-a-real-pdf"), "application/pdf")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /sessions/{id}/plan-draft
# ---------------------------------------------------------------------------


class TestGetPlanDraft:
    def test_returns_draft_after_upload(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        sid = _create_session(client)
        _upload_pdf(client, sid, sample_plan_pdf_bytes)
        resp = client.get(f"/sessions/{sid}/plan-draft")
        assert resp.status_code == 200
        assert "deductible_cents" in resp.json()

    def test_unknown_session_returns_404(self, client: TestClient) -> None:
        resp = client.get("/sessions/ghost/plan-draft")
        assert resp.status_code == 404

    def test_no_document_yet_returns_404(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.get(f"/sessions/{sid}/plan-draft")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /sessions/{id}/plan
# ---------------------------------------------------------------------------


class TestConfirmPlan:
    def test_returns_full_estimate(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/plan",
            json=_confirm_plan_payload(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "prediction" in body
        assert "plan" in body
        assert body["plan"]["name"] == "Test Plan"
        assert body["annual_plan_share_median"] is not None
        assert body["annual_plan_share_mean"] is not None

    def test_cost_share_adds_up(self, client: TestClient) -> None:
        sid = _create_session(client)
        body = client.post(
            f"/sessions/{sid}/plan", json=_confirm_plan_payload()
        ).json()
        share = body["annual_plan_share_median"]
        assert share["member_pays_cents"] + share["plan_pays_cents"] == share["charges_cents"]

    def test_document_id_included_when_doc_uploaded(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        sid = _create_session(client)
        upload = _upload_pdf(client, sid, sample_plan_pdf_bytes)
        body = client.post(
            f"/sessions/{sid}/plan", json=_confirm_plan_payload()
        ).json()
        assert body["document_id"] == upload["document_id"]

    def test_document_id_null_without_upload(self, client: TestClient) -> None:
        sid = _create_session(client)
        body = client.post(
            f"/sessions/{sid}/plan", json=_confirm_plan_payload()
        ).json()
        assert body["document_id"] is None

    def test_unknown_session_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/sessions/ghost/plan", json=_confirm_plan_payload()
        )
        assert resp.status_code == 404

    def test_negative_deductible_returns_422(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/plan",
            json=_confirm_plan_payload(deductible_cents=-1),
        )
        assert resp.status_code == 422

    def test_coinsurance_above_one_returns_422(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/plan",
            json=_confirm_plan_payload(coinsurance_rate=1.5),
        )
        assert resp.status_code == 422

    def test_higher_deductible_increases_member_share(
        self, client: TestClient
    ) -> None:
        """A higher deductible should never decrease member out-of-pocket."""
        sid_low = _create_session(client)
        sid_high = _create_session(client)
        low = client.post(
            f"/sessions/{sid_low}/plan",
            json=_confirm_plan_payload(deductible_cents=50_000),
        ).json()
        high = client.post(
            f"/sessions/{sid_high}/plan",
            json=_confirm_plan_payload(deductible_cents=500_000),
        ).json()
        assert (
            high["annual_plan_share_median"]["member_pays_cents"]
            >= low["annual_plan_share_median"]["member_pays_cents"]
        )


# ---------------------------------------------------------------------------
# GET /sessions/{id}/estimate
# ---------------------------------------------------------------------------


class TestGetEstimate:
    def test_before_plan_returns_null_shares(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.get(f"/sessions/{sid}/estimate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["annual_plan_share_median"] is None
        assert body["annual_plan_share_mean"] is None
        assert body["plan"] is None

    def test_after_plan_confirmed_returns_full_estimate(
        self, client: TestClient
    ) -> None:
        sid = _create_session(client)
        client.post(f"/sessions/{sid}/plan", json=_confirm_plan_payload())
        resp = client.get(f"/sessions/{sid}/estimate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["annual_plan_share_median"] is not None
        assert body["plan"]["name"] == "Test Plan"

    def test_unknown_session_returns_404(self, client: TestClient) -> None:
        resp = client.get("/sessions/ghost/estimate")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /sessions/{id}/whatif
# ---------------------------------------------------------------------------


class TestSessionWhatIf:
    def test_age_sweep_returns_correct_points(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/whatif",
            json={"feature": "age", "values": [25, 40, 55]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["feature"] == "age"
        assert [p["value"] for p in body["points"]] == [25, 40, 55]

    def test_smoker_sweep_two_points(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/whatif",
            json={"feature": "smoker", "values": ["no", "yes"]},
        )
        assert resp.status_code == 200
        assert len(resp.json()["points"]) == 2

    def test_all_points_have_prediction(self, client: TestClient) -> None:
        sid = _create_session(client)
        body = client.post(
            f"/sessions/{sid}/whatif",
            json={"feature": "bmi", "values": [20.0, 30.0, 40.0]},
        ).json()
        for pt in body["points"]:
            assert pt["prediction"]["median_charges_cents"] >= 0

    def test_unknown_session_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/sessions/ghost/whatif",
            json={"feature": "age", "values": [30]},
        )
        assert resp.status_code == 404

    def test_invalid_feature_returns_400(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/whatif",
            json={"feature": "magic_field", "values": [1, 2]},
        )
        assert resp.status_code == 400

    def test_empty_values_returns_empty_points(self, client: TestClient) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/whatif",
            json={"feature": "age", "values": []},
        )
        assert resp.status_code == 200
        assert resp.json()["points"] == []

    def test_out_of_range_value_returns_400(self, client: TestClient) -> None:
        """sweep() raises ValueError for invalid values; endpoint maps it to 400."""
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/whatif",
            json={"feature": "age", "values": [-999]},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /sessions/{id}/chat
# ---------------------------------------------------------------------------


class TestSessionChat:
    def test_chat_returns_answer_and_citations(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        sid = _create_session(client)
        _upload_pdf(client, sid, sample_plan_pdf_bytes)
        resp = client.post(
            f"/sessions/{sid}/chat",
            json={"question": "What is the deductible?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"]
        assert isinstance(body["citations"], list)

    def test_chat_without_document_returns_404(
        self, client: TestClient
    ) -> None:
        sid = _create_session(client)
        resp = client.post(
            f"/sessions/{sid}/chat",
            json={"question": "What is covered?"},
        )
        assert resp.status_code == 404

    def test_unknown_session_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/sessions/ghost/chat",
            json={"question": "Hello?"},
        )
        assert resp.status_code == 404

    def test_empty_question_returns_422(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        sid = _create_session(client)
        _upload_pdf(client, sid, sample_plan_pdf_bytes)
        resp = client.post(
            f"/sessions/{sid}/chat",
            json={"question": ""},
        )
        assert resp.status_code == 422

    def test_question_too_long_returns_422(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        sid = _create_session(client)
        _upload_pdf(client, sid, sample_plan_pdf_bytes)
        resp = client.post(
            f"/sessions/{sid}/chat",
            json={"question": "x" * 2_001},
        )
        assert resp.status_code == 422

    def test_top_k_above_max_returns_422(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        sid = _create_session(client)
        _upload_pdf(client, sid, sample_plan_pdf_bytes)
        resp = client.post(
            f"/sessions/{sid}/chat",
            json={"question": "What is the copay?", "top_k": 99},
        )
        assert resp.status_code == 422

    def test_chat_document_deleted_after_attach_returns_404(
        self, client: TestClient, sample_plan_pdf_bytes: bytes
    ) -> None:
        """If the document is deleted from the store after attachment, chat returns 404."""
        sid = _create_session(client)
        upload_body = _upload_pdf(client, sid, sample_plan_pdf_bytes)
        doc_id = upload_body["document_id"]

        # Delete the document from the store.
        del_resp = client.delete(f"/documents/{doc_id}")
        assert del_resp.status_code == 204

        # Chat now references a session whose document no longer exists.
        resp = client.post(
            f"/sessions/{sid}/chat",
            json={"question": "What is covered?"},
        )
        assert resp.status_code == 404
