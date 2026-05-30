"""End-to-end tests for the guided session flow.

Each test simulates a complete multi-step user journey through the public
HTTP surface — the same sequence of calls the ``IntakeWizard`` frontend
component makes.  All LLM calls use ``EchoLLM`` so no API key is required.

Journeys covered:

1. Full happy path — demographics → PDF upload → confirm plan → estimate →
   what-if → chat.
2. Skip-PDF path — demographics → manual plan entry → estimate → what-if
   (no document means no chat).
3. Multi-session isolation — two sessions in parallel don't bleed state.
4. Re-confirmation — confirming a plan twice updates the stored plan.
5. Error guard — calling out-of-order returns the right HTTP codes.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from health_app.api import create_app
from health_app.benefits.seed import build_default_repository
from health_app.docchat.llm import EchoLLM
from health_app.docchat.service import DocumentChatService
from health_app.plan_extract.extractor import PlanExtractor
from health_app.predictor.model import CostPredictor
from health_app.session.store import InMemorySessionStore

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(trained_predictor: CostPredictor) -> TestClient:
    llm = EchoLLM()
    chat_service = DocumentChatService(llm=llm)
    return TestClient(
        create_app(
            repository=build_default_repository(),
            predictor=trained_predictor,
            chat_service=chat_service,
            session_store=InMemorySessionStore(),
            plan_extractor=PlanExtractor(llm=llm),
        )
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _features(smoker: str = "no", age: int = 35) -> dict:
    return {
        "age": age,
        "sex": "female",
        "bmi": 27.5,
        "children": 1,
        "smoker": smoker,
        "region": "northeast",
    }


def _plan_payload(**overrides) -> dict:
    base = {
        "deductible_cents": 150_000,
        "out_of_pocket_max_cents": 600_000,
        "coinsurance_rate": 0.20,
        "copays_cents": {"office_visit": 3_000},
        "plan_name": "Journey Plan",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Journey 1 — full happy path with PDF upload
# ---------------------------------------------------------------------------


def test_journey_full_guided_flow(
    client: TestClient, sample_plan_pdf_bytes: bytes
) -> None:
    """Complete four-step flow: create → upload → confirm → estimate + what-if + chat."""

    # Step 1: create session
    create_resp = client.post("/sessions", json={"features": _features()})
    assert create_resp.status_code == 200
    sid = create_resp.json()["session_id"]
    initial_prediction = create_resp.json()["prediction"]
    assert initial_prediction["median_charges_cents"] > 0

    # Step 2: upload plan PDF
    upload_resp = client.post(
        f"/sessions/{sid}/document",
        files={"file": ("plan.pdf", io.BytesIO(sample_plan_pdf_bytes), "application/pdf")},
    )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["document_id"]
    draft = upload_resp.json()["plan_draft"]
    assert doc_id

    # Step 2b: fetch draft via dedicated endpoint
    draft_resp = client.get(f"/sessions/{sid}/plan-draft")
    assert draft_resp.status_code == 200
    assert draft_resp.json()["deductible_cents"] == draft["deductible_cents"]

    # Step 3: confirm plan
    confirm_resp = client.post(f"/sessions/{sid}/plan", json=_plan_payload())
    assert confirm_resp.status_code == 200
    estimate = confirm_resp.json()
    assert estimate["plan"]["name"] == "Journey Plan"
    assert estimate["document_id"] == doc_id
    share = estimate["annual_plan_share_median"]
    assert share is not None
    assert share["member_pays_cents"] + share["plan_pays_cents"] == share["charges_cents"]

    # Step 4a: fetch estimate via GET
    get_est = client.get(f"/sessions/{sid}/estimate")
    assert get_est.status_code == 200
    assert get_est.json()["plan"]["name"] == "Journey Plan"

    # Step 4b: what-if sweep
    wi_resp = client.post(
        f"/sessions/{sid}/whatif",
        json={"feature": "age", "values": [25, 35, 45, 55]},
    )
    assert wi_resp.status_code == 200
    points = wi_resp.json()["points"]
    assert len(points) == 4
    assert all(p["prediction"]["median_charges_cents"] >= 0 for p in points)

    # Step 4c: chat against uploaded document
    chat_resp = client.post(
        f"/sessions/{sid}/chat",
        json={"question": "What is the annual deductible?"},
    )
    assert chat_resp.status_code == 200
    body = chat_resp.json()
    assert body["answer"]
    assert isinstance(body["citations"], list)


# ---------------------------------------------------------------------------
# Journey 2 — skip PDF, enter plan details manually
# ---------------------------------------------------------------------------


def test_journey_skip_pdf_manual_plan_entry(client: TestClient) -> None:
    """User skips PDF upload and goes straight to confirming plan details."""

    # Step 1: create session
    sid = client.post("/sessions", json={"features": _features()}).json()["session_id"]

    # Plan draft endpoint should 404 (no upload yet) — correct guard behaviour.
    assert client.get(f"/sessions/{sid}/plan-draft").status_code == 404

    # Chat should also 404 (no document).
    assert (
        client.post(
            f"/sessions/{sid}/chat",
            json={"question": "Is telehealth covered?"},
        ).status_code
        == 404
    )

    # Step 3: confirm plan without a document
    confirm = client.post(f"/sessions/{sid}/plan", json=_plan_payload()).json()
    assert confirm["annual_plan_share_median"] is not None
    assert confirm["document_id"] is None

    # Step 4: what-if still works
    wi = client.post(
        f"/sessions/{sid}/whatif",
        json={"feature": "smoker", "values": ["no", "yes"]},
    ).json()
    assert len(wi["points"]) == 2


# ---------------------------------------------------------------------------
# Journey 3 — two sessions in parallel do not bleed state
# ---------------------------------------------------------------------------


def test_journey_multi_session_isolation(
    client: TestClient, sample_plan_pdf_bytes: bytes
) -> None:
    """Concurrent sessions hold independent state."""
    sid_a = client.post("/sessions", json={"features": _features(smoker="no")}).json()["session_id"]
    sid_b = client.post("/sessions", json={"features": _features(smoker="yes")}).json()["session_id"]

    # Confirm only session A.
    client.post(f"/sessions/{sid_a}/plan", json=_plan_payload(plan_name="Plan A"))

    # Session B should have no plan yet.
    est_b = client.get(f"/sessions/{sid_b}/estimate").json()
    assert est_b["plan"] is None

    # Session A should have its plan.
    est_a = client.get(f"/sessions/{sid_a}/estimate").json()
    assert est_a["plan"]["name"] == "Plan A"

    # Confirm session B with a different plan.
    client.post(f"/sessions/{sid_b}/plan", json=_plan_payload(plan_name="Plan B"))
    est_b_after = client.get(f"/sessions/{sid_b}/estimate").json()
    assert est_b_after["plan"]["name"] == "Plan B"

    # Session A is unchanged.
    assert client.get(f"/sessions/{sid_a}/estimate").json()["plan"]["name"] == "Plan A"


# ---------------------------------------------------------------------------
# Journey 4 — re-confirming a plan overwrites the previous one
# ---------------------------------------------------------------------------


def test_journey_reconfirm_plan_updates_estimate(client: TestClient) -> None:
    sid = client.post("/sessions", json={"features": _features()}).json()["session_id"]

    # First confirmation: low deductible.
    client.post(
        f"/sessions/{sid}/plan",
        json=_plan_payload(deductible_cents=10_000, plan_name="Bronze"),
    )
    est1 = client.get(f"/sessions/{sid}/estimate").json()
    assert est1["plan"]["name"] == "Bronze"
    assert est1["plan"]["deductible_cents"] == 10_000

    # Re-confirm with a different plan.
    client.post(
        f"/sessions/{sid}/plan",
        json=_plan_payload(deductible_cents=500_000, plan_name="Gold"),
    )
    est2 = client.get(f"/sessions/{sid}/estimate").json()
    assert est2["plan"]["name"] == "Gold"
    assert est2["plan"]["deductible_cents"] == 500_000

    # A higher deductible should mean equal or higher member OOP.
    assert (
        est2["annual_plan_share_median"]["member_pays_cents"]
        >= est1["annual_plan_share_median"]["member_pays_cents"]
    )


# ---------------------------------------------------------------------------
# Journey 5 — error guard: out-of-order and bad inputs
# ---------------------------------------------------------------------------


def test_journey_out_of_order_calls_return_useful_errors(
    client: TestClient,
) -> None:
    """Calls made against a non-existent session return clear 404s."""
    fake_id = "doesnotexist123"

    assert client.get(f"/sessions/{fake_id}/estimate").status_code == 404
    assert client.get(f"/sessions/{fake_id}/plan-draft").status_code == 404
    assert (
        client.post(
            f"/sessions/{fake_id}/plan", json=_plan_payload()
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/sessions/{fake_id}/whatif",
            json={"feature": "age", "values": [30]},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/sessions/{fake_id}/chat",
            json={"question": "Hello?"},
        ).status_code
        == 404
    )


def test_journey_smoker_prediction_higher_than_non_smoker(
    client: TestClient,
) -> None:
    """ML sanity check: smoker estimate >= non-smoker estimate on median."""
    ns_id = client.post("/sessions", json={"features": _features(smoker="no")}).json()["session_id"]
    s_id = client.post("/sessions", json={"features": _features(smoker="yes")}).json()["session_id"]

    client.post(f"/sessions/{ns_id}/plan", json=_plan_payload())
    client.post(f"/sessions/{s_id}/plan", json=_plan_payload())

    ns_est = client.get(f"/sessions/{ns_id}/estimate").json()
    s_est = client.get(f"/sessions/{s_id}/estimate").json()

    assert (
        s_est["prediction"]["median_charges_cents"]
        >= ns_est["prediction"]["median_charges_cents"]
    )


def test_journey_oop_cap_respected_in_estimate(client: TestClient) -> None:
    """Member OOP should never exceed the plan's out-of-pocket maximum."""
    # Force a high predicted spend by using a smoker with high age.
    sid = client.post(
        "/sessions", json={"features": _features(smoker="yes", age=60)}
    ).json()["session_id"]

    oop_max_cents = 300_000
    est = client.post(
        f"/sessions/{sid}/plan",
        json=_plan_payload(
            deductible_cents=50_000,
            out_of_pocket_max_cents=oop_max_cents,
            coinsurance_rate=0.20,
        ),
    ).json()

    share_median = est["annual_plan_share_median"]
    share_mean = est["annual_plan_share_mean"]
    assert share_median["member_pays_cents"] <= oop_max_cents
    assert share_mean["member_pays_cents"] <= oop_max_cents
