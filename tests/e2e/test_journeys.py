"""End-to-end tests on the benefits side: multi-step user journeys.

These exercise the public HTTP surface as a coordinated sequence of calls
the way a real client would.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gauge.api import create_app
from gauge.benefits.seed import build_default_repository
from gauge.predictor.model import CostPredictor

pytestmark = pytest.mark.e2e


@pytest.fixture
def client(trained_predictor: CostPredictor) -> TestClient:
    return TestClient(
        create_app(build_default_repository(), trained_predictor)
    )


def test_journey_compare_in_network_vs_out_of_network(
    client: TestClient,
) -> None:
    """A user wants to know how much more an out-of-network MRI will cost."""
    procedure = client.get("/procedures/73721").json()
    assert procedure["in_network_rate_cents"] == 85_000
    assert procedure["billed_amount_cents"] == 240_000

    in_network = client.post(
        "/estimate",
        json={
            "member_id": "m2",
            "procedure_code": "73721",
            "in_network": True,
        },
    ).json()
    out_of_network = client.post(
        "/estimate",
        json={
            "member_id": "m2",
            "procedure_code": "73721",
            "in_network": False,
        },
    ).json()

    assert (
        out_of_network["member_pays_cents"]
        > in_network["member_pays_cents"]
    )
    assert any("Out-of-network" in n for n in out_of_network["notes"])


def test_journey_close_to_oop_max(client: TestClient) -> None:
    """A user near the OOP cap sees the cap respected on a big claim."""
    member = client.get("/members/m3").json()
    plan = client.get(f"/plans/{member['plan_id']}").json()
    remaining_oop = (
        plan["out_of_pocket_max_cents"] - member["ytd_out_of_pocket_cents"]
    )

    estimate = client.post(
        "/estimate",
        json={
            "member_id": "m3",
            "procedure_code": "29881",
            "in_network": True,
        },
    ).json()
    assert estimate["member_pays_cents"] <= remaining_oop
    assert any("Out-of-pocket maximum" in n for n in estimate["notes"])


def test_journey_copay_eligible_visit_skips_deductible(
    client: TestClient,
) -> None:
    """A fresh member's office visit on a copay plan does not touch deductible."""
    before = client.get("/members/m1").json()
    assert before["ytd_deductible_cents"] == 0

    estimate = client.post(
        "/estimate",
        json={
            "member_id": "m1",
            "procedure_code": "99213",
            "in_network": True,
        },
    ).json()
    assert estimate["deductible_applied_cents"] == 0
    assert estimate["copay_cents"] == 2_500

    after = client.get("/members/m1").json()
    assert after == before


def test_journey_unknown_inputs_surface_clear_errors(
    client: TestClient,
) -> None:
    """Garbage input produces actionable 404s, not 500s."""
    bad_member = client.post(
        "/estimate",
        json={
            "member_id": "ghost",
            "procedure_code": "99213",
            "in_network": True,
        },
    )
    assert bad_member.status_code == 404

    bad_proc = client.post(
        "/estimate",
        json={
            "member_id": "m1",
            "procedure_code": "00000",
            "in_network": True,
        },
    )
    assert bad_proc.status_code == 404
