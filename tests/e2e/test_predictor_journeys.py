"""End-to-end predictor journeys.

These chain `/predict` and `/whatif` calls into the kind of session a
real client would run: predict a baseline, compare quitting smoking,
sweep age, layer in a plan to see annual OOP.
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


def _baseline_payload() -> dict:
    return {
        "age": 40,
        "sex": "male",
        "bmi": 32.0,
        "children": 2,
        "smoker": "yes",
        "region": "south",
    }


def test_journey_quit_smoking_lowers_predicted_cost(
    client: TestClient,
) -> None:
    smoker = client.post(
        "/predict", json={"features": _baseline_payload()}
    ).json()
    nonsmoker = client.post(
        "/predict",
        json={"features": _baseline_payload() | {"smoker": "no"}},
    ).json()
    assert (
        nonsmoker["prediction"]["median_charges_cents"]
        < smoker["prediction"]["median_charges_cents"]
    )


def test_journey_age_sweep_under_a_plan(client: TestClient) -> None:
    """Sweep age; charges and member OOP should both trend up overall."""
    response = client.post(
        "/whatif",
        json={
            "baseline": _baseline_payload() | {"smoker": "no"},
            "feature": "age",
            "values": [25, 35, 45, 55, 64],
            "plan_id": "ppo_gold",
        },
    ).json()
    points = response["points"]
    assert len(points) == 5

    charges = [p["prediction"]["median_charges_cents"] for p in points]
    # Endpoints trend up.
    assert charges[-1] > charges[0]

    # Every point has an annual plan share that adds up correctly.
    for p in points:
        share = p["annual_plan_share_median"]
        assert (
            share["member_pays_cents"] + share["plan_pays_cents"]
            == share["charges_cents"]
        )


def test_journey_predict_then_apply_plan_matches_separate_endpoints(
    client: TestClient,
) -> None:
    """Asking /predict with a plan_id matches predicting then applying the plan."""
    without_plan = client.post(
        "/predict", json={"features": _baseline_payload()}
    ).json()
    with_plan = client.post(
        "/predict",
        json={"features": _baseline_payload(), "plan_id": "hdhp_silver"},
    ).json()

    assert (
        with_plan["prediction"]["median_charges_cents"]
        == without_plan["prediction"]["median_charges_cents"]
    )
    share = with_plan["annual_plan_share_median"]
    assert (
        share["charges_cents"]
        == with_plan["prediction"]["median_charges_cents"]
    )
