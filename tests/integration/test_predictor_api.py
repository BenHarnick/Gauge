"""Integration tests for the predictor API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from health_app.api import create_app
from health_app.benefits.repository import InMemoryRepository
from health_app.predictor.model import CostPredictor

pytestmark = pytest.mark.integration


@pytest.fixture
def client(
    seeded_repository: InMemoryRepository,
    trained_predictor: CostPredictor,
) -> TestClient:
    return TestClient(create_app(seeded_repository, trained_predictor))


def _baseline_payload() -> dict:
    return {
        "age": 35,
        "sex": "female",
        "bmi": 27.5,
        "children": 1,
        "smoker": "no",
        "region": "northeast",
    }


class TestPredict:
    def test_predict_returns_interval(self, client: TestClient) -> None:
        response = client.post(
            "/predict", json={"features": _baseline_payload()}
        )
        assert response.status_code == 200
        body = response.json()
        pred = body["prediction"]
        assert pred["lower_bound_cents"] <= pred["median_charges_cents"]
        assert pred["median_charges_cents"] <= pred["upper_bound_cents"]
        assert pred["mean_charges_cents"] >= 0
        assert body["annual_plan_share_median"] is None
        assert body["annual_plan_share_mean"] is None

    def test_predict_with_plan_returns_annual_share(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/predict",
            json={
                "features": _baseline_payload(),
                "plan_id": "ppo_gold",
            },
        )
        assert response.status_code == 200
        body = response.json()
        share_median = body["annual_plan_share_median"]
        share_mean = body["annual_plan_share_mean"]
        assert share_median is not None and share_mean is not None
        assert (
            share_median["member_pays_cents"] + share_median["plan_pays_cents"]
            == share_median["charges_cents"]
        )
        assert (
            share_mean["member_pays_cents"] + share_mean["plan_pays_cents"]
            == share_mean["charges_cents"]
        )

    def test_predict_unknown_plan_404(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            json={"features": _baseline_payload(), "plan_id": "missing"},
        )
        assert response.status_code == 404

    def test_predict_validation_rejects_bad_age(
        self, client: TestClient
    ) -> None:
        bad = _baseline_payload() | {"age": -5}
        response = client.post("/predict", json={"features": bad})
        assert response.status_code == 422

    def test_predict_validation_rejects_bad_region(
        self, client: TestClient
    ) -> None:
        bad = _baseline_payload() | {"region": "atlantis"}
        response = client.post("/predict", json={"features": bad})
        assert response.status_code == 422


class TestWhatIf:
    def test_whatif_age_sweep(self, client: TestClient) -> None:
        response = client.post(
            "/whatif",
            json={
                "baseline": _baseline_payload(),
                "feature": "age",
                "values": [25, 40, 55],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["feature"] == "age"
        assert [p["value"] for p in body["points"]] == [25, 40, 55]

    def test_whatif_with_plan_includes_annual_share(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/whatif",
            json={
                "baseline": _baseline_payload(),
                "feature": "smoker",
                "values": ["no", "yes"],
                "plan_id": "ppo_gold",
            },
        )
        assert response.status_code == 200
        for point in response.json()["points"]:
            assert point["annual_plan_share_median"] is not None
            assert point["annual_plan_share_mean"] is not None

    def test_whatif_rejects_unknown_feature(self, client: TestClient) -> None:
        response = client.post(
            "/whatif",
            json={
                "baseline": _baseline_payload(),
                "feature": "magic",
                "values": [1, 2, 3],
            },
        )
        # Literal[...] validation kicks in at the request body level,
        # so this is a 422 from pydantic.
        assert response.status_code == 422

    def test_whatif_unknown_plan_404(self, client: TestClient) -> None:
        response = client.post(
            "/whatif",
            json={
                "baseline": _baseline_payload(),
                "feature": "age",
                "values": [30],
                "plan_id": "missing",
            },
        )
        assert response.status_code == 404
