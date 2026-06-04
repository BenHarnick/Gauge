"""Integration tests for the benefits side of the FastAPI surface.

These use FastAPI's TestClient, which exercises the full request
lifecycle (routing, validation, dependency injection, serialization)
without binding to a real network socket.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from health_app.api import create_app
from health_app.predictor.model import CostPredictor
from health_app.benefits.repository import InMemoryRepository

pytestmark = pytest.mark.integration


@pytest.fixture
def client(
    seeded_repository: InMemoryRepository,
    trained_predictor: CostPredictor,
) -> TestClient:
    return TestClient(create_app(seeded_repository, trained_predictor))


class TestHealth:
    def test_healthz(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestCatalogEndpoints:
    def test_get_plan(self, client: TestClient) -> None:
        response = client.get("/plans/ppo_gold")
        assert response.status_code == 200
        body = response.json()
        assert body["plan_id"] == "ppo_gold"
        assert body["deductible_cents"] == 100_000

    def test_get_plan_missing(self, client: TestClient) -> None:
        response = client.get("/plans/does_not_exist")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_member(self, client: TestClient) -> None:
        response = client.get("/members/m1")
        assert response.status_code == 200
        assert response.json()["plan_id"] == "ppo_gold"

    def test_get_member_missing(self, client: TestClient) -> None:
        response = client.get("/members/nobody")
        assert response.status_code == 404

    def test_get_procedure(self, client: TestClient) -> None:
        response = client.get("/procedures/99213")
        assert response.status_code == 200
        assert response.json()["category"] == "office_visit"

    def test_get_procedure_missing(self, client: TestClient) -> None:
        response = client.get("/procedures/00000")
        assert response.status_code == 404


class TestEstimateEndpoint:
    def test_estimate_office_visit_uses_copay(self, client: TestClient) -> None:
        response = client.post(
            "/estimate",
            json={
                "member_id": "m1",
                "procedure_code": "99213",
                "in_network": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["copay_cents"] == 2_500
        assert body["member_pays_cents"] == 2_500
        assert body["plan_pays_cents"] == 12_500

    def test_estimate_unknown_member(self, client: TestClient) -> None:
        response = client.post(
            "/estimate",
            json={
                "member_id": "ghost",
                "procedure_code": "99213",
                "in_network": True,
            },
        )
        assert response.status_code == 404
        assert "Member 'ghost'" in response.json()["detail"]

    def test_estimate_unknown_procedure(self, client: TestClient) -> None:
        response = client.post(
            "/estimate",
            json={
                "member_id": "m1",
                "procedure_code": "00000",
                "in_network": True,
            },
        )
        assert response.status_code == 404

    def test_estimate_validation_error_on_missing_fields(
        self, client: TestClient
    ) -> None:
        response = client.post("/estimate", json={"member_id": "m1"})
        assert response.status_code == 422

    def test_estimate_defaults_to_in_network(self, client: TestClient) -> None:
        response = client.post(
            "/estimate",
            json={"member_id": "m1", "procedure_code": "99213"},
        )
        assert response.status_code == 200
        assert response.json()["copay_cents"] == 2_500

    def test_estimate_member_plan_not_in_repo(
        self,
        trained_predictor: CostPredictor,
    ) -> None:
        """Member exists but references a plan_id that isn't in the repo -> 404."""
        from health_app.benefits.models import Member
        from health_app.benefits.repository import InMemoryRepository

        member = Member(member_id="m_orphan", name="Orphan", plan_id="nonexistent_plan")
        repo = InMemoryRepository(plans=[], members=[member], procedures=[])
        orphan_client = TestClient(create_app(repo, trained_predictor))
        response = orphan_client.post(
            "/estimate",
            json={
                "member_id": "m_orphan",
                "procedure_code": "99213",
                "in_network": True,
            },
        )
        assert response.status_code == 404
        assert "nonexistent_plan" in response.json()["detail"]


def test_create_app_with_unfitted_predictor_raises() -> None:
    """create_app must reject a predictor that hasn't been fitted yet."""
    from health_app.api import create_app
    from health_app.benefits.seed import build_default_repository

    unfitted = CostPredictor()
    with pytest.raises(ValueError, match="fitted"):
        create_app(build_default_repository(), unfitted)
