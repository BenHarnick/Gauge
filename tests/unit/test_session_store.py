"""Unit tests for the session store and session/estimate models.

Covers ``InMemorySessionStore`` (CRUD, concurrency) and the Pydantic schemas
in ``health_app.session.models`` (validation, defaults, serialisation).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from health_app.plan_extract.schemas import PlanDraft
from health_app.predictor.schemas import PredictionFeatures
from health_app.session.models import (
    ConfirmPlanRequest,
    CreateSessionRequest,
    Session,
    SessionEstimate,
)
from health_app.session.store import InMemorySessionStore
from health_app.predictor.model import CostPrediction

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _features() -> PredictionFeatures:
    return PredictionFeatures(
        age=35,
        sex="female",
        bmi=27.5,
        children=1,
        smoker="no",
        region="northeast",
    )


def _session(session_id: str = "abc123") -> Session:
    return Session(session_id=session_id, features=_features())


# ---------------------------------------------------------------------------
# InMemorySessionStore — basic CRUD
# ---------------------------------------------------------------------------


class TestInMemorySessionStoreCreate:
    def test_create_then_get_returns_session(self) -> None:
        store = InMemorySessionStore()
        s = _session()
        store.create(s)
        retrieved = store.get(s.session_id)
        assert retrieved is not None
        assert retrieved.session_id == s.session_id

    def test_get_missing_returns_none(self) -> None:
        store = InMemorySessionStore()
        assert store.get("nonexistent") is None

    def test_create_overwrites_existing(self) -> None:
        """Creating a session with the same ID silently replaces it."""
        store = InMemorySessionStore()
        s1 = _session("dup")
        s2 = _session("dup")
        s2.document_id = "doc-xyz"
        store.create(s1)
        store.create(s2)
        assert store.get("dup").document_id == "doc-xyz"


class TestInMemorySessionStoreUpdate:
    def test_update_persists_changes(self) -> None:
        store = InMemorySessionStore()
        s = _session()
        store.create(s)
        s.document_id = "doc-001"
        store.update(s)
        assert store.get(s.session_id).document_id == "doc-001"

    def test_update_nonexistent_raises_key_error(self) -> None:
        store = InMemorySessionStore()
        with pytest.raises(KeyError):
            store.update(_session("ghost"))


class TestInMemorySessionStoreDelete:
    def test_delete_existing_returns_true(self) -> None:
        store = InMemorySessionStore()
        s = _session()
        store.create(s)
        assert store.delete(s.session_id) is True
        assert store.get(s.session_id) is None

    def test_delete_nonexistent_returns_false(self) -> None:
        store = InMemorySessionStore()
        assert store.delete("ghost") is False

    def test_delete_then_create_is_fresh(self) -> None:
        store = InMemorySessionStore()
        s = _session()
        store.create(s)
        store.delete(s.session_id)
        s2 = _session()
        s2.document_id = "new-doc"
        store.create(s2)
        assert store.get(s2.session_id).document_id == "new-doc"


# ---------------------------------------------------------------------------
# InMemorySessionStore — thread safety
# ---------------------------------------------------------------------------


class TestInMemorySessionStoreConcurrency:
    def test_concurrent_creates_do_not_corrupt_store(self) -> None:
        """100 concurrent creates should all succeed without data loss."""
        store = InMemorySessionStore()
        sessions = [_session(f"sess-{i}") for i in range(100)]
        errors: list[Exception] = []

        def create(s: Session) -> None:
            try:
                store.create(s)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=create, args=(s,)) for s in sessions]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for s in sessions:
            assert store.get(s.session_id) is not None

    def test_concurrent_updates_do_not_raise(self) -> None:
        """Concurrent updates to different sessions should not interfere."""
        store = InMemorySessionStore()
        for i in range(20):
            store.create(_session(f"s{i}"))

        errors: list[Exception] = []

        def mutate(session_id: str, doc_id: str) -> None:
            try:
                s = store.get(session_id)
                assert s is not None
                s.document_id = doc_id
                store.update(s)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=mutate, args=(f"s{i}", f"doc-{i}"))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------


class TestSessionModel:
    def test_defaults(self) -> None:
        s = _session()
        assert s.document_id is None
        assert s.plan_draft is None
        assert s.plan is None
        assert isinstance(s.created_at, datetime)

    def test_created_at_is_utc(self) -> None:
        s = _session()
        assert s.created_at.tzinfo == timezone.utc

    def test_mutable_fields_can_be_set(self) -> None:
        s = _session()
        s.document_id = "doc-abc"
        assert s.document_id == "doc-abc"

    def test_plan_draft_round_trip(self) -> None:
        s = _session()
        s.plan_draft = PlanDraft(deductible_cents=100_000)
        dumped = s.model_dump()
        assert dumped["plan_draft"]["deductible_cents"] == 100_000


# ---------------------------------------------------------------------------
# CreateSessionRequest
# ---------------------------------------------------------------------------


class TestCreateSessionRequest:
    def test_valid_request(self) -> None:
        req = CreateSessionRequest(features=_features())
        assert req.features.age == 35

    def test_invalid_age_raises(self) -> None:
        with pytest.raises(Exception):
            CreateSessionRequest(
                features=PredictionFeatures(
                    age=-1, sex="male", bmi=25.0,
                    children=0, smoker="no", region="south",
                )
            )


# ---------------------------------------------------------------------------
# ConfirmPlanRequest validation
# ---------------------------------------------------------------------------


class TestConfirmPlanRequest:
    def test_valid_request(self) -> None:
        req = ConfirmPlanRequest(
            deductible_cents=150_000,
            out_of_pocket_max_cents=600_000,
            coinsurance_rate=0.20,
        )
        assert req.plan_name == "My Plan"

    def test_custom_plan_name(self) -> None:
        req = ConfirmPlanRequest(
            deductible_cents=0,
            out_of_pocket_max_cents=0,
            coinsurance_rate=0.0,
            plan_name="Acme Bronze",
        )
        assert req.plan_name == "Acme Bronze"

    def test_negative_deductible_rejected(self) -> None:
        with pytest.raises(Exception):
            ConfirmPlanRequest(
                deductible_cents=-1,
                out_of_pocket_max_cents=500_000,
                coinsurance_rate=0.20,
            )

    def test_negative_oop_max_rejected(self) -> None:
        with pytest.raises(Exception):
            ConfirmPlanRequest(
                deductible_cents=100_000,
                out_of_pocket_max_cents=-1,
                coinsurance_rate=0.20,
            )

    def test_coinsurance_above_one_rejected(self) -> None:
        with pytest.raises(Exception):
            ConfirmPlanRequest(
                deductible_cents=100_000,
                out_of_pocket_max_cents=500_000,
                coinsurance_rate=1.5,
            )

    def test_coinsurance_below_zero_rejected(self) -> None:
        with pytest.raises(Exception):
            ConfirmPlanRequest(
                deductible_cents=100_000,
                out_of_pocket_max_cents=500_000,
                coinsurance_rate=-0.1,
            )

    def test_copays_default_empty(self) -> None:
        req = ConfirmPlanRequest(
            deductible_cents=0,
            out_of_pocket_max_cents=0,
            coinsurance_rate=0.0,
        )
        assert req.copays_cents == {}


# ---------------------------------------------------------------------------
# SessionEstimate model
# ---------------------------------------------------------------------------


class TestSessionEstimate:
    def _prediction(self) -> CostPrediction:
        return CostPrediction(
            median_charges_cents=200_000,
            mean_charges_cents=400_000,
            lower_bound_cents=50_000,
            upper_bound_cents=800_000,
        )

    def test_no_plan_fields_are_none(self) -> None:
        est = SessionEstimate(
            features=_features(),
            prediction=self._prediction(),
        )
        assert est.plan is None
        assert est.annual_plan_share_median is None
        assert est.annual_plan_share_mean is None
        assert est.document_id is None

    def test_is_frozen(self) -> None:
        """SessionEstimate must be immutable after construction."""
        est = SessionEstimate(
            features=_features(),
            prediction=self._prediction(),
        )
        with pytest.raises(Exception):
            est.document_id = "mutated"  # type: ignore[misc]
