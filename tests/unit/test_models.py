"""Unit tests for the pydantic domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gauge.benefits.models import Member, Plan, Procedure, ServiceCategory

pytestmark = pytest.mark.unit


class TestPlan:
    def test_oop_max_must_be_at_least_deductible(self) -> None:
        with pytest.raises(ValidationError):
            Plan(
                plan_id="p",
                name="Bad",
                deductible_cents=500_000,
                out_of_pocket_max_cents=100_000,
                coinsurance_rate=0.20,
            )

    def test_coinsurance_rate_bounded(self) -> None:
        with pytest.raises(ValidationError):
            Plan(
                plan_id="p",
                name="Bad",
                deductible_cents=0,
                out_of_pocket_max_cents=1,
                coinsurance_rate=1.5,
            )

    def test_negative_amounts_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Plan(
                plan_id="p",
                name="Bad",
                deductible_cents=-1,
                out_of_pocket_max_cents=100_000,
                coinsurance_rate=0.20,
            )


class TestMember:
    def test_ytd_deductible_cannot_exceed_oop(self) -> None:
        with pytest.raises(ValidationError):
            Member(
                member_id="m",
                name="Bad",
                plan_id="p",
                ytd_deductible_cents=500,
                ytd_out_of_pocket_cents=100,
            )

    def test_defaults_to_zero_accumulators(self) -> None:
        m = Member(member_id="m", name="OK", plan_id="p")
        assert m.ytd_deductible_cents == 0
        assert m.ytd_out_of_pocket_cents == 0


class TestProcedure:
    def test_billed_must_be_at_least_negotiated(self) -> None:
        with pytest.raises(ValidationError):
            Procedure(
                code="x",
                description="Bad",
                category=ServiceCategory.OFFICE_VISIT,
                in_network_rate_cents=20_000,
                billed_amount_cents=10_000,
            )
