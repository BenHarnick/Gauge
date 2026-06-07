"""Unit tests for the annual cost integration."""

from __future__ import annotations

import pytest

from gauge.benefits.models import Plan
from gauge.predictor.annual_cost import apply_plan_to_annual_spend

pytestmark = pytest.mark.unit


def test_charges_below_deductible_all_on_member(ppo_gold: Plan) -> None:
    share = apply_plan_to_annual_spend(ppo_gold, charges_cents=50_000)
    assert share.deductible_applied_cents == 50_000
    assert share.coinsurance_cents == 0
    assert share.member_pays_cents == 50_000
    assert share.plan_pays_cents == 0
    assert share.capped_at_oop_max is False


def test_charges_above_deductible_split_coinsurance(ppo_gold: Plan) -> None:
    # $1,000 deductible + ($5,000 - $1,000) * 0.20 = $1,800 member share
    share = apply_plan_to_annual_spend(ppo_gold, charges_cents=500_000)
    assert share.deductible_applied_cents == 100_000
    assert share.coinsurance_cents == 80_000
    assert share.member_pays_cents == 180_000
    assert share.plan_pays_cents == 320_000


def test_oop_max_caps_member_share(ppo_gold: Plan) -> None:
    # Big claim: $50,000.
    # Without cap: 100_000 + 0.20 * (5_000_000 - 100_000) = 1_080_000.
    # OOP max is 500_000, so member is capped there.
    share = apply_plan_to_annual_spend(ppo_gold, charges_cents=5_000_000)
    assert share.member_pays_cents == 500_000
    assert share.plan_pays_cents == 4_500_000
    assert share.capped_at_oop_max is True


def test_zero_charges(ppo_gold: Plan) -> None:
    share = apply_plan_to_annual_spend(ppo_gold, charges_cents=0)
    assert share.member_pays_cents == 0
    assert share.plan_pays_cents == 0
    assert share.capped_at_oop_max is False


def test_negative_charges_rejected(ppo_gold: Plan) -> None:
    with pytest.raises(ValueError):
        apply_plan_to_annual_spend(ppo_gold, charges_cents=-1)


def test_components_sum_to_charges(hdhp_silver: Plan) -> None:
    share = apply_plan_to_annual_spend(hdhp_silver, charges_cents=1_234_567)
    assert share.member_pays_cents + share.plan_pays_cents == 1_234_567
