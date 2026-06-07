"""Unit tests for the cost-share calculator.

Each test computes the expected breakdown by hand so the assertions
double as worked examples of the math.
"""

from __future__ import annotations

import pytest

from gauge.benefits.calculator import estimate_cost_share
from gauge.benefits.models import Member, Plan, Procedure, ServiceCategory

pytestmark = pytest.mark.unit


def test_copay_only_when_category_has_copay(
    ppo_gold: Plan, fresh_member: Member, office_visit: Procedure
) -> None:
    result = estimate_cost_share(
        ppo_gold, fresh_member, office_visit, in_network=True
    )
    assert result.allowed_amount_cents == 15_000
    assert result.copay_cents == 2_500
    assert result.deductible_applied_cents == 0
    assert result.coinsurance_cents == 0
    assert result.member_pays_cents == 2_500
    assert result.plan_pays_cents == 12_500
    assert result.notes == []


def test_deductible_absorbs_full_allowed_when_below_remaining(
    ppo_gold: Plan, fresh_member: Member, imaging_procedure: Procedure
) -> None:
    # Imaging has no copay on PPO Gold, so deductible is in play. The full
    # $850 is below the $1,000 deductible, so the member pays all of it.
    result = estimate_cost_share(
        ppo_gold, fresh_member, imaging_procedure, in_network=True
    )
    assert result.deductible_applied_cents == 85_000
    assert result.coinsurance_cents == 0
    assert result.member_pays_cents == 85_000
    assert result.plan_pays_cents == 0


def test_partial_deductible_then_coinsurance(
    ppo_gold: Plan, fresh_member: Member, surgery_procedure: Procedure
) -> None:
    # Surgery is $5,400. Deductible is $1,000, so $1,000 toward deductible
    # and $4,400 splits 80/20.
    result = estimate_cost_share(
        ppo_gold, fresh_member, surgery_procedure, in_network=True
    )
    assert result.deductible_applied_cents == 100_000
    assert result.coinsurance_cents == 88_000
    assert result.member_pays_cents == 188_000
    assert result.plan_pays_cents == 352_000


def test_deductible_already_met(
    ppo_gold: Plan, imaging_procedure: Procedure
) -> None:
    met = Member(
        member_id="m_met",
        name="Met",
        plan_id=ppo_gold.plan_id,
        ytd_deductible_cents=100_000,
        ytd_out_of_pocket_cents=100_000,
    )
    result = estimate_cost_share(
        ppo_gold, met, imaging_procedure, in_network=True
    )
    assert result.deductible_applied_cents == 0
    assert result.coinsurance_cents == 17_000  # 20% of $850
    assert result.member_pays_cents == 17_000
    assert result.plan_pays_cents == 68_000


def test_out_of_network_uses_billed_amount(
    hdhp_silver: Plan, imaging_procedure: Procedure
) -> None:
    member = Member(
        member_id="m_partial",
        name="Partial",
        plan_id=hdhp_silver.plan_id,
        ytd_deductible_cents=200_000,
        ytd_out_of_pocket_cents=200_000,
    )
    result = estimate_cost_share(
        hdhp_silver, member, imaging_procedure, in_network=False
    )
    assert result.allowed_amount_cents == 240_000
    # Remaining deductible is $1,000 (of $3,000 minus $2,000 already paid).
    assert result.deductible_applied_cents == 100_000
    # 20% of $1,400 after deductible.
    assert result.coinsurance_cents == 28_000
    assert result.member_pays_cents == 128_000
    assert any("Out-of-network" in n for n in result.notes)


def test_out_of_pocket_max_caps_member_share(
    ppo_gold: Plan,
    near_oop_member: Member,
    surgery_procedure: Procedure,
) -> None:
    # near_oop_member has $100 remaining toward the $5,000 OOP max.
    result = estimate_cost_share(
        ppo_gold, near_oop_member, surgery_procedure, in_network=True
    )
    assert result.member_pays_cents == 10_000
    assert result.plan_pays_cents == 540_000 - 10_000
    assert any("Out-of-pocket maximum" in n for n in result.notes)


def test_oop_max_can_absorb_into_copay(
    ppo_gold: Plan, office_visit: Procedure
) -> None:
    # Member already $4,990 deep; copay would push past $5,000 cap.
    almost_capped = Member(
        member_id="m_almost",
        name="Almost",
        plan_id=ppo_gold.plan_id,
        ytd_deductible_cents=100_000,
        ytd_out_of_pocket_cents=499_000,
    )
    result = estimate_cost_share(
        ppo_gold, almost_capped, office_visit, in_network=True
    )
    # Only $10 of headroom, so copay is reduced from $25 to $10.
    assert result.copay_cents == 1_000
    assert result.member_pays_cents == 1_000
    assert result.plan_pays_cents == 14_000
    assert any("Out-of-pocket maximum" in n for n in result.notes)


def test_zero_remaining_oop_means_plan_pays_everything(
    ppo_gold: Plan, surgery_procedure: Procedure
) -> None:
    capped = Member(
        member_id="m_capped",
        name="Capped",
        plan_id=ppo_gold.plan_id,
        ytd_deductible_cents=100_000,
        ytd_out_of_pocket_cents=500_000,
    )
    result = estimate_cost_share(
        ppo_gold, capped, surgery_procedure, in_network=True
    )
    assert result.member_pays_cents == 0
    assert result.plan_pays_cents == 540_000


def test_mismatched_plan_raises(
    hdhp_silver: Plan, fresh_member: Member, office_visit: Procedure
) -> None:
    # fresh_member is on ppo_gold; passing hdhp_silver should error.
    with pytest.raises(ValueError, match="Member m1"):
        estimate_cost_share(
            hdhp_silver, fresh_member, office_visit, in_network=True
        )


def test_coinsurance_breakdown_sums_to_allowed(
    hdhp_silver: Plan, imaging_procedure: Procedure
) -> None:
    # Odd allowed amount to exercise rounding. 12345 * 0.20 = 2469.
    odd_procedure = Procedure(
        code="oddx",
        description="Odd",
        category=ServiceCategory.LAB,
        in_network_rate_cents=12_345,
        billed_amount_cents=12_345,
    )
    member = Member(
        member_id="m_done",
        name="Done",
        plan_id=hdhp_silver.plan_id,
        ytd_deductible_cents=300_000,
        ytd_out_of_pocket_cents=300_000,
    )
    result = estimate_cost_share(
        hdhp_silver, member, odd_procedure, in_network=True
    )
    assert result.coinsurance_cents == 2_469
    assert result.member_pays_cents + result.plan_pays_cents == 12_345
