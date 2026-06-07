"""Sample plans, members, and procedures used to seed the prototype."""

from __future__ import annotations

from gauge.benefits.models import (
    Member,
    Plan,
    Procedure,
    ServiceCategory,
)
from gauge.benefits.repository import InMemoryRepository

PLANS: list[Plan] = [
    Plan(
        plan_id="hdhp_silver",
        name="HDHP Silver",
        deductible_cents=300_000,
        out_of_pocket_max_cents=700_000,
        coinsurance_rate=0.20,
        copays_cents={},
    ),
    Plan(
        plan_id="ppo_gold",
        name="PPO Gold",
        deductible_cents=100_000,
        out_of_pocket_max_cents=500_000,
        coinsurance_rate=0.20,
        copays_cents={
            ServiceCategory.OFFICE_VISIT: 2_500,
            ServiceCategory.SPECIALIST: 5_000,
            ServiceCategory.URGENT_CARE: 7_500,
            ServiceCategory.GENERIC_DRUG: 1_000,
        },
    ),
    Plan(
        plan_id="ppo_platinum",
        name="PPO Platinum",
        deductible_cents=25_000,
        out_of_pocket_max_cents=200_000,
        coinsurance_rate=0.10,
        copays_cents={
            ServiceCategory.OFFICE_VISIT: 1_500,
            ServiceCategory.SPECIALIST: 3_000,
        },
    ),
]


MEMBERS: list[Member] = [
    Member(
        member_id="m1",
        name="Alex Carter",
        plan_id="ppo_gold",
    ),
    Member(
        member_id="m2",
        name="Jordan Lee",
        plan_id="hdhp_silver",
        ytd_deductible_cents=200_000,
        ytd_out_of_pocket_cents=200_000,
    ),
    Member(
        member_id="m3",
        name="Sam Rivera",
        plan_id="ppo_platinum",
        ytd_deductible_cents=25_000,
        ytd_out_of_pocket_cents=180_000,
    ),
]


PROCEDURES: list[Procedure] = [
    Procedure(
        code="99213",
        description="Office visit, established patient, low complexity",
        category=ServiceCategory.OFFICE_VISIT,
        in_network_rate_cents=15_000,
        billed_amount_cents=22_000,
    ),
    Procedure(
        code="99244",
        description="Specialist consultation",
        category=ServiceCategory.SPECIALIST,
        in_network_rate_cents=28_000,
        billed_amount_cents=42_000,
    ),
    Procedure(
        code="73721",
        description="MRI of knee without contrast",
        category=ServiceCategory.IMAGING,
        in_network_rate_cents=85_000,
        billed_amount_cents=240_000,
    ),
    Procedure(
        code="29881",
        description="Knee arthroscopy with meniscectomy",
        category=ServiceCategory.SURGERY,
        in_network_rate_cents=540_000,
        billed_amount_cents=1_200_000,
    ),
]


def build_default_repository() -> InMemoryRepository:
    """Return a repository seeded with the sample catalogue."""
    return InMemoryRepository(
        plans=list(PLANS),
        members=list(MEMBERS),
        procedures=list(PROCEDURES),
    )
