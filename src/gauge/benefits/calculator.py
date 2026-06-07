"""Pure cost-share math for a single procedure.

Assumptions baked into this prototype, called out explicitly so they are
easy to find and revise later:

* A single deductible and OOP max per plan. No separate in-network vs.
  out-of-network accumulators.
* Copays apply only in-network and only when the plan has a copay listed
  for the procedure's category. When a copay applies, that copay is the
  member's entire share for that service (no separate coinsurance).
* For out-of-network, the billed amount is used as the allowed amount and
  the same deductible and coinsurance rules apply.
* When the OOP max is reached, the excess member share is absorbed by
  reducing coinsurance first, then deductible applied, then copay.
* Rounding for coinsurance uses Python's banker's rounding to the nearest
  cent. The plan pays whatever cent residual the rounding produces.
"""

from __future__ import annotations

from gauge.benefits.models import (
    EstimateResult,
    Member,
    Plan,
    Procedure,
)


def estimate_cost_share(
    plan: Plan,
    member: Member,
    procedure: Procedure,
    in_network: bool,
) -> EstimateResult:
    """Estimate member and plan responsibility for a single procedure.

    Parameters
    ----------
    plan : Plan
        The member's plan. Must match ``member.plan_id``.
    member : Member
        The member, including year-to-date accumulators.
    procedure : Procedure
        The procedure being estimated.
    in_network : bool
        ``True`` for in-network providers, ``False`` otherwise.

    Returns
    -------
    EstimateResult
        Breakdown of allowed amount, copay, deductible applied, coinsurance,
        and total member vs. plan responsibility, plus any advisory notes
        flagging unusual conditions (out-of-network, OOP cap reached, etc.).

    Raises
    ------
    ValueError
        If the member's plan ID does not match the supplied plan.
    """
    if member.plan_id != plan.plan_id:
        raise ValueError(
            f"Member {member.member_id} is on plan {member.plan_id}, "
            f"not {plan.plan_id}."
        )

    notes: list[str] = []

    allowed = (
        procedure.in_network_rate_cents
        if in_network
        else procedure.billed_amount_cents
    )
    if not in_network:
        notes.append(
            "Out-of-network: billed amount used as allowed; deductible "
            "and coinsurance still apply."
        )

    copay = 0
    deductible_applied = 0
    coinsurance = 0

    has_copay = in_network and procedure.category in plan.copays_cents
    if has_copay:
        copay = min(plan.copays_cents[procedure.category], allowed)
    else:
        remaining_deductible = max(
            0, plan.deductible_cents - member.ytd_deductible_cents
        )
        deductible_applied = min(remaining_deductible, allowed)
        after_deductible = allowed - deductible_applied
        coinsurance = round(after_deductible * plan.coinsurance_rate)

    gross_member = copay + deductible_applied + coinsurance

    remaining_oop = max(
        0, plan.out_of_pocket_max_cents - member.ytd_out_of_pocket_cents
    )
    if gross_member > remaining_oop:
        excess = gross_member - remaining_oop
        coinsurance, excess = _absorb(coinsurance, excess)
        deductible_applied, excess = _absorb(deductible_applied, excess)
        copay, _ = _absorb(copay, excess)
        notes.append(
            "Out-of-pocket maximum reached; plan absorbs the remainder."
        )

    member_pays = copay + deductible_applied + coinsurance
    plan_pays = allowed - member_pays

    return EstimateResult(
        allowed_amount_cents=allowed,
        copay_cents=copay,
        deductible_applied_cents=deductible_applied,
        coinsurance_cents=coinsurance,
        member_pays_cents=member_pays,
        plan_pays_cents=plan_pays,
        notes=notes,
    )


def _absorb(bucket: int, excess: int) -> tuple[int, int]:
    """Reduce ``bucket`` by up to ``excess``, returning the remainder.

    Parameters
    ----------
    bucket : int
        Current balance of a cost-share component in cents.
    excess : int
        Amount to absorb from the bucket in cents.

    Returns
    -------
    tuple[int, int]
        ``(new_bucket, leftover_excess)`` where ``new_bucket`` is the bucket
        after absorption and ``leftover_excess`` is any unabsorbed remainder.
    """
    if excess <= 0:
        return bucket, 0
    if bucket >= excess:
        return bucket - excess, 0
    return 0, excess - bucket
