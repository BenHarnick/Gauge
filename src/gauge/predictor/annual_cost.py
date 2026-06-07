"""Bridge predicted annual charges into the benefits engine.

The benefits calculator works at the per-procedure level. To produce an
annual out-of-pocket estimate we make one simplifying assumption: treat
the predicted total charges as a single lump that hits the plan over the
year. That ignores per-visit copays (we don't know visit counts) but
captures deductible plus coinsurance behaviour accurately, which is the
dominant driver of annual cost-share for higher-spend members.

This file is intentionally tiny so the assumption is easy to audit and
swap out for a richer model (for example, a visit-count generator) later.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gauge.benefits.models import Plan


class AnnualPlanShare(BaseModel):
    """Member and plan share of a single annual charges figure."""

    model_config = ConfigDict(frozen=True)

    charges_cents: int = Field(ge=0)
    deductible_applied_cents: int = Field(ge=0)
    coinsurance_cents: int = Field(ge=0)
    member_pays_cents: int = Field(ge=0)
    plan_pays_cents: int = Field(ge=0)
    capped_at_oop_max: bool = False


def apply_plan_to_annual_spend(plan: Plan, charges_cents: int) -> AnnualPlanShare:
    """Distribute annual charges across deductible, coinsurance, and OOP cap.

    Parameters
    ----------
    plan : Plan
        The plan whose cost-share rules apply.
    charges_cents : int
        Predicted annual gross charges in cents.

    Returns
    -------
    AnnualPlanShare
        Breakdown of deductible applied, coinsurance, and total member vs.
        plan responsibility for the given annual spend.

    Raises
    ------
    ValueError
        If ``charges_cents`` is negative.
    """
    if charges_cents < 0:
        raise ValueError("charges_cents must be non-negative.")

    deductible_applied = min(charges_cents, plan.deductible_cents)
    after_deductible = charges_cents - deductible_applied
    coinsurance = round(after_deductible * plan.coinsurance_rate)
    gross_member = deductible_applied + coinsurance

    capped = False
    if gross_member > plan.out_of_pocket_max_cents:
        excess = gross_member - plan.out_of_pocket_max_cents
        # Reduce coinsurance first; deductible only if coinsurance can't absorb.
        if coinsurance >= excess:
            coinsurance -= excess
        else:
            excess -= coinsurance
            coinsurance = 0
            deductible_applied = max(0, deductible_applied - excess)
        capped = True

    member_pays = deductible_applied + coinsurance
    plan_pays = charges_cents - member_pays

    return AnnualPlanShare(
        charges_cents=charges_cents,
        deductible_applied_cents=deductible_applied,
        coinsurance_cents=coinsurance,
        member_pays_cents=member_pays,
        plan_pays_cents=plan_pays,
        capped_at_oop_max=capped,
    )
