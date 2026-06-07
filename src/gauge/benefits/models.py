"""Domain models for the benefits engine.

All monetary fields are expressed in whole US cents to sidestep floating
point drift during cost-share math. Callers that prefer to work in dollars
should convert at the boundary.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Cents = Annotated[int, Field(ge=0, description="Non-negative amount in US cents.")]
Rate = Annotated[float, Field(ge=0.0, le=1.0, description="Fraction in [0, 1].")]


class ServiceCategory(str, Enum):
    """High-level grouping used to look up copay rules on a plan."""

    OFFICE_VISIT = "office_visit"
    SPECIALIST = "specialist"
    URGENT_CARE = "urgent_care"
    EMERGENCY = "emergency"
    IMAGING = "imaging"
    LAB = "lab"
    SURGERY = "surgery"
    GENERIC_DRUG = "generic_drug"


class Plan(BaseModel):
    """A health insurance plan's cost-share rules.

    The estimator supports a single shared deductible and OOP max per plan.
    Family-vs-individual tracking is out of scope for this prototype.
    """

    model_config = ConfigDict(frozen=True)

    plan_id: str
    name: str
    deductible_cents: Cents
    out_of_pocket_max_cents: Cents
    coinsurance_rate: Rate = Field(
        description="Member share after deductible (e.g. 0.20 means member pays 20%)."
    )
    copays_cents: dict[ServiceCategory, Cents] = Field(
        default_factory=dict,
        description="Flat copays that replace deductible/coinsurance for that category.",
    )

    @model_validator(mode="after")
    def _oop_max_at_least_deductible(self) -> "Plan":
        if self.out_of_pocket_max_cents < self.deductible_cents:
            raise ValueError(
                "out_of_pocket_max_cents must be >= deductible_cents"
            )
        return self


class Member(BaseModel):
    """A plan member with running accumulators for the current benefit year."""

    model_config = ConfigDict(frozen=True)

    member_id: str
    name: str
    plan_id: str
    ytd_deductible_cents: Cents = 0
    ytd_out_of_pocket_cents: Cents = 0

    @model_validator(mode="after")
    def _deductible_within_oop(self) -> "Member":
        if self.ytd_deductible_cents > self.ytd_out_of_pocket_cents:
            raise ValueError(
                "ytd_deductible_cents cannot exceed ytd_out_of_pocket_cents; "
                "every dollar toward the deductible also counts toward OOP."
            )
        return self


class Procedure(BaseModel):
    """A billable service with negotiated and billed rates."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Procedure code, CPT-style (e.g. '99213').")
    description: str
    category: ServiceCategory
    in_network_rate_cents: Cents
    billed_amount_cents: Cents

    @model_validator(mode="after")
    def _billed_at_least_negotiated(self) -> "Procedure":
        if self.billed_amount_cents < self.in_network_rate_cents:
            raise ValueError(
                "billed_amount_cents should be >= in_network_rate_cents"
            )
        return self


class EstimateRequest(BaseModel):
    """Input payload for a per-procedure estimate."""

    member_id: str
    procedure_code: str
    in_network: bool = True


class EstimateResult(BaseModel):
    """Breakdown of who pays what for a single procedure."""

    model_config = ConfigDict(frozen=True)

    allowed_amount_cents: Cents
    copay_cents: Cents
    deductible_applied_cents: Cents
    coinsurance_cents: Cents
    member_pays_cents: Cents
    plan_pays_cents: Cents
    notes: list[str] = Field(default_factory=list)

    @property
    def member_pays_dollars(self) -> float:
        """Convenience: member share rendered as dollars."""
        return self.member_pays_cents / 100

    @property
    def plan_pays_dollars(self) -> float:
        """Convenience: plan share rendered as dollars."""
        return self.plan_pays_cents / 100
