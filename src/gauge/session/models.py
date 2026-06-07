"""Session models tying user demographics, plan, and document together.

A ``Session`` is the central context object for the guided flow.  It is
created when the user submits their demographics, enriched when they upload
a plan document (which triggers automatic extraction into a ``PlanDraft``),
and finalised when they confirm or correct the extracted plan fields.

All request / response schemas used by the session API endpoints live here
alongside the core session model so callers have a single import target.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from gauge.benefits.models import Plan
from gauge.plan_extract.schemas import PlanDraft
from gauge.predictor.annual_cost import AnnualPlanShare
from gauge.predictor.model import CostPrediction
from gauge.predictor.schemas import PredictionFeatures


# ---------------------------------------------------------------------------
# Core session state
# ---------------------------------------------------------------------------


class Session(BaseModel):
    """A user's in-progress or completed estimation session.

    Created when the user submits their demographics; enriched as they upload
    a document and confirm their plan details.

    Parameters
    ----------
    session_id : str
        Unique identifier assigned at creation.
    features : PredictionFeatures
        User's demographic inputs.
    document_id : str or None
        ID of the uploaded PDF, populated after a document is attached.
    plan_draft : PlanDraft or None
        LLM-extracted plan fields pending user confirmation.
    plan : Plan or None
        Confirmed plan; populated after the user accepts the draft.
    created_at : datetime
        UTC timestamp of session creation.
    """

    model_config = ConfigDict(frozen=False)

    session_id: str
    features: PredictionFeatures
    document_id: str | None = None
    plan_draft: PlanDraft | None = None
    plan: Plan | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Estimate response
# ---------------------------------------------------------------------------


class SessionEstimate(BaseModel):
    """The full personalised estimate for a completed session.

    Combines the raw ML prediction with plan-specific cost-share breakdowns
    when a confirmed plan is available.

    Parameters
    ----------
    features : PredictionFeatures
        Demographics used for the prediction.
    prediction : CostPrediction
        Raw ML prediction (median, mean, 80 % interval) in cents.
    plan : Plan or None
        The confirmed plan, or ``None`` if no plan has been set yet.
    annual_plan_share_median : AnnualPlanShare or None
        How the plan splits the *median* predicted spend.  ``None`` when
        no plan is available.
    annual_plan_share_mean : AnnualPlanShare or None
        How the plan splits the *mean* predicted spend.  ``None`` when no
        plan is available.
    document_id : str or None
        ID of the uploaded plan document, for subsequent Q&A calls.
    """

    model_config = ConfigDict(frozen=True)

    features: PredictionFeatures
    prediction: CostPrediction
    plan: Plan | None = None
    annual_plan_share_median: AnnualPlanShare | None = None
    annual_plan_share_mean: AnnualPlanShare | None = None
    document_id: str | None = None


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Request body for ``POST /sessions``.

    Parameters
    ----------
    features : PredictionFeatures
        The user's demographic inputs.
    """

    features: PredictionFeatures


class CreateSessionResponse(BaseModel):
    """Response from ``POST /sessions``.

    Parameters
    ----------
    session_id : str
        Identifier to pass in subsequent session-scoped requests.
    prediction : CostPrediction
        First-pass cost prediction based on demographics alone, before any
        plan context has been added.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    prediction: CostPrediction


class AttachDocumentResponse(BaseModel):
    """Response from ``POST /sessions/{id}/document``.

    Parameters
    ----------
    document_id : str
        ID of the newly uploaded document.
    plan_draft : PlanDraft
        Automatically extracted plan fields.  ``None`` values indicate
        fields the LLM could not find; the user should review and complete
        them on the confirmation form.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    plan_draft: PlanDraft


class ConfirmPlanRequest(BaseModel):
    """Request body for ``POST /sessions/{id}/plan``.

    Carries the user-reviewed (and possibly edited) plan fields.  The three
    required numeric fields must be provided and non-negative.

    Parameters
    ----------
    deductible_cents : int
        Individual in-network annual deductible in cents.
    out_of_pocket_max_cents : int
        Individual in-network annual out-of-pocket maximum in cents.
    coinsurance_rate : float
        Member share after deductible as a fraction in ``[0, 1]``.
    copays_cents : dict[str, int]
        Copay amounts keyed by ``ServiceCategory`` value strings.
    plan_name : str
        Human-readable name for the confirmed plan.
    """

    deductible_cents: int = Field(ge=0)
    out_of_pocket_max_cents: int = Field(ge=0)
    coinsurance_rate: float = Field(ge=0.0, le=1.0)
    copays_cents: dict[str, int] = Field(default_factory=dict)
    plan_name: str = "My Plan"
