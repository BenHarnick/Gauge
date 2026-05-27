"""What-if simulator.

Holds a baseline feature vector fixed, varies a single feature across a
list of values, and returns the prediction at each value. Optionally
also pipes each prediction's median charges through a plan to give the
annual out-of-pocket curve.

The sweep is implemented with batched prediction so dozens of points run
in roughly the same time as a single point.
"""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, ConfigDict

from health_app.benefits.models import Plan
from health_app.predictor.annual_cost import (
    AnnualPlanShare,
    apply_plan_to_annual_spend,
)
from health_app.predictor.model import CostPrediction, CostPredictor
from health_app.predictor.schemas import PredictionFeatures

SweepValue = Union[int, float, str]
SWEEPABLE_FEATURES: frozenset[str] = frozenset(PredictionFeatures.model_fields)


class WhatIfPoint(BaseModel):
    """A single point on a what-if sweep."""

    model_config = ConfigDict(frozen=True)

    value: SweepValue
    prediction: CostPrediction
    annual_plan_share_median: AnnualPlanShare | None = None
    annual_plan_share_mean: AnnualPlanShare | None = None


class WhatIfResponse(BaseModel):
    """Result of varying one feature across a list of values."""

    model_config = ConfigDict(frozen=True)

    feature: str
    points: list[WhatIfPoint]


def sweep(
    predictor: CostPredictor,
    baseline: PredictionFeatures,
    feature: str,
    values: list[SweepValue],
    plan: Plan | None = None,
) -> WhatIfResponse:
    """Sweep one feature, optionally annotating each point with plan cost-share.

    Args:
        predictor: A fitted `CostPredictor`.
        baseline: Feature vector held constant except for `feature`.
        feature: Name of the feature to vary. Must be a field of
            `PredictionFeatures`.
        values: Values to substitute for `feature`. Each must validate
            against the field's type.
        plan: Optional plan. When provided, each prediction's median
            charges are run through `apply_plan_to_annual_spend` to give
            an annual out-of-pocket figure alongside the raw prediction.

    Returns:
        A `WhatIfResponse` whose `points` are aligned to `values`.

    Raises:
        ValueError: If `feature` is not a sweepable field or any value is
            invalid for that field's type.
    """
    if feature not in SWEEPABLE_FEATURES:
        raise ValueError(
            f"Cannot sweep {feature!r}; valid features are "
            f"{sorted(SWEEPABLE_FEATURES)}."
        )
    if not values:
        return WhatIfResponse(feature=feature, points=[])

    # Re-validate each modified vector through the model so bad values
    # surface here with a clean error rather than crashing the pipeline
    # downstream. (model_copy(update=...) intentionally skips validation.)
    base_payload = baseline.model_dump()
    feature_rows: list[PredictionFeatures] = []
    for v in values:
        try:
            feature_rows.append(
                PredictionFeatures.model_validate({**base_payload, feature: v})
            )
        except Exception as e:
            raise ValueError(
                f"Invalid value {v!r} for feature {feature!r}: {e}"
            ) from e
    predictions = predictor.predict_many(feature_rows)

    points: list[WhatIfPoint] = []
    for value, prediction in zip(values, predictions):
        if plan is not None:
            share_median = apply_plan_to_annual_spend(
                plan, prediction.median_charges_cents
            )
            share_mean = apply_plan_to_annual_spend(
                plan, prediction.mean_charges_cents
            )
        else:
            share_median = None
            share_mean = None
        points.append(
            WhatIfPoint(
                value=value,
                prediction=prediction,
                annual_plan_share_median=share_median,
                annual_plan_share_mean=share_mean,
            )
        )

    return WhatIfResponse(feature=feature, points=points)
