"""Unit tests for the what-if sweep."""

from __future__ import annotations

import pytest

from health_app.benefits.models import Plan
from health_app.predictor.model import CostPredictor
from health_app.predictor.schemas import PredictionFeatures
from health_app.predictor.whatif import sweep

pytestmark = pytest.mark.unit


def test_sweep_rejects_unknown_feature(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    with pytest.raises(ValueError, match="Cannot sweep 'income'"):
        sweep(
            predictor=trained_predictor,
            baseline=baseline_features,
            feature="income",
            values=[1, 2, 3],
        )


def test_sweep_returns_one_point_per_value(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    values = [25, 35, 45, 55]
    response = sweep(
        predictor=trained_predictor,
        baseline=baseline_features,
        feature="age",
        values=values,
    )
    assert response.feature == "age"
    assert [p.value for p in response.points] == values
    assert all(p.annual_plan_share_median is None for p in response.points)
    assert all(p.annual_plan_share_mean is None for p in response.points)


def test_sweep_empty_values_returns_empty(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    response = sweep(
        predictor=trained_predictor,
        baseline=baseline_features,
        feature="age",
        values=[],
    )
    assert response.points == []


def test_sweep_charges_increase_with_age(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    response = sweep(
        predictor=trained_predictor,
        baseline=baseline_features,
        feature="age",
        values=[20, 40, 60],
    )
    charges = [p.prediction.median_charges_cents for p in response.points]
    assert charges[0] < charges[2]


def test_sweep_with_plan_includes_annual_plan_share(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
    ppo_gold: Plan,
) -> None:
    response = sweep(
        predictor=trained_predictor,
        baseline=baseline_features,
        feature="smoker",
        values=["no", "yes"],
        plan=ppo_gold,
    )
    assert len(response.points) == 2
    for point in response.points:
        assert point.annual_plan_share_median is not None
        assert point.annual_plan_share_mean is not None
        assert (
            point.annual_plan_share_median.member_pays_cents
            + point.annual_plan_share_median.plan_pays_cents
            == point.prediction.median_charges_cents
        )
        assert (
            point.annual_plan_share_mean.member_pays_cents
            + point.annual_plan_share_mean.plan_pays_cents
            == point.prediction.mean_charges_cents
        )


def test_sweep_invalid_value_for_feature_raises(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    """A string in an age sweep should be rejected with a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid value 'twenty'"):
        sweep(
            predictor=trained_predictor,
            baseline=baseline_features,
            feature="age",
            values=["twenty"],
        )
