"""Unit tests for the CostPredictor."""

from __future__ import annotations

import pytest

from health_app.predictor.dataset import generate_synthetic_dataset
from health_app.predictor.model import (
    ALL_FEATURES,
    CostPredictor,
)
from health_app.predictor.schemas import PredictionFeatures

pytestmark = pytest.mark.unit


def test_predictor_rejects_invalid_quantiles() -> None:
    with pytest.raises(ValueError):
        CostPredictor(quantiles=(0.5, 0.3, 0.9))


def test_predictor_must_be_fitted_before_predicting() -> None:
    predictor = CostPredictor()
    features = PredictionFeatures(
        age=30,
        sex="male",
        bmi=25.0,
        children=0,
        smoker="no",
        region="northeast",
    )
    with pytest.raises(RuntimeError):
        predictor.predict(features)


def test_predictor_fit_requires_target_column() -> None:
    df = generate_synthetic_dataset(n_rows=50, seed=1).drop(columns=["charges"])
    with pytest.raises(ValueError):
        CostPredictor().fit(df)


def test_prediction_interval_is_ordered_and_nonneg(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    pred = trained_predictor.predict(baseline_features)
    assert 0 <= pred.lower_bound_cents
    assert pred.lower_bound_cents <= pred.median_charges_cents
    assert pred.median_charges_cents <= pred.upper_bound_cents


def test_smoker_costs_more_than_non_smoker(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    """Directional sanity check; small dataset, so we only test the sign."""
    smoker = baseline_features.model_copy(update={"smoker": "yes"})
    smoker_pred = trained_predictor.predict(smoker).median_charges_cents
    nonsmoker_pred = trained_predictor.predict(
        baseline_features
    ).median_charges_cents
    assert smoker_pred > nonsmoker_pred


def test_higher_age_generally_costs_more(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    young = baseline_features.model_copy(update={"age": 22})
    older = baseline_features.model_copy(update={"age": 60})
    assert (
        trained_predictor.predict(older).median_charges_cents
        > trained_predictor.predict(young).median_charges_cents
    )


def test_predict_many_matches_predict_single(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    features = [
        baseline_features,
        baseline_features.model_copy(update={"age": 50}),
        baseline_features.model_copy(update={"smoker": "yes"}),
    ]
    batch = trained_predictor.predict_many(features)
    singles = [trained_predictor.predict(f) for f in features]
    assert batch == singles


def test_predict_many_on_empty_returns_empty(
    trained_predictor: CostPredictor,
) -> None:
    assert trained_predictor.predict_many([]) == []


def test_save_and_load_round_trip(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
    tmp_path,
) -> None:
    cache = tmp_path / "predictor.joblib"
    trained_predictor.save(cache)
    restored = CostPredictor.load(cache)

    assert restored.is_fitted
    assert restored.predict(baseline_features) == trained_predictor.predict(
        baseline_features
    )


def test_feature_columns_match_schema_fields() -> None:
    """Predictor's expected features must mirror the pydantic schema."""
    assert set(ALL_FEATURES) == set(PredictionFeatures.model_fields)
