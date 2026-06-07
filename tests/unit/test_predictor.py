"""Unit tests for the CostPredictor."""

from __future__ import annotations

import pytest

import numpy as np

from gauge.predictor.dataset import generate_synthetic_dataset
from gauge.predictor.model import (
    ALL_FEATURES,
    CostPredictor,
)
from gauge.predictor.schemas import PredictionFeatures

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


def test_cost_prediction_dollar_properties(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    pred = trained_predictor.predict(baseline_features)
    assert pred.median_charges_dollars == pred.median_charges_cents / 100
    assert pred.mean_charges_dollars == pred.mean_charges_cents / 100


def test_cost_prediction_interval_width(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
) -> None:
    pred = trained_predictor.predict(baseline_features)
    assert pred.interval_width_cents == pred.upper_bound_cents - pred.lower_bound_cents
    assert pred.interval_width_cents >= 0


def test_save_raises_when_predictor_not_fitted(tmp_path) -> None:
    predictor = CostPredictor()
    with pytest.raises(RuntimeError, match="not been fitted"):
        predictor.save(tmp_path / "model.joblib")


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


def test_predictor_is_calibrated_after_fit(
    trained_predictor: CostPredictor,
) -> None:
    assert trained_predictor.is_calibrated


def test_conformal_interval_wider_than_raw_quantile() -> None:
    """CQR-calibrated interval should be at least as wide as the raw interval.

    We train two predictors on the same data: one with calibration (default)
    and one without (calibration_frac=0 is not valid, so we manually skip
    calibration by inspecting q_hat).  If the data is typical, q_hat > 0
    and the calibrated interval is strictly wider.
    """
    from gauge.predictor.dataset import generate_synthetic_dataset

    df = generate_synthetic_dataset(n_rows=600, seed=99)
    features = PredictionFeatures(
        age=40, sex="male", bmi=28.0, children=0, smoker="no", region="south"
    )

    # Calibrated predictor (default).
    cal = CostPredictor().fit(df)
    assert cal.is_calibrated
    pred_cal = cal.predict(features)
    assert pred_cal.conformal_calibrated is True
    assert pred_cal.calibration_coverage == pytest.approx(0.80)

    # Uncalibrated: fit on all data, then zero out q_hat manually.
    uncal = CostPredictor().fit(df)
    uncal._q_hat = None
    uncal._calibration_coverage = None
    pred_uncal = uncal.predict(features)
    assert pred_uncal.conformal_calibrated is False
    assert pred_uncal.calibration_coverage is None

    # Calibrated interval must be at least as wide (q_hat >= 0 on real data).
    cal_width = pred_cal.upper_bound_cents - pred_cal.lower_bound_cents
    uncal_width = pred_uncal.upper_bound_cents - pred_uncal.lower_bound_cents
    assert cal_width >= uncal_width


def test_conformal_coverage_holds_on_held_out_data() -> None:
    """Empirical coverage on a fresh test set should meet the target."""
    from gauge.predictor.dataset import generate_synthetic_dataset

    rng = np.random.default_rng(7)
    df_all = generate_synthetic_dataset(n_rows=2000, seed=7)
    # Use 70% to train+calibrate, 30% as unseen test set.
    n_test = int(0.30 * len(df_all))
    test_idx = rng.choice(len(df_all), size=n_test, replace=False)
    train_idx = np.setdiff1d(np.arange(len(df_all)), test_idx)

    predictor = CostPredictor().fit(df_all.iloc[train_idx])
    assert predictor.is_calibrated

    df_test = df_all.iloc[test_idx].reset_index(drop=True)
    feature_rows = [
        PredictionFeatures(**row[ALL_FEATURES].to_dict())
        for _, row in df_test.iterrows()
    ]
    preds = predictor.predict_many(feature_rows)
    y_true = df_test["charges"].to_numpy() * 100  # dollars -> cents

    covered = sum(
        p.lower_bound_cents <= y <= p.upper_bound_cents
        for p, y in zip(preds, y_true)
    )
    empirical_coverage = covered / len(preds)
    # Allow a small margin below the 80% target given finite test size.
    assert empirical_coverage >= 0.70, (
        f"Empirical coverage {empirical_coverage:.2%} is too low"
    )


def test_save_and_load_preserves_calibration(
    trained_predictor: CostPredictor,
    baseline_features: PredictionFeatures,
    tmp_path,
) -> None:
    cache = tmp_path / "predictor_cal.joblib"
    trained_predictor.save(cache)
    restored = CostPredictor.load(cache)

    assert restored.is_calibrated
    assert restored._q_hat == trained_predictor._q_hat
    assert restored._calibration_coverage == trained_predictor._calibration_coverage
    # Predictions must be identical.
    assert restored.predict(baseline_features) == trained_predictor.predict(
        baseline_features
    )


def test_is_calibrated_false_before_fit() -> None:
    assert not CostPredictor().is_calibrated


def test_fit_raises_on_invalid_calibration_frac() -> None:
    df = generate_synthetic_dataset(n_rows=50, seed=1)
    with pytest.raises(ValueError, match="calibration_frac"):
        CostPredictor().fit(df, calibration_frac=0.0)
    with pytest.raises(ValueError, match="calibration_frac"):
        CostPredictor().fit(df, calibration_frac=1.5)


def test_fit_raises_on_invalid_calibration_coverage() -> None:
    df = generate_synthetic_dataset(n_rows=50, seed=1)
    with pytest.raises(ValueError, match="calibration_coverage"):
        CostPredictor().fit(df, calibration_coverage=0.0)
    with pytest.raises(ValueError, match="calibration_coverage"):
        CostPredictor().fit(df, calibration_coverage=1.0)


def test_load_old_model_without_calibration_keys(
    trained_predictor: CostPredictor,
    tmp_path,
) -> None:
    """Models saved before CQR was added load without error as uncalibrated."""
    import joblib

    # Simulate an old-format save: no q_hat or calibration_coverage keys.
    cache = tmp_path / "old_model.joblib"
    old_blob = {
        "quantiles": trained_predictor.quantiles,
        "pipelines": trained_predictor._trained.pipelines,  # type: ignore[union-attr]
    }
    joblib.dump(old_blob, cache)

    loaded = CostPredictor.load(cache)
    assert loaded.is_fitted
    assert not loaded.is_calibrated
    pred = loaded.predict(
        PredictionFeatures(
            age=35, sex="female", bmi=27.5, children=1, smoker="no", region="northeast"
        )
    )
    assert pred.conformal_calibrated is False
