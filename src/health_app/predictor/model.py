"""Quantile + mean cost predictor.

Four gradient-boosted regressors are trained per model:

* Quantile 0.1: the lower bound of the 80% prediction interval.
* Quantile 0.5: the median (a "typical year" estimate).
* Quantile 0.9: the upper bound of the 80% prediction interval.
* Mean (squared-error loss): the expected long-run cost.

Healthcare costs are heavily right-skewed. The median is what most
people experience in any given year; the mean is the long-run average
that incorporates the rare-but-expensive tail. Showing both is much
more honest than showing either alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from health_app.predictor.schemas import PredictionFeatures

NUMERIC_FEATURES: list[str] = ["age", "bmi", "children"]
CATEGORICAL_FEATURES: list[str] = ["sex", "smoker", "region"]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)

# Keys for the internal pipelines dict.
KEY_LOWER = "q0.10"
KEY_MEDIAN = "q0.50"
KEY_UPPER = "q0.90"
KEY_MEAN = "mean"


class CostPrediction(BaseModel):
    """Predicted annual medical charges, in cents.

    `median_charges_cents` is the 50th percentile (a "typical year").
    `mean_charges_cents` is the squared-error expectation (the long-run
    average; pulled up by the heavy tail). `lower_bound_cents` and
    `upper_bound_cents` form an 80% prediction interval.
    """

    model_config = ConfigDict(frozen=True)

    median_charges_cents: int
    mean_charges_cents: int
    lower_bound_cents: int
    upper_bound_cents: int

    @property
    def median_charges_dollars(self) -> float:
        return self.median_charges_cents / 100

    @property
    def mean_charges_dollars(self) -> float:
        return self.mean_charges_cents / 100

    @property
    def interval_width_cents(self) -> int:
        return self.upper_bound_cents - self.lower_bound_cents


@dataclass
class _TrainedPipelines:
    """Internal container for the four fitted pipelines, keyed by role."""

    pipelines: dict[str, Pipeline]


def _build_quantile_pipeline(quantile: float) -> Pipeline:
    """Construct a fresh untrained quantile pipeline."""
    return _build_pipeline(
        HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=200,
            max_depth=6,
            learning_rate=0.08,
            random_state=0,
        )
    )


def _build_mean_pipeline() -> Pipeline:
    """Construct a fresh untrained mean (squared-error) pipeline."""
    return _build_pipeline(
        HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=200,
            max_depth=6,
            learning_rate=0.08,
            random_state=0,
        )
    )


def _build_pipeline(regressor) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", regressor)]
    )


class CostPredictor:
    """Trainable cost predictor with point estimates and a prediction interval."""

    def __init__(
        self,
        quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    ) -> None:
        lower, point, upper = quantiles
        if not (0.0 < lower < point < upper < 1.0):
            raise ValueError(
                "quantiles must be three increasing values in (0, 1); "
                f"got {quantiles}"
            )
        self.quantiles = quantiles
        self._trained: _TrainedPipelines | None = None

    @property
    def is_fitted(self) -> bool:
        return self._trained is not None

    def fit(self, df: pd.DataFrame, target_column: str = "charges") -> "CostPredictor":
        """Fit four regressors: three quantile and one mean."""
        missing = set(ALL_FEATURES + [target_column]) - set(df.columns)
        if missing:
            raise ValueError(f"Training data missing columns: {sorted(missing)}")

        X = df[ALL_FEATURES]
        y = df[target_column].to_numpy()

        lower_q, point_q, upper_q = self.quantiles
        pipelines: dict[str, Pipeline] = {
            KEY_LOWER: _build_quantile_pipeline(lower_q),
            KEY_MEDIAN: _build_quantile_pipeline(point_q),
            KEY_UPPER: _build_quantile_pipeline(upper_q),
            KEY_MEAN: _build_mean_pipeline(),
        }
        for pipe in pipelines.values():
            pipe.fit(X, y)
        self._trained = _TrainedPipelines(pipelines=pipelines)
        return self

    def predict(self, features: PredictionFeatures) -> CostPrediction:
        """Predict charges for a single feature vector."""
        if self._trained is None:
            raise RuntimeError("CostPredictor has not been fitted.")

        df = _features_to_dataframe(features)
        lower = float(self._trained.pipelines[KEY_LOWER].predict(df)[0])
        median = float(self._trained.pipelines[KEY_MEDIAN].predict(df)[0])
        upper = float(self._trained.pipelines[KEY_UPPER].predict(df)[0])
        mean = float(self._trained.pipelines[KEY_MEAN].predict(df)[0])

        # Clamp at zero and enforce sane ordering of the quantile-derived
        # values (lower <= median <= upper) so the interval is always coherent.
        lower = max(0.0, lower)
        median = max(lower, median)
        upper = max(median, upper)
        mean = max(0.0, mean)

        return CostPrediction(
            median_charges_cents=_dollars_to_cents(median),
            mean_charges_cents=_dollars_to_cents(mean),
            lower_bound_cents=_dollars_to_cents(lower),
            upper_bound_cents=_dollars_to_cents(upper),
        )

    def predict_many(
        self, feature_rows: list[PredictionFeatures]
    ) -> list[CostPrediction]:
        """Batch predict for many feature vectors."""
        if self._trained is None:
            raise RuntimeError("CostPredictor has not been fitted.")
        if not feature_rows:
            return []

        df = pd.DataFrame([f.model_dump() for f in feature_rows])[ALL_FEATURES]
        lower = self._trained.pipelines[KEY_LOWER].predict(df)
        median = self._trained.pipelines[KEY_MEDIAN].predict(df)
        upper = self._trained.pipelines[KEY_UPPER].predict(df)
        mean = self._trained.pipelines[KEY_MEAN].predict(df)

        lower = np.maximum(0.0, lower)
        median = np.maximum(lower, median)
        upper = np.maximum(median, upper)
        mean = np.maximum(0.0, mean)

        return [
            CostPrediction(
                median_charges_cents=_dollars_to_cents(median[i]),
                mean_charges_cents=_dollars_to_cents(mean[i]),
                lower_bound_cents=_dollars_to_cents(lower[i]),
                upper_bound_cents=_dollars_to_cents(upper[i]),
            )
            for i in range(len(feature_rows))
        ]

    def save(self, path: Path | str) -> None:
        if self._trained is None:
            raise RuntimeError("Nothing to save; predictor has not been fitted.")
        joblib.dump(
            {
                "quantiles": self.quantiles,
                "pipelines": self._trained.pipelines,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "CostPredictor":
        blob = joblib.load(path)
        inst = cls(quantiles=tuple(blob["quantiles"]))
        inst._trained = _TrainedPipelines(pipelines=blob["pipelines"])
        return inst


def _features_to_dataframe(features: PredictionFeatures) -> pd.DataFrame:
    return pd.DataFrame([features.model_dump()])[ALL_FEATURES]


def _dollars_to_cents(amount: float) -> int:
    return max(0, int(round(amount * 100)))
