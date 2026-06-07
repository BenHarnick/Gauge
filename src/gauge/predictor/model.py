"""Quantile + mean cost predictor with conformal calibration.

Four gradient-boosted regressors are trained per model:

* Quantile 0.1: lower bound of the raw quantile interval.
* Quantile 0.5: median (a "typical year" estimate).
* Quantile 0.9: upper bound of the raw quantile interval.
* Mean (squared-error loss): the expected long-run cost.

After fitting, the predictor automatically calibrates its prediction
interval using Conformal Quantile Regression (CQR) on a held-out
calibration split.  CQR guarantees that the reported interval contains
the true value with at least ``coverage`` marginal probability for any
data distribution, without assuming normality.

The calibration procedure:

1. Hold out ``calibration_frac`` (default 20 %) of the training data
   before fitting the four regressors.
2. On the held-out set, compute the CQR nonconformity score for each
   row: ``score = max(q_lo(x) - y,  y - q_hi(x))``.  A negative score
   means the true value was already inside the raw interval; a positive
   score is how far outside it fell.
3. Set ``q_hat`` to the ``ceil((n+1)*(1-alpha))/n`` empirical quantile
   of those scores (the standard finite-sample correction that guarantees
   marginal coverage ≥ 1-alpha).
4. At prediction time expand the raw interval symmetrically:
   ``[q_lo(x) - q_hat,  q_hi(x) + q_hat]``.

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

from gauge.predictor.schemas import PredictionFeatures

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

    ``median_charges_cents`` is the 50th percentile (a "typical year").
    ``mean_charges_cents`` is the squared-error expectation (the long-run
    average; pulled up by the heavy tail). ``lower_bound_cents`` and
    ``upper_bound_cents`` form an 80% prediction interval.

    When the predictor has been conformal-calibrated (see
    :class:`CostPredictor`), ``conformal_calibrated`` is ``True`` and the
    interval has a marginal coverage guarantee of at least
    ``calibration_coverage``.  When ``False`` the interval is a raw
    quantile interval with no coverage guarantee.
    """

    model_config = ConfigDict(frozen=True)

    median_charges_cents: int
    mean_charges_cents: int
    lower_bound_cents: int
    upper_bound_cents: int
    conformal_calibrated: bool = False
    calibration_coverage: float | None = None

    @property
    def median_charges_dollars(self) -> float:
        """Median predicted charges converted from cents to dollars.

        Returns
        -------
        float
            ``median_charges_cents / 100``.
        """
        return self.median_charges_cents / 100

    @property
    def mean_charges_dollars(self) -> float:
        """Mean predicted charges converted from cents to dollars.

        Returns
        -------
        float
            ``mean_charges_cents / 100``.
        """
        return self.mean_charges_cents / 100

    @property
    def interval_width_cents(self) -> int:
        """Width of the prediction interval in cents.

        Returns
        -------
        int
            ``upper_bound_cents - lower_bound_cents``.  Always non-negative
            because the predictor enforces ``lower_bound_cents <=
            upper_bound_cents``.
        """
        return self.upper_bound_cents - self.lower_bound_cents


@dataclass
class _TrainedPipelines:
    """Internal container for the four fitted pipelines, keyed by role."""

    pipelines: dict[str, Pipeline]


def _build_quantile_pipeline(quantile: float) -> Pipeline:
    """Construct a fresh untrained quantile-regression pipeline.

    Parameters
    ----------
    quantile : float
        Target quantile in ``(0, 1)`` for the
        ``HistGradientBoostingRegressor``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Untrained pipeline with a ``ColumnTransformer`` preprocessor and a
        quantile-loss gradient-boosted regressor.
    """
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
    """Construct a fresh untrained mean (squared-error) pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Untrained pipeline with a ``ColumnTransformer`` preprocessor and a
        squared-error gradient-boosted regressor.
    """
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
    """Wrap a regressor in a standard preprocessing pipeline.

    Parameters
    ----------
    regressor : estimator
        A scikit-learn regressor conforming to the ``fit``/``predict``
        interface.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Two-step pipeline: a ``ColumnTransformer`` that one-hot encodes
        categorical features and passes numerics through, followed by
        ``regressor``.
    """
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
    """Trainable cost predictor with point estimates and a conformal interval.

    After :meth:`fit`, the predictor is automatically calibrated with
    Conformal Quantile Regression (CQR) on a held-out split of the
    training data.  The reported interval then has a marginal coverage
    guarantee of at least ``calibration_coverage`` for any data
    distribution.
    """

    def __init__(
        self,
        quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    ) -> None:
        """Initialise the predictor with a chosen prediction interval.

        Parameters
        ----------
        quantiles : tuple[float, float, float], optional
            Three strictly increasing values in ``(0, 1)`` representing the
            lower bound, median, and upper bound of the raw quantile
            interval.  Defaults to ``(0.1, 0.5, 0.9)`` (an 80 % interval),
            which matches the ``calibration_coverage`` default of 0.80.

        Raises
        ------
        ValueError
            If the values are not strictly increasing or any value falls
            outside ``(0, 1)``.
        """
        lower, point, upper = quantiles
        if not (0.0 < lower < point < upper < 1.0):
            raise ValueError(
                "quantiles must be three increasing values in (0, 1); "
                f"got {quantiles}"
            )
        self.quantiles = quantiles
        self._trained: _TrainedPipelines | None = None
        # Set by _run_calibration(); None means not yet calibrated.
        self._q_hat: float | None = None
        self._calibration_coverage: float | None = None

    @property
    def is_fitted(self) -> bool:
        """``True`` once :meth:`fit` has been called successfully.

        Returns
        -------
        bool
            ``True`` if the four regressors have been trained; ``False``
            otherwise.
        """
        return self._trained is not None

    @property
    def is_calibrated(self) -> bool:
        """``True`` once CQR calibration has been run and ``q_hat`` is stored.

        Returns
        -------
        bool
            ``True`` if :meth:`_run_calibration` has stored a finite
            ``q_hat``; ``False`` otherwise.
        """
        return self._q_hat is not None

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str = "charges",
        calibration_frac: float = 0.2,
        calibration_coverage: float = 0.80,
        seed: int = 0,
    ) -> "CostPredictor":
        """Fit four regressors and run CQR calibration on a held-out split.

        The data is shuffled and split into a training set
        (``1 - calibration_frac``) and a calibration set
        (``calibration_frac``).  The four regressors are fitted on the
        training set only; the calibration set is used exclusively to
        compute the CQR adjustment ``q_hat`` (see module docstring).

        Parameters
        ----------
        df : pd.DataFrame
            Training data. Must include all columns in ``ALL_FEATURES``
            plus ``target_column``.
        target_column : str, optional
            Name of the target column in ``df``. Default is ``"charges"``.
        calibration_frac : float, optional
            Fraction of rows to hold out for CQR calibration.  Must be
            in ``(0, 1)``.  Default is 0.20 (20 %).
        calibration_coverage : float, optional
            Target marginal coverage for the conformal interval, e.g.
            0.80 for an 80 % interval.  Must be in ``(0, 1)``.  Default
            is 0.80.
        seed : int, optional
            Random seed for the train/calibration shuffle.  Default is 0.

        Returns
        -------
        CostPredictor
            Self, to allow method chaining.

        Raises
        ------
        ValueError
            If any required column is missing from ``df``, or if
            ``calibration_frac`` or ``calibration_coverage`` are out of
            range.
        """
        missing = set(ALL_FEATURES + [target_column]) - set(df.columns)
        if missing:
            raise ValueError(f"Training data missing columns: {sorted(missing)}")
        if not (0.0 < calibration_frac < 1.0):
            raise ValueError(
                f"calibration_frac must be in (0, 1); got {calibration_frac}"
            )
        if not (0.0 < calibration_coverage < 1.0):
            raise ValueError(
                f"calibration_coverage must be in (0, 1); got {calibration_coverage}"
            )

        # --- train / calibration split ---
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n_cal = max(1, int(len(df) * calibration_frac))
        cal_idx = idx[:n_cal]
        train_idx = idx[n_cal:]

        df_train = df.iloc[train_idx]
        df_cal = df.iloc[cal_idx]

        X_train = df_train[ALL_FEATURES]
        y_train = df_train[target_column].to_numpy()

        # --- fit regressors on training split ---
        lower_q, point_q, upper_q = self.quantiles
        pipelines: dict[str, Pipeline] = {
            KEY_LOWER: _build_quantile_pipeline(lower_q),
            KEY_MEDIAN: _build_quantile_pipeline(point_q),
            KEY_UPPER: _build_quantile_pipeline(upper_q),
            KEY_MEAN: _build_mean_pipeline(),
        }
        for pipe in pipelines.values():
            pipe.fit(X_train, y_train)
        self._trained = _TrainedPipelines(pipelines=pipelines)

        # --- CQR calibration on held-out split ---
        self._run_calibration(df_cal, target_column, calibration_coverage)
        return self

    def _run_calibration(
        self,
        df_cal: pd.DataFrame,
        target_column: str,
        coverage: float,
    ) -> None:
        """Compute and store the CQR adjustment ``q_hat``.

        Uses the Conformalized Quantile Regression nonconformity score::

            score_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))

        A negative score means the true value was inside the raw quantile
        interval; a positive score records by how much it fell outside.
        ``q_hat`` is set to the finite-sample corrected empirical quantile
        of these scores at level ``ceil((n+1)*(1-alpha))/n``.

        Parameters
        ----------
        df_cal : pd.DataFrame
            Calibration rows (held out from training).
        target_column : str
            Name of the target column in ``df_cal``.
        coverage : float
            Target marginal coverage, e.g. 0.80.

        Raises
        ------
        RuntimeError
            If called before the regressors have been fitted.
        """
        if self._trained is None:
            raise RuntimeError(
                "_run_calibration called before regressors were fitted."
            )
        X_cal = df_cal[ALL_FEATURES]
        y_cal = df_cal[target_column].to_numpy()

        q_lo = self._trained.pipelines[KEY_LOWER].predict(X_cal)
        q_hi = self._trained.pipelines[KEY_UPPER].predict(X_cal)

        # CQR nonconformity scores.
        scores = np.maximum(q_lo - y_cal, y_cal - q_hi)

        n = len(scores)
        alpha = 1.0 - coverage
        # Finite-sample correction: use ceil((n+1)*(1-alpha))/n as the
        # quantile level.  Clamp to 1.0 when n is very small.
        level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
        self._q_hat = float(np.quantile(scores, level))
        self._calibration_coverage = coverage

    def predict(self, features: PredictionFeatures) -> CostPrediction:
        """Predict charges for a single feature vector.

        Parameters
        ----------
        features : PredictionFeatures
            Input feature vector for one individual.

        Returns
        -------
        CostPrediction
            Predicted charges with median, mean, lower bound, and upper
            bound all expressed in cents, clamped to zero and monotonically
            ordered.  When the predictor is calibrated, the interval is
            expanded by ``q_hat`` (CQR adjustment) and
            ``conformal_calibrated`` is ``True``.

        Raises
        ------
        RuntimeError
            If the predictor has not been fitted yet.
        """
        if self._trained is None:
            raise RuntimeError("CostPredictor has not been fitted.")

        df = _features_to_dataframe(features)
        lower = float(self._trained.pipelines[KEY_LOWER].predict(df)[0])
        median = float(self._trained.pipelines[KEY_MEDIAN].predict(df)[0])
        upper = float(self._trained.pipelines[KEY_UPPER].predict(df)[0])
        mean = float(self._trained.pipelines[KEY_MEAN].predict(df)[0])

        # Apply CQR adjustment when calibrated.
        if self._q_hat is not None:
            lower = lower - self._q_hat
            upper = upper + self._q_hat

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
            conformal_calibrated=self.is_calibrated,
            calibration_coverage=self._calibration_coverage,
        )

    def predict_many(
        self, feature_rows: list[PredictionFeatures]
    ) -> list[CostPrediction]:
        """Batch-predict charges for multiple feature vectors.

        Runs all four pipelines once per batch rather than once per row,
        so dozens of predictions take roughly the same time as a single one.

        Parameters
        ----------
        feature_rows : list[PredictionFeatures]
            Feature vectors for each individual.

        Returns
        -------
        list[CostPrediction]
            Predictions aligned to ``feature_rows``, clamped and ordered
            identically to :meth:`predict`.  CQR adjustment is applied to
            every row when the predictor is calibrated.  Returns an empty
            list when ``feature_rows`` is empty.

        Raises
        ------
        RuntimeError
            If the predictor has not been fitted yet.
        """
        if self._trained is None:
            raise RuntimeError("CostPredictor has not been fitted.")
        if not feature_rows:
            return []

        df = pd.DataFrame([f.model_dump() for f in feature_rows])[ALL_FEATURES]
        lower = self._trained.pipelines[KEY_LOWER].predict(df)
        median = self._trained.pipelines[KEY_MEDIAN].predict(df)
        upper = self._trained.pipelines[KEY_UPPER].predict(df)
        mean = self._trained.pipelines[KEY_MEAN].predict(df)

        # Apply CQR adjustment when calibrated.
        if self._q_hat is not None:
            lower = lower - self._q_hat
            upper = upper + self._q_hat

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
                conformal_calibrated=self.is_calibrated,
                calibration_coverage=self._calibration_coverage,
            )
            for i in range(len(feature_rows))
        ]

    def save(self, path: Path | str) -> None:
        """Serialise the fitted predictor to disk with :mod:`joblib`.

        Parameters
        ----------
        path : Path or str
            Destination file path. Parent directories must exist.

        Raises
        ------
        RuntimeError
            If the predictor has not been fitted yet.
        """
        if self._trained is None:
            raise RuntimeError("Nothing to save; predictor has not been fitted.")
        joblib.dump(
            {
                "quantiles": self.quantiles,
                "pipelines": self._trained.pipelines,
                "q_hat": self._q_hat,
                "calibration_coverage": self._calibration_coverage,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "CostPredictor":
        """Deserialise a predictor previously saved with :meth:`save`.

        Parameters
        ----------
        path : Path or str
            Path to the ``.joblib`` file written by :meth:`save`.

        Returns
        -------
        CostPredictor
            A fully fitted predictor ready to call :meth:`predict`.
            ``is_calibrated`` will be ``True`` when the file was saved after
            CQR calibration; models saved before calibration was introduced
            load as uncalibrated (``is_calibrated`` is ``False``).
        """
        blob = joblib.load(path)
        inst = cls(quantiles=tuple(blob["quantiles"]))
        inst._trained = _TrainedPipelines(pipelines=blob["pipelines"])
        # Graceful backward compatibility: older saved models won't have
        # these keys; treat them as uncalibrated.
        inst._q_hat = blob.get("q_hat", None)
        inst._calibration_coverage = blob.get("calibration_coverage", None)
        return inst


def _features_to_dataframe(features: PredictionFeatures) -> pd.DataFrame:
    """Convert a single feature vector into a one-row DataFrame.

    Parameters
    ----------
    features : PredictionFeatures
        Input feature vector.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns ordered as ``ALL_FEATURES``.
    """
    return pd.DataFrame([features.model_dump()])[ALL_FEATURES]


def _dollars_to_cents(amount: float) -> int:
    """Round a dollar amount to the nearest whole cent, clamped at zero.

    Parameters
    ----------
    amount : float
        Dollar amount to convert.

    Returns
    -------
    int
        Equivalent amount in cents, never negative.
    """
    return max(0, int(round(amount * 100)))
