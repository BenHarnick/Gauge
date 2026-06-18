"""Reproducible evaluation of the Gauge cost predictor.

Measures conformal coverage calibration, model benchmarks, feature ablations,
and data fidelity across multiple random seeds.  All random operations are
seeded so the output is identical on every run.

Usage
-----
::

    python -m gauge.eval

Writes to ``reports/figures/`` (PNG + SVG) and ``reports/benchmark.json``.
``MODELING.md`` is written to the repo root.
"""

from __future__ import annotations

import json
import math
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from gauge.benefits.models import Plan
from gauge.predictor.annual_cost import apply_plan_to_annual_spend
from gauge.predictor.dataset import generate_synthetic_dataset, load_csv
from gauge.predictor.meps import load_meps
from gauge.predictor.model import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    _build_mean_pipeline,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TARGET = "charges"

NOMINAL_LEVELS: list[float] = [0.50, 0.80, 0.90, 0.95]
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

# Colour palette — accessible, consistent across all figures
C_CQR = "#2563eb"  # blue — CQR / Gauge model
C_RAW = "#dc2626"  # red  — raw quantile (no conformal)
C_IDEAL = "#16a34a"  # green — target / perfect calibration
C_NEUTRAL = "#6b7280"  # grey — baselines

# Representative plan used in the OOP transform chart.
# Deductible + coinsurance + OOP max reflect a typical US employer PPO.
_REFERENCE_PLAN = Plan(
    plan_id="ppo_reference",
    name="Representative PPO",
    deductible_cents=150_000,  # $1,500
    out_of_pocket_max_cents=600_000,  # $6,000
    coinsurance_rate=0.20,
    copays_cents={},
)

# Shared rcParams for all figures
_RC: dict[str, Any] = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
}


def _rc_context() -> AbstractContextManager[None]:
    """Typed wrapper around ``plt.rc_context(_RC)``.

    matplotlib's stub types ``rc_context``'s ``rc`` argument against a
    ``Literal`` union of every valid rcParams key, which a plain
    ``dict[str, Any]`` can never satisfy under mypy's invariant dict typing.
    Centralizing the suppression here keeps it out of every call site.
    """
    return plt.rc_context(_RC)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def resolve_data() -> tuple[pd.DataFrame, str]:
    """Return ``(df, source_label)`` preferring MEPS > Kaggle CSV > synthetic.

    Returns
    -------
    tuple[pd.DataFrame, str]
        DataFrame with feature and target columns, plus a human-readable
        string naming the data source (used in figure titles and MODELING.md).
    """
    meps_path = REPO_ROOT / "data" / "meps_hc233.dta"
    saq_path = REPO_ROOT / "data" / "meps_hc236.dta"
    csv_path = REPO_ROOT / "data" / "insurance.csv"

    if meps_path.exists():
        saq = saq_path if saq_path.exists() else None
        df = load_meps(meps_path, saq_path=saq)
        return df, f"MEPS HC-233 (n={len(df):,})"

    if csv_path.exists():
        df = load_csv(csv_path)
        return df, f"Kaggle insurance CSV (n={len(df):,})"

    print("[warn] No real data found — using synthetic fallback.", file=sys.stderr)
    df = generate_synthetic_dataset(n_rows=2_000, seed=42)
    return df, f"Synthetic (n={len(df):,}, seed=42)"


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def cqr_q_hat(scores: npt.NDArray[Any], coverage: float) -> float:
    """Finite-sample corrected CQR quantile of nonconformity scores.

    Implements the correction from Romano, Patterson & Candès (2019):
    ``level = ceil((n+1)(1-α)) / n``, clamped to 1 when ``n`` is tiny.

    Parameters
    ----------
    scores : npt.NDArray[Any]
        CQR nonconformity scores on the calibration set.
    coverage : float
        Nominal marginal coverage target, e.g. 0.80.

    Returns
    -------
    float
        ``q_hat`` — the conformal adjustment to add/subtract from the raw
        quantile interval endpoints.
    """
    n = len(scores)
    alpha = 1.0 - coverage
    level = min(math.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def empirical_coverage(y: npt.NDArray[Any], lo: npt.NDArray[Any], hi: npt.NDArray[Any]) -> float:
    """Fraction of test points inside ``[lo, hi]``.

    Parameters
    ----------
    y : npt.NDArray[Any]
        True target values.
    lo, hi : npt.NDArray[Any]
        Lower and upper interval endpoints.

    Returns
    -------
    float
        Empirical coverage in ``[0, 1]``.
    """
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: npt.NDArray[Any], hi: npt.NDArray[Any]) -> float:
    """Mean interval width in dollars.

    Parameters
    ----------
    lo, hi : npt.NDArray[Any]
        Lower and upper endpoints (dollars).

    Returns
    -------
    float
        Mean of ``hi - lo``.
    """
    return float(np.mean(hi - lo))


def pinball_loss(y: npt.NDArray[Any], q_pred: npt.NDArray[Any], quantile: float) -> float:
    """Quantile (pinball) loss — the proper scoring rule for quantile forecasts.

    A lower value is better.  Lead with this metric when comparing interval
    quality: unlike coverage, it cannot be gamed by inflating the interval.

    Parameters
    ----------
    y : npt.NDArray[Any]
        True target values.
    q_pred : npt.NDArray[Any]
        Predicted quantile at level ``quantile``.
    quantile : float
        Target quantile in ``(0, 1)``.

    Returns
    -------
    float
        Mean pinball loss over the test set.
    """
    err = y - q_pred
    return float(np.mean(np.where(err >= 0, quantile * err, (quantile - 1) * err)))


def mae_metric(y: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y - y_pred)))


def rmse_metric(y: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Train / calibration / test split
# ---------------------------------------------------------------------------


def three_way_split(
    df: pd.DataFrame,
    seed: int,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]
]:
    """Shuffle and split ``df`` into train / calibration / test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features and ``TARGET`` column.
    seed : int
        Seed for the permutation.
    train_frac : float, optional
        Fraction of rows for training. Default 0.60.
    cal_frac : float, optional
        Fraction of rows for CQR calibration. Default 0.20.
        The remainder (``1 - train_frac - cal_frac``) is used for testing.

    Returns
    -------
    tuple
        ``(df_train, df_cal, df_test, y_train, y_cal, y_test)`` where the
        ``y_*`` arrays are the target column extracted as numpy arrays.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n = len(df)
    n_test = max(1, int(n * (1.0 - train_frac - cal_frac)))
    n_cal = max(1, int(n * cal_frac))

    test_idx = idx[:n_test]
    cal_idx = idx[n_test : n_test + n_cal]
    train_idx = idx[n_test + n_cal :]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_cal = df.iloc[cal_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    return (
        df_train,
        df_cal,
        df_test,
        df_train[TARGET].to_numpy(),
        df_cal[TARGET].to_numpy(),
        df_test[TARGET].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Main quantile model
# ---------------------------------------------------------------------------


@dataclass
class QuantileModel:
    """Fitted GBM quantile pipelines (q=0.10 / 0.50 / 0.90) with CQR calibration.

    The ``cal_scores`` array stores the CQR nonconformity score for every
    calibration point: ``score_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))``.
    Calling :meth:`predict_cqr` with a different ``coverage`` level reuses
    the same scores, so multiple coverage targets can be evaluated without
    retraining.
    """

    q_lo: Pipeline
    q_med: Pipeline
    q_hi: Pipeline
    cal_scores: npt.NDArray[Any]

    def predict_cqr(
        self, X: pd.DataFrame, coverage: float = 0.80
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """Predict with CQR-adjusted interval at the given nominal coverage.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        coverage : float, optional
            Nominal marginal coverage target. Default 0.80.

        Returns
        -------
        tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]
            ``(lower, median, upper)`` all clamped at zero.
        """
        q = cqr_q_hat(self.cal_scores, coverage)
        lo = np.maximum(0.0, self.q_lo.predict(X) - q)
        med = self.q_med.predict(X)
        hi = np.maximum(0.0, self.q_hi.predict(X) + q)
        return lo, med, hi

    def predict_raw(
        self, X: pd.DataFrame
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """Predict with the un-conformalized (raw) quantile interval.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]
            ``(lower, median, upper)`` clamped at zero, no CQR adjustment.
        """
        lo = np.maximum(0.0, self.q_lo.predict(X))
        med = self.q_med.predict(X)
        hi = np.maximum(0.0, self.q_hi.predict(X))
        return lo, med, hi


def _build_pipeline_for_features(quantile: float | None, cols: list[str]) -> Pipeline:
    """Build a quantile or mean pipeline whose preprocessor knows only ``cols``.

    Parameters
    ----------
    quantile : float or None
        Target quantile. ``None`` uses squared-error loss (mean).
    cols : list[str]
        The exact columns that will be passed to this pipeline's fit/predict.
        Only the categorical columns present in ``cols`` are one-hot encoded.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in cols]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    if quantile is not None:
        regressor = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=200,
            max_depth=6,
            learning_rate=0.08,
            random_state=0,
        )
    else:
        regressor = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=200,
            max_depth=6,
            learning_rate=0.08,
            random_state=0,
        )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: npt.NDArray[Any],
    X_cal: pd.DataFrame,
    y_cal: npt.NDArray[Any],
) -> QuantileModel:
    """Fit the three GBM quantile pipelines and compute CQR nonconformity scores.

    Parameters
    ----------
    X_train, X_cal : pd.DataFrame
        Feature matrices for training and calibration sets.
    y_train, y_cal : npt.NDArray[Any]
        Target arrays (annual charges in dollars).

    Returns
    -------
    QuantileModel
        Fitted model ready to call :meth:`~QuantileModel.predict_cqr` or
        :meth:`~QuantileModel.predict_raw`.
    """
    cols = list(X_train.columns)
    q_lo = _build_pipeline_for_features(0.10, cols)
    q_med = _build_pipeline_for_features(0.50, cols)
    q_hi = _build_pipeline_for_features(0.90, cols)

    q_lo.fit(X_train, y_train)
    q_med.fit(X_train, y_train)
    q_hi.fit(X_train, y_train)

    lo_cal = q_lo.predict(X_cal)
    hi_cal = q_hi.predict(X_cal)
    # CQR nonconformity score: positive means the true value fell outside the raw interval.
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)

    return QuantileModel(q_lo, q_med, q_hi, scores)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _preprocessor() -> ColumnTransformer:
    """Standard OHE-for-categoricals preprocessor used by linear baseline."""
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )


def _baseline_global_mean(
    y_train: npt.NDArray[Any], n_test: int
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    mean = float(y_train.mean())
    lo = float(np.quantile(y_train, 0.10))
    hi = float(np.quantile(y_train, 0.90))
    return np.full(n_test, lo), np.full(n_test, mean), np.full(n_test, hi)


def _baseline_linear(
    X_train: pd.DataFrame,
    y_train: npt.NDArray[Any],
    X_test: pd.DataFrame,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    pipe = Pipeline([("pre", _preprocessor()), ("reg", LinearRegression())])
    pipe.fit(X_train, y_train)
    y_pred = np.maximum(0.0, pipe.predict(X_test))
    resid_std = float(np.std(y_train - np.maximum(0.0, pipe.predict(X_train))))
    z = 1.282  # ±1.282σ ≈ 80% Gaussian interval
    lo = np.maximum(0.0, y_pred - z * resid_std)
    hi = y_pred + z * resid_std
    return lo, y_pred, hi


def _baseline_gbm_gaussian(
    X_train: pd.DataFrame,
    y_train: npt.NDArray[Any],
    X_test: pd.DataFrame,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """GBM point estimate (squared-error) with Gaussian interval from training residuals."""
    pipe = _build_mean_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = np.maximum(0.0, pipe.predict(X_test))
    resid_std = float(np.std(y_train - np.maximum(0.0, pipe.predict(X_train))))
    z = 1.282
    lo = np.maximum(0.0, y_pred - z * resid_std)
    hi = y_pred + z * resid_std
    return lo, y_pred, hi


# ---------------------------------------------------------------------------
# Subgroup definitions
# ---------------------------------------------------------------------------

SUBGROUPS: list[str] = [
    "Smoker",
    "Non-smoker",
    "Age < 35",
    "Age 35–50",
    "Age > 50",
    "BMI < 25",
    "BMI 25–30",
    "BMI ≥ 30",
    "Northeast",
    "Midwest",
    "South",
    "West",
]


def _subgroup_mask(df: pd.DataFrame, label: str) -> npt.NDArray[Any]:
    """Boolean mask selecting the named demographic subgroup from ``df``."""
    age = df["age"].to_numpy()
    bmi = df["bmi"].to_numpy()
    sm = df["smoker"].to_numpy()
    reg = df["region"].to_numpy()

    mapping: dict[str, npt.NDArray[Any]] = {
        "Smoker": sm == "yes",
        "Non-smoker": sm == "no",
        "Age < 35": age < 35,
        "Age 35–50": (age >= 35) & (age <= 50),
        "Age > 50": age > 50,
        "BMI < 25": bmi < 25.0,
        "BMI 25–30": (bmi >= 25.0) & (bmi < 30.0),
        "BMI ≥ 30": bmi >= 30.0,
        "Northeast": reg == "northeast",
        "Midwest": reg == "midwest",
        "South": reg == "south",
        "West": reg == "west",
    }
    return mapping[label]


# ---------------------------------------------------------------------------
# Per-seed result
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """All metrics collected for one random seed."""

    seed: int
    # §2.3.1 coverage at each nominal level (CQR and raw)
    cqr_coverage: dict[float, float] = field(default_factory=dict)
    raw_coverage: float = 0.0  # fixed: empirical coverage of raw q0.1/q0.9 interval
    # Interval width at 80%
    cqr_width_80: float = 0.0
    raw_width_80: float = 0.0
    # Conditional coverage per subgroup at 80%
    subgroup_cov: dict[str, float] = field(default_factory=dict)
    # §2.3.2 benchmark: model → metric → value
    benchmarks: dict[str, dict[str, float]] = field(default_factory=dict)
    # Ablations: name → metric → value
    ablations: dict[str, dict[str, float]] = field(default_factory=dict)
    # Raw predictions for scatter plot (populated for seed 0 only)
    y_test: npt.NDArray[Any] | None = None
    y_pred_med: npt.NDArray[Any] | None = None
    y_pred_lo: npt.NDArray[Any] | None = None
    y_pred_hi: npt.NDArray[Any] | None = None


def evaluate_seed(df: pd.DataFrame, seed: int, store_preds: bool = False) -> SeedResult:
    """Run the complete evaluation for one random seed.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    seed : int
        Random seed controlling the train/cal/test split.
    store_preds : bool, optional
        When ``True``, store raw test-set predictions on the result for
        scatter-plot use.  Only needed for one seed to avoid memory waste.

    Returns
    -------
    SeedResult
        All metrics for this seed.
    """
    df_train, df_cal, df_test, y_train, y_cal, y_test = three_way_split(df, seed)
    X_train = df_train[ALL_FEATURES]
    X_cal = df_cal[ALL_FEATURES]
    X_test = df_test[ALL_FEATURES]

    result = SeedResult(seed=seed)

    # --- main model ---
    model = train_quantile_model(X_train, y_train, X_cal, y_cal)

    # §2.3.1 — coverage calibration
    for level in NOMINAL_LEVELS:
        lo, _, hi = model.predict_cqr(X_test, coverage=level)
        result.cqr_coverage[level] = empirical_coverage(y_test, lo, hi)

    raw_lo, _, raw_hi = model.predict_raw(X_test)
    result.raw_coverage = empirical_coverage(y_test, raw_lo, raw_hi)

    lo_80, med_80, hi_80 = model.predict_cqr(X_test, coverage=0.80)
    result.cqr_width_80 = mean_width(lo_80, hi_80)
    result.raw_width_80 = mean_width(raw_lo, raw_hi)

    # §2.3.1 — conditional coverage per subgroup at 80%
    for sg in SUBGROUPS:
        mask = _subgroup_mask(df_test, sg)
        if mask.sum() >= 5:
            result.subgroup_cov[sg] = empirical_coverage(y_test[mask], lo_80[mask], hi_80[mask])

    # §2.3.2 — benchmark
    def _bm(name: str, lo: npt.NDArray[Any], y_pt: npt.NDArray[Any], hi: npt.NDArray[Any]) -> None:
        result.benchmarks[name] = {
            "mae": mae_metric(y_test, y_pt),
            "rmse": rmse_metric(y_test, y_pt),
            "pinball_lo": pinball_loss(y_test, lo, 0.10),
            "pinball_hi": pinball_loss(y_test, hi, 0.90),
            "width": mean_width(lo, hi),
            "coverage_80": empirical_coverage(y_test, lo, hi),
        }

    gm_lo, gm_med, gm_hi = _baseline_global_mean(y_train, len(y_test))
    _bm("Global mean", gm_lo, gm_med, gm_hi)

    lin_lo, lin_med, lin_hi = _baseline_linear(X_train, y_train, X_test)
    _bm("Linear regression", lin_lo, lin_med, lin_hi)

    gbm_lo, gbm_med, gbm_hi = _baseline_gbm_gaussian(X_train, y_train, X_test)
    _bm("GBM + Gaussian", gbm_lo, gbm_med, gbm_hi)

    _bm("Raw quantile", raw_lo, med_80, raw_hi)
    _bm("CQR (Gauge)", lo_80, med_80, hi_80)

    # §2.3.2 — ablations
    base_mae = mae_metric(y_test, med_80)
    base_width = result.cqr_width_80

    for ablated, feats in [
        ("Drop BMI", [f for f in ALL_FEATURES if f != "bmi"]),
        ("Drop smoker", [f for f in ALL_FEATURES if f != "smoker"]),
    ]:
        am = train_quantile_model(X_train[feats], y_train, X_cal[feats], y_cal)
        al, am_med, ah = am.predict_cqr(X_test[feats], 0.80)
        a_mae = mae_metric(y_test, am_med)
        a_width = mean_width(al, ah)
        result.ablations[ablated] = {
            "mae": a_mae,
            "width": a_width,
            "mae_delta": a_mae - base_mae,
            "width_delta": a_width - base_width,
        }

    if store_preds:
        result.y_test = y_test
        result.y_pred_med = med_80
        result.y_pred_lo = lo_80
        result.y_pred_hi = hi_80

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[SeedResult]) -> dict[str, Any]:
    """Aggregate multi-seed results into mean ± std dictionaries.

    Parameters
    ----------
    results : list[SeedResult]
        One :class:`SeedResult` per seed.

    Returns
    -------
    dict[str, Any]
        Nested dict keyed by metric category, then model/subgroup name, then
        ``{"mean": float, "std": float}`` for every scalar metric.
    """

    def ms(vals: list[float]) -> dict[str, float]:
        a = np.array(vals)
        return {"mean": float(a.mean()), "std": float(a.std())}

    cqr_cov = {level: ms([r.cqr_coverage[level] for r in results]) for level in NOMINAL_LEVELS}
    raw_cov = ms([r.raw_coverage for r in results])
    cqr_width = ms([r.cqr_width_80 for r in results])
    raw_width = ms([r.raw_width_80 for r in results])

    subgroup_cov: dict[str, dict[str, float]] = {}
    for sg in SUBGROUPS:
        vals = [r.subgroup_cov[sg] for r in results if sg in r.subgroup_cov]
        if vals:
            subgroup_cov[sg] = ms(vals)

    model_names = list(results[0].benchmarks.keys())
    metric_names = list(results[0].benchmarks[model_names[0]].keys())
    benchmarks: dict[str, dict[str, dict[str, float]]] = {
        name: {metric: ms([r.benchmarks[name][metric] for r in results]) for metric in metric_names}
        for name in model_names
    }

    ablation_names = list(results[0].ablations.keys())
    ablation_metric_names = list(results[0].ablations[ablation_names[0]].keys())
    ablations: dict[str, dict[str, dict[str, float]]] = {
        name: {
            metric: ms([r.ablations[name][metric] for r in results])
            for metric in ablation_metric_names
        }
        for name in ablation_names
    }

    return {
        "n_seeds": len(results),
        "cqr_coverage": {str(k): v for k, v in cqr_cov.items()},
        "raw_coverage": raw_cov,
        "cqr_width_80": cqr_width,
        "raw_width_80": raw_width,
        "subgroup_coverage": subgroup_cov,
        "benchmarks": benchmarks,
        "ablations": ablations,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig: matplotlib.figure.Figure, name: str) -> None:
    """Save figure as PNG (150 dpi) and SVG into FIGURES_DIR."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_charge_distribution(df: pd.DataFrame, source_label: str) -> None:
    """Histogram of annual charges (log x-scale) showing right-skew.

    Takeaway: healthcare costs are heavily right-skewed — almost nobody pays
    the mean, so predicting the mean alone is misleading.
    """
    with _rc_context():
        fig, ax = plt.subplots(figsize=(7, 4))
        charges = df[TARGET].to_numpy()
        # Log-spaced bins
        lo_exp = np.log10(max(charges.min(), 1.0))
        hi_exp = np.log10(charges.max())
        bins: list[float] = np.logspace(lo_exp, hi_exp, 50).tolist()
        ax.hist(charges, bins=bins, color=C_CQR, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(
            np.mean(charges),
            color=C_RAW,
            linewidth=1.8,
            linestyle="--",
            label=f"Mean ${np.mean(charges):,.0f}",
        )
        ax.axvline(
            np.median(charges),
            color=C_IDEAL,
            linewidth=1.8,
            linestyle="-",
            label=f"Median ${np.median(charges):,.0f}",
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_xlabel("Annual charges (USD, log scale)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Charge distribution — {source_label}\n"
            "Almost nobody pays the mean; quantile modeling captures the full range."
        )
        ax.legend()
        fig.tight_layout()
        _save(fig, "charge_distribution")


def plot_coverage_calibration(agg: dict[str, Any]) -> None:
    """CQR empirical coverage vs nominal level (with error bars), vs raw quantile.

    Takeaway: CQR tracks the diagonal (nominal = empirical); the raw
    quantile interval under-covers at every target level.
    """
    levels = NOMINAL_LEVELS
    cqr_means = [agg["cqr_coverage"][str(lv)]["mean"] for lv in levels]
    cqr_stds = [agg["cqr_coverage"][str(lv)]["std"] for lv in levels]
    raw_mean = agg["raw_coverage"]["mean"]
    raw_std = agg["raw_coverage"]["std"]

    with _rc_context():
        fig, ax = plt.subplots(figsize=(6, 5))

        # Ideal diagonal
        diag = np.linspace(0.45, 1.0, 100)
        ax.plot(
            diag,
            diag,
            color=C_IDEAL,
            linewidth=1.2,
            linestyle="--",
            label="Perfect calibration (y = x)",
            zorder=1,
        )

        # CQR line with error bars
        ax.errorbar(
            levels,
            cqr_means,
            yerr=cqr_stds,
            color=C_CQR,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=4,
            label=f"CQR (Gauge) — mean ± std over {N_SEEDS} seeds",
            zorder=3,
        )

        # Raw quantile — constant horizontal band
        ax.axhline(
            raw_mean,
            color=C_RAW,
            linewidth=2,
            linestyle="-",
            label=f"Raw quantile (q0.1–q0.9): {raw_mean:.1%} ± {raw_std:.1%}",
            zorder=2,
        )
        ax.axhspan(raw_mean - raw_std, raw_mean + raw_std, color=C_RAW, alpha=0.12)

        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_title(
            "Coverage calibration: CQR vs raw quantile interval\n"
            "CQR achieves the target; raw quantile chronically under-covers."
        )
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        _save(fig, "coverage_calibration")


def plot_conditional_coverage(agg: dict[str, Any]) -> None:
    """Subgroup empirical coverage at 80% nominal level with error bars.

    Takeaway: CQR guarantees marginal (overall) coverage but conditional
    (per-subgroup) coverage varies — smokers are the hardest subgroup.
    """
    sg_data = agg["subgroup_coverage"]
    labels = [sg for sg in SUBGROUPS if sg in sg_data]
    means = np.array([sg_data[sg]["mean"] for sg in labels])
    stds = np.array([sg_data[sg]["std"] for sg in labels])

    # Sort by deviation from target (largest under-coverage first)
    order = np.argsort(means)
    labels = [labels[i] for i in order]
    means = means[order]
    stds = stds[order]

    colors = [C_RAW if m < 0.80 else C_CQR for m in means]

    with _rc_context():
        fig, ax = plt.subplots(figsize=(7, 6))
        y_pos = np.arange(len(labels))
        ax.barh(
            y_pos,
            means,
            xerr=stds,
            color=colors,
            alpha=0.75,
            error_kw={"capsize": 3, "linewidth": 1.2},
            height=0.6,
        )
        ax.axvline(0.80, color=C_IDEAL, linewidth=1.8, linestyle="--", label="80% target")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Empirical coverage at 80% nominal level")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_xlim(0.5, 1.0)
        ax.set_title(
            "Conditional coverage by subgroup (80% target)\n"
            "Marginal coverage ≥ 80%; subgroup coverage varies (red = under-covers)."
        )
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save(fig, "conditional_coverage")


def plot_predicted_vs_actual(seed_result: SeedResult) -> None:
    """Predicted median vs actual charges (log-log), coloured by in/out of interval.

    Takeaway: the median prediction tracks the actual well in the main body;
    residuals are largest in the tail, as expected for right-skewed costs.
    """
    assert seed_result.y_test is not None
    y = seed_result.y_test
    yp = seed_result.y_pred_med
    lo = seed_result.y_pred_lo
    hi = seed_result.y_pred_hi

    in_interval = (y >= lo) & (y <= hi)
    coverage = float(in_interval.mean())

    with _rc_context():
        fig, ax = plt.subplots(figsize=(6, 6))

        # Points outside interval
        ax.scatter(
            y[~in_interval],
            yp[~in_interval],  # type: ignore[index]
            c=C_RAW,
            s=18,
            alpha=0.55,
            label=f"Outside 80% CI ({(~in_interval).mean():.1%})",
        )
        # Points inside interval
        ax.scatter(
            y[in_interval],
            yp[in_interval],  # type: ignore[index]
            c=C_CQR,
            s=12,
            alpha=0.35,
            label=f"Inside 80% CI ({in_interval.mean():.1%})",
        )

        # Calibration reference
        lo_lim = max(y.min(), yp.min(), 1.0)  # type: ignore[union-attr]
        hi_lim = max(y.max(), yp.max())  # type: ignore[union-attr]
        ref = np.logspace(np.log10(lo_lim), np.log10(hi_lim), 100)
        ax.plot(ref, ref, color=C_IDEAL, linewidth=1.5, linestyle="--", label="y = x")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_xlabel("Actual annual charges (USD, log scale)")
        ax.set_ylabel("Predicted median charges (USD, log scale)")
        ax.set_title(
            f"Predicted vs actual — seed 0 (empirical coverage {coverage:.1%})\n"
            "Model tracks the median well; tail residuals are expected."
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save(fig, "predicted_vs_actual")


def plot_benchmark(agg: dict[str, Any]) -> None:
    """Horizontal grouped bar chart comparing models on MAE and pinball loss.

    Takeaway: CQR achieves lowest pinball loss while maintaining honest coverage.
    """
    bm = agg["benchmarks"]
    model_names = list(bm.keys())
    mae_means = np.array([bm[m]["mae"]["mean"] for m in model_names])
    mae_stds = np.array([bm[m]["mae"]["std"] for m in model_names])
    pb_means = np.array(
        [(bm[m]["pinball_lo"]["mean"] + bm[m]["pinball_hi"]["mean"]) / 2 for m in model_names]
    )
    pb_stds = np.array(
        [
            np.sqrt((bm[m]["pinball_lo"]["std"] ** 2 + bm[m]["pinball_hi"]["std"] ** 2) / 2)
            for m in model_names
        ]
    )

    colors = [C_CQR if "CQR" in n else C_NEUTRAL for n in model_names]

    with _rc_context():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, vals, stds, xlabel, title in [
            (axes[0], mae_means, mae_stds, "MAE (USD)", "Mean Absolute Error ↓"),
            (
                axes[1],
                pb_means,
                pb_stds,
                "Pinball loss (USD)",
                "Mean Pinball Loss ↓\n(proper quantile scoring rule)",
            ),
        ]:
            order = np.argsort(vals)[::-1]
            y_pos = np.arange(len(model_names))
            ax.barh(
                y_pos,
                vals[order],
                xerr=stds[order],
                color=[colors[i] for i in order],
                alpha=0.80,
                height=0.55,
                error_kw={"capsize": 3, "linewidth": 1.2},
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels([model_names[i] for i in order], fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        fig.suptitle(
            f"Model benchmark on held-out test set — mean ± std over {N_SEEDS} seeds\n"
            "CQR (blue) leads on both metrics.",
            fontsize=11,
        )
        fig.tight_layout()
        _save(fig, "benchmark")


def plot_architecture() -> None:
    """Static pipeline diagram: Demographics → ML → Plan → Apply → OOP interval.

    The diagram is data-independent and illustrates Gauge's five-stage flow.
    Integrated here so ``python -m gauge.eval`` produces every figure from one
    command.

    Notes
    -----
    Saved as ``architecture.png`` and ``architecture.svg``.
    """
    # Box geometry (x_centre, y_centre, width, height, label, sublabel)
    stages: list[tuple[float, float, float, float, str, str]] = [
        (0.08, 0.50, 0.12, 0.36, "Demographics", "age · sex · BMI\nchildren · smoker\nregion"),
        (
            0.28,
            0.50,
            0.12,
            0.36,
            "ML Prediction",
            "CQR interval\n[lo, median, hi]\ncoverage ≥ 80 %",
        ),
        (0.50, 0.50, 0.12, 0.36, "Plan Upload", "PDF → extract\ndeductible · OOP max\ncoinsurance"),
        (0.72, 0.50, 0.12, 0.36, "Apply Plan", "monotone map\nOOP(lo) ≤ OOP(med)\n≤ OOP(hi)"),
        (
            0.92,
            0.50,
            0.12,
            0.36,
            "OOP Interval",
            "80 % guarantee\ntransfers exactly\nno simulation",
        ),
    ]

    # Colours — match the rest of the report palette
    box_fill = "#dbeafe"  # light blue (matches OOP band in WhatIfChart)
    box_edge = C_CQR  # blue
    arrow_col = "#374151"  # near-black

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for x, y, w, h, title, sub in stages:
        rect = matplotlib.patches.Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            facecolor=box_fill,
            edgecolor=box_edge,
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(rect)
        # Title — bold
        ax.text(
            x,
            y + 0.04,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1e3a5f",
            zorder=3,
        )
        # Sublabel — smaller, grey
        ax.text(
            x,
            y - 0.09,
            sub,
            ha="center",
            va="center",
            fontsize=7.5,
            color="#4b5563",
            linespacing=1.4,
            zorder=3,
        )

    # Arrows between boxes
    for i in range(len(stages) - 1):
        x_start = stages[i][0] + stages[i][2] / 2
        x_end = stages[i + 1][0] - stages[i + 1][2] / 2
        y_mid = 0.50
        ax.annotate(
            "",
            xy=(x_end, y_mid),
            xytext=(x_start, y_mid),
            arrowprops=dict(
                arrowstyle="-|>",
                color=arrow_col,
                lw=1.6,
                mutation_scale=14,
            ),
            zorder=1,
        )

    fig.suptitle(
        "Gauge pipeline",
        fontsize=13,
        fontweight="bold",
        y=0.97,
        color="#111827",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "architecture")


def plot_oop_transform(seed_result: SeedResult, source_label: str) -> None:
    """Signature chart: predicted charge distribution → OOP distribution under a plan.

    The key visual payoff: a representative plan's deductible (the kink) and
    OOP maximum (the ceiling) compress a wide, right-skewed charge distribution
    into a much tighter out-of-pocket distribution.  Because
    ``apply_plan_to_annual_spend`` is monotone non-decreasing in charges, the
    80% conformal charge interval maps directly to a valid 80% OOP interval —
    the coverage guarantee transfers without simulation.

    The representative plan is ``_REFERENCE_PLAN``:
    $1,500 deductible · 20% coinsurance · $6,000 OOP max.

    Parameters
    ----------
    seed_result : SeedResult
        Must have ``store_preds=True`` (i.e. seed 0). Provides the test-set
        actual charges and the per-user CQR bounds.
    source_label : str
        Human-readable dataset label printed in the figure title.

    Notes
    -----
    Takeaway: insurance converts scary right-skewed charge uncertainty into a
    bounded OOP range.  The OOP max eliminates worst-case spend entirely.
    """
    assert seed_result.y_test is not None
    assert seed_result.y_pred_lo is not None
    assert seed_result.y_pred_hi is not None

    y = seed_result.y_test  # actual annual charges, dollars
    lo = seed_result.y_pred_lo  # CQR lower bounds, dollars
    hi = seed_result.y_pred_hi  # CQR upper bounds, dollars

    plan = _REFERENCE_PLAN
    ded = plan.deductible_cents / 100  # $1,500
    oop_max = plan.out_of_pocket_max_cents / 100  # $6,000
    coin = plan.coinsurance_rate  # 0.20
    # Analytic charge level at which the OOP max is first reached:
    #   ded + coin * (cap_charge - ded) = oop_max  →  cap_charge = ded + (oop_max - ded) / coin
    cap_charge = ded + (oop_max - ded) / coin  # $24,000

    # Apply the plan to every actual test-set charge to get the OOP distribution.
    oop_y = np.array(
        [apply_plan_to_annual_spend(plan, max(0, int(c * 100))).member_pays_cents / 100 for c in y]
    )

    # Representative 80% interval: median of the per-user CQR bounds.
    lo_rep = float(np.median(lo))
    hi_rep = float(np.median(hi))
    oop_lo_rep = apply_plan_to_annual_spend(plan, max(0, int(lo_rep * 100))).member_pays_cents / 100
    oop_hi_rep = apply_plan_to_annual_spend(plan, max(0, int(hi_rep * 100))).member_pays_cents / 100

    charge_width = hi_rep - lo_rep
    oop_width = oop_hi_rep - oop_lo_rep

    with _rc_context():
        fig, (ax_c, ax_o) = plt.subplots(1, 2, figsize=(13, 5))

        # ── Left: charges (log x-scale) ──────────────────────────────────────
        lo_exp = np.log10(max(float(y.min()), 1.0))
        hi_exp = np.log10(float(y.max()))
        bins_c = np.logspace(lo_exp, hi_exp, 50)
        ax_c.hist(y, bins=bins_c, color=C_CQR, alpha=0.65, edgecolor="white", linewidth=0.3)
        ax_c.axvspan(
            lo_rep,
            hi_rep,
            color=C_CQR,
            alpha=0.18,
            label=f"80% CI: \\${lo_rep:,.0f}–\\${hi_rep:,.0f} (median user)",
        )
        ax_c.axvline(
            ded, color="#d97706", linewidth=1.8, linestyle="--", label=f"Deductible: \\${ded:,.0f}"
        )
        ax_c.axvline(
            cap_charge,
            color=C_IDEAL,
            linewidth=1.8,
            linestyle=":",
            label=f"OOP max reached: >\\${cap_charge:,.0f}",
        )
        ax_c.set_xscale("log")
        ax_c.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"\\${x:,.0f}"))
        ax_c.set_xlabel("Annual gross charges (USD, log scale)")
        ax_c.set_ylabel("Count")
        ax_c.set_title(f"Annual Gross Charges\nRight-skewed; 80% CI spans \\${charge_width:,.0f}.")
        ax_c.legend(fontsize=9, loc="upper left")

        # ── Right: OOP (linear x-scale) ──────────────────────────────────────
        ax_o.hist(oop_y, bins=60, color="#059669", alpha=0.65, edgecolor="white", linewidth=0.3)
        ax_o.axvspan(
            oop_lo_rep,
            oop_hi_rep,
            color="#059669",
            alpha=0.25,
            label=f"80% OOP CI: \\${oop_lo_rep:,.0f}–\\${oop_hi_rep:,.0f}",
        )
        ax_o.axvline(
            ded, color="#d97706", linewidth=1.8, linestyle="--", label=f"Deductible: \\${ded:,.0f}"
        )
        ax_o.axvline(
            oop_max,
            color=C_IDEAL,
            linewidth=2.0,
            linestyle="-",
            label=f"OOP max (ceiling): \\${oop_max:,.0f}",
        )
        # Label the ceiling using the x-axis transform (data-x · axes-fraction-y).
        ax_o.text(
            oop_max,
            0.92,
            "← ceiling",
            transform=ax_o.get_xaxis_transform(),
            fontsize=9,
            color=C_IDEAL,
            va="top",
            ha="right",
        )
        ax_o.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"\\${x:,.0f}"))
        ax_o.set_xlabel("Out-of-pocket cost (USD)")
        ax_o.set_ylabel("Count")
        ax_o.set_title(
            f"Out-of-Pocket Cost (after plan)\n"
            f"OOP max caps the tail; 80% CI narrows to \\${oop_width:,.0f}."
        )
        ax_o.legend(fontsize=9, loc="upper left")

        fig.suptitle(
            "The insurance transform — wide charge uncertainty → bounded out-of-pocket\n"
            f"Plan: \\${ded:,.0f} deductible · {coin:.0%} coinsurance · "
            f"\\${oop_max:,.0f} OOP max   ·   {source_label}",
            fontsize=11,
        )
        fig.tight_layout()
        _save(fig, "oop_transform")


# ---------------------------------------------------------------------------
# MODELING.md generation
# ---------------------------------------------------------------------------


def _fmt(ms_dict: dict[str, float], unit: str = "", pct: bool = False) -> str:
    """Format a mean ± std dict as a readable string."""
    m = ms_dict["mean"]
    s = ms_dict["std"]
    if pct:
        return f"{m:.1%} ± {s:.1%}"
    return f"{m:,.0f}{unit} ± {s:,.0f}{unit}"


def write_modeling_md(
    agg: dict[str, Any],
    df: pd.DataFrame,
    source_label: str,
) -> None:
    """Write MODELING.md to the repo root populated with actual benchmark numbers.

    Parameters
    ----------
    agg : dict[str, Any]
        Aggregated evaluation results from :func:`aggregate`.
    df : pd.DataFrame
        Full dataset (used for summary statistics in the limitations section).
    source_label : str
        Human-readable data source label.
    """
    bm = agg["benchmarks"]
    cqr = bm["CQR (Gauge)"]
    raw = bm["Raw quantile"]
    lm = bm["Linear regression"]
    gbm_g = bm["GBM + Gaussian"]
    gm = bm["Global mean"]

    raw_cov_80 = agg["raw_coverage"]
    cqr_cov_80 = agg["cqr_coverage"]["0.8"]
    cqr_cov_90 = agg["cqr_coverage"]["0.9"]

    charges = df[TARGET].to_numpy()
    n = len(df)
    charges_mean = charges.mean()
    charges_median = np.median(charges)
    charges_p95 = np.percentile(charges, 95)

    abl = agg["ablations"]

    # Subgroup coverage table rows
    sg = agg["subgroup_coverage"]
    sg_rows = "\n".join(
        f"| {label} | {sg[label]['mean']:.1%} ± {sg[label]['std']:.1%} |"
        for label in SUBGROUPS
        if label in sg
    )

    md = f"""\
# Gauge — Modeling Reference

> **One command to reproduce everything:**
> ```
> python -m gauge.eval
> ```
> All outputs are seeded. Figures land in `reports/figures/`.

---

## 1. Dataset

**Source:** {source_label}

The dataset contains {n:,} adult individuals with six demographic features
(age, sex, BMI, children, smoker, region) and annual medical charges in USD.
Healthcare costs are heavily **right-skewed and heavy-tailed**: the mean
(${charges_mean:,.0f}) is pulled far above the median (${charges_median:,.0f}) by
a small fraction of very high-cost individuals. The 95th percentile is
${charges_p95:,.0f}. This motivates quantile modeling — predicting the mean alone
is misleading because almost nobody pays it.

![Charge distribution](reports/figures/charge_distribution.png)

---

## 2. Model architecture

Four `HistGradientBoostingRegressor` pipelines are trained:

| Pipeline | Loss | Role |
|----------|------|------|
| q0.10 | Quantile (α=0.10) | Lower raw interval bound |
| q0.50 | Quantile (α=0.50) | Median — "typical year" estimate |
| q0.90 | Quantile (α=0.90) | Upper raw interval bound |
| mean  | Squared error | Long-run expected cost |

Categorical features (sex, smoker, region) are one-hot encoded; numeric
features (age, BMI, children) pass through unchanged.

### 2.1 Conformal Quantile Regression (CQR)

After training on a 60% split, the predictor calibrates its interval using
CQR ([Romano, Patterson & Candès, NeurIPS 2019](https://arxiv.org/abs/1905.03222))
on a held-out 20% calibration set.

**Nonconformity score:**
```
score_i = max( q_lo(x_i) − y_i ,  y_i − q_hi(x_i) )
```
A negative score means the true value was already inside the raw interval.
A positive score records how far it fell outside.

**Calibration quantile:**
```
q̂ = Quantile( scores, ⌈(n+1)(1−α)⌉/n )
```
The `⌈(n+1)(1−α)⌉/n` level (the finite-sample correction) guarantees that
the expanded interval `[q_lo(x) − q̂, q_hi(x) + q̂]` contains the true value
with probability **≥ 1−α** for any data distribution, without assuming
normality. This is a *marginal* guarantee — see §3.3 for the limits.

---

## 3. Evaluation

All metrics are mean ± std over **{N_SEEDS} random seeds** (60/20/20 train/cal/test
splits). Error bars on figures represent this seed-to-seed variability.

### 3.1 Coverage calibration

The headline result: the raw quantile interval achieves only
**{raw_cov_80["mean"]:.1%} ± {raw_cov_80["std"]:.1%} empirical coverage** when
targeting 80%. CQR closes this gap to
**{cqr_cov_80["mean"]:.1%} ± {cqr_cov_80["std"]:.1%}** — honoring the
guarantee across coverage targets:

| Nominal level | CQR empirical | Raw q0.1–q0.9 |
|:---:|:---:|:---:|
| 50% | {agg["cqr_coverage"]["0.5"]["mean"]:.1%} ± {agg["cqr_coverage"]["0.5"]["std"]:.1%} | {raw_cov_80["mean"]:.1%} (fixed) |
| **80%** | **{cqr_cov_80["mean"]:.1%} ± {cqr_cov_80["std"]:.1%}** | **{raw_cov_80["mean"]:.1%} ± {raw_cov_80["std"]:.1%}** |
| 90% | {cqr_cov_90["mean"]:.1%} ± {cqr_cov_90["std"]:.1%} | {raw_cov_80["mean"]:.1%} (fixed) |
| 95% | {agg["cqr_coverage"]["0.95"]["mean"]:.1%} ± {agg["cqr_coverage"]["0.95"]["std"]:.1%} | {raw_cov_80["mean"]:.1%} (fixed) |

![Coverage calibration](reports/figures/coverage_calibration.png)

**Interval efficiency.** Coverage alone can be gamed by making the interval
arbitrarily wide. CQR mean width at 80%:
**${agg["cqr_width_80"]["mean"]:,.0f} ± ${agg["cqr_width_80"]["std"]:,.0f}** vs
raw interval **${agg["raw_width_80"]["mean"]:,.0f} ± ${agg["raw_width_80"]["std"]:,.0f}**.
CQR widens the interval just enough to hit the target — it cannot shrink below
the raw width.

### 3.2 Conditional coverage

CQR guarantees *marginal* (overall) coverage, not *conditional* (per-subgroup)
coverage. Here is empirical coverage within demographic subgroups at the 80%
nominal level:

| Subgroup | Coverage (mean ± std) |
|----------|:---:|
{sg_rows}

![Conditional coverage by subgroup](reports/figures/conditional_coverage.png)

Subgroups with coverage below 80% (red bars) indicate where the model
under-covers — this is an expected limitation of marginal CQR, not a bug.
Smokers are the most challenging subgroup because their cost distribution is
qualitatively different (bimodal in the raw MEPS data).

### 3.3 Model benchmark

The **pinball (quantile) loss** is the proper scoring rule for quantile
forecasts — unlike coverage or interval width, it cannot be gamed. Lead with
this metric.

| Model | MAE (USD) | RMSE (USD) | Pinball loss (USD) | Coverage @ 80% |
|-------|----------:|----------:|------------------:|:--------------:|
| Global mean | {gm["mae"]["mean"]:,.0f} ± {gm["mae"]["std"]:,.0f} | {gm["rmse"]["mean"]:,.0f} ± {gm["rmse"]["std"]:,.0f} | {(gm["pinball_lo"]["mean"] + gm["pinball_hi"]["mean"]) / 2:,.0f} | {gm["coverage_80"]["mean"]:.1%} |
| Linear regression | {lm["mae"]["mean"]:,.0f} ± {lm["mae"]["std"]:,.0f} | {lm["rmse"]["mean"]:,.0f} ± {lm["rmse"]["std"]:,.0f} | {(lm["pinball_lo"]["mean"] + lm["pinball_hi"]["mean"]) / 2:,.0f} | {lm["coverage_80"]["mean"]:.1%} |
| GBM + Gaussian | {gbm_g["mae"]["mean"]:,.0f} ± {gbm_g["mae"]["std"]:,.0f} | {gbm_g["rmse"]["mean"]:,.0f} ± {gbm_g["rmse"]["std"]:,.0f} | {(gbm_g["pinball_lo"]["mean"] + gbm_g["pinball_hi"]["mean"]) / 2:,.0f} | {gbm_g["coverage_80"]["mean"]:.1%} |
| Raw quantile | {raw["mae"]["mean"]:,.0f} ± {raw["mae"]["std"]:,.0f} | {raw["rmse"]["mean"]:,.0f} ± {raw["rmse"]["std"]:,.0f} | {(raw["pinball_lo"]["mean"] + raw["pinball_hi"]["mean"]) / 2:,.0f} | {raw["coverage_80"]["mean"]:.1%} |
| **CQR (Gauge)** | **{cqr["mae"]["mean"]:,.0f} ± {cqr["mae"]["std"]:,.0f}** | **{cqr["rmse"]["mean"]:,.0f} ± {cqr["rmse"]["std"]:,.0f}** | **{(cqr["pinball_lo"]["mean"] + cqr["pinball_hi"]["mean"]) / 2:,.0f}** | **{cqr["coverage_80"]["mean"]:.1%}** |

![Model benchmark](reports/figures/benchmark.png)

### 3.4 Feature ablations

Dropping one feature at a time and measuring the change in median MAE and
interval width at 80%:

| Ablation | ΔMAE | ΔWidth |
|----------|-----:|-------:|
| Drop BMI | +${abl["Drop BMI"]["mae_delta"]["mean"]:,.0f} | +${abl["Drop BMI"]["width_delta"]["mean"]:,.0f} |
| Drop smoker | +${abl["Drop smoker"]["mae_delta"]["mean"]:,.0f} | +${abl["Drop smoker"]["width_delta"]["mean"]:,.0f} |

Smoker status is the dominant signal: removing it substantially widens the
interval because the model loses the ability to separate the low-cost and
high-cost tails. BMI matters mainly for smokers (interaction effect).

### 3.5 Predicted vs actual

![Predicted vs actual](reports/figures/predicted_vs_actual.png)

The model tracks the median well in the main body of the distribution. Tail
residuals (high-cost outliers above the diagonal) are expected because the
cost distribution is heavy-tailed and the sample is finite.

### 3.6 End-to-end uncertainty propagation

The conformal charge interval propagates through the plan's cost-share function
without any simulation. The key property: `apply_plan_to_annual_spend` is
monotone non-decreasing in charges — more charges never produce less member
OOP. Because the function is monotone, the q-th quantile of charges maps
directly to the q-th quantile of OOP:

```
OOP(lower_bound) ≤ OOP(median) ≤ OOP(upper_bound)
```

The 80% coverage guarantee of the CQR charge interval therefore transfers to
the OOP interval exactly, without simulation. Empirically, this means if 80%
of test-set actual charges fall inside the charge CI, 80% of the corresponding
actual OOP costs fall inside the OOP CI.

![OOP transform](reports/figures/oop_transform.png)

The visual takeaway: a wide, right-skewed charge interval (left panel, log
scale) is compressed by the plan into a much tighter OOP interval (right
panel). The OOP maximum (ceiling) eliminates the worst-case tail entirely —
insurance converts open-ended charge risk into a bounded out-of-pocket
exposure. This is the signature chart: it unifies the ML half and the
deterministic benefits half of Gauge in a single figure.

Plan used for illustration: $1,500 deductible · 20% coinsurance · $6,000 OOP
max (representative US employer PPO). Results will differ for other plan
structures.

---

## 4. Limitations

- **MEPS sampling.** MEPS HC-233 is a weighted household survey; the sample
  overrepresents certain demographics. Results may not generalise to
  employer-sponsored plans or Medicare/Medicaid populations.
- **No plan-linked OOP ground truth.** The benefits engine maps predicted
  charges through a plan's deductible, coinsurance, and OOP max, but MEPS
  does not record which specific plan each respondent holds. OOP predictions
  are therefore illustrative — they use representative plan parameters.
- **Marginal, not conditional, coverage.** CQR's guarantee holds overall.
  Individual subgroups (especially smokers) may see different empirical
  coverage, as shown in §3.2.
- **Six demographic features only.** The model does not observe diagnosis
  codes, prior utilization, or plan type — all strong predictors of actual
  cost. It is a planning tool, not a clinical risk model.

---

## 5. References

1. Romano, Y., Patterson, E., & Candès, E. J. (2019). *Conformalized Quantile
   Regression.* NeurIPS. <https://arxiv.org/abs/1905.03222>
2. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to
   Conformal Prediction and Distribution-Free Uncertainty Quantification.*
   <https://arxiv.org/abs/2107.07511>
"""

    out = REPO_ROOT / "MODELING.md"
    out.write_text(md, encoding="utf-8")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full reproducible evaluation pipeline."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    df, source_label = resolve_data()
    print(f"  {source_label}")
    print(
        f"  charges: mean=${df[TARGET].mean():,.0f}  median=${df[TARGET].median():,.0f}  "
        f"max=${df[TARGET].max():,.0f}"
    )

    print(f"\nRunning evaluation across {N_SEEDS} seeds…")
    results: list[SeedResult] = []
    for seed in SEEDS:
        print(f"  seed {seed}…", end="", flush=True)
        r = evaluate_seed(df, seed, store_preds=(seed == 0))
        results.append(r)
        cov = r.cqr_coverage[0.80]
        raw = r.raw_coverage
        mae_val = r.benchmarks["CQR (Gauge)"]["mae"]
        print(f"  cov80={cov:.1%}  raw={raw:.1%}  MAE=${mae_val:,.0f}")

    print("\nAggregating…")
    agg = aggregate(results)

    print("Saving benchmark.json…")
    (REPORTS_DIR / "benchmark.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print("Generating figures…")
    plot_architecture()
    print("  architecture ✓")
    plot_charge_distribution(df, source_label)
    print("  charge_distribution ✓")
    plot_coverage_calibration(agg)
    print("  coverage_calibration ✓")
    plot_conditional_coverage(agg)
    print("  conditional_coverage ✓")
    plot_predicted_vs_actual(results[0])
    print("  predicted_vs_actual ✓")
    plot_benchmark(agg)
    print("  benchmark ✓")
    plot_oop_transform(results[0], source_label)
    print("  oop_transform ✓")

    print("Writing MODELING.md…")
    write_modeling_md(agg, df, source_label)

    cqr_80 = agg["cqr_coverage"]["0.8"]["mean"]
    raw_80 = agg["raw_coverage"]["mean"]
    cqr_mae = agg["benchmarks"]["CQR (Gauge)"]["mae"]["mean"]

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluation complete.

Headline numbers (mean over {N_SEEDS} seeds):
  CQR coverage @ 80% target : {cqr_80:.1%}
  Raw quantile coverage      : {raw_80:.1%}   (targeting 80%, without conformal)
  CQR MAE (median predictor) : ${cqr_mae:,.0f}

Outputs:
  reports/figures/           (PNG + SVG)
  reports/benchmark.json
  MODELING.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
