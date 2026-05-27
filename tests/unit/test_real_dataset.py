"""Tests that exercise the real Kaggle insurance.csv if it is present.

These tests are skipped automatically when `data/insurance.csv` is not
available, so the suite stays green for anyone who has not downloaded
the dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from health_app.predictor.dataset import FEATURE_COLUMNS, TARGET_COLUMN, load_csv
from health_app.predictor.model import CostPredictor
from health_app.predictor.schemas import PredictionFeatures

pytestmark = pytest.mark.unit

_REAL_CSV = (
    Path(__file__).resolve().parents[2] / "data" / "insurance.csv"
)
_HAS_REAL = _REAL_CSV.exists()
_skip_no_real = pytest.mark.skipif(
    not _HAS_REAL,
    reason="data/insurance.csv not present; download from Kaggle to enable.",
)


@_skip_no_real
def test_real_csv_has_expected_schema() -> None:
    df = load_csv(_REAL_CSV)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    # Categorical values must be inside the schema's literal sets so the
    # OneHotEncoder doesn't pick up surprise categories. Regions get
    # remapped from cardinal (Kaggle) to Census (canonical) at load time.
    assert set(df["sex"].unique()) <= {"male", "female"}
    assert set(df["smoker"].unique()) <= {"yes", "no"}
    assert set(df["region"].unique()) <= {
        "northeast",
        "midwest",
        "south",
        "west",
    }
    assert (df["charges"] > 0).all()


@_skip_no_real
def test_real_csv_predictor_directional_sanity() -> None:
    """Train on the real data and check basic monotonicity holds."""
    df = load_csv(_REAL_CSV)
    predictor = CostPredictor().fit(df)

    base = PredictionFeatures(
        age=35,
        sex="female",
        bmi=27.5,
        children=1,
        smoker="no",
        region="northeast",
    )
    smoker = base.model_copy(update={"smoker": "yes"})
    older = base.model_copy(update={"age": 60})

    base_pred = predictor.predict(base).median_charges_cents
    assert predictor.predict(smoker).median_charges_cents > base_pred
    assert predictor.predict(older).median_charges_cents > base_pred
