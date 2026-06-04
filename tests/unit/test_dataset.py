"""Unit tests for the synthetic dataset generator."""

from __future__ import annotations

import pytest

from health_app.predictor.dataset import (
    FEATURE_COLUMNS,
    REGIONS,
    TARGET_COLUMN,
    generate_synthetic_dataset,
)

pytestmark = pytest.mark.unit


def test_dataset_has_expected_columns_and_size() -> None:
    df = generate_synthetic_dataset(n_rows=300, seed=1)
    assert len(df) == 300
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_dataset_is_reproducible_with_same_seed() -> None:
    a = generate_synthetic_dataset(n_rows=100, seed=7)
    b = generate_synthetic_dataset(n_rows=100, seed=7)
    assert a.equals(b)


def test_dataset_values_are_in_valid_ranges() -> None:
    df = generate_synthetic_dataset(n_rows=500, seed=2)
    assert df["age"].between(18, 64).all()
    assert df["bmi"].between(16.0, 53.0).all()
    assert df["children"].between(0, 5).all()
    assert df["smoker"].isin(["yes", "no"]).all()
    assert df["sex"].isin(["male", "female"]).all()
    assert df["region"].isin(REGIONS).all()
    assert (df["charges"] > 0).all()


def test_smokers_have_higher_average_charges() -> None:
    """Sanity check: the data-generating process makes smokers cost more."""
    df = generate_synthetic_dataset(n_rows=2_000, seed=3)
    smoker_mean = df.loc[df["smoker"] == "yes", "charges"].mean()
    nonsmoker_mean = df.loc[df["smoker"] == "no", "charges"].mean()
    assert smoker_mean > nonsmoker_mean * 2


def test_load_csv_raises_on_missing_columns(tmp_path) -> None:
    import pandas as pd

    from health_app.predictor.dataset import load_csv

    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"age": [30], "bmi": [25.0]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_csv(bad_csv)


def test_load_dataset_with_csv_path_returns_dataframe(tmp_path) -> None:
    """load_dataset delegates to load_csv when csv_path is provided."""
    import pandas as pd

    from health_app.predictor.dataset import (
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        load_dataset,
    )

    # Build a minimal valid CSV using the real insurance.csv column names
    # (cardinal directions; the loader remaps them).
    rows = [
        {
            "age": 30, "sex": "male", "bmi": 25.0,
            "children": 0, "smoker": "no", "region": "northwest",
            "charges": 4000.0,
        }
    ]
    csv_path = tmp_path / "insurance.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = load_dataset(csv_path=csv_path)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert df.iloc[0]["region"] == "midwest"  # northwest -> midwest via KAGGLE_REGION_MAP
