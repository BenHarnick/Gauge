"""Unit tests for the MEPS HC loader.

We don't ship the real MEPS file, so the tests synthesize a tiny
MEPS-shaped Stata file in-memory and verify the loader handles the
quirks (negative missing-value codes, adult-only filter, family-size
to children count, region/sex/smoker code mapping).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from health_app.predictor.meps import (
    MEPS_AGE,
    MEPS_BMI,
    MEPS_FAMID,
    MEPS_REGION,
    MEPS_SEX,
    MEPS_SMOKER,
    MEPS_TOTEXP,
    load_meps,
)

pytestmark = pytest.mark.unit


def _write_meps_fixture(rows: list[dict], path) -> None:
    """Write a MEPS-shaped DataFrame to disk as a .dta file."""
    df = pd.DataFrame(rows)
    # Stata requires explicit dtypes; cast numeric columns to int/float.
    df = df.astype(
        {
            MEPS_AGE: np.int32,
            MEPS_SEX: np.int32,
            MEPS_REGION: np.int32,
            MEPS_BMI: np.float64,
            MEPS_SMOKER: np.int32,
            MEPS_FAMID: "string",
            MEPS_TOTEXP: np.float64,
        }
    )
    df.to_stata(path, write_index=False)


def test_load_meps_maps_codes_to_schema(tmp_path) -> None:
    rows = [
        # Two adults in one family with one kid; one adult in another family.
        {
            MEPS_AGE: 40, MEPS_SEX: 1, MEPS_REGION: 1,
            MEPS_BMI: 27.5, MEPS_SMOKER: 2, MEPS_FAMID: "f1",
            MEPS_TOTEXP: 5400.0,
        },
        {
            MEPS_AGE: 38, MEPS_SEX: 2, MEPS_REGION: 1,
            MEPS_BMI: 30.0, MEPS_SMOKER: 1, MEPS_FAMID: "f1",
            MEPS_TOTEXP: 12000.0,
        },
        {
            MEPS_AGE: 8, MEPS_SEX: 1, MEPS_REGION: 1,
            MEPS_BMI: -1.0, MEPS_SMOKER: -1, MEPS_FAMID: "f1",
            MEPS_TOTEXP: 1200.0,
        },
        {
            MEPS_AGE: 55, MEPS_SEX: 1, MEPS_REGION: 4,
            MEPS_BMI: 32.0, MEPS_SMOKER: 2, MEPS_FAMID: "f2",
            MEPS_TOTEXP: 9800.0,
        },
    ]
    path = tmp_path / "meps.dta"
    _write_meps_fixture(rows, path)

    df = load_meps(path)
    assert list(df.columns) == [
        "age", "sex", "bmi", "children", "smoker", "region", "charges",
    ]
    assert len(df) == 3  # the child row was filtered out
    assert set(df["sex"].unique()) <= {"male", "female"}
    assert set(df["region"].unique()) <= {
        "northeast", "midwest", "south", "west",
    }
    assert set(df["smoker"].unique()) <= {"yes", "no"}
    # Adults in family f1 should have children=1 (one kid in their family).
    f1 = df[df["age"].isin([38, 40])]
    assert (f1["children"] == 1).all()
    # Adult in family f2 has no kids.
    assert (df.loc[df["age"] == 55, "children"] == 0).all()


def test_load_meps_drops_rows_with_missing_required(tmp_path) -> None:
    rows = [
        # Valid adult.
        {
            MEPS_AGE: 30, MEPS_SEX: 2, MEPS_REGION: 2,
            MEPS_BMI: 25.0, MEPS_SMOKER: 2, MEPS_FAMID: "a",
            MEPS_TOTEXP: 2000.0,
        },
        # Adult with missing BMI: should be dropped.
        {
            MEPS_AGE: 45, MEPS_SEX: 1, MEPS_REGION: 2,
            MEPS_BMI: -7.0, MEPS_SMOKER: 2, MEPS_FAMID: "b",
            MEPS_TOTEXP: 3000.0,
        },
        # Adult with missing smoker: should be dropped.
        {
            MEPS_AGE: 50, MEPS_SEX: 1, MEPS_REGION: 3,
            MEPS_BMI: 27.0, MEPS_SMOKER: -9, MEPS_FAMID: "c",
            MEPS_TOTEXP: 4500.0,
        },
    ]
    path = tmp_path / "meps.dta"
    _write_meps_fixture(rows, path)

    df = load_meps(path)
    assert len(df) == 1
    assert df.iloc[0]["age"] == 30


def test_load_meps_clamps_negative_charges_to_zero(tmp_path) -> None:
    """Defensive: a negative TOTEXP value (shouldn't happen in MEPS) is clamped."""
    rows = [
        {
            MEPS_AGE: 35, MEPS_SEX: 1, MEPS_REGION: 1,
            MEPS_BMI: 26.0, MEPS_SMOKER: 2, MEPS_FAMID: "x",
            MEPS_TOTEXP: -100.0,
        },
    ]
    path = tmp_path / "meps.dta"
    _write_meps_fixture(rows, path)

    df = load_meps(path)
    assert len(df) == 1
    assert df.iloc[0]["charges"] == 0.0


def test_load_meps_errors_on_missing_columns(tmp_path) -> None:
    bad = pd.DataFrame({"foo": [1, 2, 3]})
    path = tmp_path / "bad.dta"
    bad.to_stata(path, write_index=False)
    with pytest.raises(ValueError):
        load_meps(path)


def test_missing_bmi_without_saq_gives_actionable_error(tmp_path) -> None:
    """When BMI is absent and no SAQ is supplied, the error names the fix."""
    # Build a MEPS-like file with everything EXCEPT a BMI column.
    rows = pd.DataFrame(
        [
            {
                "DUPERSID": "p1",
                MEPS_AGE: 35,
                MEPS_SEX: 1,
                MEPS_REGION: 1,
                MEPS_SMOKER: 2,
                MEPS_FAMID: "f1",
                MEPS_TOTEXP: 4000.0,
            }
        ]
    ).astype(
        {
            MEPS_AGE: np.int32,
            MEPS_SEX: np.int32,
            MEPS_REGION: np.int32,
            MEPS_SMOKER: np.int32,
            MEPS_FAMID: "string",
            MEPS_TOTEXP: np.float64,
        }
    )
    path = tmp_path / "meps_no_bmi.dta"
    rows.to_stata(path, write_index=False)
    with pytest.raises(ValueError, match="SAQ Supplement"):
        load_meps(path)


def test_load_meps_bad_path_raises_value_error(tmp_path) -> None:
    """A non-existent or unparseable file raises ValueError with context."""
    bad_path = tmp_path / "nonexistent.dta"
    with pytest.raises(ValueError, match="Could not read MEPS Stata file"):
        load_meps(bad_path)


def test_load_meps_bad_saq_path_raises_value_error(tmp_path) -> None:
    """A bad SAQ path raises a ValueError that names the SAQ file."""
    # Build a valid main file (no BMI column so we need SAQ).
    main_rows = pd.DataFrame(
        [
            {
                "DUPERSID": "p1",
                MEPS_AGE: 40,
                MEPS_SEX: 1,
                MEPS_REGION: 1,
                MEPS_SMOKER: 2,
                MEPS_FAMID: "f1",
                MEPS_TOTEXP: 5000.0,
            }
        ]
    ).astype(
        {
            MEPS_AGE: np.int32,
            MEPS_SEX: np.int32,
            MEPS_REGION: np.int32,
            MEPS_SMOKER: np.int32,
            MEPS_FAMID: "string",
            MEPS_TOTEXP: np.float64,
        }
    )
    main_path = tmp_path / "main.dta"
    main_rows.to_stata(main_path, write_index=False)

    bad_saq = tmp_path / "missing_saq.dta"
    with pytest.raises(ValueError, match="Could not read MEPS SAQ Stata file"):
        load_meps(main_path, saq_path=bad_saq)


def test_load_meps_saq_missing_dupersid_raises_value_error(tmp_path) -> None:
    """If either file lacks DUPERSID the merge raises a clear ValueError."""
    main_rows = pd.DataFrame(
        [
            {
                "DUPERSID": "p1",
                MEPS_AGE: 40,
                MEPS_SEX: 1,
                MEPS_REGION: 1,
                MEPS_SMOKER: 2,
                MEPS_FAMID: "f1",
                MEPS_TOTEXP: 5000.0,
            }
        ]
    ).astype(
        {
            MEPS_AGE: np.int32,
            MEPS_SEX: np.int32,
            MEPS_REGION: np.int32,
            MEPS_SMOKER: np.int32,
            MEPS_FAMID: "string",
            MEPS_TOTEXP: np.float64,
        }
    )
    # SAQ without DUPERSID.
    saq_rows = pd.DataFrame([{"ADBMI42": 27.5}]).astype({"ADBMI42": np.float64})

    main_path = tmp_path / "main.dta"
    saq_path = tmp_path / "saq_no_id.dta"
    main_rows.to_stata(main_path, write_index=False)
    saq_rows.to_stata(saq_path, write_index=False)

    with pytest.raises(ValueError, match="DUPERSID"):
        load_meps(main_path, saq_path=saq_path)


def test_load_meps_saq_with_no_bmi_columns_raises_value_error(tmp_path) -> None:
    """SAQ that contains no recognised BMI column raises a clear ValueError."""
    main_rows = pd.DataFrame(
        [
            {
                "DUPERSID": "p1",
                MEPS_AGE: 40,
                MEPS_SEX: 1,
                MEPS_REGION: 1,
                MEPS_SMOKER: 2,
                MEPS_FAMID: "f1",
                MEPS_TOTEXP: 5000.0,
            }
        ]
    ).astype(
        {
            MEPS_AGE: np.int32,
            MEPS_SEX: np.int32,
            MEPS_REGION: np.int32,
            MEPS_SMOKER: np.int32,
            MEPS_FAMID: "string",
            MEPS_TOTEXP: np.float64,
        }
    )
    # SAQ has DUPERSID but no BMI column.
    saq_rows = pd.DataFrame(
        [{"DUPERSID": "p1", "UNRELATED_COL": 42.0}]
    ).astype({"UNRELATED_COL": np.float64})

    main_path = tmp_path / "main.dta"
    saq_path = tmp_path / "saq_no_bmi.dta"
    main_rows.to_stata(main_path, write_index=False)
    saq_rows.to_stata(saq_path, write_index=False)

    with pytest.raises(ValueError, match="no known BMI column"):
        load_meps(main_path, saq_path=saq_path)


def test_saq_merge_supplies_bmi(tmp_path) -> None:
    """An SAQ file with BMI gets merged onto the main file by DUPERSID."""
    main_rows = pd.DataFrame(
        [
            {
                "DUPERSID": "p1",
                MEPS_AGE: 40,
                MEPS_SEX: 1,
                MEPS_REGION: 1,
                MEPS_SMOKER: 2,
                MEPS_FAMID: "f1",
                MEPS_TOTEXP: 5500.0,
            },
            {
                "DUPERSID": "p2",
                MEPS_AGE: 50,
                MEPS_SEX: 2,
                MEPS_REGION: 2,
                MEPS_SMOKER: 1,
                MEPS_FAMID: "f2",
                MEPS_TOTEXP: 9000.0,
            },
        ]
    ).astype(
        {
            MEPS_AGE: np.int32,
            MEPS_SEX: np.int32,
            MEPS_REGION: np.int32,
            MEPS_SMOKER: np.int32,
            MEPS_FAMID: "string",
            MEPS_TOTEXP: np.float64,
        }
    )
    saq_rows = pd.DataFrame(
        [
            {"DUPERSID": "p1", "ADBMI42": 27.5},
            {"DUPERSID": "p2", "ADBMI42": 31.2},
        ]
    ).astype({"ADBMI42": np.float64})

    main_path = tmp_path / "main.dta"
    saq_path = tmp_path / "saq.dta"
    main_rows.to_stata(main_path, write_index=False)
    saq_rows.to_stata(saq_path, write_index=False)

    df = load_meps(main_path, saq_path=saq_path)
    assert len(df) == 2
    assert set(df["bmi"]) == {27.5, 31.2}
