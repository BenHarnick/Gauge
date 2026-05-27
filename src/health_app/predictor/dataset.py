"""Training data for the cost predictor.

The Kaggle `insurance.csv` dataset is the canonical small-scale teaching
set in this space (columns: age, sex, bmi, children, smoker, region,
charges). To keep this prototype self-contained and reproducible, we
generate a synthetic dataset with the same shape and roughly the same
relationships. A `load_csv` path is provided for swapping in the real
file later without touching anything else.

The data-generating process is documented inline; tests rely on its
deterministic seeded output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
]
TARGET_COLUMN: str = "charges"
# Census regions (matches MEPS). The Kaggle CSV uses cardinal regions
# (northeast/northwest/southeast/southwest) which we remap on load.
REGIONS: list[str] = ["northeast", "midwest", "south", "west"]
KAGGLE_REGION_MAP: dict[str, str] = {
    "northeast": "northeast",
    "northwest": "midwest",
    "southeast": "south",
    "southwest": "west",
}


def generate_synthetic_dataset(
    n_rows: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic, Kaggle-insurance-shaped dataset.

    The data-generating function approximates published analyses of the
    real Kaggle dataset::

        charges = base
                + 250 * age
                + 425 * children
                + 24_000 * smoker
                + (1_400 if smoker else 60) * max(0, bmi - 30)
                + lognormal multiplicative noise

    Parameters
    ----------
    n_rows : int, optional
        Number of synthetic rows to draw. Default is 1500.
    seed : int, optional
        Numpy random seed for reproducibility. Default is 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``age``, ``sex``, ``bmi``, ``children``,
        ``smoker``, ``region``, ``charges`` (charges in dollars).
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(low=18, high=65, size=n_rows)
    sex = rng.choice(["male", "female"], size=n_rows)
    bmi = np.clip(rng.normal(loc=30.6, scale=6.0, size=n_rows), 16.0, 53.0)
    children = rng.poisson(lam=1.1, size=n_rows).clip(0, 5)
    smoker_flag = rng.random(size=n_rows) < 0.20
    smoker = np.where(smoker_flag, "yes", "no")
    region = rng.choice(REGIONS, size=n_rows)

    bmi_over_30 = np.maximum(0.0, bmi - 30.0)
    base = 2_500.0
    age_effect = 250.0 * age
    children_effect = 425.0 * children
    smoker_effect = np.where(smoker_flag, 24_000.0, 0.0)
    bmi_effect = np.where(smoker_flag, 1_400.0, 60.0) * bmi_over_30
    noise = rng.lognormal(mean=0.0, sigma=0.15, size=n_rows)

    charges = (
        base + age_effect + children_effect + smoker_effect + bmi_effect
    ) * noise

    return pd.DataFrame(
        {
            "age": age.astype(int),
            "sex": sex,
            "bmi": bmi.round(3),
            "children": children.astype(int),
            "smoker": smoker,
            "region": region,
            "charges": charges.round(2),
        }
    )


def load_csv(path: Path | str) -> pd.DataFrame:
    """Load a Kaggle-style insurance CSV from disk.

    The CSV's cardinal regions (northwest, southwest, southeast, northeast)
    are remapped to the canonical Census regions used elsewhere in the
    schema. The mapping is documented in ``KAGGLE_REGION_MAP``.

    Parameters
    ----------
    path : Path or str
        Path to the CSV file. Must contain the columns named in
        ``FEATURE_COLUMNS`` plus ``charges``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``FEATURE_COLUMNS + [TARGET_COLUMN]``,
        with regions normalised to Census nomenclature.

    Raises
    ------
    ValueError
        If any expected column is missing from the file.
    """
    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    df["region"] = df["region"].map(KAGGLE_REGION_MAP).fillna(df["region"])
    return df


def load_dataset(
    csv_path: Path | str | None = None,
    n_rows: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Return training data, preferring a CSV if supplied, else synthetic.

    Parameters
    ----------
    csv_path : Path or str or None, optional
        Path to a Kaggle-style insurance CSV. When supplied the file is
        loaded via :func:`load_csv`; otherwise synthetic data is generated.
    n_rows : int, optional
        Number of synthetic rows when no CSV is given. Default is 1500.
    seed : int, optional
        Random seed passed to :func:`generate_synthetic_dataset`. Default
        is 42.

    Returns
    -------
    pd.DataFrame
        Training DataFrame with columns ``FEATURE_COLUMNS + [TARGET_COLUMN]``.
    """
    if csv_path is not None:
        return load_csv(csv_path)
    return generate_synthetic_dataset(n_rows=n_rows, seed=seed)
