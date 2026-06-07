"""Loader for MEPS Household Component public-use files.

MEPS variable names change subtly between years (e.g. `AGE21X` in 2021,
`AGE22X` in 2022). To stay tolerant, the loader knows several candidate
names per role and uses whichever one is actually present in the file.
If none match, the error message lists relevant columns it found so
it's easy to add a new candidate.

BMI is not in the main HC file; MEPS publishes it in the SAQ
(Self-Administered Questionnaire) Supplement, a separate file. The
loader takes an optional `saq_path` and merges it in on `DUPERSID`.

Variable mapping (MEPS role -> our schema)::

    age      -> age           (years; rows < 18 dropped)
    sex      -> sex           (1=male, 2=female)
    region   -> region        (1=NE, 2=MW, 3=S, 4=W)
    bmi      -> bmi           (self-reported adult BMI, from SAQ)
    smoker   -> smoker        (1=yes, 2=no; negatives = missing)
    famid    -> children      (count of family members aged <18)
    + age
    totexp   -> charges       (annual total medical expenditure)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from gauge.predictor.dataset import FEATURE_COLUMNS, TARGET_COLUMN

# Candidate columns per role, in priority order. The first one that
# exists in the loaded DataFrame wins.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "age": ["AGE21X", "AGE20X", "AGE22X", "AGE23X", "AGEYR", "AGEX"],
    "sex": ["SEX"],
    "region": [
        "REGION21",
        "REGION20",
        "REGION22",
        "REGION23",
        "REGION42",
        "REGION53",
        "REGION",
    ],
    "bmi": [
        # Panel longitudinal files (HC-236 covers Panel 23 / 2021).
        # ADBMI6 is round 6 (2021); ADBMI2 is round 2 (2020) fallback.
        "ADBMI6",
        "ADBMI2",
        # Other MEPS years use these names in either HC or SAQ files.
        "BMINDX53",
        "ADBMI42",
        "ADBMI53",
        "BMINDX42",
        "BMINDX31",
        "ADBMI31",
        "BMI_M18",
    ],
    "smoker": ["ADSMOK42", "ADSMOK53", "ADSMOK31"],
    "famid": ["FAMID21", "FAMID20", "FAMID22", "FAMID23", "FAMIDYR"],
    "totexp": ["TOTEXP21", "TOTEXP20", "TOTEXP22", "TOTEXP23"],
}

# When a role can't be matched, hint to the user by showing columns in
# the file whose names contain these substrings.
ROLE_KEYWORDS: dict[str, list[str]] = {
    "age": ["AGE"],
    "sex": ["SEX"],
    "region": ["REGION"],
    "bmi": ["BMI"],
    "smoker": ["SMOK"],
    "famid": ["FAMID"],
    "totexp": ["TOTEXP"],
}

# Kept as module aliases for backward compat with the unit tests. Tests
# use these names to build fixture .dta files.
MEPS_AGE = "AGE21X"
MEPS_SEX = "SEX"
MEPS_REGION = "REGION21"
MEPS_BMI = "ADBMI42"
MEPS_SMOKER = "ADSMOK42"
MEPS_FAMID = "FAMID21"
MEPS_TOTEXP = "TOTEXP21"

MEPS_REGION_MAP: dict[int, str] = {
    1: "northeast",
    2: "midwest",
    3: "south",
    4: "west",
}
MEPS_SEX_MAP: dict[int, str] = {1: "male", 2: "female"}
MEPS_SMOKER_MAP: dict[int, str] = {1: "yes", 2: "no"}


def _pick_column(df: pd.DataFrame, role: str) -> str:
    """Pick the first candidate column for ``role`` that exists in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns are searched.
    role : str
        Key into ``COLUMN_CANDIDATES`` (e.g. ``"age"``, ``"bmi"``).

    Returns
    -------
    str
        The first candidate column name that is present in ``df``.

    Raises
    ------
    ValueError
        With a hint listing columns in ``df`` whose names contain the
        role's keyword(s), so the candidate list can be extended.
    """
    candidates = COLUMN_CANDIDATES[role]
    for name in candidates:
        if name in df.columns:
            return name
    keywords = ROLE_KEYWORDS.get(role, [])
    hint_columns = sorted(
        {
            c
            for c in df.columns
            if any(re.search(kw, c, re.IGNORECASE) for kw in keywords)
        }
    )
    raise ValueError(
        f"Could not find a MEPS column for role {role!r}. "
        f"Tried {candidates}. "
        f"Columns in the file that look related: "
        f"{hint_columns or '(none)'}. "
        f"If your year of MEPS uses a different name, add it to "
        f"COLUMN_CANDIDATES[{role!r}] in src/gauge/predictor/meps.py."
    )


PERSON_ID = "DUPERSID"


def load_meps(
    path: Path | str,
    saq_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load a MEPS HC Stata file and shape it to match the predictor schema.

    Parameters
    ----------
    path : Path or str
        Path to the MEPS HC ``.dta`` file (Full-Year Consolidated File).
    saq_path : Path or str or None, optional
        Path to the SAQ supplement ``.dta`` file, which contains BMI.
        Merged onto the main file by ``DUPERSID``. If omitted and the main
        file does not have BMI, a clear error tells you to fetch it.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``FEATURE_COLUMNS + [TARGET_COLUMN]``.
        Adults only, complete cases; charges may include zeros.

    Raises
    ------
    ValueError
        If a file cannot be read or a required role cannot be matched to
        any candidate column name.
    """
    try:
        df = pd.read_stata(path, convert_categoricals=False)
    except Exception as e:
        raise ValueError(
            f"Could not read MEPS Stata file at {path}: {e}"
        ) from e

    if saq_path is not None:
        try:
            saq = pd.read_stata(saq_path, convert_categoricals=False)
        except Exception as e:
            raise ValueError(
                f"Could not read MEPS SAQ Stata file at {saq_path}: {e}"
            ) from e
        if PERSON_ID not in df.columns or PERSON_ID not in saq.columns:
            raise ValueError(
                f"Cannot merge SAQ: both files must contain a "
                f"{PERSON_ID!r} column."
            )
        # Only bring across BMI columns (and the person ID). The Panel
        # Longitudinal file has 5k+ columns we don't need.
        bmi_cols_in_saq = [
            c for c in COLUMN_CANDIDATES["bmi"] if c in saq.columns
        ]
        if not bmi_cols_in_saq:
            raise ValueError(
                f"SAQ file {saq_path} contains no known BMI column. "
                f"Add the actual column name to COLUMN_CANDIDATES['bmi'] "
                f"in src/gauge/predictor/meps.py. Run "
                f"`python scripts/inspect_meps.py <path>` to see what's there."
            )
        df = df.merge(
            saq[[PERSON_ID, *bmi_cols_in_saq]],
            on=PERSON_ID,
            how="left",
        )

    col_age = _pick_column(df, "age")
    col_sex = _pick_column(df, "sex")
    col_region = _pick_column(df, "region")
    try:
        col_bmi = _pick_column(df, "bmi")
    except ValueError as e:
        if saq_path is None:
            raise ValueError(
                "BMI is not in the MEPS HC file. It lives in the SAQ "
                "Supplement. Download HC-236 (or your year's SAQ) and "
                "pass its path as `saq_path` (or place it at "
                "data/meps_hc236.dta to be auto-detected)."
            ) from e
        raise
    col_smoker = _pick_column(df, "smoker")
    col_famid = _pick_column(df, "famid")
    col_totexp = _pick_column(df, "totexp")

    keep = [col_age, col_sex, col_region, col_bmi, col_smoker, col_famid, col_totexp]
    df = df[keep].copy()

    # MEPS uses negative integers as missing-value markers. Null them.
    for col in [col_age, col_sex, col_region, col_bmi, col_smoker]:
        df.loc[df[col] < 0, col] = pd.NA

    # Count children under 18 per family identifier. Done before the
    # adult-only filter so child rows contribute to their family's count.
    kids_per_family = (
        df.loc[df[col_age].fillna(99) < 18, [col_famid]]
        .assign(_n=1)
        .groupby(col_famid)["_n"]
        .sum()
    )
    df["_children"] = (
        df[col_famid].map(kids_per_family).fillna(0).astype(int)
    )

    df = df[df[col_age] >= 18].copy()

    required = [col_age, col_sex, col_region, col_bmi, col_smoker, col_totexp]
    df = df.dropna(subset=required).copy()

    df["age"] = df[col_age].astype(int)
    df["sex"] = df[col_sex].astype(int).map(MEPS_SEX_MAP)
    df["region"] = df[col_region].astype(int).map(MEPS_REGION_MAP)
    df["bmi"] = df[col_bmi].astype(float)
    df["smoker"] = df[col_smoker].astype(int).map(MEPS_SMOKER_MAP)
    df["children"] = df["_children"].clip(lower=0)
    df["charges"] = df[col_totexp].astype(float).clip(lower=0)

    df = df.dropna(subset=["sex", "region", "smoker"]).copy()
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].reset_index(drop=True)
