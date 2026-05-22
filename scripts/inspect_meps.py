"""Print all columns in the MEPS .dta that match the roles our loader cares about.

Useful when the per-year variable names differ from what the loader's
candidate list expects. Run:

    python scripts/inspect_meps.py data/meps_hc233.dta
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROLE_KEYWORDS: dict[str, list[str]] = {
    "age": ["AGE"],
    "sex": ["SEX"],
    "region": ["REGION"],
    "bmi": ["BMI"],
    "smoker": ["SMOK"],
    "famid": ["FAMID"],
    "totexp": ["TOTEXP", "TOTAL.*EXP"],
}


def main(path_arg: str | None = None) -> int:
    path = Path(
        path_arg
        or Path(__file__).resolve().parents[1] / "data" / "meps_hc233.dta"
    )
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    print(f"Inspecting {path}...")
    df = pd.read_stata(path, convert_categoricals=False)
    print(f"Total columns: {len(df.columns)}  rows: {len(df)}")
    print()
    for role, keywords in ROLE_KEYWORDS.items():
        matches = sorted(
            {
                c
                for c in df.columns
                if any(re.search(kw, c, re.IGNORECASE) for kw in keywords)
            }
        )
        print(f"  {role:<8} candidates in file: {matches or '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
