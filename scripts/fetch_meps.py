"""Download MEPS 2021 data files (HC-233 + HC-236 SAQ supplement) into data/.

Run with:

    python scripts/fetch_meps.py

The script downloads the Stata-format public-use files from
meps.ahrq.gov and unzips them into `data/`:

    data/meps_hc233.dta   Full-Year Consolidated File (demographics, expenditure)
    data/meps_hc236.dta   SAQ Supplement (adult BMI lives here)

If an AHRQ URL ever changes, the script prints the canonical download
page so you can fetch the file manually.
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MepsFile:
    name: str           # human-friendly label
    dest: Path          # final .dta location
    urls: list[str]     # try in order

    @property
    def page(self) -> str:
        """URL of the AHRQ download page for manual fallback."""
        # The download page URL pattern uses the file code.
        code = self.urls[0].rsplit("/", 2)[-2]  # 'h233' etc.
        return (
            "https://meps.ahrq.gov/mepsweb/data_stats/"
            f"download_data_files_detail.jsp?cboPufNumber={code.upper()}"
        )


FILES: list[MepsFile] = [
    MepsFile(
        name="HC-233 (Full-Year Consolidated, 2021)",
        dest=REPO_ROOT / "data" / "meps_hc233.dta",
        urls=[
            "https://meps.ahrq.gov/mepsweb/data_files/pufs/h233/h233dta.zip",
            "https://meps.ahrq.gov/data_files/pufs/h233/h233dta.zip",
        ],
    ),
    MepsFile(
        name="HC-236 (SAQ Supplement, 2021)",
        dest=REPO_ROOT / "data" / "meps_hc236.dta",
        urls=[
            "https://meps.ahrq.gov/mepsweb/data_files/pufs/h236/h236dta.zip",
            "https://meps.ahrq.gov/data_files/pufs/h236/h236dta.zip",
        ],
    ),
]


def _try_download(url: str) -> bytes | None:
    print(f"  trying {url} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "health-app/0.2 (prototype fetcher)"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        print(f"ok ({len(data) / (1024 * 1024):.1f} MB)")
        return data
    except Exception as e:
        print(f"failed ({e.__class__.__name__})")
        return None


def _fetch_one(spec: MepsFile) -> bool:
    spec.dest.parent.mkdir(parents=True, exist_ok=True)
    if spec.dest.exists():
        size_mb = spec.dest.stat().st_size / (1024 * 1024)
        print(f"{spec.name}: already at {spec.dest} ({size_mb:.1f} MB).")
        return True

    print(f"Fetching {spec.name}...")
    blob: bytes | None = None
    for url in spec.urls:
        blob = _try_download(url)
        if blob is not None:
            break

    if blob is None:
        print(f"  Automatic download failed for {spec.name}.")
        print(f"  Manual fallback: {spec.page}")
        print(f"    1. Download the Stata file.")
        print(f"    2. Unzip and rename the .dta to: {spec.dest}")
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            dta_names = [n for n in zf.namelist() if n.lower().endswith(".dta")]
            if not dta_names:
                raise ValueError("zip did not contain a .dta file")
            with zf.open(dta_names[0]) as f, open(spec.dest, "wb") as out:
                out.write(f.read())
    except Exception as e:
        print(f"  Unzip failed: {e}")
        return False

    size_mb = spec.dest.stat().st_size / (1024 * 1024)
    digest = hashlib.sha1(spec.dest.read_bytes()).hexdigest()[:12]
    print(f"  Saved {spec.dest} ({size_mb:.1f} MB, sha1 {digest})")
    return True


def main() -> int:
    print("Fetching MEPS 2021 files (HC-233 + HC-236 SAQ).")
    print()
    all_ok = True
    for spec in FILES:
        all_ok = _fetch_one(spec) and all_ok
        print()

    if all_ok:
        print(
            "Next steps:\n"
            "  1. Invalidate cached model: rm -rf ~/.cache/health_app\n"
            "  2. Restart backend:        uvicorn health_app.main:app --reload\n"
            "  3. The predictor will retrain on MEPS + SAQ automatically."
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
