"""Module entry point: `uvicorn health_app.main:app`.

Trains (or loads from cache) the cost predictor on startup, then wires
it to the FastAPI app along with the seeded benefits repository.

Dataset resolution order:

1. `HEALTH_APP_DATASET_CSV` env var, if set, must point to a CSV.
2. `HEALTH_APP_MEPS_DTA` env var, if set, must point to a MEPS .dta file.
3. `data/meps_hc233.dta` if present (preferred over Kaggle).
4. `data/insurance.csv` if present.
5. Synthetic Kaggle-shaped dataset, generated deterministically.

The model cache is keyed by the chosen data source so swapping inputs
forces a clean retrain instead of silently reusing the old model.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from health_app.api import create_app
from health_app.benefits.seed import build_default_repository
from health_app.docchat.llm import auto_select_llm
from health_app.docchat.service import DocumentChatService
from health_app.predictor.dataset import load_dataset
from health_app.predictor.meps import load_meps
from health_app.predictor.model import CostPredictor

# Paths to optional local datasets. Resolved relative to this file so they
# work regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_CSV_PATH = _REPO_ROOT / "data" / "insurance.csv"
_LOCAL_MEPS_PATH = _REPO_ROOT / "data" / "meps_hc233.dta"
_LOCAL_MEPS_SAQ_PATH = _REPO_ROOT / "data" / "meps_hc236.dta"

_CACHE_DIR = Path(
    os.environ.get(
        "HEALTH_APP_CACHE_DIR",
        str(Path.home() / ".cache" / "health_app"),
    )
)

# Marker used in the source tag to dispatch to the right loader.
_KIND_MEPS = "meps"
_KIND_CSV = "csv"
_KIND_SYNTHETIC = "synthetic"


def _resolve_dataset_source() -> tuple[str, str, Path | None, Path | None]:
    """Pick the dataset source and return its kind, tag, and file paths.

    The source tag namespaces the model cache file so swapping datasets
    forces a retrain rather than silently reusing the old model.

    Returns
    -------
    tuple[str, str, Path or None, Path or None]
        ``(kind, source_tag, primary_path, saq_path)`` where ``kind`` is
        one of ``_KIND_MEPS``, ``_KIND_CSV``, or ``_KIND_SYNTHETIC``;
        ``source_tag`` is a stable string used to key the cache;
        ``primary_path`` is the dataset file path (``None`` for synthetic);
        and ``saq_path`` is the MEPS SAQ supplement path or ``None``.
    """
    env_csv = os.environ.get("HEALTH_APP_DATASET_CSV")
    if env_csv:
        return _KIND_CSV, f"env_csv:{env_csv}", Path(env_csv), None
    env_meps = os.environ.get("HEALTH_APP_MEPS_DTA")
    env_saq = os.environ.get("HEALTH_APP_MEPS_SAQ")
    if env_meps:
        saq = Path(env_saq) if env_saq else None
        tag = f"env_meps:{env_meps}|saq:{env_saq or ''}"
        return _KIND_MEPS, tag, Path(env_meps), saq
    if _LOCAL_MEPS_PATH.exists():
        saq = _LOCAL_MEPS_SAQ_PATH if _LOCAL_MEPS_SAQ_PATH.exists() else None
        tag = f"meps:{_LOCAL_MEPS_PATH}|saq:{saq or ''}"
        return _KIND_MEPS, tag, _LOCAL_MEPS_PATH, saq
    if _LOCAL_CSV_PATH.exists():
        return _KIND_CSV, f"csv:{_LOCAL_CSV_PATH}", _LOCAL_CSV_PATH, None
    return _KIND_SYNTHETIC, "synthetic", None, None


def _cache_path_for(source_tag: str) -> Path:
    """Return the stable, source-keyed cache file path for a trained model.

    Parameters
    ----------
    source_tag : str
        The source tag returned by :func:`_resolve_dataset_source`.

    Returns
    -------
    Path
        Path under ``_CACHE_DIR`` whose filename embeds a 12-character
        SHA-1 digest of ``source_tag``.
    """
    digest = hashlib.sha1(source_tag.encode("utf-8")).hexdigest()[:12]
    return _CACHE_DIR / f"cost_predictor.{digest}.joblib"


def _load_or_train_predictor() -> CostPredictor:
    """Return a fitted predictor, loading from cache or training on first run.

    On the first call for a given dataset source, trains the model and
    writes it to ``_CACHE_DIR``. Subsequent calls load the cached file
    directly. Swapping the dataset source (via env vars or local files)
    produces a different cache key and triggers a fresh retrain.

    Returns
    -------
    CostPredictor
        A fully fitted predictor ready to serve predictions.
    """
    kind, source_tag, path, saq_path = _resolve_dataset_source()
    cache_path = _cache_path_for(source_tag)

    if cache_path.exists():
        return CostPredictor.load(cache_path)

    if kind == _KIND_MEPS:
        assert path is not None
        df = load_meps(path, saq_path=saq_path)
    elif kind == _KIND_CSV:
        df = load_dataset(csv_path=path)
    else:
        df = load_dataset()

    predictor = CostPredictor()
    predictor.fit(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(cache_path)
    return predictor


app = create_app(
    repository=build_default_repository(),
    predictor=_load_or_train_predictor(),
    chat_service=DocumentChatService(llm=auto_select_llm()),
)
