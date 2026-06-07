"""Module entry point: ``uvicorn gauge.main:app``.

Trains (or loads from cache) the cost predictor on startup, then wires
it to the FastAPI app along with the seeded benefits repository and
SQLite-backed persistence stores.

Dataset resolution order
------------------------
1. ``GAUGE_DATASET_CSV`` env var, if set, must point to a CSV.
2. ``GAUGE_MEPS_DTA`` env var, if set, must point to a MEPS .dta file.
3. ``data/meps_hc233.dta`` if present (preferred over Kaggle).
4. ``data/insurance.csv`` if present.
5. Synthetic Kaggle-shaped dataset, generated deterministically.

The model cache is keyed by the chosen data source so swapping inputs
forces a clean retrain instead of silently reusing the old model.

Persistence
-----------
Sessions and uploaded documents are stored in a SQLite database at
``~/.cache/gauge/gauge.db`` by default.  Override the path
with the ``GAUGE_DB_PATH`` environment variable.  The in-memory
stores are used instead when ``GAUGE_NO_PERSIST=1`` is set.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from gauge.api import create_app
from gauge.benefits.seed import build_default_repository
from gauge.docchat.llm import auto_select_llm
from gauge.docchat.service import DocumentChatService
from gauge.docchat.sqlite_store import SqliteDocumentStore
from gauge.docchat.store import InMemoryDocumentStore
from gauge.plan_extract.extractor import PlanExtractor
from gauge.predictor.dataset import load_dataset
from gauge.predictor.meps import load_meps
from gauge.predictor.model import CostPredictor
from gauge.session.sqlite_store import SqliteSessionStore
from gauge.session.store import InMemorySessionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Datasets — resolved relative to the repo root so they work regardless of
# the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_CSV_PATH = _REPO_ROOT / "data" / "insurance.csv"
_LOCAL_MEPS_PATH = _REPO_ROOT / "data" / "meps_hc233.dta"
_LOCAL_MEPS_SAQ_PATH = _REPO_ROOT / "data" / "meps_hc236.dta"

# Model cache directory.
_CACHE_DIR = Path(
    os.environ.get(
        "GAUGE_CACHE_DIR",
        str(Path.home() / ".cache" / "gauge"),
    )
)

# SQLite database path.
_DB_PATH = Path(
    os.environ.get(
        "GAUGE_DB_PATH",
        str(_CACHE_DIR / "gauge.db"),
    )
)

# When set to "1" the in-memory stores are used and nothing is persisted.
_NO_PERSIST = os.environ.get("GAUGE_NO_PERSIST", "0") == "1"

# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------

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
    env_csv = os.environ.get("GAUGE_DATASET_CSV")
    if env_csv:
        return _KIND_CSV, f"env_csv:{env_csv}", Path(env_csv), None

    env_meps = os.environ.get("GAUGE_MEPS_DTA")
    env_saq = os.environ.get("GAUGE_MEPS_SAQ")
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

    _log_dataset_source(kind, path, saq_path)

    if cache_path.exists():
        logger.info("Loading cached model from %s", cache_path)
        return CostPredictor.load(cache_path)

    logger.info("No cached model found — training now (this may take a minute)…")

    if kind == _KIND_MEPS:
        assert path is not None
        df = load_meps(path, saq_path=saq_path)
    elif kind == _KIND_CSV:
        df = load_dataset(csv_path=path)
    else:
        df = load_dataset()

    logger.info("Loaded %d training rows — fitting model…", len(df))
    predictor = CostPredictor()
    predictor.fit(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(cache_path)
    logger.info("Model trained and cached at %s", cache_path)
    return predictor


def _log_dataset_source(
    kind: str, path: Path | None, saq_path: Path | None
) -> None:
    """Emit an INFO log describing the chosen data source.

    Parameters
    ----------
    kind : str
        One of ``_KIND_MEPS``, ``_KIND_CSV``, or ``_KIND_SYNTHETIC``.
    path : Path or None
        Primary dataset file path, or ``None`` for synthetic.
    saq_path : Path or None
        SAQ supplement path (MEPS only), or ``None``.

    Returns
    -------
    None
    """
    if kind == _KIND_MEPS:
        saq_note = f" + SAQ {saq_path}" if saq_path else " (no SAQ — BMI may be missing)"
        logger.info("Dataset: MEPS  %s%s", path, saq_note)
    elif kind == _KIND_CSV:
        logger.info("Dataset: CSV   %s", path)
    else:
        logger.info(
            "Dataset: synthetic (place data/meps_hc233.dta or data/insurance.csv"
            " in the repo root for real data)"
        )


# ---------------------------------------------------------------------------
# Persistence stores
# ---------------------------------------------------------------------------


def _make_session_store() -> SqliteSessionStore | InMemorySessionStore:
    """Return the appropriate session store based on env configuration.

    Returns
    -------
    SqliteSessionStore or InMemorySessionStore
        ``SqliteSessionStore`` by default; ``InMemorySessionStore`` when
        ``GAUGE_NO_PERSIST=1``.
    """
    if _NO_PERSIST:
        logger.info("Session store: in-memory (GAUGE_NO_PERSIST=1)")
        return InMemorySessionStore()
    logger.info("Session store: SQLite at %s", _DB_PATH)
    return SqliteSessionStore(_DB_PATH)


def _make_document_store() -> SqliteDocumentStore | InMemoryDocumentStore:
    """Return the appropriate document store based on env configuration.

    Returns
    -------
    SqliteDocumentStore or InMemoryDocumentStore
        ``SqliteDocumentStore`` by default; ``InMemoryDocumentStore`` when
        ``GAUGE_NO_PERSIST=1``.
    """
    if _NO_PERSIST:
        logger.info("Document store: in-memory (GAUGE_NO_PERSIST=1)")
        return InMemoryDocumentStore()
    logger.info("Document store: SQLite at %s", _DB_PATH)
    return SqliteDocumentStore(_DB_PATH)


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

_llm = auto_select_llm()
logger.info("LLM backend: %s", _llm.name)

_document_store = _make_document_store()
_chat_service = DocumentChatService(store=_document_store, llm=_llm)

app = create_app(
    repository=build_default_repository(),
    predictor=_load_or_train_predictor(),
    chat_service=_chat_service,
    session_store=_make_session_store(),
    plan_extractor=PlanExtractor(llm=_llm),
)

# ---------------------------------------------------------------------------
# Static file serving (Docker / production)
#
# When the built React SPA is present at frontend/dist, mount it so that
# a single container can serve both the API and the UI.  In local dev the
# Vite dev server handles the frontend and this block is skipped.
# ---------------------------------------------------------------------------

_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    _ASSETS_DIR = _FRONTEND_DIST / "assets"
    if _ASSETS_DIR.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_ASSETS_DIR)),
            name="assets",
        )
        logger.info("Serving frontend assets from %s", _ASSETS_DIR)

    _INDEX_HTML = _FRONTEND_DIST / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _serve_spa(full_path: str) -> FileResponse:  # noqa: ARG001
        """Catch-all route: return index.html for any unknown path.

        Parameters
        ----------
        full_path : str
            Any URL path not matched by an API route.

        Returns
        -------
        FileResponse
            The React SPA entry point, allowing client-side routing to take
            over.
        """
        return FileResponse(str(_INDEX_HTML))

    logger.info("SPA catch-all active — serving %s", _INDEX_HTML)
