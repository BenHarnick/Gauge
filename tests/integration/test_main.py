"""Smoke tests for the application bootstrap module (main.py).

These tests exercise the dataset-resolution helpers, store factories, and
the application entry point without requiring a full server process.

Because ``main.py`` imports at module level (to build ``app``), the module
under test is imported inside each test to keep side-effects predictable and
to allow env-var overrides via ``monkeypatch``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


class TestDatasetResolution:
    """Unit tests for ``_resolve_dataset_source``.

    These call the function directly (no module reload) so they don't
    trigger model training — they just verify the resolution logic.
    """

    def test_env_csv_takes_priority(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """GAUGE_DATASET_CSV overrides all other sources."""
        import gauge.main as m

        csv = tmp_path / "custom.csv"
        csv.touch()
        monkeypatch.setenv("GAUGE_DATASET_CSV", str(csv))

        kind, tag, path, saq = m._resolve_dataset_source()
        assert kind == m._KIND_CSV
        assert str(csv) in tag
        assert path == csv
        assert saq is None

    def test_env_meps_second_priority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GAUGE_MEPS_DTA is used when no CSV env var is set."""
        import gauge.main as m

        dta = tmp_path / "meps.dta"
        dta.touch()
        monkeypatch.delenv("GAUGE_DATASET_CSV", raising=False)
        monkeypatch.setenv("GAUGE_MEPS_DTA", str(dta))

        kind, tag, path, saq = m._resolve_dataset_source()
        assert kind == m._KIND_MEPS
        assert path == dta

    def test_synthetic_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to synthetic when no data source is configured."""
        import gauge.main as m

        monkeypatch.delenv("GAUGE_DATASET_CSV", raising=False)
        monkeypatch.delenv("GAUGE_MEPS_DTA", raising=False)
        # Patch the local file paths so they don't exist in the test environment.
        with (
            patch.object(m, "_LOCAL_MEPS_PATH", tmp_path / "nonexistent.dta"),
            patch.object(m, "_LOCAL_CSV_PATH", tmp_path / "nonexistent.csv"),
        ):
            kind, tag, path, saq = m._resolve_dataset_source()

        assert kind == m._KIND_SYNTHETIC
        assert path is None

    def test_cache_path_is_deterministic(self) -> None:
        """Same source tag always produces the same cache file path."""
        import gauge.main as m

        p1 = m._cache_path_for("synthetic")
        p2 = m._cache_path_for("synthetic")
        assert p1 == p2

    def test_cache_path_differs_for_different_tags(self) -> None:
        """Different source tags produce different cache paths (no collision)."""
        import gauge.main as m

        assert m._cache_path_for("synthetic") != m._cache_path_for("csv:/some/path")


class TestStoreFactories:
    """Tests for ``_make_session_store`` and ``_make_document_store``."""

    def test_in_memory_session_store_when_no_persist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GAUGE_NO_PERSIST=1 returns an in-memory session store."""
        import gauge.main as m
        from gauge.session.store import InMemorySessionStore

        monkeypatch.setenv("GAUGE_NO_PERSIST", "1")
        monkeypatch.setenv("GAUGE_CACHE_DIR", str(tmp_path))

        store = m._make_session_store()
        assert isinstance(store, InMemorySessionStore)

    def test_sqlite_session_store_by_default(self, tmp_path: Path) -> None:
        """Without GAUGE_NO_PERSIST, returns a SQLite-backed session store.

        ``_NO_PERSIST`` is a module-level constant set at import time, so we
        patch it directly rather than setting the env var after import.
        """
        import gauge.main as m
        from gauge.session.sqlite_store import SqliteSessionStore

        with (
            patch.object(m, "_NO_PERSIST", False),
            patch.object(m, "_DB_PATH", tmp_path / "test.db"),
        ):
            store = m._make_session_store()

        assert isinstance(store, SqliteSessionStore)

    def test_in_memory_document_store_when_no_persist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GAUGE_NO_PERSIST=1 returns an in-memory document store."""
        import gauge.main as m
        from gauge.docchat.store import InMemoryDocumentStore

        monkeypatch.setenv("GAUGE_NO_PERSIST", "1")

        store = m._make_document_store()
        assert isinstance(store, InMemoryDocumentStore)


class TestAppBoot:
    """Smoke tests for the top-level ``app`` object exposed by main.py.

    ``main.py`` builds the app at import time (model training runs once).
    These tests reuse the already-imported module to avoid re-training.
    """

    def test_healthz_via_main_app(self) -> None:
        """The app built by main.py responds to /healthz with status ok."""
        import gauge.main as m

        client = TestClient(m.app, raise_server_exceptions=True)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_app_is_fastapi_instance(self) -> None:
        """main.app is a FastAPI application (not None or a bare dict)."""
        from fastapi import FastAPI

        import gauge.main as m

        assert isinstance(m.app, FastAPI)
