"""Unit tests for the SQLite-backed session and document stores.

Tests verify:
- CRUD operations behave identically to the in-memory equivalents.
- Data survives a store close + reopen (the key property of SQLite persistence).
- Thread safety under concurrent access.
- The document store rebuilds TF-IDF indexes correctly from persisted chunks.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from gauge.docchat.schemas import Chunk, DocumentMeta
from gauge.docchat.sqlite_store import SqliteDocumentStore
from gauge.docchat.index import TfidfRetrievalIndex
from gauge.predictor.schemas import PredictionFeatures
from gauge.session.models import Session
from gauge.session.sqlite_store import SqliteSessionStore

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _features() -> PredictionFeatures:
    return PredictionFeatures(
        age=35, sex="female", bmi=27.5, children=1, smoker="no", region="northeast"
    )


def _session(session_id: str = "sess-001") -> Session:
    return Session(session_id=session_id, features=_features())


def _doc_meta(doc_id: str = "doc-001") -> DocumentMeta:
    from datetime import datetime, timezone

    return DocumentMeta(
        document_id=doc_id,
        filename="plan.pdf",
        n_pages=3,
        n_chunks=5,
        uploaded_at=datetime.now(timezone.utc),
    )


def _chunks(doc_id: str = "doc-001", n: int = 3) -> list[Chunk]:
    return [
        Chunk(
            document_id=doc_id,
            chunk_index=i,
            text=f"Chunk {i}: deductible $1,000 coinsurance 20% oop max $5,000",
            page_numbers=[i + 1],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# SqliteSessionStore — CRUD
# ---------------------------------------------------------------------------


class TestSqliteSessionStoreCRUD:
    def test_create_and_get(self, tmp_path: Path) -> None:
        store = SqliteSessionStore(tmp_path / "test.db")
        s = _session()
        store.create(s)
        retrieved = store.get(s.session_id)
        assert retrieved is not None
        assert retrieved.session_id == s.session_id
        assert retrieved.features.age == 35

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        store = SqliteSessionStore(tmp_path / "test.db")
        assert store.get("nonexistent") is None

    def test_create_overwrites_existing(self, tmp_path: Path) -> None:
        store = SqliteSessionStore(tmp_path / "test.db")
        s = _session("dup")
        store.create(s)
        s2 = _session("dup")
        s2.document_id = "doc-xyz"
        store.create(s2)
        assert store.get("dup").document_id == "doc-xyz"

    def test_update_persists_change(self, tmp_path: Path) -> None:
        store = SqliteSessionStore(tmp_path / "test.db")
        s = _session()
        store.create(s)
        s.document_id = "doc-updated"
        store.update(s)
        assert store.get(s.session_id).document_id == "doc-updated"

    def test_update_missing_raises_key_error(self, tmp_path: Path) -> None:
        store = SqliteSessionStore(tmp_path / "test.db")
        with pytest.raises(KeyError):
            store.update(_session("ghost"))

    def test_delete_existing_returns_true(self, tmp_path: Path) -> None:
        store = SqliteSessionStore(tmp_path / "test.db")
        s = _session()
        store.create(s)
        assert store.delete(s.session_id) is True
        assert store.get(s.session_id) is None

    def test_delete_missing_returns_false(self, tmp_path: Path) -> None:
        store = SqliteSessionStore(tmp_path / "test.db")
        assert store.delete("ghost") is False


# ---------------------------------------------------------------------------
# SqliteSessionStore — persistence across reconnect
# ---------------------------------------------------------------------------


class TestSqliteSessionStorePersistence:
    def test_session_survives_close_and_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        # First connection: write.
        store1 = SqliteSessionStore(db)
        s = _session()
        s.document_id = "persisted-doc"
        store1.create(s)
        store1.close()

        # Second connection: should still see the session.
        store2 = SqliteSessionStore(db)
        retrieved = store2.get(s.session_id)
        assert retrieved is not None
        assert retrieved.document_id == "persisted-doc"

    def test_update_survives_reconnect(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store1 = SqliteSessionStore(db)
        s = _session()
        store1.create(s)
        s.document_id = "updated-doc"
        store1.update(s)
        store1.close()

        store2 = SqliteSessionStore(db)
        assert store2.get(s.session_id).document_id == "updated-doc"

    def test_delete_persists_across_reconnect(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store1 = SqliteSessionStore(db)
        store1.create(_session())
        store1.delete("sess-001")
        store1.close()

        store2 = SqliteSessionStore(db)
        assert store2.get("sess-001") is None

    def test_multiple_sessions_survive_reconnect(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store1 = SqliteSessionStore(db)
        for i in range(10):
            store1.create(_session(f"s{i}"))
        store1.close()

        store2 = SqliteSessionStore(db)
        for i in range(10):
            assert store2.get(f"s{i}") is not None


# ---------------------------------------------------------------------------
# SqliteSessionStore — thread safety
# ---------------------------------------------------------------------------


class TestSqliteSessionStoreThreadSafety:
    def test_concurrent_creates(self, tmp_path: Path) -> None:
        store = SqliteSessionStore(tmp_path / "test.db")
        sessions = [_session(f"t{i}") for i in range(50)]
        errors: list[Exception] = []

        def create(s: Session) -> None:
            try:
                store.create(s)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=create, args=(s,)) for s in sessions]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for s in sessions:
            assert store.get(s.session_id) is not None


# ---------------------------------------------------------------------------
# SqliteDocumentStore — CRUD
# ---------------------------------------------------------------------------


class TestSqliteDocumentStoreCRUD:
    def test_add_and_get(self, tmp_path: Path) -> None:
        store = SqliteDocumentStore(tmp_path / "test.db")
        meta = _doc_meta()
        chunks = _chunks()
        store.add(meta, chunks)
        stored = store.get(meta.document_id)
        assert stored is not None
        assert stored.meta.document_id == meta.document_id
        assert len(stored.chunks) == len(chunks)

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        store = SqliteDocumentStore(tmp_path / "test.db")
        assert store.get("missing") is None

    def test_list_meta_returns_all(self, tmp_path: Path) -> None:
        store = SqliteDocumentStore(tmp_path / "test.db")
        for i in range(3):
            store.add(_doc_meta(f"doc-{i}"), _chunks(f"doc-{i}"))
        metas = store.list_meta()
        assert len(metas) == 3
        ids = {m.document_id for m in metas}
        assert ids == {"doc-0", "doc-1", "doc-2"}

    def test_delete_existing_returns_true(self, tmp_path: Path) -> None:
        store = SqliteDocumentStore(tmp_path / "test.db")
        meta = _doc_meta()
        store.add(meta, _chunks())
        assert store.delete(meta.document_id) is True
        assert store.get(meta.document_id) is None

    def test_delete_removes_chunks(self, tmp_path: Path) -> None:
        """After delete + reopen, chunks should not be rebuilt for deleted doc."""
        db = tmp_path / "test.db"
        store1 = SqliteDocumentStore(db)
        store1.add(_doc_meta(), _chunks())
        store1.delete("doc-001")
        store1.close()

        store2 = SqliteDocumentStore(db)
        assert store2.get("doc-001") is None
        assert store2.list_meta() == []

    def test_delete_missing_returns_false(self, tmp_path: Path) -> None:
        store = SqliteDocumentStore(tmp_path / "test.db")
        assert store.delete("ghost") is False

    def test_add_replaces_existing(self, tmp_path: Path) -> None:
        store = SqliteDocumentStore(tmp_path / "test.db")
        meta = _doc_meta()
        store.add(meta, _chunks(n=2))
        new_chunks = _chunks(n=5)
        store.add(meta, new_chunks)
        stored = store.get(meta.document_id)
        assert len(stored.chunks) == 5


# ---------------------------------------------------------------------------
# SqliteDocumentStore — index quality
# ---------------------------------------------------------------------------


class TestSqliteDocumentStoreIndex:
    def test_index_is_searchable(self, tmp_path: Path) -> None:
        store = SqliteDocumentStore(tmp_path / "test.db")
        chunks = _chunks()
        store.add(_doc_meta(), chunks)
        stored = store.get("doc-001")
        assert isinstance(stored.index, TfidfRetrievalIndex)
        results = stored.index.search("deductible", k=2)
        assert len(results) > 0

    def test_rebuilt_index_is_searchable(self, tmp_path: Path) -> None:
        """Index rebuilt from persisted chunks should answer queries correctly."""
        db = tmp_path / "test.db"
        store1 = SqliteDocumentStore(db)
        store1.add(_doc_meta(), _chunks())
        store1.close()

        store2 = SqliteDocumentStore(db)
        stored = store2.get("doc-001")
        assert stored is not None
        results = stored.index.search("coinsurance", k=2)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# SqliteDocumentStore — persistence across reconnect
# ---------------------------------------------------------------------------


class TestSqliteDocumentStorePersistence:
    def test_document_survives_close_and_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store1 = SqliteDocumentStore(db)
        store1.add(_doc_meta(), _chunks())
        store1.close()

        store2 = SqliteDocumentStore(db)
        stored = store2.get("doc-001")
        assert stored is not None
        assert stored.meta.filename == "plan.pdf"
        assert len(stored.chunks) == 3

    def test_multiple_documents_survive_reconnect(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store1 = SqliteDocumentStore(db)
        for i in range(5):
            store1.add(_doc_meta(f"doc-{i}"), _chunks(f"doc-{i}"))
        store1.close()

        store2 = SqliteDocumentStore(db)
        assert len(store2.list_meta()) == 5
        for i in range(5):
            assert store2.get(f"doc-{i}") is not None

    def test_page_numbers_preserved_across_reconnect(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        chunks = [
            Chunk(
                document_id="doc-001",
                chunk_index=0,
                text="Deductible information",
                page_numbers=[2, 3],
            )
        ]
        store1 = SqliteDocumentStore(db)
        store1.add(_doc_meta(), chunks)
        store1.close()

        store2 = SqliteDocumentStore(db)
        stored = store2.get("doc-001")
        assert stored.chunks[0].page_numbers == [2, 3]


# ---------------------------------------------------------------------------
# SqliteDocumentStore — thread safety
# ---------------------------------------------------------------------------


class TestSqliteDocumentStoreThreadSafety:
    def test_concurrent_adds(self, tmp_path: Path) -> None:
        store = SqliteDocumentStore(tmp_path / "test.db")
        errors: list[Exception] = []

        def add_doc(i: int) -> None:
            try:
                store.add(_doc_meta(f"doc-{i}"), _chunks(f"doc-{i}"))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=add_doc, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store.list_meta()) == 20
