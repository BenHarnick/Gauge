"""SQLite-backed document store.

Provides the same public interface as :class:`InMemoryDocumentStore` but
persists document metadata and text chunks to SQLite so they survive server
restarts.

On startup, all previously stored chunks are loaded from the database and
used to rebuild the in-memory TF-IDF indexes.  Subsequent reads hit the
in-memory cache; writes are mirrored to SQLite atomically.

Schema
------
``documents``
    One row per document.  Stores metadata (filename, page count, etc.).
``chunks``
    One row per chunk.  Foreign-keyed to ``documents`` with CASCADE DELETE
    so removing a document also purges its chunks.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from gauge.docchat.index import TfidfRetrievalIndex
from gauge.docchat.schemas import Chunk, DocumentMeta
from gauge.docchat.store import StoredDocument

logger = logging.getLogger(__name__)

_CREATE_DOCUMENTS = """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        filename    TEXT NOT NULL,
        n_pages     INTEGER NOT NULL,
        n_chunks    INTEGER NOT NULL,
        uploaded_at TEXT NOT NULL
    )
"""

_CREATE_CHUNKS = """
    CREATE TABLE IF NOT EXISTS chunks (
        document_id  TEXT    NOT NULL,
        chunk_index  INTEGER NOT NULL,
        text         TEXT    NOT NULL,
        page_numbers TEXT    NOT NULL,
        PRIMARY KEY (document_id, chunk_index),
        FOREIGN KEY (document_id) REFERENCES documents(document_id)
            ON DELETE CASCADE
    )
"""


class SqliteDocumentStore:
    """SQLite-backed document store with in-memory index cache.

    Documents and their text chunks are persisted to SQLite so they
    survive server restarts.  TF-IDF retrieval indexes are held in memory
    (rebuilt from the database on startup) so query latency is identical
    to the in-memory implementation.

    Parameters
    ----------
    db_path : Path or str
        Path to the SQLite database file.  Created (along with any missing
        parent directories) if it does not already exist.
    """

    def __init__(self, db_path: Path | str) -> None:
        """Open the database, create tables if needed, and rebuild indexes.

        Parameters
        ----------
        db_path : Path or str
            Path to the SQLite ``.db`` file.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(_CREATE_DOCUMENTS)
        self._conn.execute(_CREATE_CHUNKS)
        self._conn.commit()

        # In-memory index cache: document_id -> StoredDocument
        self._docs: dict[str, StoredDocument] = {}
        self._rebuild_indexes()

    # ------------------------------------------------------------------
    # Public interface (mirrors InMemoryDocumentStore)
    # ------------------------------------------------------------------

    def add(self, meta: DocumentMeta, chunks: list[Chunk]) -> None:
        """Persist a document and its chunks, then build the retrieval index.

        Parameters
        ----------
        meta : DocumentMeta
            Document metadata (ID, filename, page/chunk counts, timestamp).
        chunks : list[Chunk]
            Text chunks to persist and index.
        """
        index = TfidfRetrievalIndex(chunks)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents"
                " (document_id, filename, n_pages, n_chunks, uploaded_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    meta.document_id,
                    meta.filename,
                    meta.n_pages,
                    meta.n_chunks,
                    meta.uploaded_at.isoformat(),
                ),
            )
            # Remove old chunks first (handles the OR REPLACE case).
            self._conn.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (meta.document_id,),
            )
            self._conn.executemany(
                "INSERT INTO chunks (document_id, chunk_index, text, page_numbers)"
                " VALUES (?, ?, ?, ?)",
                [
                    (
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.text,
                        json.dumps(chunk.page_numbers),
                    )
                    for chunk in chunks
                ],
            )
            self._conn.commit()
            self._docs[meta.document_id] = StoredDocument(
                meta=meta, chunks=chunks, index=index
            )

    def get(self, document_id: str) -> StoredDocument | None:
        """Return the stored document for ``document_id``, or ``None``.

        Parameters
        ----------
        document_id : str
            Identifier assigned at upload time.

        Returns
        -------
        StoredDocument or None
            In-memory cached document, or ``None`` if not found.
        """
        with self._lock:
            return self._docs.get(document_id)

    def list_meta(self) -> list[DocumentMeta]:
        """Return metadata for all stored documents.

        Returns
        -------
        list[DocumentMeta]
            Snapshot of metadata in insertion order.
        """
        with self._lock:
            return [d.meta for d in self._docs.values()]

    def delete(self, document_id: str) -> bool:
        """Remove a document and its chunks from the store.

        Parameters
        ----------
        document_id : str
            Identifier of the document to remove.

        Returns
        -------
        bool
            ``True`` if the document existed and was removed, ``False``
            if it was not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            self._conn.commit()
            existed = cursor.rowcount > 0
            if existed:
                self._docs.pop(document_id, None)
            return existed

    def close(self) -> None:
        """Close the underlying database connection.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rebuild_indexes(self) -> None:
        """Load all persisted documents from SQLite and build their indexes.

        Called once at startup.  Logs a summary of what was restored.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rows = self._conn.execute(
            "SELECT document_id, filename, n_pages, n_chunks, uploaded_at"
            " FROM documents"
        ).fetchall()

        restored = 0
        for doc_id, filename, n_pages, n_chunks, uploaded_at_str in rows:
            chunk_rows = self._conn.execute(
                "SELECT chunk_index, text, page_numbers"
                " FROM chunks WHERE document_id = ?"
                " ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()

            if not chunk_rows:
                logger.warning(
                    "Document %r has no stored chunks — skipping index rebuild.",
                    doc_id,
                )
                continue

            chunks = [
                Chunk(
                    document_id=doc_id,
                    chunk_index=ci,
                    text=text,
                    page_numbers=json.loads(pn),
                )
                for ci, text, pn in chunk_rows
            ]

            try:
                uploaded_at = datetime.fromisoformat(uploaded_at_str)
            except ValueError:
                uploaded_at = datetime.now(timezone.utc)

            meta = DocumentMeta(
                document_id=doc_id,
                filename=filename,
                n_pages=n_pages,
                n_chunks=n_chunks,
                uploaded_at=uploaded_at,
            )
            self._docs[doc_id] = StoredDocument(
                meta=meta,
                chunks=chunks,
                index=TfidfRetrievalIndex(chunks),
            )
            restored += 1

        if restored:
            logger.info(
                "Restored %d document(s) from %s", restored, self._db_path
            )
