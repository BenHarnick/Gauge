"""In-memory document store with one TF-IDF index per document.

Persistence is intentionally out of scope for this prototype. Restarting
the backend wipes the uploaded corpus, and clients are expected to
re-upload as needed. A SQLite-backed store would slot in cleanly behind
the same public methods.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from health_app.docchat.index import TfidfRetrievalIndex
from health_app.docchat.schemas import Chunk, DocumentMeta


@dataclass
class _StoredDocument:
    meta: DocumentMeta
    chunks: list[Chunk]
    index: TfidfRetrievalIndex


class InMemoryDocumentStore:
    """Thread-safe in-memory store keyed by `document_id`."""

    def __init__(self) -> None:
        """Initialise an empty, thread-safe document store."""
        self._docs: dict[str, _StoredDocument] = {}
        self._lock = threading.Lock()

    def add(self, meta: DocumentMeta, chunks: list[Chunk]) -> None:
        """Insert (or replace) a document and build its retrieval index.

        Parameters
        ----------
        meta : DocumentMeta
            Metadata for the document.
        chunks : list[Chunk]
            Text chunks extracted from the document.
        """
        index = TfidfRetrievalIndex(chunks)
        with self._lock:
            self._docs[meta.document_id] = _StoredDocument(
                meta=meta, chunks=chunks, index=index
            )

    def get(self, document_id: str) -> _StoredDocument | None:
        """Return the stored document for ``document_id``, or ``None``.

        Parameters
        ----------
        document_id : str
            Identifier assigned at upload time.

        Returns
        -------
        _StoredDocument or None
            The stored document object, or ``None`` if not found.
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
        """Remove a document from the store.

        Parameters
        ----------
        document_id : str
            Identifier of the document to remove.

        Returns
        -------
        bool
            ``True`` if the document existed and was removed, ``False`` if
            it was not found.
        """
        with self._lock:
            return self._docs.pop(document_id, None) is not None
