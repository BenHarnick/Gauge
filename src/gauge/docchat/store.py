"""Document store interface, in-memory implementation, and shared types.

``StoredDocument`` and the ``DocumentStore`` Protocol are defined here so
both the in-memory and SQLite implementations can share them without
circular imports.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from gauge.docchat.index import TfidfRetrievalIndex
from gauge.docchat.schemas import Chunk, DocumentMeta


@dataclass
class StoredDocument:
    """A document held in a store, with its chunks and retrieval index.

    Parameters
    ----------
    meta : DocumentMeta
        Document metadata (ID, filename, page/chunk counts, upload time).
    chunks : list[Chunk]
        Text chunks extracted from the document.
    index : TfidfRetrievalIndex
        Pre-built retrieval index for answering questions against this doc.
    """

    meta: DocumentMeta
    chunks: list[Chunk]
    index: TfidfRetrievalIndex


# Keep the old private name as an alias for any code that used it directly.
_StoredDocument = StoredDocument


@runtime_checkable
class DocumentStore(Protocol):
    """Structural protocol satisfied by any document store implementation.

    Both :class:`InMemoryDocumentStore` and
    :class:`~gauge.docchat.sqlite_store.SqliteDocumentStore` satisfy
    this protocol without explicit inheritance.
    """

    def add(self, meta: DocumentMeta, chunks: list[Chunk]) -> None:
        """Insert or replace a document and build its retrieval index."""
        ...

    def get(self, document_id: str) -> StoredDocument | None:
        """Return the stored document for ``document_id``, or ``None``."""
        ...

    def list_meta(self) -> list[DocumentMeta]:
        """Return metadata for all stored documents."""
        ...

    def delete(self, document_id: str) -> bool:
        """Remove a document; return ``True`` if it existed."""
        ...


class InMemoryDocumentStore:
    """Thread-safe in-memory store keyed by ``document_id``.

    Sessions are lost when the server restarts. A SQLite-backed store
    (:class:`~gauge.docchat.sqlite_store.SqliteDocumentStore`) uses
    the same public interface and survives restarts.
    """

    def __init__(self) -> None:
        """Initialise an empty, thread-safe document store."""
        self._docs: dict[str, StoredDocument] = {}
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
