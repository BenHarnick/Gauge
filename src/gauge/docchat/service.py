"""High-level orchestration for upload-then-ask flows."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from gauge.docchat.chunker import chunk_pages
from gauge.docchat.extractor import extract_pages
from gauge.docchat.llm import EchoLLM, LLMClient
from gauge.docchat.schemas import (
    ChatResponse,
    Citation,
    DocumentMeta,
)
from gauge.docchat.store import DocumentStore, InMemoryDocumentStore

CITATION_SNIPPET_LEN = 240


class DocumentChatService:
    """Glue between the store, retrieval index, and LLM client."""

    def __init__(
        self,
        store: DocumentStore | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        """Initialise the service with a document store and LLM client.

        Parameters
        ----------
        store : DocumentStore or None, optional
            Document store to use.  Any object satisfying the
            :class:`~gauge.docchat.store.DocumentStore` protocol is
            accepted — both :class:`~gauge.docchat.store.InMemoryDocumentStore`
            and :class:`~gauge.docchat.sqlite_store.SqliteDocumentStore`
            qualify.  A fresh :class:`InMemoryDocumentStore` is created
            when not supplied.
        llm : LLMClient or None, optional
            LLM backend for answering questions. Defaults to
            :class:`EchoLLM` when not supplied.
        """
        self.store: DocumentStore = store or InMemoryDocumentStore()
        self.llm = llm or EchoLLM()

    def upload_pdf(self, filename: str, pdf_bytes: bytes) -> DocumentMeta:
        """Extract, chunk, index, and persist a PDF.

        Parameters
        ----------
        filename : str
            Original filename stored in metadata and shown to clients.
        pdf_bytes : bytes
            Raw PDF file contents.

        Returns
        -------
        DocumentMeta
            Metadata for the newly stored document, including its assigned
            ``document_id``.

        Raises
        ------
        ValueError
            If the PDF cannot be parsed or contains no extractable text.
        """
        pages = extract_pages(pdf_bytes)
        if not pages:
            raise ValueError("PDF contained zero pages.")

        document_id = uuid.uuid4().hex[:12]
        chunks = chunk_pages(pages, document_id=document_id)
        if not chunks:
            raise ValueError(
                "PDF parsed but produced no extractable text. It may be a "
                "scanned document that needs OCR."
            )

        meta = DocumentMeta(
            document_id=document_id,
            filename=filename,
            n_pages=len(pages),
            n_chunks=len(chunks),
            uploaded_at=datetime.now(timezone.utc),
        )
        self.store.add(meta, chunks)
        return meta

    def ask(
        self, document_id: str, question: str, top_k: int = 4
    ) -> ChatResponse:
        """Answer a question against a previously uploaded document.

        Parameters
        ----------
        document_id : str
            Identifier returned by :meth:`upload_pdf`.
        question : str
            Free-text question from the user.
        top_k : int, optional
            Number of chunks to retrieve and pass to the LLM. Default is 4.

        Returns
        -------
        ChatResponse
            Answer text, citations, and the LLM identifier used.

        Raises
        ------
        KeyError
            If no document with ``document_id`` exists in the store.
        """
        stored = self.store.get(document_id)
        if stored is None:
            raise KeyError(document_id)

        results = stored.index.search(question, k=top_k)
        contexts = [chunk for chunk, _ in results]
        answer = self.llm.answer(question, contexts)
        citations = [
            Citation(
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                page_numbers=chunk.page_numbers,
                snippet=_short(chunk.text),
            )
            for chunk in contexts
        ]
        return ChatResponse(
            document_id=document_id,
            question=question,
            answer=answer,
            citations=citations,
            llm_used=self.llm.name,
        )


def _short(text: str) -> str:
    """Produce a single-line preview of a chunk for the UI.

    Parameters
    ----------
    text : str
        Raw chunk text, potentially multiline.

    Returns
    -------
    str
        Whitespace-collapsed string truncated to at most
        ``CITATION_SNIPPET_LEN`` characters, with ``"..."`` appended if
        truncated.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= CITATION_SNIPPET_LEN:
        return cleaned
    return cleaned[: CITATION_SNIPPET_LEN - 1].rstrip() + "..."
