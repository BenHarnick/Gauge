"""Unit tests for the chunker."""

from __future__ import annotations

import pytest

from gauge.docchat.chunker import chunk_pages

pytestmark = pytest.mark.unit


def test_chunker_preserves_text_and_assigns_pages() -> None:
    pages = [
        (1, "A" * 500),
        (2, "B" * 500),
        (3, "C" * 500),
    ]
    chunks = chunk_pages(
        pages, document_id="d", chunk_size=400, overlap=50
    )
    assert chunks, "expected at least one chunk"
    # Every chunk is at most chunk_size characters.
    for c in chunks:
        assert len(c.text) <= 400
    # Page numbers are 1-indexed and only contain pages that actually exist.
    for c in chunks:
        assert all(1 <= p <= 3 for p in c.page_numbers)
    # The first chunk touches page 1.
    assert 1 in chunks[0].page_numbers


def test_chunker_overlap_creates_overlap() -> None:
    pages = [(1, "abcdefghij" * 50)]  # 500 chars on a single page
    chunks = chunk_pages(
        pages, document_id="d", chunk_size=200, overlap=50
    )
    # With chunk_size=200, overlap=50, step=150, we expect at least three
    # chunks and adjacent ones must share their overlap region.
    assert len(chunks) >= 2
    tail = chunks[0].text[-30:]
    assert tail in chunks[1].text


def test_chunker_rejects_overlap_ge_chunk_size() -> None:
    with pytest.raises(ValueError, match="greater than overlap"):
        chunk_pages([(1, "abc")], document_id="d", chunk_size=10, overlap=10)


def test_chunker_returns_empty_for_blank_input() -> None:
    assert chunk_pages([(1, "")], document_id="d") == []


def test_chunker_chunks_a_real_extracted_pdf(
    sample_plan_pdf_bytes: bytes,
) -> None:
    from gauge.docchat.extractor import extract_pages

    pages = extract_pages(sample_plan_pdf_bytes)
    chunks = chunk_pages(pages, document_id="d", chunk_size=400, overlap=80)
    assert chunks
    # The deductible content lives on page 1; the first chunk should
    # include it and cite page 1.
    first_chunk_text = chunks[0].text.lower()
    assert "deductible" in first_chunk_text
    assert 1 in chunks[0].page_numbers
