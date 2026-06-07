"""Unit tests for the TF-IDF retrieval index."""

from __future__ import annotations

import pytest

from gauge.docchat.index import TfidfRetrievalIndex
from gauge.docchat.schemas import Chunk

pytestmark = pytest.mark.unit


def _chunk(i: int, text: str, pages: list[int]) -> Chunk:
    return Chunk(
        document_id="d", chunk_index=i, text=text, page_numbers=pages
    )


def test_index_requires_non_empty_chunks() -> None:
    with pytest.raises(ValueError):
        TfidfRetrievalIndex([])


def test_index_returns_topk_in_relevance_order() -> None:
    chunks = [
        _chunk(0, "Annual deductible is one thousand dollars individual.", [1]),
        _chunk(1, "Office visit copay is twenty five dollars.", [2]),
        _chunk(2, "Generic drug copay is ten dollars per prescription.", [3]),
    ]
    index = TfidfRetrievalIndex(chunks)
    results = index.search("how much is the deductible", k=2)
    assert len(results) >= 1
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_index == 0
    assert top_score > 0.0


def test_index_returns_empty_for_blank_query() -> None:
    # Use a non-stop-word so the TF-IDF vocabulary isn't empty.
    chunks = [_chunk(0, "deductible details apply to all members", [1])]
    index = TfidfRetrievalIndex(chunks)
    assert index.search("   ") == []


def test_index_filters_zero_similarity() -> None:
    chunks = [
        _chunk(0, "deductible details", [1]),
        _chunk(1, "specialty drugs coinsurance", [2]),
    ]
    index = TfidfRetrievalIndex(chunks)
    results = index.search("xyzzy unrelated query", k=5)
    # All scores should be zero (no overlap), so search returns nothing.
    assert results == []


def test_size_property_matches_chunk_count() -> None:
    chunks = [
        _chunk(0, "deductible applies annually", [1]),
        _chunk(1, "copay required per visit", [2]),
        _chunk(2, "coinsurance twenty percent", [3]),
    ]
    index = TfidfRetrievalIndex(chunks)
    assert index.size == 3


def test_stop_words_only_content_falls_back_gracefully() -> None:
    """When all tokens are English stop words the index rebuilds without the filter."""
    # These are all common English stop words that TfidfVectorizer would strip.
    chunks = [_chunk(0, "a the is in on at by", [1])]
    # Should not raise; the fallback vectorizer handles it.
    index = TfidfRetrievalIndex(chunks)
    assert index.size == 1
    # Search should still return something (vocabulary is now the stop words).
    results = index.search("a the", k=1)
    assert isinstance(results, list)
