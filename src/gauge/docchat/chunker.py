"""Sliding-window chunker that preserves page-level provenance.

Each chunk records which pages its text spans, so citations can point
back to the right pages in the source PDF.
"""

from __future__ import annotations

from gauge.docchat.schemas import Chunk

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150


def chunk_pages(
    pages: list[tuple[int, str]],
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Slice page text into overlapping chunks tagged with page numbers.

    Parameters
    ----------
    pages : list[tuple[int, str]]
        List of ``(page_number, text)`` pairs from :func:`extract_pages`.
    document_id : str
        Identifier stamped on every produced chunk.
    chunk_size : int, optional
        Target chunk length in characters. Default is 800.
    overlap : int, optional
        Number of characters adjacent chunks share, so a relevant span is
        not truncated at a boundary. Default is 150.

    Returns
    -------
    list[Chunk]
        Chunks in document order. Empty pages are silently skipped but
        their page numbers are still honoured when computing provenance.

    Raises
    ------
    ValueError
        If ``chunk_size <= overlap``, which would cause an infinite loop
        in the sliding window.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    # Concatenate page texts and remember where each page starts so we
    # can map any character offset back to its source page. A separator
    # is inserted between pages (but not after the last one) so the last
    # word of one page doesn't fuse with the first word of the next.
    combined_parts: list[str] = []
    page_starts: list[tuple[int, int]] = []  # (char_offset, page_number)
    cursor = 0
    for i, (page_number, text) in enumerate(pages):
        page_starts.append((cursor, page_number))
        combined_parts.append(text)
        cursor += len(text)
        is_last = i == len(pages) - 1
        if text and not text.endswith("\n") and not is_last:
            combined_parts.append("\n")
            cursor += 1
    combined = "".join(combined_parts)

    if not combined.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap
    while start < len(combined):
        end = min(start + chunk_size, len(combined))
        text = combined[start:end].strip()
        if text:
            pages_touched = _pages_for_span(start, end, page_starts)
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=text,
                    page_numbers=pages_touched,
                )
            )
            chunk_index += 1
        if end >= len(combined):
            break
        start += step
    return chunks


def _pages_for_span(
    span_start: int,
    span_end: int,
    page_starts: list[tuple[int, int]],
) -> list[int]:
    """Return the sorted page numbers a character span touches.

    Parameters
    ----------
    span_start : int
        Inclusive start offset in the concatenated document string.
    span_end : int
        Exclusive end offset in the concatenated document string.
    page_starts : list[tuple[int, int]]
        ``(char_offset, page_number)`` pairs marking where each page begins
        in the concatenated string.

    Returns
    -------
    list[int]
        Sorted list of 1-indexed page numbers whose text the span
        ``[span_start, span_end)`` overlaps.
    """
    pages: set[int] = set()
    for i, (offset, page_number) in enumerate(page_starts):
        next_offset = (
            page_starts[i + 1][0]
            if i + 1 < len(page_starts)
            else float("inf")
        )
        if offset < span_end and next_offset > span_start:
            pages.add(page_number)
    return sorted(pages)
