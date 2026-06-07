"""PDF text extraction with page provenance.

This is the dumb-but-reliable approach: pypdf's per-page text extraction.
Plan documents lean heavily on tables, which pypdf renders as best-effort
text. For richer table fidelity we can swap in pdfplumber later without
touching anything downstream.
"""

from __future__ import annotations

import io

from pypdf import PdfReader


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return ``(page_number, text)`` tuples extracted from a PDF, 1-indexed.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file contents.

    Returns
    -------
    list[tuple[int, str]]
        One entry per page. Pages with no extractable text return an empty
        string rather than being omitted, preserving the page count for
        downstream consumers.

    Raises
    ------
    ValueError
        If ``pdf_bytes`` cannot be parsed as a PDF.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        raise ValueError(f"Not a parseable PDF: {e}") from e

    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # Some PDFs have malformed pages; carry on with an empty page
            # rather than failing the entire upload.
            text = ""
        out.append((i, text))
    return out
