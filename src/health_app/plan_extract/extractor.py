"""LLM-powered extraction of structured plan fields from a PDF document.

The extractor reuses the same retrieval + LLM pipeline that powers the
document Q&A chat.  For each plan field it asks a targeted single-answer
question, then runs a regex parser over the LLM's response to pull out a
dollar amount or percentage.

Fields that cannot be parsed are left as ``None`` in the returned
``PlanDraft`` with a note explaining what the LLM said, so the user can
fill them in manually via the confirmation form.
"""

from __future__ import annotations

import re
import uuid

from health_app.benefits.models import Plan, ServiceCategory
from health_app.docchat.index import TfidfRetrievalIndex
from health_app.docchat.llm import LLMClient
from health_app.docchat.schemas import Chunk
from health_app.plan_extract.schemas import FieldExtraction, PlanDraft

# ---------------------------------------------------------------------------
# Extraction questions
# ---------------------------------------------------------------------------

# One question per field.  The phrasing instructs the LLM to return a single
# numeric answer so the downstream regex has the best possible signal.
_FIELD_QUESTIONS: dict[str, str] = {
    "deductible": (
        "What is the individual in-network annual deductible? "
        "Reply with the dollar amount only, for example '$1,500'."
    ),
    "oop_max": (
        "What is the individual in-network annual out-of-pocket maximum? "
        "Reply with the dollar amount only, for example '$5,000'."
    ),
    "coinsurance": (
        "After the deductible is met, what percentage of costs does the "
        "member pay (coinsurance rate)? "
        "Reply with the percentage only, for example '20%'."
    ),
    "copay_office_visit": (
        "What is the copay for a primary care or office visit? "
        "Reply with the dollar amount only, for example '$30'."
    ),
    "copay_specialist": (
        "What is the specialist visit copay? "
        "Reply with the dollar amount only, for example '$50'."
    ),
    "copay_urgent_care": (
        "What is the urgent care copay? "
        "Reply with the dollar amount only, for example '$75'."
    ),
    "copay_generic_drug": (
        "What is the copay for a generic prescription drug? "
        "Reply with the dollar amount only, for example '$10'."
    ),
}

# Maps extraction field names to the ServiceCategory enum used by the
# benefits engine.
_COPAY_FIELD_MAP: dict[str, ServiceCategory] = {
    "copay_office_visit": ServiceCategory.OFFICE_VISIT,
    "copay_specialist": ServiceCategory.SPECIALIST,
    "copay_urgent_care": ServiceCategory.URGENT_CARE,
    "copay_generic_drug": ServiceCategory.GENERIC_DRUG,
}

_DEFAULT_TOP_K = 4
_NOTE_RAW_CLIP = 120  # characters of raw LLM text shown in extraction notes


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_dollars(text: str) -> int | None:
    """Parse the first dollar-amount from an LLM answer.

    Parameters
    ----------
    text : str
        Raw LLM response string.

    Returns
    -------
    int or None
        Amount in whole US cents, or ``None`` if no numeric value was found.
    """
    cleaned = text.replace(",", "")
    match = re.search(r"\$?\s*(\d+(?:\.\d{1,2})?)", cleaned)
    if match:
        return int(round(float(match.group(1)) * 100))
    return None


def _parse_percent(text: str) -> float | None:
    """Parse the first percentage from an LLM answer.

    Parameters
    ----------
    text : str
        Raw LLM response string.

    Returns
    -------
    float or None
        Rate as a fraction in ``[0, 1]``, or ``None`` if not found.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1)) / 100.0
    return None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class PlanExtractor:
    """Extracts structured plan fields from an already-indexed PDF document.

    Reuses the TF-IDF retrieval index and LLM backend from the document chat
    service to ask targeted single-answer questions about each plan field,
    then parses the responses into a ``PlanDraft``.  Fields that cannot be
    parsed are left as ``None`` with a human-readable note.
    """

    def __init__(self, llm: LLMClient) -> None:
        """Initialise the extractor with an LLM backend.

        Parameters
        ----------
        llm : LLMClient
            LLM client used to answer each extraction question.  Any
            implementation conforming to the ``LLMClient`` protocol works,
            including ``EchoLLM`` for testing without an API key.
        """
        self._llm = llm

    def extract(
        self,
        index: TfidfRetrievalIndex,
        top_k: int = _DEFAULT_TOP_K,
    ) -> PlanDraft:
        """Run all field-extraction questions against a document index.

        Parameters
        ----------
        index : TfidfRetrievalIndex
            Retrieval index built from the uploaded document.
        top_k : int, optional
            Number of chunks to retrieve per question. Default is 4.

        Returns
        -------
        PlanDraft
            Extracted plan fields.  ``None`` values indicate fields the LLM
            could not locate or that the parser could not read.
        """
        draft = PlanDraft()
        notes: list[str] = []
        unresolved: list[str] = []

        for field, question in _FIELD_QUESTIONS.items():
            extraction = self._ask_field(index, field, question, top_k)
            self._apply(draft, field, extraction)

            if not extraction.confident:
                unresolved.append(field)
                clip = extraction.raw[:_NOTE_RAW_CLIP]
                notes.append(
                    f"Could not parse {field!r} -- document said: {clip!r}"
                )

        draft.unresolved_fields = unresolved
        draft.extraction_notes = notes
        return draft

    def draft_to_plan(
        self,
        draft: PlanDraft,
        plan_id: str | None = None,
        name: str | None = None,
    ) -> Plan:
        """Convert a confirmed ``PlanDraft`` into a usable ``Plan`` object.

        The caller is responsible for ensuring all required fields are
        non-``None`` before calling this (i.e. after the user has reviewed
        and filled in any gaps on the confirmation form).

        Parameters
        ----------
        draft : PlanDraft
            A draft whose required numeric fields have been confirmed by the
            user and are non-``None``.
        plan_id : str or None, optional
            Identifier for the new plan.  A random 12-character hex ID is
            used when not supplied.
        name : str or None, optional
            Human-readable plan name.  Defaults to ``"My Plan"``.

        Returns
        -------
        Plan
            A fully constructed ``Plan`` ready for the benefits engine.

        Raises
        ------
        ValueError
            If ``deductible_cents``, ``out_of_pocket_max_cents``, or
            ``coinsurance_rate`` are ``None`` on ``draft``.
        """
        missing = [
            field
            for field, val in [
                ("deductible_cents", draft.deductible_cents),
                ("out_of_pocket_max_cents", draft.out_of_pocket_max_cents),
                ("coinsurance_rate", draft.coinsurance_rate),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"Cannot create a Plan -- required fields are still missing: "
                f"{missing}. Ask the user to fill them in manually."
            )
        return Plan(
            plan_id=plan_id or uuid.uuid4().hex[:12],
            name=name or "My Plan",
            deductible_cents=draft.deductible_cents,  # type: ignore[arg-type]
            out_of_pocket_max_cents=draft.out_of_pocket_max_cents,  # type: ignore[arg-type]
            coinsurance_rate=draft.coinsurance_rate,  # type: ignore[arg-type]
            copays_cents=dict(draft.copays_cents),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ask_field(
        self,
        index: TfidfRetrievalIndex,
        field: str,
        question: str,
        top_k: int,
    ) -> FieldExtraction:
        """Retrieve relevant chunks, ask the LLM, and parse the answer.

        Parameters
        ----------
        index : TfidfRetrievalIndex
            Document index to search.
        field : str
            Field name used only for deciding whether to parse dollars or a
            percentage.
        question : str
            Extraction question to send to the LLM.
        top_k : int
            Number of chunks to retrieve.

        Returns
        -------
        FieldExtraction
            Raw LLM response with the parsed value and confidence flag.
        """
        results = index.search(question, k=top_k)
        chunks: list[Chunk] = [chunk for chunk, _ in results]
        raw = self._llm.answer(question, chunks)

        if field == "coinsurance":
            value = _parse_percent(raw)
        else:
            value = _parse_dollars(raw)

        return FieldExtraction(raw=raw, value=value, confident=value is not None)

    @staticmethod
    def _apply(draft: PlanDraft, field: str, extraction: FieldExtraction) -> None:
        """Write a parsed extraction result into the appropriate draft field.

        Parameters
        ----------
        draft : PlanDraft
            The draft to mutate in place.
        field : str
            The field key from ``_FIELD_QUESTIONS``.
        extraction : FieldExtraction
            The parsed extraction result for that field.
        """
        if not extraction.confident:
            return
        v = extraction.value
        if field == "deductible":
            draft.deductible_cents = int(v)  # type: ignore[arg-type]
        elif field == "oop_max":
            draft.out_of_pocket_max_cents = int(v)  # type: ignore[arg-type]
        elif field == "coinsurance":
            draft.coinsurance_rate = float(v)  # type: ignore[arg-type]
        elif field in _COPAY_FIELD_MAP:
            draft.copays_cents[_COPAY_FIELD_MAP[field]] = int(v)  # type: ignore[arg-type]
