"""Unit tests for the plan_extract package.

Covers the two regex parsers (``_parse_dollars``, ``_parse_percent``) and the
``PlanExtractor`` class, including field extraction, draft-to-plan conversion,
and the ``_apply`` static method.

All LLM calls are replaced by a stub so no API key is required.
"""

from __future__ import annotations

import pytest

from health_app.benefits.models import Plan, ServiceCategory
from health_app.docchat.index import TfidfRetrievalIndex
from health_app.docchat.llm import EchoLLM
from health_app.docchat.schemas import Chunk
from health_app.plan_extract.extractor import (
    PlanExtractor,
    _parse_dollars,
    _parse_percent,
)
from health_app.plan_extract.schemas import FieldExtraction, PlanDraft

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_dollars
# ---------------------------------------------------------------------------


class TestParseDollars:
    def test_plain_dollar_amount(self) -> None:
        assert _parse_dollars("$1,500") == 150_000

    def test_no_dollar_sign(self) -> None:
        assert _parse_dollars("1500") == 150_000

    def test_with_cents(self) -> None:
        assert _parse_dollars("$25.00") == 2_500

    def test_embedded_in_sentence(self) -> None:
        assert _parse_dollars("The deductible is $3,000 per year.") == 300_000

    def test_comma_separated_thousands(self) -> None:
        assert _parse_dollars("$10,000") == 1_000_000

    def test_zero(self) -> None:
        assert _parse_dollars("$0") == 0

    def test_no_match_returns_none(self) -> None:
        assert _parse_dollars("I could not find a dollar amount.") is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_dollars("") is None

    def test_picks_first_amount(self) -> None:
        # Should pick the first numeric value found.
        result = _parse_dollars("individual $1,000 / family $2,000")
        assert result == 100_000

    def test_dollar_with_space(self) -> None:
        assert _parse_dollars("$ 500") == 50_000


# ---------------------------------------------------------------------------
# _parse_percent
# ---------------------------------------------------------------------------


class TestParsePercent:
    def test_integer_percent(self) -> None:
        assert _parse_percent("20%") == pytest.approx(0.20)

    def test_decimal_percent(self) -> None:
        assert _parse_percent("12.5%") == pytest.approx(0.125)

    def test_embedded_in_sentence(self) -> None:
        assert _parse_percent("You pay 30% coinsurance after deductible.") == pytest.approx(0.30)

    def test_zero_percent(self) -> None:
        assert _parse_percent("0%") == pytest.approx(0.0)

    def test_no_match_returns_none(self) -> None:
        assert _parse_percent("No percentage found here.") is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_percent("") is None

    def test_picks_first_percent(self) -> None:
        # First percentage wins.
        result = _parse_percent("20% in-network, 40% out-of-network")
        assert result == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index(texts: list[str]) -> TfidfRetrievalIndex:
    """Build a minimal TF-IDF index from raw text strings."""
    chunks = [
        Chunk(
            text=t,
            chunk_index=i,
            document_id="test-doc",
            page_numbers=[1],
        )
        for i, t in enumerate(texts)
    ]
    return TfidfRetrievalIndex(chunks)


class _FixedAnswerLLM:
    """LLM stub that returns a pre-configured answer for every question."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def answer(self, question: str, chunks: list[Chunk]) -> str:  # noqa: ARG002
        return self._answer


# ---------------------------------------------------------------------------
# PlanExtractor.extract
# ---------------------------------------------------------------------------


class TestPlanExtractorExtract:
    def _extractor_with_answer(self, answer: str) -> PlanExtractor:
        return PlanExtractor(llm=_FixedAnswerLLM(answer))  # type: ignore[arg-type]

    def test_extracts_deductible(self) -> None:
        index = _make_index(["Annual deductible is $1,500."])
        extractor = self._extractor_with_answer("$1,500")
        draft = extractor.extract(index)
        assert draft.deductible_cents == 150_000

    def test_extracts_oop_max(self) -> None:
        index = _make_index(["Out-of-pocket maximum: $5,000."])
        extractor = self._extractor_with_answer("$5,000")
        draft = extractor.extract(index)
        assert draft.out_of_pocket_max_cents == 500_000

    def test_extracts_coinsurance(self) -> None:
        index = _make_index(["Coinsurance rate 20%."])
        extractor = self._extractor_with_answer("20%")
        draft = extractor.extract(index)
        assert draft.coinsurance_rate == pytest.approx(0.20)

    def test_extracts_copay(self) -> None:
        index = _make_index(["Office visit copay $30."])
        extractor = self._extractor_with_answer("$30")
        draft = extractor.extract(index)
        assert draft.copays_cents.get(ServiceCategory.OFFICE_VISIT) == 3_000

    def test_unresolvable_fields_flagged(self) -> None:
        index = _make_index(["No relevant information here."])
        extractor = self._extractor_with_answer("I could not find an answer.")
        draft = extractor.extract(index)
        assert "deductible" in draft.unresolved_fields
        assert "oop_max" in draft.unresolved_fields
        assert "coinsurance" in draft.unresolved_fields
        assert len(draft.extraction_notes) == len(draft.unresolved_fields)

    def test_partial_extraction(self) -> None:
        """When only some fields are parseable, others remain None."""
        responses = iter(["$2,000", "I don't know", "15%",
                          "I don't know", "I don't know", "I don't know", "I don't know"])

        class _SequentialLLM:
            def answer(self, question: str, chunks: list[Chunk]) -> str:  # noqa: ARG002
                return next(responses)

        index = _make_index(["Some plan text."])
        extractor = PlanExtractor(llm=_SequentialLLM())  # type: ignore[arg-type]
        draft = extractor.extract(index)
        assert draft.deductible_cents == 200_000
        assert draft.out_of_pocket_max_cents is None
        assert draft.coinsurance_rate == pytest.approx(0.15)

    def test_echo_llm_produces_draft(self, sample_plan_pdf_bytes: bytes) -> None:
        """EchoLLM always returns text; extraction should complete without error."""
        from health_app.docchat.chunker import chunk_pages
        from health_app.docchat.extractor import extract_pages

        pages = extract_pages(sample_plan_pdf_bytes)
        chunks = chunk_pages(pages, "echo-doc")
        index = TfidfRetrievalIndex(chunks)
        extractor = PlanExtractor(llm=EchoLLM())
        draft = extractor.extract(index)
        # EchoLLM returns chunk text, which may or may not contain a parseable
        # value — we just assert the method completes and returns a PlanDraft.
        assert isinstance(draft, PlanDraft)


# ---------------------------------------------------------------------------
# PlanExtractor.draft_to_plan
# ---------------------------------------------------------------------------


class TestDraftToPlan:
    def _complete_draft(self) -> PlanDraft:
        return PlanDraft(
            deductible_cents=100_000,
            out_of_pocket_max_cents=500_000,
            coinsurance_rate=0.20,
            copays_cents={ServiceCategory.OFFICE_VISIT: 2_500},
        )

    def test_returns_plan_with_correct_fields(self) -> None:
        extractor = PlanExtractor(llm=EchoLLM())
        plan = extractor.draft_to_plan(self._complete_draft(), plan_id="test", name="Test Plan")
        assert isinstance(plan, Plan)
        assert plan.deductible_cents == 100_000
        assert plan.out_of_pocket_max_cents == 500_000
        assert plan.coinsurance_rate == pytest.approx(0.20)
        assert plan.copays_cents[ServiceCategory.OFFICE_VISIT] == 2_500

    def test_auto_assigns_plan_id_when_not_supplied(self) -> None:
        extractor = PlanExtractor(llm=EchoLLM())
        plan = extractor.draft_to_plan(self._complete_draft())
        assert plan.plan_id  # non-empty string
        assert plan.name == "My Plan"

    def test_custom_name(self) -> None:
        extractor = PlanExtractor(llm=EchoLLM())
        plan = extractor.draft_to_plan(self._complete_draft(), name="Acme PPO")
        assert plan.name == "Acme PPO"

    def test_raises_when_deductible_missing(self) -> None:
        draft = PlanDraft(
            out_of_pocket_max_cents=500_000,
            coinsurance_rate=0.20,
        )
        extractor = PlanExtractor(llm=EchoLLM())
        with pytest.raises(ValueError, match="deductible_cents"):
            extractor.draft_to_plan(draft)

    def test_raises_when_oop_max_missing(self) -> None:
        draft = PlanDraft(
            deductible_cents=100_000,
            coinsurance_rate=0.20,
        )
        extractor = PlanExtractor(llm=EchoLLM())
        with pytest.raises(ValueError, match="out_of_pocket_max_cents"):
            extractor.draft_to_plan(draft)

    def test_raises_when_coinsurance_missing(self) -> None:
        draft = PlanDraft(
            deductible_cents=100_000,
            out_of_pocket_max_cents=500_000,
        )
        extractor = PlanExtractor(llm=EchoLLM())
        with pytest.raises(ValueError, match="coinsurance_rate"):
            extractor.draft_to_plan(draft)


# ---------------------------------------------------------------------------
# PlanExtractor._apply (static helper)
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_deductible(self) -> None:
        draft = PlanDraft()
        extraction = FieldExtraction(raw="$1,000", value=100_000, confident=True)
        PlanExtractor._apply(draft, "deductible", extraction)
        assert draft.deductible_cents == 100_000

    def test_apply_oop_max(self) -> None:
        draft = PlanDraft()
        extraction = FieldExtraction(raw="$5,000", value=500_000, confident=True)
        PlanExtractor._apply(draft, "oop_max", extraction)
        assert draft.out_of_pocket_max_cents == 500_000

    def test_apply_coinsurance(self) -> None:
        draft = PlanDraft()
        extraction = FieldExtraction(raw="20%", value=0.20, confident=True)
        PlanExtractor._apply(draft, "coinsurance", extraction)
        assert draft.coinsurance_rate == pytest.approx(0.20)

    def test_apply_copay_office_visit(self) -> None:
        draft = PlanDraft()
        extraction = FieldExtraction(raw="$30", value=3_000, confident=True)
        PlanExtractor._apply(draft, "copay_office_visit", extraction)
        assert draft.copays_cents[ServiceCategory.OFFICE_VISIT] == 3_000

    def test_apply_skips_when_not_confident(self) -> None:
        draft = PlanDraft()
        extraction = FieldExtraction(raw="not found", value=None, confident=False)
        PlanExtractor._apply(draft, "deductible", extraction)
        assert draft.deductible_cents is None

    def test_apply_unknown_field_is_noop(self) -> None:
        """An unrecognised field name should silently do nothing."""
        draft = PlanDraft()
        extraction = FieldExtraction(raw="$100", value=10_000, confident=True)
        PlanExtractor._apply(draft, "unknown_field", extraction)
        assert draft.deductible_cents is None  # unchanged


# ---------------------------------------------------------------------------
# PlanDraft schema
# ---------------------------------------------------------------------------


class TestPlanDraftSchema:
    def test_defaults_are_none(self) -> None:
        draft = PlanDraft()
        assert draft.deductible_cents is None
        assert draft.out_of_pocket_max_cents is None
        assert draft.coinsurance_rate is None
        assert draft.copays_cents == {}
        assert draft.unresolved_fields == []
        assert draft.extraction_notes == []

    def test_round_trip_serialisation(self) -> None:
        draft = PlanDraft(
            deductible_cents=150_000,
            out_of_pocket_max_cents=600_000,
            coinsurance_rate=0.15,
            copays_cents={ServiceCategory.OFFICE_VISIT: 2_500},
        )
        restored = PlanDraft.model_validate(draft.model_dump())
        assert restored.deductible_cents == 150_000
        assert restored.coinsurance_rate == pytest.approx(0.15)
