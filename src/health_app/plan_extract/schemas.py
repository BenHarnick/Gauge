"""Pydantic schemas for the plan-extraction pipeline.

A ``PlanDraft`` holds the fields the LLM found in an uploaded PDF.  Every
field is nullable -- ``None`` means the document either didn't mention it or
the parser couldn't read the answer.  The front-end shows the draft to the
user for review before it is committed as a live ``Plan``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from health_app.benefits.models import ServiceCategory


class FieldExtraction(BaseModel):
    """The result of extracting a single plan field from a document.

    Parameters
    ----------
    raw : str
        Raw LLM response text before parsing.
    value : int or float or None
        Parsed numeric value.  Cents for monetary fields; a fraction in
        ``[0, 1]`` for the coinsurance rate.  ``None`` when parsing failed.
    confident : bool
        ``True`` when a numeric value was successfully parsed from ``raw``.
    """

    model_config = ConfigDict(frozen=True)

    raw: str = Field(description="Raw LLM response before parsing.")
    value: int | float | None = Field(
        default=None,
        description=(
            "Parsed numeric value: cents for money fields, fraction for rates."
        ),
    )
    confident: bool = Field(
        default=False,
        description="True when a numeric value was successfully parsed.",
    )


class PlanDraft(BaseModel):
    """Extracted plan fields pending user confirmation.

    Every monetary field is in whole US cents.  ``None`` values indicate
    fields that could not be found or parsed from the source document; the
    front-end should prompt the user to fill them in manually.

    Parameters
    ----------
    deductible_cents : int or None
        Individual in-network annual deductible.
    out_of_pocket_max_cents : int or None
        Individual in-network annual out-of-pocket maximum.
    coinsurance_rate : float or None
        Member share after deductible, as a fraction in ``[0, 1]``
        (e.g. ``0.20`` means the member pays 20 %).
    copays_cents : dict[ServiceCategory, int]
        Flat copays by service category that replace deductible/coinsurance
        for that category.
    unresolved_fields : list[str]
        Names of fields that could not be parsed from the document.
    extraction_notes : list[str]
        Human-readable notes explaining what was or wasn't found.
    """

    model_config = ConfigDict(frozen=False)

    deductible_cents: int | None = None
    out_of_pocket_max_cents: int | None = None
    coinsurance_rate: float | None = None
    copays_cents: dict[ServiceCategory, int] = Field(default_factory=dict)
    unresolved_fields: list[str] = Field(
        default_factory=list,
        description="Field names that could not be parsed from the document.",
    )
    extraction_notes: list[str] = Field(
        default_factory=list,
        description="Human-readable notes about what was or wasn't found.",
    )
