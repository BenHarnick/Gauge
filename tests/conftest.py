"""Shared test fixtures.

The cost predictor is fit once per session because training takes more
time than the rest of the suite combined. Every test that needs a model
reuses this single instance.
"""

from __future__ import annotations

import pytest

from gauge.benefits.models import (
    Member,
    Plan,
    Procedure,
    ServiceCategory,
)
from gauge.benefits.repository import InMemoryRepository
from gauge.benefits.seed import build_default_repository
from gauge.predictor.dataset import generate_synthetic_dataset
from gauge.predictor.model import CostPredictor
from gauge.predictor.schemas import PredictionFeatures


# --- benefits fixtures ---------------------------------------------------


@pytest.fixture
def ppo_gold() -> Plan:
    """A typical PPO plan with office-visit copays."""
    return Plan(
        plan_id="ppo_gold",
        name="PPO Gold",
        deductible_cents=100_000,
        out_of_pocket_max_cents=500_000,
        coinsurance_rate=0.20,
        copays_cents={
            ServiceCategory.OFFICE_VISIT: 2_500,
            ServiceCategory.SPECIALIST: 5_000,
        },
    )


@pytest.fixture
def hdhp_silver() -> Plan:
    """A high-deductible plan with no copays."""
    return Plan(
        plan_id="hdhp_silver",
        name="HDHP Silver",
        deductible_cents=300_000,
        out_of_pocket_max_cents=700_000,
        coinsurance_rate=0.20,
    )


@pytest.fixture
def fresh_member(ppo_gold: Plan) -> Member:
    return Member(
        member_id="m1",
        name="Alex Carter",
        plan_id=ppo_gold.plan_id,
    )


@pytest.fixture
def near_oop_member(ppo_gold: Plan) -> Member:
    return Member(
        member_id="m_oop",
        name="OOP Near",
        plan_id=ppo_gold.plan_id,
        ytd_deductible_cents=100_000,
        ytd_out_of_pocket_cents=490_000,
    )


@pytest.fixture
def office_visit() -> Procedure:
    return Procedure(
        code="99213",
        description="Office visit",
        category=ServiceCategory.OFFICE_VISIT,
        in_network_rate_cents=15_000,
        billed_amount_cents=22_000,
    )


@pytest.fixture
def imaging_procedure() -> Procedure:
    return Procedure(
        code="73721",
        description="MRI knee",
        category=ServiceCategory.IMAGING,
        in_network_rate_cents=85_000,
        billed_amount_cents=240_000,
    )


@pytest.fixture
def surgery_procedure() -> Procedure:
    return Procedure(
        code="29881",
        description="Knee arthroscopy",
        category=ServiceCategory.SURGERY,
        in_network_rate_cents=540_000,
        billed_amount_cents=1_200_000,
    )


@pytest.fixture
def seeded_repository() -> InMemoryRepository:
    """A fresh copy of the default seeded repository for each test."""
    return build_default_repository()


# --- predictor fixtures --------------------------------------------------


@pytest.fixture(scope="session")
def trained_predictor() -> CostPredictor:
    """Fit a predictor once per session on a modest synthetic dataset.

    The dataset is intentionally smaller than the production seed so the
    test suite stays snappy. Quality is good enough for the assertions
    we make (directional sanity checks, not strict accuracy).
    """
    df = generate_synthetic_dataset(n_rows=800, seed=42)
    return CostPredictor().fit(df)


@pytest.fixture
def baseline_features() -> PredictionFeatures:
    """A reasonable middle-of-the-road feature vector."""
    return PredictionFeatures(
        age=35,
        sex="female",
        bmi=27.5,
        children=1,
        smoker="no",
        region="northeast",
    )


# --- docchat fixtures ----------------------------------------------------


def _make_pdf(pages: list[str]) -> bytes:
    """Build an in-memory PDF for testing. One page per supplied string."""
    from io import BytesIO

    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=LETTER)
    for body in pages:
        text_obj = pdf.beginText(72, 720)
        for line in body.split("\n"):
            text_obj.textLine(line)
        pdf.drawText(text_obj)
        pdf.showPage()
    pdf.save()
    return buf.getvalue()


@pytest.fixture
def sample_plan_pdf_bytes() -> bytes:
    """A tiny three-page "plan document" PDF with known content per page."""
    pages = [
        "Summary of Benefits and Coverage\n"
        "Plan name: Acme PPO Gold\n"
        "Annual deductible: $1,000 individual / $2,000 family.\n"
        "Annual out-of-pocket maximum: $5,000 individual.",
        "Cost sharing details.\n"
        "Office visit copay: $25 for primary care, $50 for specialists.\n"
        "Coinsurance after deductible: 20%.\n"
        "Emergency room copay: $250, waived if admitted.",
        "Prescription drug coverage.\n"
        "Generic drugs: $10 copay.\n"
        "Preferred brand: $40 copay.\n"
        "Specialty drugs: 30% coinsurance up to $250 per prescription.",
    ]
    return _make_pdf(pages)
