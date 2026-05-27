"""Unified FastAPI application factory.

The factory takes both the benefits-side `CatalogRepository` and a
fitted `CostPredictor`, then wires both routers onto a single app. Tests
inject fresh dependencies per case via override.
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from health_app.docchat.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentMeta,
    UploadResponse,
)
from health_app.docchat.service import DocumentChatService

from health_app.benefits.calculator import estimate_cost_share
from health_app.benefits.models import (
    EstimateRequest,
    EstimateResult,
    Member,
    Plan,
    Procedure,
)
from health_app.benefits.repository import CatalogRepository
from health_app.predictor.annual_cost import (
    AnnualPlanShare,
    apply_plan_to_annual_spend,
)
from health_app.predictor.model import CostPrediction, CostPredictor
from health_app.predictor.schemas import PredictionFeatures
from health_app.predictor.whatif import (
    SWEEPABLE_FEATURES,
    SweepValue,
    WhatIfResponse,
    sweep,
)


class PredictRequest(BaseModel):
    """POST /predict body."""

    features: PredictionFeatures
    plan_id: str | None = None


class PredictResponse(BaseModel):
    """POST /predict response."""

    prediction: CostPrediction
    annual_plan_share_median: AnnualPlanShare | None = None
    annual_plan_share_mean: AnnualPlanShare | None = None


class WhatIfRequest(BaseModel):
    """POST /whatif body."""

    baseline: PredictionFeatures
    feature: Literal["age", "sex", "bmi", "children", "smoker", "region"]
    values: list[SweepValue]
    plan_id: str | None = None


MAX_PDF_BYTES = 25 * 1024 * 1024  # 25 MB upload cap


def create_app(
    repository: CatalogRepository,
    predictor: CostPredictor,
    chat_service: DocumentChatService | None = None,
) -> FastAPI:
    """Build a FastAPI app wired to the supplied dependencies.

    Parameters
    ----------
    repository : CatalogRepository
        Catalog source for plans, members, and procedures.
    predictor : CostPredictor
        A *fitted* predictor. Callers are responsible for training or
        loading the model before constructing the app.
    chat_service : DocumentChatService or None, optional
        Document-chat orchestrator. A fresh instance is created when not
        supplied; tests can inject their own.

    Returns
    -------
    FastAPI
        A configured FastAPI instance ready to be served or wrapped in a
        ``TestClient``.

    Raises
    ------
    ValueError
        If ``predictor`` has not been fitted yet.
    """
    if not predictor.is_fitted:
        raise ValueError(
            "CostPredictor must be fitted before being passed to create_app."
        )

    chat_service = chat_service or DocumentChatService()

    app = FastAPI(
        title="Health App",
        version="0.2.0",
        description="Benefits engine plus ML cost predictor (prototype).",
    )

    # CORS for the React dev server. Default allows the standard Vite
    # ports; tighten or override via HEALTH_APP_CORS_ORIGINS in production.
    raw_origins = os.environ.get(
        "HEALTH_APP_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    allowed_origins = [
        o.strip() for o in raw_origins.split(",") if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def get_repository() -> CatalogRepository:
        return repository

    def get_predictor() -> CostPredictor:
        return predictor

    def get_chat_service() -> DocumentChatService:
        return chat_service

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, str]:
        """Return ``{"status": "ok"}`` for liveness checks."""
        return {"status": "ok"}

    # --- benefits routes -------------------------------------------------

    @app.get("/plans/{plan_id}", response_model=Plan, tags=["catalog"])
    def get_plan(
        plan_id: str,
        repo: CatalogRepository = Depends(get_repository),
    ) -> Plan:
        """Return the plan matching ``plan_id``, or 404 if not found."""
        plan = repo.get_plan(plan_id)
        if plan is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plan '{plan_id}' not found.",
            )
        return plan

    @app.get("/members/{member_id}", response_model=Member, tags=["catalog"])
    def get_member(
        member_id: str,
        repo: CatalogRepository = Depends(get_repository),
    ) -> Member:
        """Return the member matching ``member_id``, or 404 if not found."""
        member = repo.get_member(member_id)
        if member is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Member '{member_id}' not found.",
            )
        return member

    @app.get(
        "/procedures/{code}", response_model=Procedure, tags=["catalog"]
    )
    def get_procedure(
        code: str,
        repo: CatalogRepository = Depends(get_repository),
    ) -> Procedure:
        """Return the procedure matching ``code``, or 404 if not found."""
        proc = repo.get_procedure(code)
        if proc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Procedure '{code}' not found.",
            )
        return proc

    @app.post(
        "/estimate",
        response_model=EstimateResult,
        tags=["estimate"],
    )
    def post_estimate(
        request: EstimateRequest,
        repo: CatalogRepository = Depends(get_repository),
    ) -> EstimateResult:
        """Compute an out-of-pocket estimate for a single procedure."""
        member = repo.get_member(request.member_id)
        if member is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Member '{request.member_id}' not found.",
            )
        plan = repo.get_plan(member.plan_id)
        if plan is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Plan '{member.plan_id}' for member "
                    f"'{member.member_id}' not found."
                ),
            )
        procedure = repo.get_procedure(request.procedure_code)
        if procedure is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Procedure '{request.procedure_code}' not found.",
            )
        return estimate_cost_share(
            plan=plan,
            member=member,
            procedure=procedure,
            in_network=request.in_network,
        )

    # --- predictor routes ------------------------------------------------

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["predictor"],
    )
    def post_predict(
        request: PredictRequest,
        model: CostPredictor = Depends(get_predictor),
        repo: CatalogRepository = Depends(get_repository),
    ) -> PredictResponse:
        """Predict annual medical charges; optionally annotate with plan OOP."""
        prediction = model.predict(request.features)
        share_median, share_mean = _annual_shares_for(
            request.plan_id, prediction, repo
        )
        return PredictResponse(
            prediction=prediction,
            annual_plan_share_median=share_median,
            annual_plan_share_mean=share_mean,
        )

    @app.post(
        "/whatif",
        response_model=WhatIfResponse,
        tags=["predictor"],
    )
    def post_whatif(
        request: WhatIfRequest,
        model: CostPredictor = Depends(get_predictor),
        repo: CatalogRepository = Depends(get_repository),
    ) -> WhatIfResponse:
        """Vary one feature across values and return the prediction curve."""
        if request.feature not in SWEEPABLE_FEATURES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot sweep '{request.feature}'.",
            )
        plan = _resolve_plan(request.plan_id, repo)
        try:
            return sweep(
                predictor=model,
                baseline=request.baseline,
                feature=request.feature,
                values=request.values,
                plan=plan,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e

    # --- document chat routes ------------------------------------------

    @app.post(
        "/documents",
        response_model=UploadResponse,
        tags=["docchat"],
    )
    async def post_document(
        file: UploadFile = File(...),
        service: DocumentChatService = Depends(get_chat_service),
    ) -> UploadResponse:
        """Upload a PDF and build a retrieval index over it."""
        if file.content_type not in (
            "application/pdf",
            "application/x-pdf",
            None,
        ) and not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF uploads are accepted.",
            )
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        if len(contents) > MAX_PDF_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB cap.",
            )
        try:
            meta = service.upload_pdf(
                filename=file.filename or "uploaded.pdf",
                pdf_bytes=contents,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e
        return UploadResponse(document=meta)

    @app.get(
        "/documents",
        response_model=list[DocumentMeta],
        tags=["docchat"],
    )
    def list_documents(
        service: DocumentChatService = Depends(get_chat_service),
    ) -> list[DocumentMeta]:
        """Return metadata for all currently stored documents."""
        return service.store.list_meta()

    @app.delete(
        "/documents/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["docchat"],
    )
    def delete_document(
        document_id: str,
        service: DocumentChatService = Depends(get_chat_service),
    ) -> None:
        """Delete a document by ID; returns 404 if not found."""
        if not service.store.delete(document_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

    @app.post(
        "/chat",
        response_model=ChatResponse,
        tags=["docchat"],
    )
    def post_chat(
        request: ChatRequest,
        service: DocumentChatService = Depends(get_chat_service),
    ) -> ChatResponse:
        """Answer a question against a previously uploaded document."""
        try:
            return service.ask(
                document_id=request.document_id,
                question=request.question,
                top_k=request.top_k,
            )
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{request.document_id}' not found.",
            ) from e

    return app


def _resolve_plan(
    plan_id: str | None, repo: CatalogRepository
) -> Plan | None:
    """Look up a plan by ID, raising HTTP 404 if provided but not found.

    Parameters
    ----------
    plan_id : str or None
        The plan identifier to look up, or ``None`` to skip the lookup.
    repo : CatalogRepository
        Catalog to query.

    Returns
    -------
    Plan or None
        The resolved plan, or ``None`` when ``plan_id`` is ``None``.

    Raises
    ------
    HTTPException
        With status 404 if ``plan_id`` is provided but not in the catalog.
    """
    if plan_id is None:
        return None
    plan = repo.get_plan(plan_id)
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan '{plan_id}' not found.",
        )
    return plan


def _annual_shares_for(
    plan_id: str | None,
    prediction: CostPrediction,
    repo: CatalogRepository,
) -> tuple[AnnualPlanShare | None, AnnualPlanShare | None]:
    """Compute median and mean annual plan cost-shares for a prediction.

    Parameters
    ----------
    plan_id : str or None
        Plan to evaluate against. Returns ``(None, None)`` when ``None``.
    prediction : CostPrediction
        Predicted charges used as annual spend input.
    repo : CatalogRepository
        Catalog used to resolve the plan.

    Returns
    -------
    tuple[AnnualPlanShare or None, AnnualPlanShare or None]
        ``(median_share, mean_share)`` when a plan is resolved, else
        ``(None, None)``.

    Raises
    ------
    HTTPException
        With status 404 if ``plan_id`` is provided but not found.
    """
    plan = _resolve_plan(plan_id, repo)
    if plan is None:
        return None, None
    median_share = apply_plan_to_annual_spend(
        plan, prediction.median_charges_cents
    )
    mean_share = apply_plan_to_annual_spend(
        plan, prediction.mean_charges_cents
    )
    return median_share, mean_share
