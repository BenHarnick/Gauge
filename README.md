# Gauge

> **What will my health insurance actually cost me this year?**

That one question is surprisingly hard to answer. Your premium is visible. Everything else, deductibles, coinsurance, copays, OOP max, and how all of those interact with your health profile, is not. Gauge makes it answerable: you enter your demographics and your plan, and Gauge returns a calibrated out-of-pocket interval with an honest 80% coverage guarantee, not a single guess dressed up as precision.

<video src="https://github.com/BenH88888/Gauge/assets/180685510/606091395-0fcaec30-f088-4587-8c81-3f44bf3d8fcc" controls width="100%" title="Gauge demo"></video>

## The signature result

A wide, right-skewed charge distribution is compressed by the plan into a much tighter out-of-pocket interval. The OOP max eliminates worst-case spend entirely. This is the figure that unifies the ML half and the deterministic benefits half:

![OOP transform](reports/figures/oop_transform.png)

## How it works

![Architecture](reports/figures/architecture.png)

```
Demographics → ML prediction → Plan upload → Apply plan → OOP interval
```

**Demographics.** Age, sex, BMI, children, smoker status, and region.

**ML prediction.** Four gradient-boosted models (HistGradientBoostingRegressor) predict the 10th, 50th, and 90th percentile annual charges, plus the mean. The interval is calibrated with Conformalized Quantile Regression (Romano, Patterson & Candès 2019), which guarantees marginal coverage ≥ 80% for any data distribution without assuming normality. The `CostPrediction` response always carries `conformal_calibrated: true` and `calibration_coverage: 0.8`.

**Plan upload.** Upload your Summary of Benefits PDF. The backend extracts deductible, OOP max, coinsurance rate, and copays automatically using targeted Q&A against the document's TF-IDF retrieval index. You review and correct any field before confirming.

**OOP interval.** `apply_plan_to_annual_spend` is monotone non-decreasing in charges, so applying it to the CQR charge interval `[lo, median, hi]` yields a valid OOP interval `[OOP(lo), OOP(median), OOP(hi)]`, no simulation required. The same 80% coverage guarantee transfers.

## Disclaimer

Estimates are illustrative. Real costs depend on actual plan documents, claim adjudication, provider contracts, and personal health history. Confirm coverage and costs with your insurer.

## Running the app

You need two terminals.

**Terminal 1 (backend):**

```bash
pip install -e ".[dev]"
uvicorn gauge.main:app --reload
```

On first startup the predictor trains and caches the model under `~/.cache/gauge/`. Subsequent starts load from cache.

**Terminal 2 (frontend):**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Enter your demographics, upload your plan PDF, review the extracted fields, and get your personalised OOP interval. The what-if simulator and plan Q&A panel appear inline below the result, no separate tabs. The **"How it works"** nav link opens a blog page explaining the project, architecture, and conformal prediction approach.

The frontend talks to the backend at `http://localhost:8000`. Override with `VITE_API_BASE` at build time.

### API examples

```bash
# Guided session flow
SESSION=$(curl -sX POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "sex": "female", "bmi": 26.5,
                    "children": 1, "smoker": "no", "region": "northeast"}}' \
  | jq -r .session_id)

curl -X POST http://localhost:8000/sessions/$SESSION/document \
  -F "file=@/path/to/my_plan.pdf"

curl -X POST http://localhost:8000/sessions/$SESSION/plan \
  -H "Content-Type: application/json" \
  -d '{"deductible_cents": 150000, "out_of_pocket_max_cents": 600000,
       "coinsurance_rate": 0.20, "copays_cents": {}, "plan_name": "My PPO"}'

curl -X POST http://localhost:8000/sessions/$SESSION/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Is telehealth covered?"}'
```

## Data

The predictor auto-detects a data source in this order:

1. `GAUGE_DATASET_CSV`: path to a Kaggle-style insurance CSV.
2. `GAUGE_MEPS_DTA`: path to a MEPS `.dta` file (optionally paired with `GAUGE_MEPS_SAQ` for BMI from HC-236).
3. `data/meps_hc233.dta` if present (run `python scripts/fetch_meps.py` to download; BMI merged from `data/meps_hc236.dta` when present).
4. `data/insurance.csv` (Kaggle insurance dataset).
5. Synthetic Kaggle-shaped data, deterministically generated.

Region labels are Census-standard (northeast, midwest, south, west) to match MEPS. Kaggle cardinal-direction regions are mapped at load time.

Enabling real LLM backends:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
# or
pip install -e ".[openai]"
export OPENAI_API_KEY=...
```

The backend auto-detects whichever key is set and reports `llm_used` on every chat response. `EchoLLM` (the default) requires no API key and returns retrieved excerpts formatted as answers.

## Session persistence

Sessions and uploaded documents persist to SQLite at `~/.cache/gauge/gauge.db` by default, so they survive a server restart. Override the path with `GAUGE_DB_PATH`. Set `GAUGE_NO_PERSIST=1` to use in-memory stores (useful for testing or ephemeral deployments).

## Guided-flow API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sessions` | Create session; returns initial prediction |
| `POST` | `/sessions/{id}/document` | Upload plan PDF; returns extracted draft |
| `GET` | `/sessions/{id}/plan-draft` | Fetch current extracted draft |
| `POST` | `/sessions/{id}/plan` | Confirm plan fields; returns full OOP interval |
| `GET` | `/sessions/{id}/estimate` | Fetch current personalised estimate |
| `POST` | `/sessions/{id}/whatif` | What-if sweep on session demographics |
| `POST` | `/sessions/{id}/chat` | Plain-English Q&A against the session's plan document |

## Saved estimates

Completed estimates can be saved under a user-supplied label and retrieved across sessions. The browser generates a UUID on first visit, sent as `X-Gauge-User-Id` on every request, no login required, but estimates are scoped to your browser.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/saved-estimates` | Save current session estimate (plan must be confirmed) |
| `GET` | `/saved-estimates` | List all saved estimates for this user, newest first |
| `PATCH` | `/saved-estimates/{id}` | Rename a saved estimate |
| `DELETE` | `/saved-estimates/{id}` | Delete a saved estimate |

All four endpoints require the `X-Gauge-User-Id` header (422 if absent). `PATCH` and `DELETE` return 403 if the requesting user does not own the estimate.

## Running the tests

```bash
pytest                 # full suite
pytest -m unit         # fast logic checks
pytest -m integration  # API endpoint checks
pytest -m e2e          # multi-step user journeys
```

The suite runs in a few seconds. The predictor is fit once per session via a session-scoped fixture.

## ML evaluation

Run the evaluation script to reproduce the figures in `MODELING.md`:

```bash
python -m gauge.eval
```

Outputs figures to `reports/figures/` and `reports/benchmark.json`.
Key results (10-seed mean): CQR coverage @80% = 80.3%, raw quantile coverage = 77.8%, MAE = $9,335.

## Project layout

```
src/gauge/
  benefits/
    models.py        Plan, Member, Procedure domain models
    calculator.py    Per-procedure cost-share math
    repository.py    CatalogRepository protocol + in-memory implementation
    seed.py          Sample plans, members, procedures
  predictor/
    schemas.py       PredictionFeatures request schema
    dataset.py       Data loader (MEPS, Kaggle, synthetic)
    model.py         CostPredictor: 3-quantile + mean gradient boosting + CQR
    whatif.py        What-if sweep (returns OOP interval per point)
    annual_cost.py   OopInterval, oop_interval_from_prediction (monotonicity proof)
  docchat/
    schemas.py       Chunk, DocumentMeta, Citation, ChatRequest/Response
    extractor.py     PDF text extraction (pypdf)
    chunker.py       Sliding-window chunker with page provenance
    index.py         TfidfRetrievalIndex (sklearn TF-IDF + cosine)
    llm.py           EchoLLM + pluggable Anthropic/OpenAI clients
    store.py         InMemoryDocumentStore (thread-safe)
    sqlite_store.py  SqliteDocumentStore (survives restarts)
    service.py       Upload + ask orchestration
  plan_extract/
    schemas.py       PlanDraft + FieldExtraction models
    extractor.py     PlanExtractor: targeted Q&A + regex parsing
  session/
    models.py        Session, SessionEstimate, request/response schemas
    store.py         InMemorySessionStore (thread-safe, UUID-keyed)
    sqlite_store.py  SqliteSessionStore (survives restarts)
  saved_estimates/
    models.py        SavedEstimate model + SavedEstimateStore protocol
                     InMemorySavedEstimateStore (tests / GAUGE_NO_PERSIST=1)
    sqlite_store.py  SqliteSavedEstimateStore (default; WAL, user-scoped)
  api.py             FastAPI app factory; all endpoints including session flow
  main.py            Entry point: model load-or-train, app startup
  eval.py            Evaluation script: coverage, MAE, ablations, figures
frontend/
  src/
    App.tsx          Brand header + nav (Estimator / How it works)
    api.ts           Typed client for the session-based backend API
    components/
      IntakeWizard.tsx  4-step flow: demographics → PDF upload → confirm → estimate
                        Step 4 includes what-if chart and plan Q&A inline
                        Landing screen shows saved estimates for returning users
      Blog.tsx          "How it works" writeup with inline SVG diagrams
      WhatIfChart.tsx   Recharts component rendering the OOP sweep
      Select.tsx        Generic select input
      Slider.tsx        Range slider with formatted label
      Toggle.tsx        Binary toggle (sex, smoker)
tests/
  unit/              Logic tests: models, calculator, predictor, OOP interval,
                     saved-estimate stores (InMemory + SQLite)
  integration/       Endpoint tests via FastAPI TestClient (incl. saved estimates)
  e2e/               Multi-step user journey tests
reports/
  figures/           PNG + SVG figures from gauge.eval
  benchmark.json     Numeric eval results
MODELING.md          ML methodology, calibration results, feature analysis
```
