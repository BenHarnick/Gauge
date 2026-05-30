# Health App

Prototype that combines four pieces into a single guided flow:

1. **Benefits engine** that, given a member and a procedure, applies a
   plan's deductible, coinsurance, copays, and OOP max to estimate the
   member's out-of-pocket cost.
2. **Machine-learning cost predictor** that takes a person's profile
   (age, sex, BMI, children, smoker status, region) and predicts annual
   medical charges with an 80% prediction interval. A what-if simulator
   sweeps any feature and shows how the prediction moves. Predictions
   can be piped through a plan to produce annual out-of-pocket
   estimates.
3. **Insurance document chatbot** that ingests a plan PDF, builds a
   retrieval index, and answers questions in plain English with
   page-level citations back to the source document.
4. **Guided session flow** that ties everything together: a user enters
   their demographics, uploads their plan PDF, reviews auto-extracted
   plan fields, and receives a personalised cost estimate with full
   cost-share breakdown and inline Q&A.

## Disclaimer

Estimates produced by this prototype are illustrative only. Real
benefits and real costs depend on actual plan documents, claim
adjudication, provider contracts, and personal health history. Members
should confirm coverage and costs with their insurer.

## ML pipeline overview

* **Model:** three scikit-learn `HistGradientBoostingRegressor`s trained
  with `loss="quantile"` at the 10th, 50th, and 90th percentiles. The
  median is the point estimate; the 10/90 pair gives an 80% prediction
  interval. Healthcare cost distributions are heavy-tailed, so intervals
  are much more honest than a single number.
* **Features:** age, sex, bmi, children, smoker, region. Categorical
  features are one-hot encoded via a `ColumnTransformer`; numerics pass
  through.
* **Data:** auto-detects in this order:
  1. `data/meps_hc233.dta` (MEPS HC 2021, run `python scripts/fetch_meps.py`)
  2. `data/insurance.csv` (Kaggle insurance dataset)
  3. Synthetic Kaggle-insurance-shaped dataset, deterministically generated.

  Region labels are Census-standard (northeast, midwest, south, west) to
  match MEPS. The Kaggle CSV's cardinal-direction regions are mapped
  onto Census regions at load time.
* **What-if simulator:** holds a baseline feature vector fixed, varies
  one feature across a list of values, and predicts the curve in a
  single batched call. Optionally annotates each point with annual
  cost-share under a chosen plan.
* **Bridge to benefits engine:** `apply_plan_to_annual_spend` runs the
  predicted annual charges through the plan's deductible, coinsurance,
  and OOP max to estimate annual out-of-pocket.

## Document chatbot pipeline

* **PDF extraction** via pypdf, page by page.
* **Chunking:** sliding-window with overlap, each chunk records the
  pages it spans so citations point back to the right page.
* **Retrieval:** TF-IDF + cosine similarity (scikit-learn). One index
  per uploaded document, which keeps vocabulary tight and tuned to
  domain-specific language.
* **LLM:** pluggable. `EchoLLM` is the default (no API key required;
  returns retrieved excerpts formatted as a coherent answer). Install
  the `anthropic` or `openai` extra and set the matching environment
  variable to upgrade to a real LLM:

  ```bash
  pip install -e ".[anthropic]"
  export ANTHROPIC_API_KEY=...
  # or
  pip install -e ".[openai]"
  export OPENAI_API_KEY=...
  ```

  The backend auto-detects whichever is available and reports
  `llm_used` on every chat response.

## Plan extraction pipeline

When a plan PDF is uploaded in the guided flow the backend automatically
extracts plan fields without any manual input:

* **Targeted Q&A:** for each field (deductible, OOP max, coinsurance
  rate, and four copay types) a plain-English question is run against
  the document's retrieval index.
* **Regex parsing:** dollar amounts (`_parse_dollars`) and percentages
  (`_parse_percent`) are extracted from the LLM answer. Anything that
  cannot be parsed produces a `null` field and is flagged in
  `unresolved_fields`.
* **Human confirmation:** the extracted draft is shown to the user on a
  review form. Fields that were found are pre-filled; missing fields are
  highlighted with an amber badge so the user can fill them in manually
  before confirming.
* **Plan creation:** on confirmation the draft is converted to a `Plan`
  object and stored in the session, enabling the full annual cost-share
  breakdown.

## Session layer

The session API (`/sessions/...`) is a thin UUID-keyed store that ties
`PredictionFeatures`, a `Plan`, and a `document_id` together for the
duration of a guided flow. Sessions are in-memory and are lost on server
restart (intentional for this prototype).

Guided-flow endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sessions` | Create session from demographics; returns initial prediction |
| `POST` | `/sessions/{id}/document` | Upload plan PDF; returns auto-extracted plan draft |
| `GET` | `/sessions/{id}/plan-draft` | Fetch the current extracted draft |
| `POST` | `/sessions/{id}/plan` | Confirm (or correct) plan fields; returns full estimate |
| `GET` | `/sessions/{id}/estimate` | Fetch the current personalised estimate |
| `POST` | `/sessions/{id}/whatif` | What-if sweep using the session's demographics as baseline |
| `POST` | `/sessions/{id}/chat` | Plain-English Q&A against the session's plan document |

## Running the app

You need two terminals: one for the Python API, one for the React frontend.

**Terminal 1: backend**

```bash
pip install -e ".[dev]"
uvicorn health_app.main:app --reload
```

On first startup the predictor trains and caches the model under
`~/.cache/health_app/`. Subsequent starts load from the cache.

**Terminal 2: frontend**

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser. The default landing
page is the guided "Get my estimate" flow: enter demographics, upload
your plan PDF, review extracted fields, and get a personalised estimate.
The "Cost predictor" and "Document chat" tabs remain available for
direct access to each module independently.

The frontend talks to the backend at `http://localhost:8000` by default.
Override at build time with the `VITE_API_BASE` env var if you deploy
the backend elsewhere.

For a production build of the frontend:

```bash
cd frontend
npm run build
# Output is in frontend/dist; serve it with any static file host.
```

### Example calls

```bash
# Predict annual charges and annual OOP under a plan
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 40, "sex": "male", "bmi": 32.0,
      "children": 2, "smoker": "yes", "region": "southeast"
    },
    "plan_id": "ppo_gold"
  }'

# What-if: sweep age from 25 to 64 under PPO Gold
curl -X POST http://127.0.0.1:8000/whatif \
  -H "Content-Type: application/json" \
  -d '{
    "baseline": {
      "age": 40, "sex": "male", "bmi": 27.0,
      "children": 0, "smoker": "no", "region": "northeast"
    },
    "feature": "age",
    "values": [25, 35, 45, 55, 64],
    "plan_id": "ppo_gold"
  }'

# Per-procedure estimate (benefits engine)
curl -X POST http://127.0.0.1:8000/estimate \
  -H "Content-Type: application/json" \
  -d '{"member_id":"m1","procedure_code":"99213","in_network":true}'

# --- Guided session flow ---

# 1. Create a session (returns session_id + initial prediction)
curl -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 35, "sex": "female", "bmi": 26.5,
      "children": 1, "smoker": "no", "region": "northeast"
    }
  }'

# 2. Upload a plan PDF and extract fields
curl -X POST http://127.0.0.1:8000/sessions/{session_id}/document \
  -F "file=@/path/to/my_plan.pdf"

# 3. Confirm (or correct) the extracted plan fields
curl -X POST http://127.0.0.1:8000/sessions/{session_id}/plan \
  -H "Content-Type: application/json" \
  -d '{
    "deductible_cents": 150000,
    "out_of_pocket_max_cents": 600000,
    "coinsurance_rate": 0.20,
    "copays_cents": {},
    "plan_name": "My PPO"
  }'

# 4. Ask a question about the plan document
curl -X POST http://127.0.0.1:8000/sessions/{session_id}/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Is telehealth covered?"}'
```

## Running the tests

```bash
pytest                 # full suite
pytest -m unit         # fast logic checks
pytest -m integration  # API endpoint checks
pytest -m e2e          # multi-step journeys
```

The full suite is around 2 seconds (70 tests). The predictor is fit
once per session via a session-scoped fixture so the ML training cost
is paid exactly once.

## Project layout

```
src/health_app/
  benefits/
    models.py        Pydantic domain models (Plan, Member, Procedure, ...)
    calculator.py    Per-procedure cost-share math
    repository.py    CatalogRepository protocol + in-memory implementation
    seed.py          Sample plans, members, procedures
  predictor/
    schemas.py       PredictionFeatures (request schema)
    dataset.py       Synthetic data generator + CSV loader
    model.py         CostPredictor (3-quantile gradient boosting)
    whatif.py        What-if sweep with validation
    annual_cost.py   Predicted charges -> annual member/plan share
  docchat/
    schemas.py       Chunk, DocumentMeta, Citation, ChatRequest/Response
    extractor.py     PDF text extraction (pypdf)
    chunker.py       Sliding-window chunker with page provenance
    index.py         TfidfRetrievalIndex (sklearn TF-IDF + cosine)
    llm.py           EchoLLM + pluggable Anthropic/OpenAI clients
    store.py         InMemoryDocumentStore (thread-safe)
    service.py       Upload + ask orchestration
  plan_extract/
    schemas.py       PlanDraft + FieldExtraction Pydantic models
    extractor.py     PlanExtractor: targeted Q&A + regex parsing
  session/
    models.py        Session, SessionEstimate, request/response schemas
    store.py         InMemorySessionStore (thread-safe, UUID-keyed)
  api.py             Unified FastAPI app factory with CORS; includes
                     all session endpoints (/sessions/...)
  main.py            Module entry point with model load-or-train
frontend/
  src/
    App.tsx          Top-level tabs: Get my estimate / Cost predictor /
                     Document chat
    api.ts           Typed client for the FastAPI backend (includes
                     session API functions)
    components/
      IntakeWizard.tsx  4-step guided flow (demographics, PDF upload,
                        plan confirmation, personalised estimate)
      PredictorPage/    Standalone cost predictor with what-if chart
      DocChatPage/      Standalone PDF upload + chat panel
tests/
  unit/              Logic tests (models, calculator, dataset,
                     predictor, whatif, annual_cost)
  integration/       Endpoint tests via TestClient
  e2e/               Multi-step user journey tests
```

## What's next

* Real LLM backend: install the `anthropic` or `openai` extra and set
  the matching API key to replace `EchoLLM`. Plan extraction quality
  improves significantly with a real model.
* Persistent session and document store (SQLite + on-disk PDFs) so
  uploads and sessions survive a restart.
* Wider, more honest prediction intervals via conformal prediction.
* Train on real MEPS data instead of the synthetic dataset for a more
  realistic predictor.
* Multi-plan comparison in the guided flow so users can compare OOP
  estimates side by side across two or more uploaded plans.
