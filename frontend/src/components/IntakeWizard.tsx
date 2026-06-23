/**
 * IntakeWizard — four-step guided flow that ties demographics, plan upload,
 * confirmation, and the final estimate together into one coherent experience.
 *
 * Step 1 — Your info:   user enters demographics and gets an instant prediction.
 * Step 2 — Your plan:   user uploads their plan PDF; the server extracts fields.
 * Step 3 — Confirm:     user reviews / corrects the extracted plan numbers.
 * Step 4 — Estimate:    full personalised estimate with breakdown, what-if, and Q&A.
 *
 * The session ID returned in step 1 is threaded through every subsequent API
 * call so the server can keep the three pieces of context (demographics, plan,
 * document) together without needing a database.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChatTurn,
  ConfirmPlanRequest,
  CreateSessionResponse,
  OopInterval,
  PlanDraft,
  Region,
  SavedEstimate,
  SessionEstimate,
  Sex,
  Smoker,
  SweepFeature,
  WhatIfResponse,
  attachSessionDocument,
  centsToDollars,
  confirmSessionPlan,
  createSession,
  deleteSavedEstimate,
  listSavedEstimates,
  renameSavedEstimate,
  saveEstimate,
  sessionChat,
  sessionWhatIf,
} from "../api";
import type { PredictionFeatures } from "../api";
import { Select } from "./Select";
import { Slider } from "./Slider";
import { Toggle } from "./Toggle";
import { WhatIfChart } from "./WhatIfChart";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Which step of the wizard the user is currently on (0 = landing, 1–4 = intake, 5 = loaded saved estimate). */
type Step = 0 | 1 | 2 | 3 | 4 | 5;

const REGIONS: Region[] = ["northeast", "midwest", "south", "west"];

const SWEEP_OPTIONS: { value: SweepFeature; label: string }[] = [
  { value: "age", label: "Age" },
  { value: "bmi", label: "BMI" },
  { value: "children", label: "Children" },
  { value: "smoker", label: "Smoker" },
  { value: "sex", label: "Sex" },
  { value: "region", label: "Region" },
];

/** Preset sweep values for each sweepable feature. */
function sweepValuesFor(feature: SweepFeature): Array<number | string> {
  switch (feature) {
    case "age":      return [20, 25, 30, 35, 40, 45, 50, 55, 60, 64];
    case "bmi":      return [18, 22, 26, 30, 34, 38, 42];
    case "children": return [0, 1, 2, 3, 4, 5];
    case "smoker":   return ["no", "yes"];
    case "sex":      return ["female", "male"];
    case "region":   return REGIONS as unknown as string[];
  }
}

// ---------------------------------------------------------------------------
// Step 0 — Landing screen
// ---------------------------------------------------------------------------

/**
 * Hero landing screen shown before the intake flow begins.
 *
 * Communicates the product's single thesis and the four load-bearing steps in
 * plain language, then hands off to the intake wizard on "Get started".
 *
 * @param onStart - Called when the user clicks the CTA.
 * @param onLoad - Called when the user loads a saved estimate.
 */
function Step0Landing({
  onStart,
  onLoad,
}: {
  onStart: () => void;
  onLoad: (est: SavedEstimate) => void;
}) {
  const steps = [
    {
      icon: "👤",
      title: "Tell us about yourself",
      body: "Age, health profile, and a few quick details to seed the prediction.",
    },
    {
      icon: "📄",
      title: "Upload your plan PDF",
      body: "Drop your Summary of Benefits and Coverage. We'll pull out deductible, OOP max, and copays automatically.",
    },
    {
      icon: "📊",
      title: "Get your estimate",
      body: "An 80% confidence range for what you'll actually pay out of pocket this year.",
    },
    {
      icon: "💬",
      title: "Ask anything",
      body: "A chatbot grounded in your real plan answers follow-up questions with page citations.",
    },
  ];

  return (
    <div className="flex min-h-[70vh] flex-col items-center justify-center px-4 py-16 text-center">
      {/* Hero */}
      <div className="max-w-2xl">
        <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-brand-600">
          Gauge
        </p>
        <h1 className="text-4xl font-bold leading-tight tracking-tight text-slate-900 sm:text-5xl">
          What will you actually pay
          <br />
          <span className="text-brand-600">this year?</span>
        </h1>
        <p className="mx-auto mt-5 max-w-xl text-lg text-slate-600">
          Gauge reads your real insurance plan, predicts your annual costs with
          calibrated uncertainty, and answers your questions.
        </p>
        <button
          type="button"
          onClick={onStart}
          className="mt-8 rounded-lg bg-brand-600 px-8 py-3.5 text-base font-semibold text-white shadow-sm hover:bg-brand-700 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2"
        >
          Get started →
        </button>
      </div>

      {/* How it works */}
      <div className="mt-16 grid w-full max-w-3xl grid-cols-2 gap-4 text-left sm:grid-cols-4">
        {steps.map((s, i) => (
          <div
            key={i}
            className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
          >
            <div className="mb-2 text-2xl">{s.icon}</div>
            <p className="text-sm font-semibold text-slate-800">{s.title}</p>
            <p className="mt-1 text-xs leading-relaxed text-slate-500">{s.body}</p>
          </div>
        ))}
      </div>

      <p className="mt-10 text-xs text-slate-400">
        Illustrative prototype — not a substitute for an actual insurance quote
        or advice from your insurer.
      </p>

      <SavedEstimatesPanel onLoad={onLoad} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Saved estimates panel (used on landing screen)
// ---------------------------------------------------------------------------

/**
 * Fetches and displays all saved estimates with inline rename and delete.
 * Shown on the landing screen so returning users can see their history.
 *
 * @param onLoad - Called when the user clicks "Load" on a saved estimate,
 *   transitioning them straight to the estimate view.
 */
function SavedEstimatesPanel({
  onLoad,
}: {
  onLoad: (est: SavedEstimate) => void;
}) {
  const [estimates, setEstimates] = useState<SavedEstimate[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editLabel, setEditLabel] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listSavedEstimates()
      .then(setEstimates)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  async function handleDelete(id: string) {
    try {
      await deleteSavedEstimate(id);
      setEstimates((prev) => prev.filter((e) => e.id !== id));
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleRename(id: string) {
    if (!editLabel.trim()) return;
    try {
      const updated = await renameSavedEstimate(id, { label: editLabel.trim() });
      setEstimates((prev) => prev.map((e) => (e.id === id ? updated : e)));
      setEditingId(null);
    } catch (e) {
      setError(String(e));
    }
  }

  if (loading) return null;
  if (estimates.length === 0) return null;

  return (
    <div className="mt-12 w-full max-w-3xl">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
        Saved estimates
      </h2>

      {error && (
        <div className="mb-3 rounded bg-red-50 px-3 py-2 text-xs text-red-700">{error}</div>
      )}

      <div className="space-y-2">
        {estimates.map((est) => (
          <div
            key={est.id}
            className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-sm"
          >
            {/* Label / rename */}
            <div className="min-w-0 flex-1">
              {editingId === est.id ? (
                <div className="flex gap-2">
                  <input
                    autoFocus
                    type="text"
                    value={editLabel}
                    onChange={(e) => setEditLabel(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") void handleRename(est.id);
                      if (e.key === "Escape") setEditingId(null);
                    }}
                    className="flex-1 rounded-md border border-slate-300 px-2 py-1 text-sm outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500"
                  />
                  <button
                    type="button"
                    onClick={() => void handleRename(est.id)}
                    className="text-xs font-semibold text-brand-600 hover:text-brand-800"
                  >
                    Save
                  </button>
                  <button
                    type="button"
                    onClick={() => setEditingId(null)}
                    className="text-xs text-slate-400 hover:text-slate-600"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <div>
                  <p className="truncate text-sm font-medium text-slate-900">{est.label}</p>
                  <p className="text-xs text-slate-500">
                    {est.oop_interval
                      ? `${centsToDollars(est.oop_interval.lower_cents)}–${centsToDollars(est.oop_interval.upper_cents)} OOP`
                      : centsToDollars(est.prediction.median_charges_cents) + " charges"}
                    {" · "}
                    {new Date(est.created_at).toLocaleDateString()}
                  </p>
                </div>
              )}
            </div>

            {/* Actions */}
            {editingId !== est.id && (
              <div className="flex shrink-0 items-center gap-2">
                <button
                  type="button"
                  onClick={() => onLoad(est)}
                  className="rounded-md bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-brand-700"
                >
                  Load
                </button>
                <button
                  type="button"
                  onClick={() => { setEditingId(est.id); setEditLabel(est.label); }}
                  className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
                >
                  Rename
                </button>
                <button
                  type="button"
                  onClick={() => void handleDelete(est.id)}
                  className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-400 hover:bg-red-50 hover:text-red-600"
                >
                  Delete
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step indicator
// ---------------------------------------------------------------------------

/**
 * Horizontal stepper showing the user's position in the four-step flow.
 *
 * @param current - Current step number (1-indexed; step 0 landing is never shown here).
 * @param labels - Display label for each step.
 */
function StepBar({
  current,
  labels,
}: {
  current: 1 | 2 | 3 | 4;
  labels: string[];
}) {
  return (
    <ol className="mb-8 flex items-center gap-0">
      {labels.map((label, i) => {
        const stepNum = (i + 1) as Step;
        const done = stepNum < current;
        const active = stepNum === current;
        return (
          <li key={label} className="flex flex-1 items-center">
            <div className="flex flex-col items-center gap-1">
              <div
                className={
                  "flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold " +
                  (done
                    ? "bg-brand-600 text-white"
                    : active
                    ? "border-2 border-brand-600 text-brand-600"
                    : "border-2 border-slate-300 text-slate-400")
                }
              >
                {done ? "✓" : stepNum}
              </div>
              <span
                className={
                  "hidden text-xs sm:block " +
                  (active ? "font-semibold text-slate-900" : "text-slate-500")
                }
              >
                {label}
              </span>
            </div>
            {i < labels.length - 1 && (
              <div
                className={
                  "mx-2 h-0.5 flex-1 " +
                  (done ? "bg-brand-600" : "bg-slate-200")
                }
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}

// ---------------------------------------------------------------------------
// Step 1 — Demographics
// ---------------------------------------------------------------------------

/**
 * Renders the demographics form and fires the createSession API call when
 * the user clicks Continue.
 *
 * @param onDone - Called with the session response once the session is created.
 */
function Step1Demographics({
  onDone,
}: {
  onDone: (resp: CreateSessionResponse, features: PredictionFeatures) => void;
}) {
  const [features, setFeatures] = useState<PredictionFeatures>({
    age: 35,
    sex: "female",
    bmi: 27.5,
    children: 1,
    smoker: "no",
    region: "northeast",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** Update a single feature field. */
  const update = useCallback(
    <K extends keyof PredictionFeatures>(key: K, val: PredictionFeatures[K]) =>
      setFeatures((prev) => ({ ...prev, [key]: val })),
    [],
  );

  async function handleContinue() {
    setLoading(true);
    setError(null);
    try {
      const resp = await createSession(features);
      onDone(resp, features);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-900">Tell us about yourself</h2>
        <p className="mt-1 text-sm text-slate-600">
          We use these details to predict your expected annual medical spend.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 rounded-xl border border-slate-200 bg-white p-6 shadow-card sm:grid-cols-2">
        <Slider label="Age" min={18} max={64} value={features.age}
          onChange={(v) => update("age", v)} format={(v) => `${v} yrs`} />
        <Toggle<Sex> label="Sex" value={features.sex}
          options={[{ value: "female", label: "Female" }, { value: "male", label: "Male" }]}
          onChange={(v) => update("sex", v)} />
        <Slider label="BMI" min={16} max={53} step={0.1} value={features.bmi}
          onChange={(v) => update("bmi", Number(v.toFixed(1)))}
          format={(v) => v.toFixed(1)} />
        <Slider label="Children" min={0} max={5} value={features.children}
          onChange={(v) => update("children", v)} />
        <Toggle<Smoker> label="Smoker" value={features.smoker}
          options={[{ value: "no", label: "No" }, { value: "yes", label: "Yes" }]}
          onChange={(v) => update("smoker", v)} />
        <Select<Region> label="Region" value={features.region}
          options={REGIONS.map((r) => ({ value: r, label: r.charAt(0).toUpperCase() + r.slice(1) }))}
          onChange={(v) => update("region", v)} />
      </div>

      {error && (
        <div className="rounded bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
      )}

      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => void handleContinue()}
          disabled={loading}
          className="rounded-md bg-brand-600 px-5 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
        >
          {loading ? "Creating session…" : "Continue →"}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 2 — Upload plan PDF
// ---------------------------------------------------------------------------

/**
 * Prompts the user to upload their insurance plan PDF.  On upload, calls the
 * attachSessionDocument API which returns extracted plan fields.
 *
 * @param sessionId - Active session to attach the document to.
 * @param onDone - Called with the extracted draft and document ID on success.
 * @param onSkip - Called when the user opts to enter plan details manually.
 */
function Step2Upload({
  sessionId,
  onDone,
  onSkip,
}: {
  sessionId: string;
  onDone: (draft: PlanDraft, documentId: string) => void;
  onSkip: () => void;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setLoading(true);
    setError(null);
    try {
      const resp = await attachSessionDocument(sessionId, file);
      onDone(resp.plan_draft, resp.document_id);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-900">Upload your plan document</h2>
        <p className="mt-1 text-sm text-slate-600">
          Drop your Summary of Benefits and Coverage or any plan PDF. We will
          automatically extract your deductible, out-of-pocket max, coinsurance
          rate, and copays -- you can review and edit them on the next screen.
        </p>
      </div>

      <div
        className="flex h-48 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 bg-white transition hover:border-brand-400 hover:bg-brand-50"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const file = e.dataTransfer.files[0];
          if (file) void handleFile(file);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="sr-only"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void handleFile(file);
          }}
        />
        {loading ? (
          <p className="text-sm text-slate-500">Uploading and extracting…</p>
        ) : (
          <>
            <p className="text-sm font-medium text-slate-700">
              Click to choose a PDF, or drag and drop
            </p>
            <p className="mt-1 text-xs text-slate-500">Max 25 MB</p>
          </>
        )}
      </div>

      {error && (
        <div className="rounded bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
      )}

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={onSkip}
          className="text-sm text-slate-500 underline-offset-2 hover:text-slate-700 hover:underline"
        >
          I don't have a PDF — enter plan details manually
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 3 — Confirm extracted plan fields
// ---------------------------------------------------------------------------

/** Display name for each ServiceCategory value. */
const CATEGORY_LABELS: Record<string, string> = {
  office_visit:  "Primary care visit",
  specialist:    "Specialist visit",
  urgent_care:   "Urgent care",
  generic_drug:  "Generic drug",
  emergency:     "Emergency room",
  imaging:       "Imaging (MRI / CT)",
  lab:           "Lab work",
  surgery:       "Surgery",
};

/**
 * Labelled dollar-amount input for a required plan field.
 *
 * Defined at module scope so React does not treat it as a new component type
 * on every render (which would unmount and remount the input, killing focus).
 */
function DollarField({
  label,
  value,
  onChange,
  missing,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  missing: boolean;
}) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="font-medium text-slate-700">
        {label}{" "}
        {missing && (
          <span className="ml-1 rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-700">
            not found
          </span>
        )}
      </span>
      <div className="flex items-center rounded-md border border-slate-300 bg-white px-3 py-1.5">
        <span className="mr-1 text-slate-400">$</span>
        <input
          type="number"
          min={0}
          step={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="flex-1 border-none bg-transparent text-slate-900 outline-none"
          placeholder="0"
        />
      </div>
    </label>
  );
}

/**
 * Shows the auto-extracted plan fields and lets the user correct any values
 * before confirming.  Fields that could not be extracted are shown as empty
 * inputs with a "not found" badge.
 *
 * @param draft - Auto-extracted plan draft from the server.
 * @param sessionId - Active session ID.
 * @param onDone - Called with the full estimate once the plan is confirmed.
 */
function Step3Confirm({
  draft,
  sessionId,
  onDone,
}: {
  draft: PlanDraft;
  sessionId: string;
  onDone: (estimate: SessionEstimate) => void;
}) {
  /** Convert cents → dollars string for display in inputs. */
  const centsToInput = (c: number | null) =>
    c !== null ? String(c / 100) : "";

  const [deductible, setDeductible] = useState(centsToInput(draft.deductible_cents));
  const [oopMax, setOopMax] = useState(centsToInput(draft.out_of_pocket_max_cents));
  const [coinsurance, setCoinsurance] = useState(
    draft.coinsurance_rate !== null ? String(draft.coinsurance_rate * 100) : "",
  );
  const [copays, setCopays] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    for (const [cat, cents] of Object.entries(draft.copays_cents)) {
      init[cat] = String(cents / 100);
    }
    return init;
  });
  const [planName, setPlanName] = useState("My Plan");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleConfirm() {
    const dedCents = Math.round(parseFloat(deductible || "0") * 100);
    const oopCents = Math.round(parseFloat(oopMax || "0") * 100);
    const coinRate = parseFloat(coinsurance || "0") / 100;

    if (isNaN(dedCents) || isNaN(oopCents) || isNaN(coinRate)) {
      setError("Please fill in all required fields (deductible, OOP max, coinsurance).");
      return;
    }
    if (oopCents < dedCents) {
      setError("Out-of-pocket maximum must be at least as large as the deductible.");
      return;
    }

    const copaysCents: Record<string, number> = {};
    for (const [cat, val] of Object.entries(copays)) {
      if (val) copaysCents[cat] = Math.round(parseFloat(val) * 100);
    }

    const req: ConfirmPlanRequest = {
      deductible_cents: dedCents,
      out_of_pocket_max_cents: oopCents,
      coinsurance_rate: coinRate,
      copays_cents: copaysCents,
      plan_name: planName,
    };

    setLoading(true);
    setError(null);
    try {
      const estimate = await confirmSessionPlan(sessionId, req);
      onDone(estimate);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  const unresolved = new Set(draft.unresolved_fields);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-900">
          Review your plan details
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          We extracted these numbers from your document. Check them against your
          plan and correct anything that looks wrong before continuing.
        </p>
      </div>

      {draft.extraction_notes.length > 0 && (
        <details className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
          <summary className="cursor-pointer text-sm font-medium text-amber-800">
            {draft.unresolved_fields.length} field
            {draft.unresolved_fields.length !== 1 ? "s" : ""} could not be
            read automatically — expand to see notes
          </summary>
          <ul className="mt-2 space-y-1 text-xs text-amber-700">
            {draft.extraction_notes.map((n, i) => (
              <li key={i}>• {n}</li>
            ))}
          </ul>
        </details>
      )}

      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-card">
        <label className="mb-5 flex flex-col gap-1 text-sm">
          <span className="font-medium text-slate-700">Plan name</span>
          <input
            type="text"
            value={planName}
            onChange={(e) => setPlanName(e.target.value)}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-slate-900 outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500"
          />
        </label>

        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Core cost-share numbers
        </h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <DollarField
            label="Annual deductible"
            value={deductible}
            onChange={setDeductible}
            missing={unresolved.has("deductible")}
          />
          <DollarField
            label="Out-of-pocket maximum"
            value={oopMax}
            onChange={setOopMax}
            missing={unresolved.has("oop_max")}
          />
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-slate-700">
              Coinsurance rate{" "}
              {unresolved.has("coinsurance") && (
                <span className="ml-1 rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-700">
                  not found
                </span>
              )}
            </span>
            <div className="flex items-center rounded-md border border-slate-300 bg-white px-3 py-1.5">
              <input
                type="number"
                min={0}
                max={100}
                step={1}
                value={coinsurance}
                onChange={(e) => setCoinsurance(e.target.value)}
                className="flex-1 border-none bg-transparent text-slate-900 outline-none"
                placeholder="20"
              />
              <span className="ml-1 text-slate-400">%</span>
            </div>
          </label>
        </div>

        <h3 className="mb-3 mt-6 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Copays (optional — leave blank if not applicable)
        </h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Object.entries(CATEGORY_LABELS).map(([cat, label]) => (
            <label key={cat} className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-slate-700">{label}</span>
              <div className="flex items-center rounded-md border border-slate-300 bg-white px-3 py-1.5">
                <span className="mr-1 text-slate-400">$</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={copays[cat] ?? ""}
                  onChange={(e) =>
                    setCopays((prev) => ({ ...prev, [cat]: e.target.value }))
                  }
                  className="flex-1 border-none bg-transparent text-slate-900 outline-none"
                  placeholder="—"
                />
              </div>
            </label>
          ))}
        </div>
      </div>

      {error && (
        <div className="rounded bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
      )}

      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => void handleConfirm()}
          disabled={loading}
          className="rounded-md bg-brand-600 px-5 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
        >
          {loading ? "Calculating…" : "See my estimate →"}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 4 — Estimate + what-if + Q&A
// ---------------------------------------------------------------------------

/**
 * Hero block displaying the OOP interval as the primary estimate output.
 *
 * @param interval - 80%-coverage OOP interval from the server, or null before
 *   a plan is confirmed.
 * @param chargesCents - Median gross charges, shown as secondary context.
 */
function EstimateBlock({
  interval,
  chargesCents,
}: {
  interval: OopInterval | null;
  chargesCents: number;
}) {
  return (
    <div className="bg-white p-6">
      {interval ? (
        <>
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            You'll likely pay out of pocket
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-bold tracking-tight text-brand-600 tabular-nums">
              {centsToDollars(interval.lower_cents)}
            </span>
            <span className="text-lg font-medium text-slate-400">to</span>
            <span className="text-3xl font-bold tracking-tight text-brand-600 tabular-nums">
              {centsToDollars(interval.upper_cents)}
            </span>
          </div>
          <div className="mt-1 text-xs text-slate-500">
            80% confidence interval · median{" "}
            <span className="tabular-nums font-medium text-slate-700">
              {centsToDollars(interval.median_cents)}
            </span>
          </div>
          {(interval.capped_at_oop_max_lower || interval.capped_at_oop_max_upper) && (
            <div className="mt-3 rounded bg-emerald-50 px-2 py-1.5 text-xs text-emerald-700">
              Upper end capped at your plan's out-of-pocket maximum.
            </div>
          )}
          <div className="mt-3 border-t border-slate-100 pt-3 text-xs text-slate-400">
            Predicted gross charges: {centsToDollars(chargesCents)}
          </div>
        </>
      ) : (
        <>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Predicted annual charges
          </div>
          <div className="mt-1 text-2xl font-semibold text-slate-900 tabular-nums">
            {centsToDollars(chargesCents)}
          </div>
          <div className="text-xs text-slate-500">
            median — confirm a plan to see your out-of-pocket cost
          </div>
        </>
      )}
    </div>
  );
}

/**
 * The final step: two-column layout with the estimate + what-if chart on the
 * left and a sticky chatbot panel on the right.
 *
 * On narrow viewports the columns stack vertically (report first, chat below).
 *
 * @param estimate - Initial estimate from the confirmSessionPlan call.
 * @param sessionId - Active session (used for what-if and chat requests).
 * @param onRestart - Callback to reset the wizard and start over.
 */
function Step4Estimate({
  estimate,
  sessionId,
  onRestart,
}: {
  estimate: SessionEstimate;
  sessionId: string;
  onRestart: () => void;
}) {
  const { prediction, plan, oop_interval, document_id } = estimate;

  // What-if state
  const [sweepFeature, setSweepFeature] = useState<SweepFeature>("age");
  const [sweep, setSweep] = useState<WhatIfResponse | null>(null);
  const [sweepLoading, setSweepLoading] = useState(false);
  const [sweepError, setSweepError] = useState<string | null>(null);

  // Q&A state
  const [question, setQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState<
    { question: string; answer: string; pages: number[] }[]
  >([]);
  const [chatError, setChatError] = useState<string | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Save state
  const [saveLabel, setSaveLabel] = useState("");
  const [saveOpen, setSaveOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedConfirm, setSavedConfirm] = useState(false);

  // Fire what-if whenever the sweep feature changes
  useEffect(() => {
    let cancelled = false;
    const values = sweepValuesFor(sweepFeature);
    setSweepLoading(true);
    setSweepError(null);

    sessionWhatIf(sessionId, sweepFeature, values)
      .then((r) => { if (!cancelled) setSweep(r); })
      .catch((e) => { if (!cancelled) setSweepError(String(e)); })
      .finally(() => { if (!cancelled) setSweepLoading(false); });

    return () => { cancelled = true; };
  }, [sessionId, sweepFeature]);

  // Scroll chat to bottom whenever a new message arrives
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  async function handleAsk() {
    if (!question.trim() || !document_id) return;
    setChatLoading(true);
    setChatError(null);
    const currentQuestion = question.trim();
    // Build history from prior messages for context continuity
    const history: ChatTurn[] = chatMessages.map((m) => ({
      question: m.question,
      answer: m.answer,
    }));
    try {
      const resp = await sessionChat(sessionId, currentQuestion, history);
      setChatMessages((prev) => [
        ...prev,
        {
          question: currentQuestion,
          answer: resp.answer,
          pages: resp.citations.flatMap((c) => c.page_numbers),
        },
      ]);
      setQuestion("");
    } catch (e) {
      setChatError(String(e));
    } finally {
      setChatLoading(false);
    }
  }

  async function handleSave() {
    if (!saveLabel.trim()) return;
    setSaving(true);
    setSaveError(null);
    try {
      await saveEstimate({ session_id: sessionId, label: saveLabel.trim() });
      setSavedConfirm(true);
      setSaveOpen(false);
      setSaveLabel("");
      setTimeout(() => setSavedConfirm(false), 3000);
    } catch (e) {
      setSaveError(String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4">
      {/* Page header */}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="text-xl font-semibold text-slate-900">Your estimate</h2>
          {plan && (
            <p className="mt-1 text-sm text-slate-600">
              Based on <span className="font-medium">{plan.name}</span> —
              ${(plan.deductible_cents / 100).toLocaleString()} deductible,{" "}
              ${(plan.out_of_pocket_max_cents / 100).toLocaleString()} OOP max,{" "}
              {Math.round(plan.coinsurance_rate * 100)}% coinsurance.
            </p>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-3">
          {/* Save estimate */}
          {savedConfirm ? (
            <span className="text-xs font-medium text-emerald-600">Saved ✓</span>
          ) : saveOpen ? (
            <div className="flex items-center gap-2">
              <input
                autoFocus
                type="text"
                value={saveLabel}
                onChange={(e) => setSaveLabel(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void handleSave();
                  if (e.key === "Escape") setSaveOpen(false);
                }}
                placeholder="Label, e.g. Blue Shield Gold"
                className="w-52 rounded-md border border-slate-300 px-2 py-1 text-xs outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500"
              />
              <button
                type="button"
                onClick={() => void handleSave()}
                disabled={saving || !saveLabel.trim()}
                className="rounded-md bg-brand-600 px-3 py-1 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-40"
              >
                {saving ? "Saving…" : "Save"}
              </button>
              <button
                type="button"
                onClick={() => setSaveOpen(false)}
                className="text-xs text-slate-400 hover:text-slate-600"
              >
                Cancel
              </button>
              {saveError && (
                <span className="text-xs text-red-600">{saveError}</span>
              )}
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setSaveOpen(true)}
              className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
            >
              Save estimate
            </button>
          )}

          <button
            type="button"
            onClick={onRestart}
            className="text-xs text-slate-400 underline-offset-2 hover:text-slate-600 hover:underline"
          >
            Start over
          </button>
        </div>
      </div>

      {/* Two-column body */}
      <div className="flex flex-col gap-6 lg:flex-row lg:items-start">

        {/* ── Left column: estimate + what-if ── */}
        <div className="min-w-0 flex-1 space-y-6">
          {/* OOP interval hero */}
          <div className="overflow-hidden rounded-xl border border-slate-200 shadow-card">
            <EstimateBlock
              interval={oop_interval}
              chargesCents={prediction.median_charges_cents}
            />
            <div className="border-t border-slate-200 bg-slate-50 px-6 py-3 text-xs text-slate-500">
              80% charge interval:{" "}
              <span className="tabular-nums font-medium text-slate-700">
                {centsToDollars(prediction.lower_bound_cents)}
              </span>{" "}
              to{" "}
              <span className="tabular-nums font-medium text-slate-700">
                {centsToDollars(prediction.upper_bound_cents)}
              </span>
            </div>
          </div>

          {/* What-if */}
          <section className="space-y-3">
            <div className="flex items-end justify-between">
              <div>
                <h3 className="text-base font-semibold text-slate-900">What-if</h3>
                <p className="text-sm text-slate-600">
                  Hold everything else fixed and vary one factor.
                </p>
              </div>
              <Select<SweepFeature>
                label="Vary"
                value={sweepFeature}
                onChange={setSweepFeature}
                options={SWEEP_OPTIONS}
              />
            </div>
            <WhatIfChart
              data={sweep}
              loading={sweepLoading}
              error={sweepError}
              feature={sweepFeature}
            />
          </section>
        </div>

        {/* ── Right column: chatbot panel ── */}
        <div className="w-full shrink-0 lg:sticky lg:top-6 lg:w-96">
          <div className="flex h-[32rem] flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card lg:h-[calc(100vh-8rem)]">
            {/* Chat header */}
            <div className="border-b border-slate-200 px-4 py-3">
              <h3 className="text-sm font-semibold text-slate-900">
                Ask your plan
              </h3>
              <p className="mt-0.5 text-xs text-slate-500">
                {document_id
                  ? "Answers grounded in the pages you uploaded."
                  : "Upload a plan PDF to enable document Q&A."}
              </p>
            </div>

            {/* Message list */}
            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
              {chatMessages.length === 0 && (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
                  <span className="text-3xl">💬</span>
                  <p className="text-sm text-slate-500">
                    {document_id
                      ? 'Try "Is physical therapy covered?" or "What\'s my emergency copay?"'
                      : "Upload a plan PDF in step 2 to chat with your document."}
                  </p>
                </div>
              )}

              {chatMessages.map((m, i) => (
                <div key={i} className="space-y-1">
                  {/* User bubble */}
                  <div className="flex justify-end">
                    <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-brand-600 px-3 py-2 text-sm text-white">
                      {m.question}
                    </div>
                  </div>
                  {/* Assistant bubble */}
                  <div className="flex justify-start">
                    <div className="max-w-[85%] space-y-1 rounded-2xl rounded-tl-sm border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800">
                      <p className="whitespace-pre-wrap">{m.answer}</p>
                      {m.pages.length > 0 && (
                        <p className="text-xs text-slate-400">
                          Pages {[...new Set(m.pages)].sort((a, b) => a - b).join(", ")}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {chatLoading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl rounded-tl-sm border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-400">
                    Thinking…
                  </div>
                </div>
              )}

              {chatError && (
                <div className="rounded bg-red-50 px-3 py-2 text-xs text-red-700">
                  {chatError}
                </div>
              )}

              <div ref={chatBottomRef} />
            </div>

            {/* Input bar */}
            <div className="border-t border-slate-200 px-3 py-3">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") void handleAsk(); }}
                  disabled={!document_id}
                  placeholder={
                    document_id
                      ? "Ask about your plan…"
                      : "Upload a PDF to enable chat"
                  }
                  className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400"
                />
                <button
                  type="button"
                  onClick={() => void handleAsk()}
                  disabled={chatLoading || !question.trim() || !document_id}
                  className="rounded-lg bg-brand-600 px-3 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-40"
                >
                  {chatLoading ? "…" : "↑"}
                </button>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root wizard
// ---------------------------------------------------------------------------

/**
 * Orchestrates the full flow: landing (step 0) followed by the four-step
 * intake wizard.  Manages step transitions and passes shared state (session
 * ID, plan draft, final estimate) between steps.
 */
export function IntakeWizard() {
  const [step, setStep] = useState<Step>(0);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [draft, setDraft] = useState<PlanDraft | null>(null);
  const [estimate, setEstimate] = useState<SessionEstimate | null>(null);
  const [loadedEstimate, setLoadedEstimate] = useState<SavedEstimate | null>(null);

  function restart() {
    setStep(0);
    setSessionId(null);
    setDraft(null);
    setEstimate(null);
    setLoadedEstimate(null);
  }

  function handleLoad(saved: SavedEstimate) {
    setLoadedEstimate(saved);
    setStep(5);
  }

  /** Blank draft used when the user skips the PDF upload step. */
  const emptyDraft: PlanDraft = {
    deductible_cents: null,
    out_of_pocket_max_cents: null,
    coinsurance_rate: null,
    copays_cents: {},
    unresolved_fields: [],
    extraction_notes: [],
  };

  return (
    <div>
      {/* Landing screen — no step bar */}
      {step === 0 && <Step0Landing onStart={() => setStep(1)} onLoad={handleLoad} />}

      {/* Loaded saved estimate — read-only view */}
      {step === 5 && loadedEstimate && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-slate-900">{loadedEstimate.label}</h2>
              {loadedEstimate.plan && (
                <p className="mt-1 text-sm text-slate-600">
                  {loadedEstimate.plan.name} · ${(loadedEstimate.plan.deductible_cents / 100).toLocaleString()} deductible,{" "}
                  ${(loadedEstimate.plan.out_of_pocket_max_cents / 100).toLocaleString()} OOP max
                </p>
              )}
            </div>
            <button
              type="button"
              onClick={restart}
              className="text-xs text-slate-400 underline-offset-2 hover:text-slate-600 hover:underline"
            >
              ← Back
            </button>
          </div>
          <div className="overflow-hidden rounded-xl border border-slate-200 shadow-card">
            <EstimateBlock
              interval={loadedEstimate.oop_interval}
              chargesCents={loadedEstimate.prediction.median_charges_cents}
            />
            <div className="border-t border-slate-200 bg-slate-50 px-6 py-3 text-xs text-slate-500">
              80% charge interval:{" "}
              <span className="tabular-nums font-medium text-slate-700">
                {centsToDollars(loadedEstimate.prediction.lower_bound_cents)}
              </span>{" "}
              to{" "}
              <span className="tabular-nums font-medium text-slate-700">
                {centsToDollars(loadedEstimate.prediction.upper_bound_cents)}
              </span>
            </div>
          </div>
          <p className="text-xs text-slate-400">
            Saved {new Date(loadedEstimate.created_at).toLocaleString()} · Age {loadedEstimate.features.age},{" "}
            {loadedEstimate.features.smoker === "yes" ? "smoker" : "non-smoker"},{" "}
            BMI {loadedEstimate.features.bmi}
          </p>
        </div>
      )}

      {/* Intake wizard steps 1–4 */}
      {step !== 0 && step !== 5 && (
        <>
          <StepBar
            current={step as 1 | 2 | 3 | 4}
            labels={["Your info", "Your plan", "Confirm", "Estimate"]}
          />

          {step === 1 && (
            <Step1Demographics
              onDone={(resp) => {
                setSessionId(resp.session_id);
                setStep(2);
              }}
            />
          )}

          {step === 2 && sessionId && (
            <Step2Upload
              sessionId={sessionId}
              onDone={(d) => { setDraft(d); setStep(3); }}
              onSkip={() => { setDraft(emptyDraft); setStep(3); }}
            />
          )}

          {step === 3 && sessionId && draft && (
            <Step3Confirm
              draft={draft}
              sessionId={sessionId}
              onDone={(est) => { setEstimate(est); setStep(4); }}
            />
          )}

          {step === 4 && sessionId && estimate && (
            <Step4Estimate
              estimate={estimate}
              sessionId={sessionId}
              onRestart={restart}
            />
          )}
        </>
      )}
    </div>
  );
}

