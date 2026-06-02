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
  AnnualPlanShare,
  ConfirmPlanRequest,
  CreateSessionResponse,
  PlanDraft,
  Region,
  SessionEstimate,
  Sex,
  Smoker,
  SweepFeature,
  WhatIfResponse,
  attachSessionDocument,
  centsToDollars,
  confirmSessionPlan,
  createSession,
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

/** Which step of the wizard the user is currently on (1-indexed). */
type Step = 1 | 2 | 3 | 4;

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
// Step indicator
// ---------------------------------------------------------------------------

/**
 * Horizontal stepper showing the user's position in the four-step flow.
 *
 * @param current - Current step number (1-indexed).
 * @param labels - Display label for each step.
 */
function StepBar({
  current,
  labels,
}: {
  current: Step;
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
          Nothing is stored beyond your browser session.
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

  /** Renders a labelled dollar-amount input for a required plan field. */
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
 * Renders a plan cost-share breakdown block.
 *
 * @param title - Section heading (e.g. "Typical year").
 * @param chargesCents - Raw predicted charges in cents.
 * @param share - Plan cost-share breakdown, or null if no plan is set.
 */
function EstimateBlock({
  title,
  chargesCents,
  share,
}: {
  title: string;
  chargesCents: number;
  share: AnnualPlanShare | null;
}) {
  return (
    <div className="bg-white p-6">
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        {title}
      </div>
      <div className="mt-1 text-2xl font-semibold text-slate-900 tabular-nums">
        {centsToDollars(chargesCents)}
      </div>
      <div className="text-xs text-slate-500">predicted gross charges</div>
      {share && (
        <div className="mt-4 border-t border-slate-100 pt-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            You pay
          </div>
          <div className="text-xl font-semibold text-brand-700 tabular-nums">
            {centsToDollars(share.member_pays_cents)}
          </div>
          <dl className="mt-2 grid grid-cols-2 gap-y-0.5 text-xs">
            <dt className="text-slate-500">Deductible</dt>
            <dd className="text-right tabular-nums">
              {centsToDollars(share.deductible_applied_cents)}
            </dd>
            <dt className="text-slate-500">Coinsurance</dt>
            <dd className="text-right tabular-nums">
              {centsToDollars(share.coinsurance_cents)}
            </dd>
            <dt className="text-slate-500">Plan pays</dt>
            <dd className="text-right tabular-nums">
              {centsToDollars(share.plan_pays_cents)}
            </dd>
          </dl>
          {share.capped_at_oop_max && (
            <div className="mt-2 rounded bg-emerald-50 px-2 py-1 text-xs text-emerald-700">
              Out-of-pocket maximum reached — plan absorbs the rest.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * The final step: shows the full personalised estimate, a what-if chart keyed
 * to the session, and an inline Q&A panel if a document was attached.
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
  const { prediction, plan, annual_plan_share_median, annual_plan_share_mean, document_id } =
    estimate;

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

  async function handleAsk() {
    if (!question.trim() || !document_id) return;
    setChatLoading(true);
    setChatError(null);
    try {
      const resp = await sessionChat(sessionId, question.trim());
      setChatMessages((prev) => [
        ...prev,
        {
          question: question.trim(),
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

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
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
        <button
          type="button"
          onClick={onRestart}
          className="text-xs text-slate-400 underline-offset-2 hover:text-slate-600 hover:underline"
        >
          Start over
        </button>
      </div>

      {/* Prediction + plan breakdown */}
      <div className="overflow-hidden rounded-xl border border-slate-200 shadow-card">
        <div className="grid grid-cols-1 gap-px bg-slate-100 sm:grid-cols-2">
          <EstimateBlock
            title="Typical year (median)"
            chargesCents={prediction.median_charges_cents}
            share={annual_plan_share_median}
          />
          <EstimateBlock
            title="Long-run average (mean)"
            chargesCents={prediction.mean_charges_cents}
            share={annual_plan_share_mean}
          />
        </div>
        <div className="border-t border-slate-200 bg-white px-6 py-3 text-xs text-slate-500">
          80% interval:{" "}
          <span className="tabular-nums">
            {centsToDollars(prediction.lower_bound_cents)}
          </span>{" "}
          to{" "}
          <span className="tabular-nums">
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

      {/* Q&A (only shown when a document is attached) */}
      {document_id && (
        <section className="space-y-4">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Ask your plan document
            </h3>
            <p className="text-sm text-slate-600">
              Ask anything about your plan in plain English — coverage, copays,
              exclusions. Answers are pulled directly from the pages you uploaded.
            </p>
          </div>

          <div className="space-y-3">
            {chatMessages.map((m, i) => (
              <div key={i} className="rounded-lg border border-slate-200 bg-white p-4 text-sm">
                <p className="font-medium text-slate-900">{m.question}</p>
                <p className="mt-2 text-slate-700 whitespace-pre-wrap">{m.answer}</p>
                {m.pages.length > 0 && (
                  <p className="mt-2 text-xs text-slate-400">
                    Source pages: {[...new Set(m.pages)].sort((a, b) => a - b).join(", ")}
                  </p>
                )}
              </div>
            ))}
          </div>

          {chatError && (
            <div className="rounded bg-red-50 px-4 py-3 text-sm text-red-700">
              {chatError}
            </div>
          )}

          <div className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") void handleAsk(); }}
              placeholder="e.g. Is physical therapy covered?"
              className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500"
            />
            <button
              type="button"
              onClick={() => void handleAsk()}
              disabled={chatLoading || !question.trim()}
              className="rounded-md bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
            >
              {chatLoading ? "…" : "Ask"}
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root wizard
// ---------------------------------------------------------------------------

/**
 * Orchestrates the four-step intake flow.  Manages step transitions and
 * passes shared state (session ID, plan draft, final estimate) between steps.
 */
export function IntakeWizard() {
  const [step, setStep] = useState<Step>(1);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [draft, setDraft] = useState<PlanDraft | null>(null);
  const [estimate, setEstimate] = useState<SessionEstimate | null>(null);

  function restart() {
    setStep(1);
    setSessionId(null);
    setDraft(null);
    setEstimate(null);
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
      <StepBar
        current={step}
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
    </div>
  );
}
