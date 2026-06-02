import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
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
import { attachSessionDocument, centsToDollars, confirmSessionPlan, createSession, sessionChat, sessionWhatIf, } from "../api";
import { Select } from "./Select";
import { Slider } from "./Slider";
import { Toggle } from "./Toggle";
import { WhatIfChart } from "./WhatIfChart";
const REGIONS = ["northeast", "midwest", "south", "west"];
const SWEEP_OPTIONS = [
    { value: "age", label: "Age" },
    { value: "bmi", label: "BMI" },
    { value: "children", label: "Children" },
    { value: "smoker", label: "Smoker" },
    { value: "sex", label: "Sex" },
    { value: "region", label: "Region" },
];
/** Preset sweep values for each sweepable feature. */
function sweepValuesFor(feature) {
    switch (feature) {
        case "age": return [20, 25, 30, 35, 40, 45, 50, 55, 60, 64];
        case "bmi": return [18, 22, 26, 30, 34, 38, 42];
        case "children": return [0, 1, 2, 3, 4, 5];
        case "smoker": return ["no", "yes"];
        case "sex": return ["female", "male"];
        case "region": return REGIONS;
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
function StepBar({ current, labels, }) {
    return (_jsx("ol", { className: "mb-8 flex items-center gap-0", children: labels.map((label, i) => {
            const stepNum = (i + 1);
            const done = stepNum < current;
            const active = stepNum === current;
            return (_jsxs("li", { className: "flex flex-1 items-center", children: [_jsxs("div", { className: "flex flex-col items-center gap-1", children: [_jsx("div", { className: "flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold " +
                                    (done
                                        ? "bg-brand-600 text-white"
                                        : active
                                            ? "border-2 border-brand-600 text-brand-600"
                                            : "border-2 border-slate-300 text-slate-400"), children: done ? "✓" : stepNum }), _jsx("span", { className: "hidden text-xs sm:block " +
                                    (active ? "font-semibold text-slate-900" : "text-slate-500"), children: label })] }), i < labels.length - 1 && (_jsx("div", { className: "mx-2 h-0.5 flex-1 " +
                            (done ? "bg-brand-600" : "bg-slate-200") }))] }, label));
        }) }));
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
function Step1Demographics({ onDone, }) {
    const [features, setFeatures] = useState({
        age: 35,
        sex: "female",
        bmi: 27.5,
        children: 1,
        smoker: "no",
        region: "northeast",
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    /** Update a single feature field. */
    const update = useCallback((key, val) => setFeatures((prev) => ({ ...prev, [key]: val })), []);
    async function handleContinue() {
        setLoading(true);
        setError(null);
        try {
            const resp = await createSession(features);
            onDone(resp, features);
        }
        catch (e) {
            setError(String(e));
        }
        finally {
            setLoading(false);
        }
    }
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-xl font-semibold text-slate-900", children: "Tell us about yourself" }), _jsx("p", { className: "mt-1 text-sm text-slate-600", children: "We use these details to predict your expected annual medical spend. Nothing is stored beyond your browser session." })] }), _jsxs("div", { className: "grid grid-cols-1 gap-5 rounded-xl border border-slate-200 bg-white p-6 shadow-card sm:grid-cols-2", children: [_jsx(Slider, { label: "Age", min: 18, max: 64, value: features.age, onChange: (v) => update("age", v), format: (v) => `${v} yrs` }), _jsx(Toggle, { label: "Sex", value: features.sex, options: [{ value: "female", label: "Female" }, { value: "male", label: "Male" }], onChange: (v) => update("sex", v) }), _jsx(Slider, { label: "BMI", min: 16, max: 53, step: 0.1, value: features.bmi, onChange: (v) => update("bmi", Number(v.toFixed(1))), format: (v) => v.toFixed(1) }), _jsx(Slider, { label: "Children", min: 0, max: 5, value: features.children, onChange: (v) => update("children", v) }), _jsx(Toggle, { label: "Smoker", value: features.smoker, options: [{ value: "no", label: "No" }, { value: "yes", label: "Yes" }], onChange: (v) => update("smoker", v) }), _jsx(Select, { label: "Region", value: features.region, options: REGIONS.map((r) => ({ value: r, label: r.charAt(0).toUpperCase() + r.slice(1) })), onChange: (v) => update("region", v) })] }), error && (_jsx("div", { className: "rounded bg-red-50 px-4 py-3 text-sm text-red-700", children: error })), _jsx("div", { className: "flex justify-end", children: _jsx("button", { type: "button", onClick: () => void handleContinue(), disabled: loading, className: "rounded-md bg-brand-600 px-5 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50", children: loading ? "Creating session…" : "Continue →" }) })] }));
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
function Step2Upload({ sessionId, onDone, onSkip, }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const inputRef = useRef(null);
    async function handleFile(file) {
        setLoading(true);
        setError(null);
        try {
            const resp = await attachSessionDocument(sessionId, file);
            onDone(resp.plan_draft, resp.document_id);
        }
        catch (e) {
            setError(String(e));
        }
        finally {
            setLoading(false);
        }
    }
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-xl font-semibold text-slate-900", children: "Upload your plan document" }), _jsx("p", { className: "mt-1 text-sm text-slate-600", children: "Drop your Summary of Benefits and Coverage or any plan PDF. We will automatically extract your deductible, out-of-pocket max, coinsurance rate, and copays -- you can review and edit them on the next screen." })] }), _jsxs("div", { className: "flex h-48 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 bg-white transition hover:border-brand-400 hover:bg-brand-50", onClick: () => inputRef.current?.click(), onDragOver: (e) => e.preventDefault(), onDrop: (e) => {
                    e.preventDefault();
                    const file = e.dataTransfer.files[0];
                    if (file)
                        void handleFile(file);
                }, children: [_jsx("input", { ref: inputRef, type: "file", accept: ".pdf,application/pdf", className: "sr-only", onChange: (e) => {
                            const file = e.target.files?.[0];
                            if (file)
                                void handleFile(file);
                        } }), loading ? (_jsx("p", { className: "text-sm text-slate-500", children: "Uploading and extracting\u2026" })) : (_jsxs(_Fragment, { children: [_jsx("p", { className: "text-sm font-medium text-slate-700", children: "Click to choose a PDF, or drag and drop" }), _jsx("p", { className: "mt-1 text-xs text-slate-500", children: "Max 25 MB" })] }))] }), error && (_jsx("div", { className: "rounded bg-red-50 px-4 py-3 text-sm text-red-700", children: error })), _jsx("div", { className: "flex items-center justify-between", children: _jsx("button", { type: "button", onClick: onSkip, className: "text-sm text-slate-500 underline-offset-2 hover:text-slate-700 hover:underline", children: "I don't have a PDF \u2014 enter plan details manually" }) })] }));
}
// ---------------------------------------------------------------------------
// Step 3 — Confirm extracted plan fields
// ---------------------------------------------------------------------------
/** Display name for each ServiceCategory value. */
const CATEGORY_LABELS = {
    office_visit: "Primary care visit",
    specialist: "Specialist visit",
    urgent_care: "Urgent care",
    generic_drug: "Generic drug",
    emergency: "Emergency room",
    imaging: "Imaging (MRI / CT)",
    lab: "Lab work",
    surgery: "Surgery",
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
function Step3Confirm({ draft, sessionId, onDone, }) {
    /** Convert cents → dollars string for display in inputs. */
    const centsToInput = (c) => c !== null ? String(c / 100) : "";
    const [deductible, setDeductible] = useState(centsToInput(draft.deductible_cents));
    const [oopMax, setOopMax] = useState(centsToInput(draft.out_of_pocket_max_cents));
    const [coinsurance, setCoinsurance] = useState(draft.coinsurance_rate !== null ? String(draft.coinsurance_rate * 100) : "");
    const [copays, setCopays] = useState(() => {
        const init = {};
        for (const [cat, cents] of Object.entries(draft.copays_cents)) {
            init[cat] = String(cents / 100);
        }
        return init;
    });
    const [planName, setPlanName] = useState("My Plan");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
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
        const copaysCents = {};
        for (const [cat, val] of Object.entries(copays)) {
            if (val)
                copaysCents[cat] = Math.round(parseFloat(val) * 100);
        }
        const req = {
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
        }
        catch (e) {
            setError(String(e));
        }
        finally {
            setLoading(false);
        }
    }
    /** Renders a labelled dollar-amount input for a required plan field. */
    function DollarField({ label, value, onChange, missing, }) {
        return (_jsxs("label", { className: "flex flex-col gap-1 text-sm", children: [_jsxs("span", { className: "font-medium text-slate-700", children: [label, " ", missing && (_jsx("span", { className: "ml-1 rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-700", children: "not found" }))] }), _jsxs("div", { className: "flex items-center rounded-md border border-slate-300 bg-white px-3 py-1.5", children: [_jsx("span", { className: "mr-1 text-slate-400", children: "$" }), _jsx("input", { type: "number", min: 0, step: 1, value: value, onChange: (e) => onChange(e.target.value), className: "flex-1 border-none bg-transparent text-slate-900 outline-none", placeholder: "0" })] })] }));
    }
    const unresolved = new Set(draft.unresolved_fields);
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-xl font-semibold text-slate-900", children: "Review your plan details" }), _jsx("p", { className: "mt-1 text-sm text-slate-600", children: "We extracted these numbers from your document. Check them against your plan and correct anything that looks wrong before continuing." })] }), draft.extraction_notes.length > 0 && (_jsxs("details", { className: "rounded-lg border border-amber-200 bg-amber-50 px-4 py-3", children: [_jsxs("summary", { className: "cursor-pointer text-sm font-medium text-amber-800", children: [draft.unresolved_fields.length, " field", draft.unresolved_fields.length !== 1 ? "s" : "", " could not be read automatically \u2014 expand to see notes"] }), _jsx("ul", { className: "mt-2 space-y-1 text-xs text-amber-700", children: draft.extraction_notes.map((n, i) => (_jsxs("li", { children: ["\u2022 ", n] }, i))) })] })), _jsxs("div", { className: "rounded-xl border border-slate-200 bg-white p-6 shadow-card", children: [_jsxs("label", { className: "mb-5 flex flex-col gap-1 text-sm", children: [_jsx("span", { className: "font-medium text-slate-700", children: "Plan name" }), _jsx("input", { type: "text", value: planName, onChange: (e) => setPlanName(e.target.value), className: "rounded-md border border-slate-300 px-3 py-1.5 text-slate-900 outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500" })] }), _jsx("h3", { className: "mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500", children: "Core cost-share numbers" }), _jsxs("div", { className: "grid grid-cols-1 gap-4 sm:grid-cols-3", children: [_jsx(DollarField, { label: "Annual deductible", value: deductible, onChange: setDeductible, missing: unresolved.has("deductible") }), _jsx(DollarField, { label: "Out-of-pocket maximum", value: oopMax, onChange: setOopMax, missing: unresolved.has("oop_max") }), _jsxs("label", { className: "flex flex-col gap-1 text-sm", children: [_jsxs("span", { className: "font-medium text-slate-700", children: ["Coinsurance rate", " ", unresolved.has("coinsurance") && (_jsx("span", { className: "ml-1 rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-700", children: "not found" }))] }), _jsxs("div", { className: "flex items-center rounded-md border border-slate-300 bg-white px-3 py-1.5", children: [_jsx("input", { type: "number", min: 0, max: 100, step: 1, value: coinsurance, onChange: (e) => setCoinsurance(e.target.value), className: "flex-1 border-none bg-transparent text-slate-900 outline-none", placeholder: "20" }), _jsx("span", { className: "ml-1 text-slate-400", children: "%" })] })] })] }), _jsx("h3", { className: "mb-3 mt-6 text-xs font-semibold uppercase tracking-wide text-slate-500", children: "Copays (optional \u2014 leave blank if not applicable)" }), _jsx("div", { className: "grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4", children: Object.entries(CATEGORY_LABELS).map(([cat, label]) => (_jsxs("label", { className: "flex flex-col gap-1 text-sm", children: [_jsx("span", { className: "font-medium text-slate-700", children: label }), _jsxs("div", { className: "flex items-center rounded-md border border-slate-300 bg-white px-3 py-1.5", children: [_jsx("span", { className: "mr-1 text-slate-400", children: "$" }), _jsx("input", { type: "number", min: 0, step: 1, value: copays[cat] ?? "", onChange: (e) => setCopays((prev) => ({ ...prev, [cat]: e.target.value })), className: "flex-1 border-none bg-transparent text-slate-900 outline-none", placeholder: "\u2014" })] })] }, cat))) })] }), error && (_jsx("div", { className: "rounded bg-red-50 px-4 py-3 text-sm text-red-700", children: error })), _jsx("div", { className: "flex justify-end", children: _jsx("button", { type: "button", onClick: () => void handleConfirm(), disabled: loading, className: "rounded-md bg-brand-600 px-5 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50", children: loading ? "Calculating…" : "See my estimate →" }) })] }));
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
function EstimateBlock({ title, chargesCents, share, }) {
    return (_jsxs("div", { className: "bg-white p-6", children: [_jsx("div", { className: "text-xs font-semibold uppercase tracking-wide text-slate-500", children: title }), _jsx("div", { className: "mt-1 text-2xl font-semibold text-slate-900 tabular-nums", children: centsToDollars(chargesCents) }), _jsx("div", { className: "text-xs text-slate-500", children: "predicted gross charges" }), share && (_jsxs("div", { className: "mt-4 border-t border-slate-100 pt-3", children: [_jsx("div", { className: "text-xs font-semibold uppercase tracking-wide text-slate-500", children: "You pay" }), _jsx("div", { className: "text-xl font-semibold text-brand-700 tabular-nums", children: centsToDollars(share.member_pays_cents) }), _jsxs("dl", { className: "mt-2 grid grid-cols-2 gap-y-0.5 text-xs", children: [_jsx("dt", { className: "text-slate-500", children: "Deductible" }), _jsx("dd", { className: "text-right tabular-nums", children: centsToDollars(share.deductible_applied_cents) }), _jsx("dt", { className: "text-slate-500", children: "Coinsurance" }), _jsx("dd", { className: "text-right tabular-nums", children: centsToDollars(share.coinsurance_cents) }), _jsx("dt", { className: "text-slate-500", children: "Plan pays" }), _jsx("dd", { className: "text-right tabular-nums", children: centsToDollars(share.plan_pays_cents) })] }), share.capped_at_oop_max && (_jsx("div", { className: "mt-2 rounded bg-emerald-50 px-2 py-1 text-xs text-emerald-700", children: "Out-of-pocket maximum reached \u2014 plan absorbs the rest." }))] }))] }));
}
/**
 * The final step: shows the full personalised estimate, a what-if chart keyed
 * to the session, and an inline Q&A panel if a document was attached.
 *
 * @param estimate - Initial estimate from the confirmSessionPlan call.
 * @param sessionId - Active session (used for what-if and chat requests).
 * @param onRestart - Callback to reset the wizard and start over.
 */
function Step4Estimate({ estimate, sessionId, onRestart, }) {
    const { prediction, plan, annual_plan_share_median, annual_plan_share_mean, document_id } = estimate;
    // What-if state
    const [sweepFeature, setSweepFeature] = useState("age");
    const [sweep, setSweep] = useState(null);
    const [sweepLoading, setSweepLoading] = useState(false);
    const [sweepError, setSweepError] = useState(null);
    // Q&A state
    const [question, setQuestion] = useState("");
    const [chatLoading, setChatLoading] = useState(false);
    const [chatMessages, setChatMessages] = useState([]);
    const [chatError, setChatError] = useState(null);
    // Fire what-if whenever the sweep feature changes
    useEffect(() => {
        let cancelled = false;
        const values = sweepValuesFor(sweepFeature);
        setSweepLoading(true);
        setSweepError(null);
        sessionWhatIf(sessionId, sweepFeature, values)
            .then((r) => { if (!cancelled)
            setSweep(r); })
            .catch((e) => { if (!cancelled)
            setSweepError(String(e)); })
            .finally(() => { if (!cancelled)
            setSweepLoading(false); });
        return () => { cancelled = true; };
    }, [sessionId, sweepFeature]);
    async function handleAsk() {
        if (!question.trim() || !document_id)
            return;
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
        }
        catch (e) {
            setChatError(String(e));
        }
        finally {
            setChatLoading(false);
        }
    }
    return (_jsxs("div", { className: "space-y-8", children: [_jsxs("div", { className: "flex items-start justify-between", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-xl font-semibold text-slate-900", children: "Your estimate" }), plan && (_jsxs("p", { className: "mt-1 text-sm text-slate-600", children: ["Based on ", _jsx("span", { className: "font-medium", children: plan.name }), " \u2014 $", (plan.deductible_cents / 100).toLocaleString(), " deductible,", " ", "$", (plan.out_of_pocket_max_cents / 100).toLocaleString(), " OOP max,", " ", Math.round(plan.coinsurance_rate * 100), "% coinsurance."] }))] }), _jsx("button", { type: "button", onClick: onRestart, className: "text-xs text-slate-400 underline-offset-2 hover:text-slate-600 hover:underline", children: "Start over" })] }), _jsxs("div", { className: "overflow-hidden rounded-xl border border-slate-200 shadow-card", children: [_jsxs("div", { className: "grid grid-cols-1 gap-px bg-slate-100 sm:grid-cols-2", children: [_jsx(EstimateBlock, { title: "Typical year (median)", chargesCents: prediction.median_charges_cents, share: annual_plan_share_median }), _jsx(EstimateBlock, { title: "Long-run average (mean)", chargesCents: prediction.mean_charges_cents, share: annual_plan_share_mean })] }), _jsxs("div", { className: "border-t border-slate-200 bg-white px-6 py-3 text-xs text-slate-500", children: ["80% interval:", " ", _jsx("span", { className: "tabular-nums", children: centsToDollars(prediction.lower_bound_cents) }), " ", "to", " ", _jsx("span", { className: "tabular-nums", children: centsToDollars(prediction.upper_bound_cents) })] })] }), _jsxs("section", { className: "space-y-3", children: [_jsxs("div", { className: "flex items-end justify-between", children: [_jsxs("div", { children: [_jsx("h3", { className: "text-base font-semibold text-slate-900", children: "What-if" }), _jsx("p", { className: "text-sm text-slate-600", children: "Hold everything else fixed and vary one factor." })] }), _jsx(Select, { label: "Vary", value: sweepFeature, onChange: setSweepFeature, options: SWEEP_OPTIONS })] }), _jsx(WhatIfChart, { data: sweep, loading: sweepLoading, error: sweepError, feature: sweepFeature })] }), document_id && (_jsxs("section", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("h3", { className: "text-base font-semibold text-slate-900", children: "Ask your plan document" }), _jsx("p", { className: "text-sm text-slate-600", children: "Ask anything about your plan in plain English \u2014 coverage, copays, exclusions. Answers are pulled directly from the pages you uploaded." })] }), _jsx("div", { className: "space-y-3", children: chatMessages.map((m, i) => (_jsxs("div", { className: "rounded-lg border border-slate-200 bg-white p-4 text-sm", children: [_jsx("p", { className: "font-medium text-slate-900", children: m.question }), _jsx("p", { className: "mt-2 text-slate-700 whitespace-pre-wrap", children: m.answer }), m.pages.length > 0 && (_jsxs("p", { className: "mt-2 text-xs text-slate-400", children: ["Source pages: ", [...new Set(m.pages)].sort((a, b) => a - b).join(", ")] }))] }, i))) }), chatError && (_jsx("div", { className: "rounded bg-red-50 px-4 py-3 text-sm text-red-700", children: chatError })), _jsxs("div", { className: "flex gap-2", children: [_jsx("input", { type: "text", value: question, onChange: (e) => setQuestion(e.target.value), onKeyDown: (e) => { if (e.key === "Enter")
                                    void handleAsk(); }, placeholder: "e.g. Is physical therapy covered?", className: "flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500" }), _jsx("button", { type: "button", onClick: () => void handleAsk(), disabled: chatLoading || !question.trim(), className: "rounded-md bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50", children: chatLoading ? "…" : "Ask" })] })] }))] }));
}
// ---------------------------------------------------------------------------
// Root wizard
// ---------------------------------------------------------------------------
/**
 * Orchestrates the four-step intake flow.  Manages step transitions and
 * passes shared state (session ID, plan draft, final estimate) between steps.
 */
export function IntakeWizard() {
    const [step, setStep] = useState(1);
    const [sessionId, setSessionId] = useState(null);
    const [draft, setDraft] = useState(null);
    const [estimate, setEstimate] = useState(null);
    function restart() {
        setStep(1);
        setSessionId(null);
        setDraft(null);
        setEstimate(null);
    }
    /** Blank draft used when the user skips the PDF upload step. */
    const emptyDraft = {
        deductible_cents: null,
        out_of_pocket_max_cents: null,
        coinsurance_rate: null,
        copays_cents: {},
        unresolved_fields: [],
        extraction_notes: [],
    };
    return (_jsxs("div", { children: [_jsx(StepBar, { current: step, labels: ["Your info", "Your plan", "Confirm", "Estimate"] }), step === 1 && (_jsx(Step1Demographics, { onDone: (resp) => {
                    setSessionId(resp.session_id);
                    setStep(2);
                } })), step === 2 && sessionId && (_jsx(Step2Upload, { sessionId: sessionId, onDone: (d) => { setDraft(d); setStep(3); }, onSkip: () => { setDraft(emptyDraft); setStep(3); } })), step === 3 && sessionId && draft && (_jsx(Step3Confirm, { draft: draft, sessionId: sessionId, onDone: (est) => { setEstimate(est); setStep(4); } })), step === 4 && sessionId && estimate && (_jsx(Step4Estimate, { estimate: estimate, sessionId: sessionId, onRestart: restart }))] }));
}
