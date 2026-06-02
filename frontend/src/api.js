/**
 * Tiny typed client for the health-app FastAPI backend.
 *
 * The base URL is read from the VITE_API_BASE env var at build time so
 * the same bundle can be pointed at a local dev backend or a deployed
 * one without code changes.
 */
const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
async function postJSON(path, body) {
    const res = await fetch(`${BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const detail = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${detail}`);
    }
    return res.json();
}
async function getJSON(path) {
    const res = await fetch(`${BASE}${path}`);
    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`);
    }
    return res.json();
}
export function predict(features, planId) {
    return postJSON("/predict", {
        features,
        plan_id: planId ?? null,
    });
}
export function whatif(baseline, feature, values, planId) {
    return postJSON("/whatif", {
        baseline,
        feature,
        values,
        plan_id: planId ?? null,
    });
}
export function getPlan(planId) {
    return getJSON(`/plans/${encodeURIComponent(planId)}`);
}
export async function uploadPDF(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/documents`, {
        method: "POST",
        body: form,
    });
    if (!res.ok) {
        const detail = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${detail}`);
    }
    return res.json();
}
export function listDocuments() {
    return getJSON("/documents");
}
export async function deleteDocument(documentId) {
    const res = await fetch(`${BASE}/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
    if (!res.ok && res.status !== 204) {
        throw new Error(`${res.status} ${res.statusText}`);
    }
}
export function askChat(documentId, question, topK = 4) {
    return postJSON("/chat", {
        document_id: documentId,
        question,
        top_k: topK,
    });
}
export const KNOWN_PLAN_IDS = ["hdhp_silver", "ppo_gold", "ppo_platinum"];
/**
 * Format a cent amount as a US dollar currency string.
 *
 * @param cents - Amount in whole US cents.
 * @returns Formatted string, e.g. `"$1,500"`.
 */
export function centsToDollars(cents) {
    return (cents / 100).toLocaleString("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
    });
}
// ---------------------------------------------------------------------------
// Session API functions
// ---------------------------------------------------------------------------
/**
 * Create a new session from the user's demographic inputs.
 *
 * @param features - User demographics (age, BMI, smoker status, etc.).
 * @returns New session ID and a first-pass cost prediction.
 */
export function createSession(features) {
    return postJSON("/sessions", { features });
}
/**
 * Upload a plan PDF and attach it to an existing session.
 *
 * The server automatically extracts plan fields (deductible, OOP max, etc.)
 * from the document and returns a draft for the user to review.
 *
 * @param sessionId - Session to attach the document to.
 * @param file - The PDF file chosen by the user.
 * @returns The new document ID and the auto-extracted plan draft.
 * @throws {Error} On network failure or a non-2xx response.
 */
export async function attachSessionDocument(sessionId, file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/document`, { method: "POST", body: form });
    if (!res.ok) {
        const detail = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${detail}`);
    }
    return res.json();
}
/**
 * Fetch the current plan draft for a session.
 *
 * @param sessionId - Target session.
 * @returns The extracted plan draft.
 * @throws {Error} If the session has no attached document yet.
 */
export function getSessionPlanDraft(sessionId) {
    return getJSON(`/sessions/${encodeURIComponent(sessionId)}/plan-draft`);
}
/**
 * Confirm (or correct) the plan fields and receive the full estimate.
 *
 * @param sessionId - Target session.
 * @param req - User-reviewed plan fields.
 * @returns Full personalised estimate with plan cost-share breakdown.
 */
export function confirmSessionPlan(sessionId, req) {
    return postJSON(`/sessions/${encodeURIComponent(sessionId)}/plan`, req);
}
/**
 * Fetch the current estimate for a session.
 *
 * @param sessionId - Target session.
 * @returns Current estimate; plan breakdown is `null` until a plan is confirmed.
 */
export function getSessionEstimate(sessionId) {
    return getJSON(`/sessions/${encodeURIComponent(sessionId)}/estimate`);
}
/**
 * Run a what-if sweep using the session's demographics as the baseline.
 *
 * @param sessionId - Target session.
 * @param feature - Feature to vary (e.g. `"age"`, `"smoker"`).
 * @param values - Values to sweep the feature across.
 * @returns Prediction at each swept value.
 */
export function sessionWhatIf(sessionId, feature, values) {
    return postJSON(`/sessions/${encodeURIComponent(sessionId)}/whatif`, { feature, values });
}
/**
 * Ask a plain-English question against the session's uploaded plan document.
 *
 * @param sessionId - Target session (must have an attached document).
 * @param question - Free-text question from the user.
 * @param topK - Number of chunks to retrieve for context. Defaults to 4.
 * @returns Answer text and page-level citations.
 */
export function sessionChat(sessionId, question, topK = 4) {
    return postJSON(`/sessions/${encodeURIComponent(sessionId)}/chat`, { question, top_k: topK });
}
