/**
 * Tiny typed client for the health-app FastAPI backend.
 *
 * The base URL is read from the VITE_API_BASE env var at build time so
 * the same bundle can be pointed at a local dev backend or a deployed
 * one without code changes.
 */

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export type Sex = "male" | "female";
export type Smoker = "yes" | "no";
export type Region = "northeast" | "midwest" | "south" | "west";

export interface PredictionFeatures {
  age: number;
  sex: Sex;
  bmi: number;
  children: number;
  smoker: Smoker;
  region: Region;
}

export interface CostPrediction {
  predicted_charges_cents: number;
  lower_bound_cents: number;
  upper_bound_cents: number;
}

export interface AnnualPlanShare {
  charges_cents: number;
  deductible_applied_cents: number;
  coinsurance_cents: number;
  member_pays_cents: number;
  plan_pays_cents: number;
  capped_at_oop_max: boolean;
}

export interface PredictResponse {
  prediction: CostPrediction;
  annual_plan_share: AnnualPlanShare | null;
}

export type SweepFeature =
  | "age"
  | "sex"
  | "bmi"
  | "children"
  | "smoker"
  | "region";

export interface WhatIfPoint {
  value: number | string;
  prediction: CostPrediction;
  annual_plan_share: AnnualPlanShare | null;
}

export interface WhatIfResponse {
  feature: SweepFeature;
  points: WhatIfPoint[];
}

export interface Plan {
  plan_id: string;
  name: string;
  deductible_cents: number;
  out_of_pocket_max_cents: number;
  coinsurance_rate: number;
  copays_cents: Record<string, number>;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function predict(
  features: PredictionFeatures,
  planId?: string
): Promise<PredictResponse> {
  return postJSON<PredictResponse>("/predict", {
    features,
    plan_id: planId ?? null,
  });
}

export function whatif(
  baseline: PredictionFeatures,
  feature: SweepFeature,
  values: Array<number | string>,
  planId?: string
): Promise<WhatIfResponse> {
  return postJSON<WhatIfResponse>("/whatif", {
    baseline,
    feature,
    values,
    plan_id: planId ?? null,
  });
}

export function getPlan(planId: string): Promise<Plan> {
  return getJSON<Plan>(`/plans/${encodeURIComponent(planId)}`);
}

// --- document chat ------------------------------------------------------

export interface DocumentMeta {
  document_id: string;
  filename: string;
  n_pages: number;
  n_chunks: number;
  uploaded_at: string;
}

export interface UploadResponse {
  document: DocumentMeta;
}

export interface Citation {
  document_id: string;
  chunk_index: number;
  page_numbers: number[];
  snippet: string;
}

export interface ChatResponse {
  document_id: string;
  question: string;
  answer: string;
  citations: Citation[];
  llm_used: string;
}

export async function uploadPDF(file: File): Promise<UploadResponse> {
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

export function listDocuments(): Promise<DocumentMeta[]> {
  return getJSON<DocumentMeta[]>("/documents");
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await fetch(
    `${BASE}/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" }
  );
  if (!res.ok && res.status !== 204) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
}

export function askChat(
  documentId: string,
  question: string,
  topK = 4
): Promise<ChatResponse> {
  return postJSON<ChatResponse>("/chat", {
    document_id: documentId,
    question,
    top_k: topK,
  });
}

export const KNOWN_PLAN_IDS = ["hdhp_silver", "ppo_gold", "ppo_platinum"] as const;
export type KnownPlanId = (typeof KNOWN_PLAN_IDS)[number];

export function centsToDollars(cents: number): string {
  return (cents / 100).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}
