import { useState } from "react";
import { DocChatPage } from "./components/DocChatPage";
import { IntakeWizard } from "./components/IntakeWizard";
import { PredictorPage } from "./components/PredictorPage";

type Tab = "guided" | "predictor" | "docchat";

const TABS: { id: Tab; label: string; description: string }[] = [
  {
    id: "guided",
    label: "Get my estimate",
    description:
      "Step-by-step guided flow: enter your details, upload your plan PDF, and get a personalised cost estimate with a full plan cost-share breakdown.",
  },
  {
    id: "predictor",
    label: "Cost predictor",
    description:
      "ML model trained on real insurance data. Adjust your details to see annual charges with an 80% prediction interval and what you'd actually pay on a chosen plan.",
  },
  {
    id: "docchat",
    label: "Document chat",
    description:
      "Upload a plan PDF — Summary of Benefits, Evidence of Coverage, anything. Ask questions in plain English and get answers with page citations.",
  },
];

/** Simple shield + pulse icon for the brand mark */
function BrandIcon() {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="h-8 w-8 flex-shrink-0"
      aria-hidden="true"
    >
      <rect width="32" height="32" rx="8" fill="#2563eb" />
      {/* shield outline */}
      <path
        d="M16 6 L24 9.5 V17 C24 21.5 20.5 25 16 26 C11.5 25 8 21.5 8 17 V9.5 Z"
        fill="white"
        fillOpacity="0.18"
        stroke="white"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      {/* cross / plus */}
      <path
        d="M16 12 V20 M12 16 H20"
        stroke="white"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("guided");
  const active = TABS.find((t) => t.id === tab)!;

  return (
    <div className="min-h-screen">
      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <div className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-6">
          {/* Brand row */}
          <div className="flex items-center gap-3 py-5">
            <BrandIcon />
            <div>
              <h1 className="text-lg font-semibold leading-none tracking-tight text-slate-900">
                ClearPlan
              </h1>
              <p className="mt-0.5 text-xs text-slate-500">
                Health insurance cost estimator
              </p>
            </div>
          </div>

          {/* Tab bar */}
          <nav
            className="-mb-px flex gap-1"
            role="tablist"
            aria-label="App sections"
          >
            {TABS.map((t) => {
              const selected = t.id === tab;
              return (
                <button
                  key={t.id}
                  type="button"
                  role="tab"
                  aria-selected={selected}
                  onClick={() => setTab(t.id)}
                  className={
                    "relative px-4 py-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2 " +
                    (selected
                      ? "text-brand-600"
                      : "text-slate-500 hover:text-slate-800")
                  }
                >
                  {t.label}
                  {/* active underline */}
                  {selected && (
                    <span className="absolute inset-x-0 bottom-0 h-0.5 rounded-full bg-brand-600" />
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* ── Main content ────────────────────────────────────────────── */}
      <main className="mx-auto max-w-5xl px-6 py-8">
        {/* Tab description */}
        <p className="mb-6 text-sm text-slate-500">{active.description}</p>

        {tab === "guided" ? (
          <IntakeWizard />
        ) : tab === "predictor" ? (
          <PredictorPage />
        ) : (
          <DocChatPage />
        )}
      </main>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="mx-auto max-w-5xl px-6 pb-8">
        <p className="text-xs text-slate-400">
          Illustrative prototype — not a substitute for an actual insurance
          quote or advice from your insurer.
        </p>
      </footer>
    </div>
  );
}
