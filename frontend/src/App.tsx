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
      "ML model trained on the Kaggle insurance dataset. Adjust your details to see annual charges with an 80% interval and what you would actually pay on a chosen plan.",
  },
  {
    id: "docchat",
    label: "Document chat",
    description:
      "Upload a plan PDF (Summary of Benefits and Coverage, plan documents, anything). Ask questions in plain English; the chatbot retrieves the relevant passages and cites the page they came from.",
  },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("guided");
  const active = TABS.find((t) => t.id === tab)!;

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-6">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
          Health app
        </h1>
        <nav className="mt-4 inline-flex rounded-md border border-slate-300 bg-white p-0.5">
          {TABS.map((t) => {
            const selected = t.id === tab;
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={
                  "rounded px-4 py-1.5 text-sm transition " +
                  (selected
                    ? "bg-brand-600 text-white"
                    : "text-slate-700 hover:bg-slate-100")
                }
              >
                {t.label}
              </button>
            );
          })}
        </nav>
        <p className="mt-3 max-w-3xl text-sm text-slate-600">
          {active.description}
        </p>
        <p className="mt-2 max-w-3xl text-xs text-slate-500">
          Illustrative prototype. Not a substitute for an actual insurance
          quote or for advice from your insurer.
        </p>
      </header>

      {tab === "guided" ? (
        <IntakeWizard />
      ) : tab === "predictor" ? (
        <PredictorPage />
      ) : (
        <DocChatPage />
      )}

      <footer className="mt-10 text-xs text-slate-500">
        Backend at{" "}
        <code className="rounded bg-slate-100 px-1.5 py-0.5">
          {import.meta.env.VITE_API_BASE ?? "http://localhost:8000"}
        </code>
        . Set <code>VITE_API_BASE</code> at build time to point elsewhere.
      </footer>
    </div>
  );
}
