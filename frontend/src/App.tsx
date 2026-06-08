/**
 * App — root component for Gauge.
 *
 * Renders a persistent brand header and footer around the single guided
 * intake flow.  There is intentionally no top-level navigation: the product
 * answers one question ("what will I actually pay?"), so every screen is a
 * step in that answer, not a separate feature.
 *
 * The what-if simulator and plan Q&A panel are embedded inside the final step
 * of IntakeWizard — they appear in context, after the estimate, not behind
 * separate tabs.
 */

import { IntakeWizard } from "./components/IntakeWizard";

/** Shield-plus brand mark used in the header. */
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
      <path
        d="M16 6 L24 9.5 V17 C24 21.5 20.5 25 16 26 C11.5 25 8 21.5 8 17 V9.5 Z"
        fill="white"
        fillOpacity="0.18"
        stroke="white"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
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
  return (
    <div className="min-h-screen">
      {/* Brand header */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-6 py-5">
          <BrandIcon />
          <div>
            <h1 className="text-lg font-semibold leading-none tracking-tight text-slate-900">
              Gauge
            </h1>
            <p className="mt-0.5 text-xs text-slate-500">
              What will you actually pay this year?
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
        <IntakeWizard />
      </main>

      <footer className="mx-auto max-w-5xl px-6 pb-8">
        <p className="text-xs text-slate-400">
          Illustrative prototype — not a substitute for an actual insurance
          quote or advice from your insurer.
        </p>
      </footer>
    </div>
  );
}
