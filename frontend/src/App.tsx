/**
 * App — root component for Gauge.
 *
 * Renders a persistent brand header and footer. The header includes a "How it
 * works" nav link that toggles between the intake wizard and the blog page.
 * No router dependency — a single boolean state is enough for two views.
 */

import { useState } from "react";
import Blog from "./components/Blog";
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

/** GitHub mark used for the repo link in the header. */
function GitHubIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="currentColor"
      xmlns="http://www.w3.org/2000/svg"
      className="h-5 w-5"
      aria-hidden="true"
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}

export default function App() {
  const [showBlog, setShowBlog] = useState(false);

  return (
    <div className="min-h-screen">
      {/* Brand header */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-6 py-5">
          {/* Logo + wordmark — clicking always returns to the tool */}
          <button
            onClick={() => setShowBlog(false)}
            className="flex items-center gap-3 focus:outline-none"
            aria-label="Go to Gauge home"
          >
            <BrandIcon />
            <div className="text-left">
              <p className="text-lg font-semibold leading-none tracking-tight text-slate-900">
                Gauge
              </p>
            </div>
          </button>

          {/* Nav */}
          <nav className="ml-auto flex items-center gap-1">
            <NavLink active={!showBlog} onClick={() => setShowBlog(false)}>
              Estimator
            </NavLink>
            <NavLink active={showBlog} onClick={() => setShowBlog(true)}>
              How it works
            </NavLink>
            <a
              href="https://github.com/BenH88888/Gauge"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="View source on GitHub"
              className="ml-1 flex items-center rounded-md p-1.5 text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900"
            >
              <GitHubIcon />
            </a>
          </nav>
        </div>
      </header>

      <main className={showBlog ? "" : "mx-auto max-w-5xl px-6 py-8"}>
        {showBlog ? <Blog /> : <IntakeWizard />}
      </main>
    </div>
  );
}

function NavLink({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
        active
          ? "bg-brand-50 text-brand-700"
          : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
      ].join(" ")}
    >
      {children}
    </button>
  );
}
