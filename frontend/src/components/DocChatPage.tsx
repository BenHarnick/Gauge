import { useEffect, useState } from "react";
import {
  DocumentMeta,
  deleteDocument,
  listDocuments,
} from "../api";
import { ChatPanel } from "./ChatPanel";
import { PDFUpload } from "./PDFUpload";

export function DocChatPage() {
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      const list = await listDocuments();
      setDocs(list);
      if (list.length > 0 && !list.some((d) => d.document_id === activeId)) {
        setActiveId(list[0].document_id);
      }
      if (list.length === 0) setActiveId(null);
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onUploaded(doc: DocumentMeta) {
    setDocs((prev) => [doc, ...prev]);
    setActiveId(doc.document_id);
    setError(null);
  }

  async function onDelete(doc: DocumentMeta) {
    try {
      await deleteDocument(doc.document_id);
      setDocs((prev) => prev.filter((d) => d.document_id !== doc.document_id));
      if (activeId === doc.document_id) {
        const remaining = docs.filter(
          (d) => d.document_id !== doc.document_id
        );
        setActiveId(remaining[0]?.document_id ?? null);
      }
    } catch (e) {
      setError(String(e));
    }
  }

  const active = docs.find((d) => d.document_id === activeId) ?? null;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[20rem_1fr]">
      <aside className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-card">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Your documents
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Uploaded PDFs live in memory for this server session. They go
            away when the backend restarts.
          </p>
        </div>
        <PDFUpload onUploaded={onUploaded} />
        {error && (
          <div className="rounded bg-red-50 px-3 py-2 text-xs text-red-700">
            {error}
          </div>
        )}
        <ul className="space-y-1">
          {docs.length === 0 && (
            <li className="text-xs text-slate-500">
              No documents uploaded yet.
            </li>
          )}
          {docs.map((d) => {
            const isActive = d.document_id === activeId;
            return (
              <li key={d.document_id}>
                <div
                  className={
                    "flex items-center justify-between gap-2 rounded px-2 py-2 text-sm " +
                    (isActive
                      ? "bg-brand-50 text-brand-700"
                      : "hover:bg-slate-100")
                  }
                >
                  <button
                    type="button"
                    onClick={() => setActiveId(d.document_id)}
                    className="flex-1 truncate text-left"
                    title={d.filename}
                  >
                    {d.filename}
                    <span className="ml-2 text-xs text-slate-500">
                      {d.n_pages}p
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => void onDelete(d)}
                    className="text-xs text-slate-400 hover:text-red-600"
                    title="Delete"
                  >
                    Delete
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </aside>

      <main>
        {active ? (
          <ChatPanel document={active} />
        ) : (
          <div className="flex h-[32rem] flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white p-10 text-center text-sm text-slate-500">
            <div className="max-w-md">
              <h2 className="text-base font-semibold text-slate-900">
                Upload an insurance document
              </h2>
              <p className="mt-2">
                Upload a Summary of Benefits and Coverage, a plan document,
                or any PDF you want to ask questions about. The chatbot
                retrieves the relevant passages and answers in plain
                English with page citations.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
