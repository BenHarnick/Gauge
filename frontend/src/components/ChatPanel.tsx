import { FormEvent, useEffect, useRef, useState } from "react";
import { ChatResponse, DocumentMeta, askChat } from "../api";

interface Turn {
  id: number;
  question: string;
  response: ChatResponse | null;
  error: string | null;
  loading: boolean;
}

interface ChatPanelProps {
  document: DocumentMeta;
}

export function ChatPanel({ document }: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Clear conversation when document changes.
  useEffect(() => {
    setTurns([]);
    setDraft("");
  }, [document.document_id]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const question = draft.trim();
    if (!question) return;

    const id = Date.now();
    setTurns((prev) => [
      ...prev,
      { id, question, response: null, error: null, loading: true },
    ]);
    setDraft("");

    try {
      const response = await askChat(document.document_id, question);
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id ? { ...t, response, loading: false } : t
        )
      );
    } catch (e) {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id ? { ...t, error: String(e), loading: false } : t
        )
      );
    }
  }

  return (
    <div className="flex h-[32rem] flex-col rounded-xl border border-slate-200 bg-white shadow-card">
      <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {turns.length === 0 ? (
          <div className="text-sm text-slate-500">
            Ask anything about <strong>{document.filename}</strong>. Try:
            "What's the deductible?", "What does an MRI cost?", "Does this
            plan cover physical therapy?"
          </div>
        ) : (
          turns.map((turn) => (
            <TurnView key={turn.id} turn={turn} />
          ))
        )}
        <div ref={bottomRef} />
      </div>
      <form
        onSubmit={submit}
        className="flex gap-2 border-t border-slate-200 px-3 py-3"
      >
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={`Ask about ${document.filename}...`}
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
        />
        <button
          type="submit"
          disabled={!draft.trim()}
          className="rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Ask
        </button>
      </form>
    </div>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  return (
    <div className="space-y-2">
      <div className="ml-auto max-w-[80%] rounded-2xl rounded-tr-sm bg-brand-50 px-4 py-2 text-sm text-slate-900">
        {turn.question}
      </div>

      {turn.loading && (
        <div className="max-w-[80%] rounded-2xl rounded-tl-sm bg-slate-100 px-4 py-2 text-sm italic text-slate-500">
          Searching...
        </div>
      )}

      {turn.error && (
        <div className="max-w-[80%] rounded-2xl rounded-tl-sm bg-red-50 px-4 py-2 text-sm text-red-700">
          {turn.error}
        </div>
      )}

      {turn.response && (
        <div className="space-y-2">
          <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-tl-sm bg-slate-100 px-4 py-3 text-sm text-slate-900">
            {turn.response.answer}
          </div>
          <div className="text-xs text-slate-500">
            Source: <code className="rounded bg-slate-100 px-1.5 py-0.5">{turn.response.llm_used}</code>
          </div>
          {turn.response.citations.length > 0 && (
            <details className="text-xs text-slate-600">
              <summary className="cursor-pointer text-slate-700 hover:text-slate-900">
                {turn.response.citations.length} excerpt
                {turn.response.citations.length === 1 ? "" : "s"} from the
                document
              </summary>
              <ul className="mt-2 space-y-2">
                {turn.response.citations.map((c) => (
                  <li
                    key={c.chunk_index}
                    className="rounded border border-slate-200 bg-slate-50 px-3 py-2"
                  >
                    <div className="mb-1 font-medium text-slate-700">
                      Page {c.page_numbers.join(", ")}
                    </div>
                    <div className="text-slate-600">{c.snippet}</div>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
