import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import { askChat } from "../api";
export function ChatPanel({ document }) {
    const [draft, setDraft] = useState("");
    const [turns, setTurns] = useState([]);
    const bottomRef = useRef(null);
    // Clear conversation when document changes.
    useEffect(() => {
        setTurns([]);
        setDraft("");
    }, [document.document_id]);
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [turns]);
    async function submit(e) {
        e.preventDefault();
        const question = draft.trim();
        if (!question)
            return;
        const id = Date.now();
        setTurns((prev) => [
            ...prev,
            { id, question, response: null, error: null, loading: true },
        ]);
        setDraft("");
        try {
            const response = await askChat(document.document_id, question);
            setTurns((prev) => prev.map((t) => t.id === id ? { ...t, response, loading: false } : t));
        }
        catch (e) {
            setTurns((prev) => prev.map((t) => t.id === id ? { ...t, error: String(e), loading: false } : t));
        }
    }
    return (_jsxs("div", { className: "flex h-[32rem] flex-col rounded-xl border border-slate-200 bg-white shadow-card", children: [_jsxs("div", { className: "flex-1 space-y-4 overflow-y-auto px-5 py-4", children: [turns.length === 0 ? (_jsxs("div", { className: "text-sm text-slate-500", children: ["Ask anything about ", _jsx("strong", { children: document.filename }), ". Try: \"What's the deductible?\", \"What does an MRI cost?\", \"Does this plan cover physical therapy?\""] })) : (turns.map((turn) => (_jsx(TurnView, { turn: turn }, turn.id)))), _jsx("div", { ref: bottomRef })] }), _jsxs("form", { onSubmit: submit, className: "flex gap-2 border-t border-slate-200 px-3 py-3", children: [_jsx("input", { type: "text", value: draft, onChange: (e) => setDraft(e.target.value), placeholder: `Ask about ${document.filename}...`, className: "flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500" }), _jsx("button", { type: "submit", disabled: !draft.trim(), className: "rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300", children: "Ask" })] })] }));
}
function TurnView({ turn }) {
    return (_jsxs("div", { className: "space-y-2", children: [_jsx("div", { className: "ml-auto max-w-[80%] rounded-2xl rounded-tr-sm bg-brand-50 px-4 py-2 text-sm text-slate-900", children: turn.question }), turn.loading && (_jsx("div", { className: "max-w-[80%] rounded-2xl rounded-tl-sm bg-slate-100 px-4 py-2 text-sm italic text-slate-500", children: "Searching..." })), turn.error && (_jsx("div", { className: "max-w-[80%] rounded-2xl rounded-tl-sm bg-red-50 px-4 py-2 text-sm text-red-700", children: turn.error })), turn.response && (_jsxs("div", { className: "space-y-2", children: [_jsx("div", { className: "max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-tl-sm bg-slate-100 px-4 py-3 text-sm text-slate-900", children: turn.response.answer }), _jsxs("div", { className: "text-xs text-slate-500", children: ["Source: ", _jsx("code", { className: "rounded bg-slate-100 px-1.5 py-0.5", children: turn.response.llm_used })] }), turn.response.citations.length > 0 && (_jsxs("details", { className: "text-xs text-slate-600", children: [_jsxs("summary", { className: "cursor-pointer text-slate-700 hover:text-slate-900", children: [turn.response.citations.length, " excerpt", turn.response.citations.length === 1 ? "" : "s", " from the document"] }), _jsx("ul", { className: "mt-2 space-y-2", children: turn.response.citations.map((c) => (_jsxs("li", { className: "rounded border border-slate-200 bg-slate-50 px-3 py-2", children: [_jsxs("div", { className: "mb-1 font-medium text-slate-700", children: ["Page ", c.page_numbers.join(", ")] }), _jsx("div", { className: "text-slate-600", children: c.snippet })] }, c.chunk_index))) })] }))] }))] }));
}
