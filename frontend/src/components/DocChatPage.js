import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { deleteDocument, listDocuments, } from "../api";
import { ChatPanel } from "./ChatPanel";
import { PDFUpload } from "./PDFUpload";
export function DocChatPage() {
    const [docs, setDocs] = useState([]);
    const [activeId, setActiveId] = useState(null);
    const [error, setError] = useState(null);
    async function refresh() {
        try {
            const list = await listDocuments();
            setDocs(list);
            if (list.length > 0 && !list.some((d) => d.document_id === activeId)) {
                setActiveId(list[0].document_id);
            }
            if (list.length === 0)
                setActiveId(null);
        }
        catch (e) {
            setError(String(e));
        }
    }
    useEffect(() => {
        void refresh();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    function onUploaded(doc) {
        setDocs((prev) => [doc, ...prev]);
        setActiveId(doc.document_id);
        setError(null);
    }
    async function onDelete(doc) {
        try {
            await deleteDocument(doc.document_id);
            setDocs((prev) => prev.filter((d) => d.document_id !== doc.document_id));
            if (activeId === doc.document_id) {
                const remaining = docs.filter((d) => d.document_id !== doc.document_id);
                setActiveId(remaining[0]?.document_id ?? null);
            }
        }
        catch (e) {
            setError(String(e));
        }
    }
    const active = docs.find((d) => d.document_id === activeId) ?? null;
    return (_jsxs("div", { className: "grid grid-cols-1 gap-6 lg:grid-cols-[20rem_1fr]", children: [_jsxs("aside", { className: "space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-card", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-base font-semibold text-slate-900", children: "Your documents" }), _jsx("p", { className: "mt-1 text-xs text-slate-500", children: "Uploaded PDFs live in memory for this server session. They go away when the backend restarts." })] }), _jsx(PDFUpload, { onUploaded: onUploaded }), error && (_jsx("div", { className: "rounded bg-red-50 px-3 py-2 text-xs text-red-700", children: error })), _jsxs("ul", { className: "space-y-1", children: [docs.length === 0 && (_jsx("li", { className: "text-xs text-slate-500", children: "No documents uploaded yet." })), docs.map((d) => {
                                const isActive = d.document_id === activeId;
                                return (_jsx("li", { children: _jsxs("div", { className: "flex items-center justify-between gap-2 rounded px-2 py-2 text-sm " +
                                            (isActive
                                                ? "bg-brand-50 text-brand-700"
                                                : "hover:bg-slate-100"), children: [_jsxs("button", { type: "button", onClick: () => setActiveId(d.document_id), className: "flex-1 truncate text-left", title: d.filename, children: [d.filename, _jsxs("span", { className: "ml-2 text-xs text-slate-500", children: [d.n_pages, "p"] })] }), _jsx("button", { type: "button", onClick: () => void onDelete(d), className: "text-xs text-slate-400 hover:text-red-600", title: "Delete", children: "Delete" })] }) }, d.document_id));
                            })] })] }), _jsx("main", { children: active ? (_jsx(ChatPanel, { document: active })) : (_jsx("div", { className: "flex h-[32rem] flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white p-10 text-center text-sm text-slate-500", children: _jsxs("div", { className: "max-w-md", children: [_jsx("h2", { className: "text-base font-semibold text-slate-900", children: "Upload an insurance document" }), _jsx("p", { className: "mt-2", children: "Upload a Summary of Benefits and Coverage, a plan document, or any PDF you want to ask questions about. The chatbot retrieves the relevant passages and answers in plain English with page citations." })] }) })) })] }));
}
