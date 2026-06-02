import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useRef, useState } from "react";
import { uploadPDF } from "../api";
export function PDFUpload({ onUploaded, disabled }) {
    const inputRef = useRef(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    async function handle(file) {
        setUploading(true);
        setError(null);
        try {
            const res = await uploadPDF(file);
            onUploaded(res.document);
        }
        catch (e) {
            setError(String(e));
        }
        finally {
            setUploading(false);
            if (inputRef.current)
                inputRef.current.value = "";
        }
    }
    return (_jsxs("div", { children: [_jsx("input", { ref: inputRef, type: "file", accept: "application/pdf,.pdf", className: "hidden", onChange: (e) => {
                    const f = e.target.files?.[0];
                    if (f)
                        void handle(f);
                } }), _jsx("button", { type: "button", disabled: disabled || uploading, onClick: () => inputRef.current?.click(), className: "rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300", children: uploading ? "Uploading..." : "Upload a plan PDF" }), error && (_jsx("div", { className: "mt-2 rounded bg-red-50 px-3 py-2 text-xs text-red-700", children: error }))] }));
}
