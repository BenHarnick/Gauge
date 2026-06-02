import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export function Select({ label, value, onChange, options, }) {
    return (_jsxs("label", { className: "flex flex-col gap-1.5 text-sm", children: [_jsx("span", { className: "font-medium text-slate-700", children: label }), _jsx("select", { className: "rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-card focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-200", value: value, onChange: (e) => onChange(e.target.value), children: options.map((opt) => (_jsx("option", { value: opt.value, children: opt.label }, opt.value))) })] }));
}
