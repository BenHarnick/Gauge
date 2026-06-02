import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export function Toggle({ label, value, options, onChange, }) {
    return (_jsxs("div", { className: "flex flex-col gap-1.5 text-sm", children: [_jsx("span", { className: "font-medium text-slate-700", children: label }), _jsx("div", { className: "inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5", children: options.map((opt) => {
                    const selected = opt.value === value;
                    return (_jsx("button", { type: "button", onClick: () => onChange(opt.value), className: "flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-all " +
                            (selected
                                ? "bg-white text-brand-700 shadow-card"
                                : "text-slate-500 hover:text-slate-700"), children: opt.label }, opt.value));
                }) })] }));
}
