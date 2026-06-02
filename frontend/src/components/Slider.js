import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export function Slider({ label, value, onChange, min, max, step = 1, format = (v) => String(v), }) {
    return (_jsxs("label", { className: "flex flex-col gap-2 text-sm", children: [_jsxs("div", { className: "flex items-baseline justify-between", children: [_jsx("span", { className: "font-medium text-slate-700", children: label }), _jsx("span", { className: "rounded-md bg-brand-50 px-2 py-0.5 text-xs font-semibold tabular-nums text-brand-700", children: format(value) })] }), _jsx("input", { type: "range", min: min, max: max, step: step, value: value, onChange: (e) => onChange(Number(e.target.value)) })] }));
}
