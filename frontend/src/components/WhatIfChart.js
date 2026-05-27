import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, } from "recharts";
import { centsToDollars } from "../api";
export function WhatIfChart({ data, loading, error, feature }) {
    const rows = useMemo(() => {
        if (!data)
            return [];
        return data.points.map((p) => ({
            value: p.value,
            median: p.prediction.median_charges_cents / 100,
            mean: p.prediction.mean_charges_cents / 100,
            lower: p.prediction.lower_bound_cents / 100,
            upper: p.prediction.upper_bound_cents / 100,
            memberPaysMedian: p.annual_plan_share_median != null
                ? p.annual_plan_share_median.member_pays_cents / 100
                : null,
            memberPaysMean: p.annual_plan_share_mean != null
                ? p.annual_plan_share_mean.member_pays_cents / 100
                : null,
        }));
    }, [data]);
    if (loading) {
        return (_jsx("div", { className: "rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-500", children: "Running what-if sweep..." }));
    }
    if (error) {
        return (_jsx("div", { className: "rounded-lg border border-red-200 bg-red-50 p-6 text-sm text-red-700", children: error }));
    }
    if (rows.length === 0) {
        return (_jsx("div", { className: "rounded-lg border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-500", children: "Pick a feature to sweep and the curve will render here." }));
    }
    return (_jsxs("div", { className: "rounded-lg border border-slate-200 bg-white p-4 shadow-sm", children: [_jsxs("div", { className: "mb-2 text-sm text-slate-600", children: ["Charges and member out-of-pocket as ", _jsx("strong", { children: feature }), " varies. Solid lines are point estimates; dashed lines are the 10th and 90th percentile bounds."] }), _jsx("div", { className: "h-80 w-full", children: _jsx(ResponsiveContainer, { width: "100%", height: "100%", children: _jsxs(LineChart, { data: rows, margin: { top: 10, right: 16, bottom: 0, left: 0 }, children: [_jsx(CartesianGrid, { stroke: "#e2e8f0", strokeDasharray: "3 3" }), _jsx(XAxis, { dataKey: "value", tick: { fontSize: 12, fill: "#475569" }, padding: { left: 8, right: 8 } }), _jsx(YAxis, { tick: { fontSize: 12, fill: "#475569" }, tickFormatter: (v) => v >= 1000 ? `$${Math.round(v / 1000)}k` : `$${v}`, width: 56 }), _jsx(Tooltip, { formatter: (v, name) => [
                                    centsToDollars(Math.round(v * 100)),
                                    name,
                                ], labelFormatter: (label) => `${feature} = ${label}`, contentStyle: { fontSize: 12 } }), _jsx(Legend, { wrapperStyle: { fontSize: 12 } }), _jsx(Line, { type: "monotone", dataKey: "median", name: "Median charges", stroke: "#1d4ed8", strokeWidth: 2, dot: { r: 3 } }), _jsx(Line, { type: "monotone", dataKey: "mean", name: "Mean charges", stroke: "#7c3aed", strokeWidth: 2, dot: { r: 3 } }), _jsx(Line, { type: "monotone", dataKey: "upper", name: "90th pct", stroke: "#93c5fd", strokeDasharray: "3 3", dot: false }), _jsx(Line, { type: "monotone", dataKey: "lower", name: "10th pct", stroke: "#93c5fd", strokeDasharray: "3 3", dot: false }), rows[0].memberPaysMean != null && (_jsx(Line, { type: "monotone", dataKey: "memberPaysMean", name: "You pay (mean)", stroke: "#059669", strokeWidth: 2, dot: { r: 3 } }))] }) }) })] }));
}
