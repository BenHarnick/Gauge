import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback, useEffect, useState } from "react";
import { KNOWN_PLAN_IDS, predict, whatif, } from "../api";
import { ResultCard } from "./ResultCard";
import { Select } from "./Select";
import { Slider } from "./Slider";
import { Toggle } from "./Toggle";
import { WhatIfChart } from "./WhatIfChart";
const REGIONS = ["northeast", "midwest", "south", "west"];
const PLAN_LABELS = {
    none: "No plan (just predict charges)",
    hdhp_silver: "HDHP Silver",
    ppo_gold: "PPO Gold",
    ppo_platinum: "PPO Platinum",
};
function sweepValuesFor(feature) {
    switch (feature) {
        case "age":
            return [20, 25, 30, 35, 40, 45, 50, 55, 60, 64];
        case "bmi":
            return [18, 22, 26, 30, 34, 38, 42];
        case "children":
            return [0, 1, 2, 3, 4, 5];
        case "smoker":
            return ["no", "yes"];
        case "sex":
            return ["female", "male"];
        case "region":
            return REGIONS;
    }
}
export function PredictorPage() {
    const [features, setFeatures] = useState({
        age: 35,
        sex: "female",
        bmi: 27.5,
        children: 1,
        smoker: "no",
        region: "northeast",
    });
    const [planId, setPlanId] = useState("ppo_gold");
    const [sweepFeature, setSweepFeature] = useState("age");
    const [result, setResult] = useState(null);
    const [resultLoading, setResultLoading] = useState(false);
    const [resultError, setResultError] = useState(null);
    const [sweep, setSweep] = useState(null);
    const [sweepLoading, setSweepLoading] = useState(false);
    const [sweepError, setSweepError] = useState(null);
    const update = useCallback((key, value) => {
        setFeatures((prev) => ({ ...prev, [key]: value }));
    }, []);
    useEffect(() => {
        let cancelled = false;
        const handle = setTimeout(async () => {
            setResultLoading(true);
            setResultError(null);
            try {
                const r = await predict(features, planId === "none" ? undefined : planId);
                if (!cancelled)
                    setResult(r);
            }
            catch (e) {
                if (!cancelled)
                    setResultError(String(e));
            }
            finally {
                if (!cancelled)
                    setResultLoading(false);
            }
        }, 200);
        return () => {
            cancelled = true;
            clearTimeout(handle);
        };
    }, [features, planId]);
    useEffect(() => {
        let cancelled = false;
        const values = sweepValuesFor(sweepFeature);
        const handle = setTimeout(async () => {
            setSweepLoading(true);
            setSweepError(null);
            try {
                const r = await whatif(features, sweepFeature, values, planId === "none" ? undefined : planId);
                if (!cancelled)
                    setSweep(r);
            }
            catch (e) {
                if (!cancelled)
                    setSweepError(String(e));
            }
            finally {
                if (!cancelled)
                    setSweepLoading(false);
            }
        }, 250);
        return () => {
            cancelled = true;
            clearTimeout(handle);
        };
    }, [features, planId, sweepFeature]);
    return (_jsxs("div", { className: "grid grid-cols-1 gap-6 lg:grid-cols-[20rem_1fr]", children: [_jsxs("aside", { className: "space-y-5 rounded-lg border border-slate-200 bg-white p-5 shadow-sm", children: [_jsx(Slider, { label: "Age", min: 18, max: 64, value: features.age, onChange: (v) => update("age", v), format: (v) => `${v} years` }), _jsx(Toggle, { label: "Sex", value: features.sex, options: [
                            { value: "female", label: "Female" },
                            { value: "male", label: "Male" },
                        ], onChange: (v) => update("sex", v) }), _jsx(Slider, { label: "BMI", min: 16, max: 53, step: 0.1, value: features.bmi, onChange: (v) => update("bmi", Number(v.toFixed(1))), format: (v) => v.toFixed(1) }), _jsx(Slider, { label: "Children", min: 0, max: 5, value: features.children, onChange: (v) => update("children", v) }), _jsx(Toggle, { label: "Smoker", value: features.smoker, options: [
                            { value: "no", label: "No" },
                            { value: "yes", label: "Yes" },
                        ], onChange: (v) => update("smoker", v) }), _jsx(Select, { label: "Region", value: features.region, options: REGIONS.map((r) => ({
                            value: r,
                            label: r.charAt(0).toUpperCase() + r.slice(1),
                        })), onChange: (v) => update("region", v) }), _jsx("hr", { className: "border-slate-200" }), _jsx(Select, { label: "Plan", value: planId, options: [
                            { value: "none", label: PLAN_LABELS.none },
                            ...KNOWN_PLAN_IDS.map((id) => ({
                                value: id,
                                label: PLAN_LABELS[id],
                            })),
                        ], onChange: setPlanId })] }), _jsxs("main", { className: "space-y-6", children: [_jsx(ResultCard, { result: result, loading: resultLoading, error: resultError }), _jsxs("section", { className: "space-y-3", children: [_jsxs("div", { className: "flex items-end justify-between", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-lg font-semibold text-slate-900", children: "What-if simulator" }), _jsx("p", { className: "text-sm text-slate-600", children: "Hold everything else fixed and vary one feature." })] }), _jsx(Select, { label: "Vary", value: sweepFeature, onChange: setSweepFeature, options: [
                                            { value: "age", label: "Age" },
                                            { value: "bmi", label: "BMI" },
                                            { value: "children", label: "Children" },
                                            { value: "smoker", label: "Smoker" },
                                            { value: "sex", label: "Sex" },
                                            { value: "region", label: "Region" },
                                        ] })] }), _jsx(WhatIfChart, { data: sweep, loading: sweepLoading, error: sweepError, feature: sweepFeature })] })] })] }));
}
