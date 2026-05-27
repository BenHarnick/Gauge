import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { WhatIfResponse, centsToDollars } from "../api";

interface WhatIfChartProps {
  data: WhatIfResponse | null;
  loading: boolean;
  error: string | null;
  feature: string;
}

interface ChartPoint {
  value: number | string;
  median: number;
  mean: number;
  lower: number;
  upper: number;
  memberPaysMedian: number | null;
  memberPaysMean: number | null;
}

export function WhatIfChart({ data, loading, error, feature }: WhatIfChartProps) {
  const rows: ChartPoint[] = useMemo(() => {
    if (!data) return [];
    return data.points.map((p) => ({
      value: p.value,
      median: p.prediction.median_charges_cents / 100,
      mean: p.prediction.mean_charges_cents / 100,
      lower: p.prediction.lower_bound_cents / 100,
      upper: p.prediction.upper_bound_cents / 100,
      memberPaysMedian:
        p.annual_plan_share_median != null
          ? p.annual_plan_share_median.member_pays_cents / 100
          : null,
      memberPaysMean:
        p.annual_plan_share_mean != null
          ? p.annual_plan_share_mean.member_pays_cents / 100
          : null,
    }));
  }, [data]);

  if (loading) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-500">
        Running what-if sweep...
      </div>
    );
  }
  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-sm text-red-700">
        {error}
      </div>
    );
  }
  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-500">
        Pick a feature to sweep and the curve will render here.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 text-sm text-slate-600">
        Charges and member out-of-pocket as <strong>{feature}</strong> varies.
        Solid lines are point estimates; dashed lines are the 10th and 90th
        percentile bounds.
      </div>
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
            <XAxis
              dataKey="value"
              tick={{ fontSize: 12, fill: "#475569" }}
              padding={{ left: 8, right: 8 }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: "#475569" }}
              tickFormatter={(v: number) =>
                v >= 1000 ? `$${Math.round(v / 1000)}k` : `$${v}`
              }
              width={56}
            />
            <Tooltip
              formatter={(v: number, name) => [
                centsToDollars(Math.round(v * 100)),
                name,
              ]}
              labelFormatter={(label) => `${feature} = ${label}`}
              contentStyle={{ fontSize: 12 }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line
              type="monotone"
              dataKey="median"
              name="Median charges"
              stroke="#1d4ed8"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="mean"
              name="Mean charges"
              stroke="#7c3aed"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="upper"
              name="90th pct"
              stroke="#93c5fd"
              strokeDasharray="3 3"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="lower"
              name="10th pct"
              stroke="#93c5fd"
              strokeDasharray="3 3"
              dot={false}
            />
            {rows[0].memberPaysMean != null && (
              <Line
                type="monotone"
                dataKey="memberPaysMean"
                name="You pay (mean)"
                stroke="#059669"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
