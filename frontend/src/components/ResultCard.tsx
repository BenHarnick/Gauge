import { AnnualPlanShare, PredictResponse, centsToDollars } from "../api";

interface ResultCardProps {
  result: PredictResponse | null;
  loading: boolean;
  error: string | null;
}

export function ResultCard({ result, loading, error }: ResultCardProps) {
  if (loading) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-500">
        Predicting...
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
  if (!result) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-500">
        Adjust the inputs on the left and the prediction will appear here.
      </div>
    );
  }

  const { prediction, annual_plan_share_median, annual_plan_share_mean } =
    result;

  return (
    <div className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="grid grid-cols-1 gap-px bg-slate-200 sm:grid-cols-2">
        <PredictionBlock
          title="Typical year"
          tooltip="50th percentile (median). Half of people like you spend less, half spend more. This is what you should expect in a normal year."
          charges_cents={prediction.median_charges_cents}
          share={annual_plan_share_median}
        />
        <PredictionBlock
          title="Long-run average"
          tooltip="Mean (expected value). Includes the rare-but-expensive years that pull the average up. Better number for budgeting across many years."
          charges_cents={prediction.mean_charges_cents}
          share={annual_plan_share_mean}
        />
      </div>

      <div className="border-t border-slate-200 px-6 py-3 text-xs text-slate-500">
        80% interval for charges:{" "}
        <span className="tabular-nums">
          {centsToDollars(prediction.lower_bound_cents)}
        </span>{" "}
        to{" "}
        <span className="tabular-nums">
          {centsToDollars(prediction.upper_bound_cents)}
        </span>{" "}
        (10th to 90th percentile). Healthcare costs are heavily
        right-skewed; the interval is wide because real costs really are.
      </div>
    </div>
  );
}

interface PredictionBlockProps {
  title: string;
  tooltip: string;
  charges_cents: number;
  share: AnnualPlanShare | null;
}

function PredictionBlock({
  title,
  tooltip,
  charges_cents,
  share,
}: PredictionBlockProps) {
  return (
    <div className="bg-white p-6">
      <div className="flex items-center gap-1.5">
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          {title}
        </div>
        <abbr
          title={tooltip}
          className="cursor-help text-xs text-slate-400 no-underline"
        >
          ?
        </abbr>
      </div>
      <div className="mt-1 text-2xl font-semibold text-slate-900 tabular-nums">
        {centsToDollars(charges_cents)}
      </div>
      <div className="text-xs text-slate-500">predicted annual charges</div>

      {share && (
        <div className="mt-4 border-t border-slate-100 pt-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            You would pay
          </div>
          <div className="text-xl font-semibold text-brand-700 tabular-nums">
            {centsToDollars(share.member_pays_cents)}
          </div>
          <dl className="mt-2 grid grid-cols-2 gap-y-0.5 text-xs">
            <dt className="text-slate-500">Deductible</dt>
            <dd className="text-right tabular-nums">
              {centsToDollars(share.deductible_applied_cents)}
            </dd>
            <dt className="text-slate-500">Coinsurance</dt>
            <dd className="text-right tabular-nums">
              {centsToDollars(share.coinsurance_cents)}
            </dd>
            <dt className="text-slate-500">Plan pays</dt>
            <dd className="text-right tabular-nums">
              {centsToDollars(share.plan_pays_cents)}
            </dd>
          </dl>
          {share.capped_at_oop_max && (
            <div className="mt-2 rounded bg-emerald-50 px-2 py-1 text-xs text-emerald-700">
              OOP max reached.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
