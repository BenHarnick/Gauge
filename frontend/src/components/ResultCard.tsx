import { AnnualPlanShare, PredictResponse, centsToDollars } from "../api";

interface ResultCardProps {
  result: PredictResponse | null;
  loading: boolean;
  error: string | null;
}

export function ResultCard({ result, loading, error }: ResultCardProps) {
  if (loading) {
    return (
      <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card">
        <div className="grid grid-cols-1 gap-px bg-slate-100 sm:grid-cols-2">
          <SkeletonBlock />
          <SkeletonBlock />
        </div>
        <div className="border-t border-slate-100 px-6 py-3">
          <div className="skeleton h-3 w-64" />
        </div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700">
        {error}
      </div>
    );
  }
  if (!result) {
    return (
      <div className="rounded-xl border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-500">
        Adjust the inputs on the left and your prediction will appear here.
      </div>
    );
  }

  const { prediction, annual_plan_share_median, annual_plan_share_mean } =
    result;

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card">
      <div className="grid grid-cols-1 gap-px bg-slate-100 sm:grid-cols-2">
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

      <div className="border-t border-slate-100 bg-slate-50 px-6 py-3 text-xs text-slate-500">
        80% interval for charges:{" "}
        <span className="tabular-nums font-medium text-slate-700">
          {centsToDollars(prediction.lower_bound_cents)}
        </span>{" "}
        to{" "}
        <span className="tabular-nums font-medium text-slate-700">
          {centsToDollars(prediction.upper_bound_cents)}
        </span>{" "}
        (10th to 90th percentile). Healthcare costs are heavily right-skewed;
        the interval is wide because real costs really are.
      </div>
    </div>
  );
}

function SkeletonBlock() {
  return (
    <div className="space-y-3 bg-white p-6">
      <div className="skeleton h-3 w-24" />
      <div className="skeleton h-8 w-32" />
      <div className="skeleton h-3 w-40" />
      <div className="mt-4 space-y-2 border-t border-slate-100 pt-3">
        <div className="skeleton h-3 w-20" />
        <div className="skeleton h-7 w-28" />
        <div className="mt-2 grid grid-cols-2 gap-y-1.5">
          <div className="skeleton h-3 w-20" />
          <div className="skeleton ml-auto h-3 w-16" />
          <div className="skeleton h-3 w-20" />
          <div className="skeleton ml-auto h-3 w-16" />
        </div>
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
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </div>
        <abbr
          title={tooltip}
          className="flex h-4 w-4 cursor-help items-center justify-center rounded-full bg-slate-100 text-[10px] font-bold text-slate-500 no-underline"
        >
          ?
        </abbr>
      </div>
      <div className="mt-1 text-3xl font-bold tracking-tight text-slate-900 tabular-nums">
        {centsToDollars(charges_cents)}
      </div>
      <div className="mt-0.5 text-xs text-slate-400">predicted annual charges</div>

      {share && (
        <div className="mt-4 border-t border-slate-100 pt-3">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            You would pay
          </div>
          <div className="mt-0.5 text-2xl font-bold tracking-tight text-brand-600 tabular-nums">
            {centsToDollars(share.member_pays_cents)}
          </div>
          <dl className="mt-3 space-y-1.5 text-xs">
            <div className="flex items-center justify-between">
              <dt className="text-slate-500">Deductible</dt>
              <dd className="tabular-nums font-medium text-slate-700">
                {centsToDollars(share.deductible_applied_cents)}
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-slate-500">Coinsurance</dt>
              <dd className="tabular-nums font-medium text-slate-700">
                {centsToDollars(share.coinsurance_cents)}
              </dd>
            </div>
            <div className="flex items-center justify-between border-t border-slate-100 pt-1.5">
              <dt className="text-slate-500">Plan pays</dt>
              <dd className="tabular-nums font-medium text-emerald-600">
                {centsToDollars(share.plan_pays_cents)}
              </dd>
            </div>
          </dl>
          {share.capped_at_oop_max && (
            <div className="mt-2.5 rounded-lg bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700">
              OOP max reached — plan absorbs the rest.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
