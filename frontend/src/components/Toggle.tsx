interface ToggleProps<T extends string> {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}

export function Toggle<T extends string>({
  label,
  value,
  options,
  onChange,
}: ToggleProps<T>) {
  return (
    <div className="flex flex-col gap-1.5 text-sm">
      <span className="font-medium text-slate-700">{label}</span>
      <div className="inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5">
        {options.map((opt) => {
          const selected = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onChange(opt.value)}
              className={
                "flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-all " +
                (selected
                  ? "bg-white text-brand-700 shadow-card"
                  : "text-slate-500 hover:text-slate-700")
              }
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
