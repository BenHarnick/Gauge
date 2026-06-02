interface SelectProps<T extends string> {
  label: string;
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string }[];
}

export function Select<T extends string>({
  label,
  value,
  onChange,
  options,
}: SelectProps<T>) {
  return (
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="font-medium text-slate-700">{label}</span>
      <select
        className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-card focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-200"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}
