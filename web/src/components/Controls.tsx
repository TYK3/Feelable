// Form controls: segmented control + switch. Both are real inputs/buttons for a11y.
import { useId } from "react";

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  const id = useId();
  return (
    <div className="field">
      <span className="field__label" id={id}>{label}</span>
      <div className="segmented" role="radiogroup" aria-labelledby={id}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={value === o.value}
            className={value === o.value ? "segmented__opt is-on" : "segmented__opt"}
            onClick={() => onChange(o.value)}
            onKeyDown={(e) => {
              if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
              e.preventDefault();
              const i = options.findIndex((x) => x.value === value);
              const n = options[(i + (e.key === "ArrowRight" ? 1 : options.length - 1)) % options.length];
              onChange(n.value);
              const parent = e.currentTarget.parentElement;
              requestAnimationFrame(() => (parent?.querySelector(".is-on") as HTMLElement | null)?.focus());
            }}
            tabIndex={value === o.value ? 0 : -1}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Switch({
  checked,
  onChange,
  label,
  hideLabel,
  disabled,
  describedBy,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hideLabel?: boolean;
  disabled?: boolean;
  describedBy?: string;
}) {
  return (
    <label className={["switch", disabled && "is-disabled"].filter(Boolean).join(" ")}>
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        disabled={disabled}
        aria-describedby={describedBy}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="switch__track" aria-hidden="true">
        <span className="switch__thumb" />
      </span>
      <span className={hideLabel ? "sr-only" : "switch__label"}>{label}</span>
    </label>
  );
}
