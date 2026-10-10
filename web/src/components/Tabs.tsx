// Accessible tabs (roving tabindex, arrow keys). Panels are rendered by the parent.
import type { ReactNode } from "react";

export type TabDef<T extends string> = { id: T; label: string; icon?: ReactNode };

export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
  idPrefix,
  label,
}: {
  tabs: TabDef<T>[];
  value: T;
  onChange: (v: T) => void;
  idPrefix: string;
  label: string;
}) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((t, i) => (
        <button
          key={t.id}
          id={`${idPrefix}-tab-${t.id}`}
          role="tab"
          type="button"
          aria-selected={value === t.id}
          aria-controls={`${idPrefix}-panel`}
          tabIndex={value === t.id ? 0 : -1}
          className={value === t.id ? "tabs__tab is-on" : "tabs__tab"}
          onClick={() => onChange(t.id)}
          onKeyDown={(e) => {
            let j = -1;
            if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
            else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
            else if (e.key === "Home") j = 0;
            else if (e.key === "End") j = tabs.length - 1;
            if (j < 0) return;
            e.preventDefault();
            onChange(tabs[j].id);
            document.getElementById(`${idPrefix}-tab-${tabs[j].id}`)?.focus();
          }}
        >
          {t.icon}
          {t.label}
        </button>
      ))}
    </div>
  );
}
