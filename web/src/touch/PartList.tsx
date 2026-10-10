import { useEffect, useRef } from "react";
import type { Part } from "../types";
import { cap } from "./Menu";

interface PartListProps {
  title: string;
  parts: Part[];
  selectedId: string | null;
  onSelect(part: Part): void;
  onClose(): void;
}

/** Screen-reader friendly path: one big button per part; activating it speaks the description. */
export default function PartList(props: PartListProps) {
  const headRef = useRef<HTMLHeadingElement>(null);
  const sheetRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    headRef.current?.focus();
    // TS-08: make the background inert while the dialog is open, restore focus on close.
    const bg = Array.from(document.body.children).filter((el) => !el.contains(sheetRef.current)) as HTMLElement[];
    const root = sheetRef.current?.closest(".ts-touch") ?? null;
    const siblings = root ? (Array.from(root.children).filter((el) => !el.contains(sheetRef.current)) as HTMLElement[]) : [];
    [...bg, ...siblings].forEach((el) => el.setAttribute("inert", ""));
    return () => {
      [...bg, ...siblings].forEach((el) => el.removeAttribute("inert"));
      if (opener && document.contains(opener)) opener.focus();
    };
  }, []);

  // TS-08: keep Tab / Shift+Tab inside the dialog.
  const trapTab = (e: React.KeyboardEvent) => {
    if (e.key !== "Tab" || !sheetRef.current) return;
    const items = Array.from(
      sheetRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'),
    ).filter((el) => el.offsetParent !== null);
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || active === headRef.current || !sheetRef.current.contains(active))) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && (active === last || !sheetRef.current.contains(active))) {
      e.preventDefault();
      first.focus();
    }
  };

  return (
    <div
      className="ts-t-scrim ts-t-scrim-solid"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          e.stopPropagation();
          props.onClose();
          return;
        }
        trapTab(e);
      }}
    >
      <div ref={sheetRef} className="ts-t-sheet ts-t-sheet-full" role="dialog" aria-modal="true" aria-labelledby="ts-t-list-title">
        <div className="ts-t-sheet-head">
          <h2 id="ts-t-list-title" ref={headRef} tabIndex={-1}>
            {cap(props.title)}: {props.parts.length} parts
          </h2>
          <button className="ts-t-btn ts-t-btn-primary" onClick={props.onClose}>
            Back to diagram
          </button>
        </div>
        {props.parts.length === 0 ? (
          <p className="ts-t-empty">This diagram has no parts to explore yet.</p>
        ) : (
          <ul className="ts-t-list" role="list">
            {props.parts.map((part) => {
              const sel = part.id === props.selectedId;
              return (
                <li key={part.id}>
                  <button
                    className={"ts-t-btn ts-t-btn-big ts-t-part" + (sel ? " is-selected" : "")}
                    aria-describedby={`ts-t-desc-${part.id}`}
                    aria-pressed={sel}
                    onClick={() => props.onSelect(part)}
                  >
                    <span className="ts-t-part-name">{cap(part.name)}</span>
                    {part.key && (
                      <span className="ts-t-key" aria-label={`braille key ${part.key}`}>
                        {part.key}
                      </span>
                    )}
                  </button>
                  <p id={`ts-t-desc-${part.id}`} className={"ts-t-desc" + (sel ? "" : " ts-t-sr")}>
                    {part.description || `${cap(part.name)}.`}
                  </p>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
