// Accessible modal dialog (focus trap, Esc closes, focus restored).
import { useEffect, useId, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";

export function Dialog({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    const el = ref.current!;
    const focusables = () => Array.from(el.querySelectorAll<HTMLElement>("button:not([disabled]), a[href], input, textarea"));
    focusables()[0]?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "Tab") {
        const f = focusables();
        if (!f.length) return;
        const i = f.indexOf(document.activeElement as HTMLElement);
        if (e.shiftKey && i <= 0) {
          e.preventDefault();
          f[f.length - 1].focus();
        } else if (!e.shiftKey && i === f.length - 1) {
          e.preventDefault();
          f[0].focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
      prev?.focus?.();
    };
  }, [onClose]);
  return createPortal(
    <div className="dlg" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dlg__box" role="dialog" aria-modal="true" aria-labelledby={titleId} ref={ref}>
        <h2 id={titleId} className="dlg__title">{title}</h2>
        {children}
      </div>
    </div>,
    document.body,
  );
}
