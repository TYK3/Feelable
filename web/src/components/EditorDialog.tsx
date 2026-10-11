// Modal used by the repair tools (Draw a missing part, Close-up lesson). UX03/UX04:
// - a visible, labelled Close button in the header (phones can't press Escape);
// - a scrollable body and a pinned footer, bounded to the dynamic viewport;
// - focus moves in, stays in (Tab/Shift+Tab wrap, including focusable SVG canvases), Escape
//   closes, focus returns to the opener; the app behind is inert and the page doesn't scroll.
import { useEffect, useId, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { IconX } from "./Icons";

const FOCUSABLE =
  'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function EditorDialog({
  title,
  intro,
  children,
  footer,
  onClose,
  initialFocus,
  fullOnPhone,
}: {
  title: string;
  intro?: ReactNode;
  children: ReactNode;
  footer: ReactNode;
  onClose: () => void;
  /** CSS selector inside the dialog to focus first (defaults to the first focusable control). */
  initialFocus?: string;
  /** small phones: use the whole screen (drawing needs every pixel) */
  fullOnPhone?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const descId = useId();
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    const box = ref.current!;
    const focusables = () => Array.from(box.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => el.offsetParent !== null || el.tagName === "svg");
    const first = (initialFocus && box.querySelector<HTMLElement>(initialFocus)) || focusables()[0];
    first?.focus();

    // background: inert + no scroll
    const root = document.getElementById("root");
    root?.setAttribute("inert", "");
    root?.setAttribute("aria-hidden", "true");
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        closeRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const f = focusables();
      if (!f.length) return;
      const i = f.indexOf(document.activeElement as HTMLElement);
      if (e.shiftKey && (i <= 0 || !box.contains(document.activeElement))) {
        e.preventDefault();
        f[f.length - 1].focus();
      } else if (!e.shiftKey && (i === f.length - 1 || !box.contains(document.activeElement))) {
        e.preventDefault();
        f[0].focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      root?.removeAttribute("inert");
      root?.removeAttribute("aria-hidden");
      document.body.style.overflow = prevOverflow;
      if (opener && document.contains(opener)) opener.focus();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return createPortal(
    <div className={`edlg${fullOnPhone ? " edlg--full" : ""}`} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        ref={ref}
        className="edlg__box"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={intro ? descId : undefined}
      >
        <header className="edlg__head">
          <h2 id={titleId} className="edlg__title">
            {title}
          </h2>
          <button type="button" className="edlg__close" onClick={onClose} aria-label={`Close ${title}`}>
            <IconX size={20} />
          </button>
        </header>
        <div className="edlg__body">
          {intro && (
            <p id={descId} className="edlg__intro">
              {intro}
            </p>
          )}
          {children}
        </div>
        <footer className="edlg__foot">{footer}</footer>
      </div>
    </div>,
    document.body,
  );
}

/** Map a pointer event to SVG user (viewBox) coordinates through the inverse screen matrix (UX02).
 *  Works whatever the letterboxing/scaling (preserveAspectRatio), zoom or scroll. */
export function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number): [number, number] | null {
  const m = svg.getScreenCTM();
  if (!m) return null;
  const p = new DOMPoint(clientX, clientY).matrixTransform(m.inverse());
  return [p.x, p.y];
}
