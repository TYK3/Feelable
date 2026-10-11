// Interactive diagram: Original ↔ Tactile compare slider, zoom, hover/tap regions (F17/F18),
// plus the braille key page and a 3D preview guarded by an error boundary (F03).
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { JobResult, Part, Polygon } from "../types";
import { ErrorBoundary } from "../ErrorBoundary";
import { BrailleWord } from "./BrailleWord";
import { IconCube, IconEye, IconMinus, IconPlus, IconRefresh } from "./Icons";
import { VisionView } from "./VisionView";
import { Tabs } from "./Tabs";
import { bust } from "./jobStore";
import { polysToPath, rgb } from "./PartOverlay";
import { sentenceCase } from "./text";

type View = "diagram" | "vision" | "legend" | "3d";

/** Fit source→tactile (uniform scale + offset) from the parts' boxes; robust to enlarged small parts. */
export function fitTransform(job: JobResult): { s: number; tx: number; ty: number } {
  const sc: number[] = [];
  const cx: [number, number][] = [];
  const cy: [number, number][] = [];
  for (const p of job.parts) {
    const pts = p.tactile_polygons.flat();
    if (pts.length < 3) continue;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of pts) {
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
    const [sx0, sy0, sx1, sy1] = p.source_bbox;
    const sw = sx1 - sx0, sh = sy1 - sy0;
    if (sw < 4 || sh < 4) continue;
    sc.push((x1 - x0) / sw, (y1 - y0) / sh);
    cx.push([(sx0 + sx1) / 2, (x0 + x1) / 2]);
    cy.push([(sy0 + sy1) / 2, (y0 + y1) / 2]);
  }
  const med = (a: number[]) => {
    const b = [...a].sort((m, n) => m - n);
    return b.length ? b[Math.floor(b.length / 2)] : NaN;
  };
  const W = job.tactile.width, H = job.tactile.height, SW = job.source.width, SH = job.source.height;
  let s = med(sc);
  if (!isFinite(s) || s <= 0) {
    s = Math.min(W / SW, H / SH) * 0.9;
    return { s, tx: (W - SW * s) / 2, ty: (H - SH * s) / 2 };
  }
  const tx = med(cx.map(([a, b]) => b - s * a));
  const ty = med(cy.map(([a, b]) => b - s * a));
  return { s, tx, ty };
}

function bboxOf(polys: Polygon[]) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const poly of polys) for (const [x, y] of poly) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  return isFinite(x0) ? { x0, y0, x1, y1 } : null;
}

function makeStl() {
  return lazy(() => import("./StlViewer"));
}

export function DiagramViewer({
  job,
  version,
  selectedId,
  hoverId,
  onSelect,
  onHover,
  names,
  compact,
}: {
  job: JobResult;
  version: number;
  selectedId: string | null;
  hoverId: string | null;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null) => void;
  names: Record<string, string>; // current (draft) names
  compact?: boolean;
}) {
  const [view, setView] = useState<View>("diagram");
  const [split, setSplit] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [stlTry, setStlTry] = useState(0);
  const Stl = useMemo(() => makeStl(), [stlTry]); // eslint-disable-line react-hooks/exhaustive-deps
  const stageRef = useRef<HTMLDivElement>(null);
  const W = job.tactile.width, H = job.tactile.height;
  const T = useMemo(() => fitTransform(job), [job]);
  const title = sentenceCase(job.title) || "Untitled diagram";

  const shapes = useMemo(
    () =>
      job.parts.map((p) => {
        const tactile = p.tactile_polygons.length > 0;
        return {
          part: p,
          d: polysToPath(tactile ? p.tactile_polygons : p.source_polygons),
          tactile,
          box: tactile ? bboxOf(p.tactile_polygons) : (() => {
            const b = bboxOf(p.source_polygons);
            return b && { x0: b.x0 * T.s + T.tx, y0: b.y0 * T.s + T.ty, x1: b.x1 * T.s + T.tx, y1: b.y1 * T.s + T.ty };
          })(),
        };
      }),
    [job, T],
  );
  const sel = shapes.find((x) => x.part.id === selectedId) ?? null;
  const scrollRef = useRef<HTMLDivElement>(null);
  // Handover (mobile): zoom centres on the selected part instead of the empty top-left corner.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || zoom <= 1 || !sel?.box) return;
    const sw = el.clientWidth * zoom;
    const sh = (sw * H) / W;
    const cx = ((sel.box.x0 + sel.box.x1) / 2 / W) * sw;
    const cy = ((sel.box.y0 + sel.box.y1) / 2 / H) * sh;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    el.scrollTo({ left: Math.max(0, cx - el.clientWidth / 2), top: Math.max(0, cy - el.clientHeight / 2), behavior: reduce ? "auto" : "smooth" });
  }, [zoom, sel?.part.id, W, H]); // eslint-disable-line react-hooks/exhaustive-deps
  const hov = shapes.find((x) => x.part.id === hoverId && x.part.id !== selectedId) ?? null;

  const dragFrom = useCallback((clientX: number) => {
    const r = stageRef.current?.getBoundingClientRect();
    if (!r) return;
    setSplit(Math.min(1, Math.max(0, (clientX - r.left) / r.width)));
  }, []);

  const label = (p: Part) => sentenceCase(names[p.id] ?? p.name);

  return (
    <section className={compact ? "viewer is-compact" : "viewer"} aria-label="Diagram preview">
      <div className="viewer__bar">
        <Tabs<View>
          idPrefix="dv"
          label="Preview"
          value={view}
          onChange={setView}
          tabs={[
            { id: "diagram", label: "Diagram" },
            { id: "vision", label: "Scan details", icon: <IconEye size={16} /> },
            { id: "legend", label: "Braille key" },
            { id: "3d", label: "3D", icon: <IconCube size={16} /> },
          ]}
        />
        {view === "diagram" && (
          <div className="zoom" role="group" aria-label="Zoom">
            <button type="button" className="zoom__btn" onClick={() => setZoom((z) => Math.max(1, +(z - 0.5).toFixed(1)))} disabled={zoom <= 1} aria-label="Zoom out">
              <IconMinus size={18} />
            </button>
            <span className="zoom__val tabular" aria-live="polite">{Math.round(zoom * 100)}%</span>
            <button type="button" className="zoom__btn" onClick={() => setZoom((z) => Math.min(3, z + 0.5))} disabled={zoom >= 3} aria-label="Zoom in">
              <IconPlus size={18} />
            </button>
            <button type="button" className="zoom__fit" onClick={() => setZoom(1)} disabled={zoom <= 1} aria-label="Fit diagram to view">
              Fit
            </button>
            <button type="button" className="zoom__fit" onClick={() => setZoom((z) => Math.max(2, z))} disabled={!sel} aria-label="Zoom in on the selected part">
              Focus
            </button>
          </div>
        )}
      </div>

      <div id="dv-panel" role="tabpanel" aria-labelledby={`dv-tab-${view}`} className="viewer__panel">
        {view === "diagram" && (
          <>
            <div className="viewer__scroll" ref={scrollRef}>
              <div className="viewer__stage" ref={stageRef} style={{ width: `${zoom * 100}%`, aspectRatio: `${W} / ${H}` }}>
                <svg viewBox={`0 0 ${W} ${H}`} className="viewer__svg" role="img" aria-label={`${title}: original photo on the left of the slider, tactile version on the right.`}>
                  <defs>
                    <clipPath id="dv-left">
                      <rect x="0" y="0" width={W * split} height={H} />
                    </clipPath>
                    <clipPath id="dv-right">
                      <rect x={W * split} y="0" width={W * (1 - split)} height={H} />
                    </clipPath>
                  </defs>
                  <rect width={W} height={H} fill="#fff" />
                  <g clipPath="url(#dv-left)">
                    <rect width={W} height={H} className="viewer__origbg" />
                    <image href={bust(job.source.url, version)} width={job.source.width} height={job.source.height} transform={`matrix(${T.s} 0 0 ${T.s} ${T.tx} ${T.ty})`} />
                  </g>
                  <g clipPath="url(#dv-right)">
                    <image href={bust(job.tactile.print_url, version)} width={W} height={H} />
                  </g>
                  {sel?.tactile && <path d={`M0 0H${W}V${H}H0Z${sel.d}`} fillRule="evenodd" className="viewer__veil" pointerEvents="none" />}
                  {shapes.map(({ part, d, tactile }) =>
                    d ? (
                      <path
                        key={part.id}
                        d={d}
                        fillRule="evenodd"
                        transform={tactile ? undefined : `matrix(${T.s} 0 0 ${T.s} ${T.tx} ${T.ty})`}
                        className={[
                          "viewer__hit",
                          part.id === selectedId && "is-sel",
                          part.id === hoverId && "is-hover",
                          !part.include && "is-off",
                        ].filter(Boolean).join(" ")}
                        style={{ ["--c" as string]: rgb(part.color) }}
                        vectorEffect="non-scaling-stroke"
                        onPointerEnter={(e) => e.pointerType === "mouse" && onHover(part.id)}
                        onPointerLeave={(e) => e.pointerType === "mouse" && onHover(null)}
                        onClick={() => onSelect(part.id === selectedId ? null : part.id)}
                      />
                    ) : null,
                  )}
                </svg>

                {(sel ?? hov)?.box && (() => {
                  const s = (sel ?? hov)!;
                  const b = s.box!;
                  const top = (b.y0 / H) * 100;
                  const left = (((b.x0 + b.x1) / 2) / W) * 100;
                  return (
                    <div className={"viewer__tag" + (top < 12 ? " is-below" : "")} style={{ left: `${Math.min(88, Math.max(12, left))}%`, top: `${top < 12 ? (b.y1 / H) * 100 : top}%` }} aria-hidden="true">
                      {s.part.key && <BrailleWord word={s.part.key.replace(/[^a-z]/gi, "").slice(0, 2)} className="viewer__tag-braille" />}
                      <span>{label(s.part)}</span>
                    </div>
                  );
                })()}

                <div
                  className="viewer__handle"
                  style={{ left: `${split * 100}%` }}
                  role="slider"
                  tabIndex={0}
                  aria-label="Compare original and tactile"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(split * 100)}
                  aria-valuetext={`${Math.round(split * 100)}% original`}
                  onPointerDown={(e) => {
                    e.preventDefault();
                    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
                    dragFrom(e.clientX);
                  }}
                  onPointerMove={(e) => {
                    if ((e.currentTarget as HTMLElement).hasPointerCapture(e.pointerId)) dragFrom(e.clientX);
                  }}
                  onKeyDown={(e) => {
                    const step = e.shiftKey ? 0.25 : 0.05;
                    if (e.key === "ArrowLeft") setSplit((v) => Math.max(0, v - step));
                    else if (e.key === "ArrowRight") setSplit((v) => Math.min(1, v + step));
                    else if (e.key === "Home") setSplit(0);
                    else if (e.key === "End") setSplit(1);
                    else return;
                    e.preventDefault();
                  }}
                >
                  <span className="viewer__grip" aria-hidden="true">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="m9 6-6 6 6 6M15 6l6 6-6 6" /></svg>
                  </span>
                </div>
              </div>
            </div>
            <div className="viewer__compare">
              <button type="button" className={split >= 0.98 ? "viewer__side is-on" : "viewer__side"} onClick={() => setSplit(1)}>
                Original
              </button>
              <button type="button" className={split > 0.02 && split < 0.98 ? "viewer__side is-on" : "viewer__side"} onClick={() => setSplit(0.5)}>
                Compare
              </button>
              <button type="button" className={split <= 0.02 ? "viewer__side is-on" : "viewer__side"} onClick={() => setSplit(0)}>
                Tactile
              </button>
            </div>
            {!compact && (
              <p className="viewer__caption">
                Tap or point at a shape to find its part. On the tactile side, black is raised; grey labels are print-only, for you.
              </p>
            )}
          </>
        )}

        {view === "vision" && <VisionView job={job} version={version} selectedId={selectedId} onSelect={onSelect} />}

        {view === "legend" && (
          <figure className="viewer__plain">
            <img src={bust(job.tactile.legend_url, version)} alt={`Braille key page for ${title}: each braille key with the part name.`} />
            <figcaption className="viewer__caption">
              Page 2 of the PDF. Braille is UEB grade 1 (uncontracted), English.
            </figcaption>
          </figure>
        )}

        {view === "3d" && (
          <>
            <ErrorBoundary
              resetKey={stlTry}
              fallback={({ retry }) => (
                <div className="stl stl--error" role="alert">
                  <p><strong>The 3D preview didn't load.</strong> Everything else on this page still works, and the STL download is fine.</p>
                  <div className="stl__actions">
                    <button type="button" className="btn btn--primary btn--md" onClick={() => { setStlTry((n) => n + 1); retry(); }}>
                      <IconRefresh size={18} /><span>Retry</span>
                    </button>
                    <button type="button" className="btn btn--secondary btn--md" onClick={() => window.location.reload()}>
                      <span>Reload page</span>
                    </button>
                  </div>
                </div>
              )}
            >
              <Suspense fallback={<div className="stl"><div className="stl__status"><span className="spinner spinner--lg" aria-hidden="true" /> Loading 3D viewer…</div></div>}>
                <Stl url={bust(job.files.stl, version)} label={title} />
              </Suspense>
            </ErrorBoundary>
            <p className="viewer__caption">{stlCaption(job)}</p>
          </>
        )}
      </div>
    </section>
  );
}

export function stlSize(job: JobResult): string | null {
  const s = job.tactile.stl_size_mm;
  if (!s || s.length < 3) return null;
  const f = (n: number) => (Math.round(n * 10) / 10).toString();
  return `${f(s[0])} × ${f(s[1])} × ${f(s[2])} mm`;
}

export function stlCaption(job: JobResult) {
  const sz = stlSize(job);
  return sz
    ? `The 3D model measures ${sz}: a base plate with raised lines and braille domes, ready to slice.`
    : "A base plate with raised lines and braille domes, ready to slice. Your slicer shows its exact size.";
}
