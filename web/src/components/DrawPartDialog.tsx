// TS-09 / handover "Required feature": add or correct a part the scan missed, on the original photo.
//
// 1. Mark it: tap inside the structure (and, if needed, mark what is NOT part of it).
// 2. Find outline: Feelable proposes an outline from the picture around the marks (never from the
//    name alone). The proposal becomes editable points: accept it, refine it, or draw it yourself.
// 3. Choose what it is: a new part, or an area to add to / remove from an existing part.
// 4. Save part: if the outline covers other parts, the teacher chooses what happens to them.
//
// Every edit advances a draft revision; an outline answer for an older revision is ignored, so a
// slow response can never overwrite newer drawing. Points are mapped through the SVG's inverse
// screen matrix, so marks land exactly where they were tapped even when the photo is letterboxed.
// Keyboard: arrows move a crosshair, Enter places a point (or a mark), Backspace undoes.
// Precision audit: Zoom in / out, Fit, Focus on the marks and a Move-picture mode (pan) so tiny
// structures can be isolated on a phone; Adjust points drags a vertex; a suggestion with several
// pieces (both kidneys) keeps every piece; the status line always describes the current draft.
import { useMemo, useRef, useState } from "react";
import { ApiError, findOutline, type OutlineOverlap } from "../api";
import type { Part } from "../types";
import { Button } from "./Button";
import { EditorDialog, svgPoint } from "./EditorDialog";
import { polysToPath } from "./PartOverlay";
import { IconMinus, IconRefresh } from "./Icons";

type Pt = [number, number];
type Tool = "inside" | "outside" | "draw" | "adjust" | "pan";
type Draft = { pts: Pt[]; more: Pt[][]; inside: Pt[]; outside: Pt[]; suggested: boolean };
export type DrawOptions = { overlap: "cut" | "keep" | "replace"; into: string | null; mode: "add" | "remove"; more?: Pt[][] };

const EMPTY: Draft = { pts: [], more: [], inside: [], outside: [], suggested: false };
const MAX_ZOOM = 8;
const MAX_POINTS = 400;

function thin(poly: Pt[]): Pt[] {
  if (poly.length <= MAX_POINTS) return poly;
  const k = Math.ceil(poly.length / MAX_POINTS);
  return poly.filter((_, i) => i % k === 0);
}

export function DrawPartDialog({
  jobId,
  sourceUrl,
  size,
  nameLimit,
  suggestions,
  initialName,
  parts,
  initialInto,
  busy,
  onAdd,
  onClose,
}: {
  jobId: string;
  sourceUrl: string;
  size: [number, number];
  nameLimit: number;
  suggestions: string[];
  initialName?: string;
  parts: Part[];
  initialInto?: string | null;
  busy: boolean;
  onAdd: (name: string, polygon: Pt[], opts: DrawOptions) => Promise<string | null> | void;
  onClose: () => void;
}) {
  const [hist, setHist] = useState<{ past: Draft[]; now: Draft; future: Draft[] }>({ past: [], now: EMPTY, future: [] });
  const d = hist.now;
  const [tool, setTool] = useState<Tool>("inside");
  const [cursor, setCursor] = useState<Pt | null>(null);
  const [name, setName] = useState(initialName ?? suggestions[0] ?? "");
  const [target, setTarget] = useState<string>(initialInto ? `add:${initialInto}` : "new");
  const [finding, setFinding] = useState(false);
  // a one-off message about the LAST action (outline found / removed / failed); cleared by any
  // later change so it can never describe a draft that no longer exists
  const [note, setNote] = useState<string>("");
  const [overlaps, setOverlaps] = useState<OutlineOverlap[] | null>(null);
  const [saveErr, setSaveErr] = useState<string | null>(null);
  const rev = useRef(0);
  const abort = useRef<AbortController | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [w, h] = size;
  // view: zoom factor and the top-left of the visible window (source px)
  const [view, setView] = useState<{ z: number; x: number; y: number }>({ z: 1, x: 0, y: 0 });
  const drag = useRef<{ kind: "pan" | "vertex"; idx: number; start: Pt; view: { z: number; x: number; y: number } } | null>(null);
  const listId = useMemo(() => `draw-sugg-${Math.random().toString(36).slice(2, 8)}`, []);
  const included = parts.filter((p) => p.include);

  const invalidate = () => {
    rev.current += 1;
    abort.current?.abort();
    abort.current = null;
    setFinding(false);
    setOverlaps(null);
    setSaveErr(null);
  };
  // every change is a new draft revision; any outline request for an older one is now stale
  const commit = (next: Draft, msg = "") => {
    invalidate();
    setNote(msg);
    setHist((hh) => ({ past: [...hh.past, hh.now].slice(-50), now: next, future: [] }));
  };
  const undo = () => {
    if (!hist.past.length) return;
    const prev = hist.past[hist.past.length - 1];
    invalidate();
    setNote(d.suggested && !prev.suggested ? "Suggested outline removed." : prev.pts.length < d.pts.length ? "Last point removed." : "Undone.");
    setHist((hh) => (hh.past.length ? { past: hh.past.slice(0, -1), now: hh.past[hh.past.length - 1], future: [hh.now, ...hh.future] } : hh));
  };
  const redo = () => {
    if (!hist.future.length) return;
    const next = hist.future[0];
    invalidate();
    setNote(next.suggested && !d.suggested ? "Suggested outline restored." : "Redone.");
    setHist((hh) => (hh.future.length ? { past: [...hh.past, hh.now], now: hh.future[0], future: hh.future.slice(1) } : hh));
  };

  // ---- view (zoom / pan) ----
  const clampView = (z: number, x: number, y: number) => {
    const zz = Math.min(MAX_ZOOM, Math.max(1, z));
    const vw = w / zz, vh = h / zz;
    return { z: zz, x: Math.min(w - vw, Math.max(0, x)), y: Math.min(h - vh, Math.max(0, y)) };
  };
  const zoomAt = (factor: number, c?: Pt) => {
    const vw = w / view.z, vh = h / view.z;
    const cx = c ? c[0] : view.x + vw / 2, cy = c ? c[1] : view.y + vh / 2;
    const z = view.z * factor;
    setView(clampView(z, cx - w / z / 2, cy - h / z / 2));
  };
  const focusMarks = () => {
    const all = [...d.pts, ...d.more.flat(), ...d.inside, ...d.outside];
    if (!all.length) return;
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = 0.15 * Math.max(x1 - x0, y1 - y0, 0.08 * Math.max(w, h));
    const z = Math.min(w / (x1 - x0 + 2 * pad), h / (y1 - y0 + 2 * pad));
    setView(clampView(z, (x0 + x1) / 2 - w / z / 2, (y0 + y1) / 2 - h / z / 2));
  };
  const place = (p: Pt) => {
    const q: Pt = [Math.round(p[0]), Math.round(p[1])];
    if (tool === "inside") commit({ ...d, inside: [...d.inside, q] });
    else if (tool === "outside") commit({ ...d, outside: [...d.outside, q] });
    else commit({ ...d, pts: [...d.pts, q], suggested: false });
  };

  const onPointer = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const p = svgPoint(svg, e.clientX, e.clientY);
    if (!p) return;
    const [x, y] = p;
    if (tool === "pan") {
      drag.current = { kind: "pan", idx: -1, start: [e.clientX, e.clientY], view };
      svg.setPointerCapture?.(e.pointerId);
      return;
    }
    if (x < 0 || y < 0 || x > w || y > h) return; // tap in the letterbox padding, not on the photo
    if (tool === "adjust") {
      const r = (Math.max(w, h) / view.z) * 0.03;
      let best = -1, bd = r;
      d.pts.forEach((q, i) => {
        const dd = Math.hypot(q[0] - x, q[1] - y);
        if (dd < bd) { best = i; bd = dd; }
      });
      if (best >= 0) {
        commit({ ...d, suggested: false }, "Drag the point to move it.");
        drag.current = { kind: "vertex", idx: best, start: [x, y], view };
        svg.setPointerCapture?.(e.pointerId);
      }
      return;
    }
    place([x, y]);
    setCursor([x, y]);
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const g = drag.current;
    const svg = svgRef.current;
    if (!g || !svg) return;
    if (g.kind === "pan") {
      const rect = svg.getBoundingClientRect();
      const scale = (w / g.view.z) / Math.max(1, rect.width);
      setView(clampView(g.view.z, g.view.x - (e.clientX - g.start[0]) * scale, g.view.y - (e.clientY - g.start[1]) * scale));
      return;
    }
    const p = svgPoint(svg, e.clientX, e.clientY);
    if (!p) return;
    const q: Pt = [Math.round(Math.min(w, Math.max(0, p[0]))), Math.round(Math.min(h, Math.max(0, p[1])))];
    setHist((hh) => ({ ...hh, now: { ...hh.now, pts: hh.now.pts.map((v, i) => (i === g.idx ? q : v)) } }));
  };
  const endDrag = () => {
    drag.current = null;
  };

  const onKey = (e: React.KeyboardEvent<SVGSVGElement>) => {
    const step = (e.shiftKey ? 0.05 : 0.01) * Math.max(w, h);
    const cur: Pt = cursor ?? [w / 2, h / 2];
    const move = (dx: number, dy: number) => {
      e.preventDefault();
      setCursor([Math.min(w, Math.max(0, cur[0] + dx)), Math.min(h, Math.max(0, cur[1] + dy))]);
    };
    if (e.key === "ArrowLeft") move(-step, 0);
    else if (e.key === "ArrowRight") move(step, 0);
    else if (e.key === "ArrowUp") move(0, -step);
    else if (e.key === "ArrowDown") move(0, step);
    else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      place(cur);
    } else if (e.key === "Backspace") {
      e.preventDefault();
      undo();
    }
  };

  const find = async () => {
    invalidate();
    const myRev = rev.current;
    const ac = new AbortController();
    abort.current = ac;
    setFinding(true);
    setNote("");
    try {
      const lasso = d.inside.length === 0 && d.pts.length >= 3 ? d.pts : null;
      const res = await findOutline(jobId, { rev: myRev, inside: d.inside, outside: d.outside, lasso }, ac.signal);
      if (res.rev !== rev.current || ac.signal.aborted) return; // a newer draft exists: never overwrite it
      if (!res.found || !res.polygons.length) {
        setNote(res.message ?? "We couldn't find a clear outline. Keep drawing or try another selection.");
        return;
      }
      // keep EVERY piece (both kidneys, several mitochondria), biggest first
      const polys = res.polygons.map((pg) => thin(pg)).filter((pg) => pg.length >= 3);
      rev.current += 1;
      setHist((hh) => ({ past: [...hh.past, hh.now].slice(-50), now: { ...hh.now, pts: polys[0], more: polys.slice(1, 13), suggested: true }, future: [] }));
      setNote(
        polys.length > 1
          ? `Suggested outline in ${polys.length} pieces — check them. All pieces are kept; use Keep largest piece if only one belongs.`
          : "Suggested outline — check it. Use it, refine the points, or draw it yourself.",
      );
    } catch (err) {
      if ((err as Error)?.name === "AbortError") return;
      setNote(err instanceof ApiError ? `${err.message} Your marks are kept; keep drawing or try again.` : "Couldn't reach Feelable. Your marks are kept; keep drawing or try again.");
    } finally {
      if (abort.current === ac) {
        abort.current = null;
        setFinding(false);
      }
    }
  };

  const opts = (): DrawOptions => {
    const more = d.more.length ? d.more : undefined;
    if (target.startsWith("add:")) return { overlap: "cut", into: target.slice(4), mode: "add", more };
    if (target.startsWith("remove:")) return { overlap: "keep", into: target.slice(7), mode: "remove", more };
    return { overlap: "cut", into: null, mode: "add", more };
  };
  const isNew = target === "new";
  const nameOk = !isNew || (name.trim().length > 0 && name.length <= nameLimit);
  const ok = d.pts.length >= 3 && nameOk;

  // Save: first check what the outline would cover; ask before changing any other part
  const save = async (overlap?: DrawOptions["overlap"]) => {
    if (!ok) return;
    const o = opts();
    if (!overlap && o.mode === "add") {
      try {
        const res = await findOutline(jobId, { rev: rev.current, polygon: d.pts });
        const others = res.overlaps.filter((x) => x.part_id !== o.into && x.covered >= 0.02);
        if (others.length) {
          setOverlaps(others);
          return;
        }
      } catch {
        /* overlap check is advisory; the server still refuses to erase a part silently */
      }
    }
    const err = await onAdd(name.trim(), d.pts, { ...o, overlap: overlap ?? o.overlap });
    if (typeof err === "string" && err) setSaveErr(err);
  };

  const stroke = Math.max(1, Math.max(w, h) / 300 / Math.sqrt(view.z));
  const toolBtn = (t: Tool, label: string) => (
    <button type="button" className={`seg__btn${tool === t ? " is-on" : ""}`} aria-pressed={tool === t} onClick={() => setTool(t)}>
      {label}
    </button>
  );
  const full = overlaps?.filter((x) => x.covered >= 0.98) ?? [];

  return (
    <EditorDialog
      fullOnPhone
      title="Add or fix a missing part"
      intro="Tap inside the part, then Find outline. Or choose Draw myself and tap points around it. Nothing changes until you save."
      onClose={onClose}
      initialFocus="input"
      footer={
        overlaps ? (
          <div className="draw__choice" role="group" aria-label="This outline covers other parts">
            <p>
              This outline covers {overlaps.map((x) => `“${x.name}” (${Math.round(x.covered * 100)}%)`).join(", ")}.
            </p>
            <div className="draw__choice-btns">
              {full.length === 0 && (
                <Button variant="primary" onClick={() => save("cut")} disabled={busy}>
                  Cut it out of them
                </Button>
              )}
              <Button variant="secondary" onClick={() => save("keep")} disabled={busy}>
                Keep both (overlap)
              </Button>
              {full.length > 0 && (
                <Button variant="secondary" onClick={() => save("replace")} disabled={busy}>
                  Replace {full.length === 1 ? `“${full[0].name}”` : "them"}
                </Button>
              )}
              <Button variant="ghost" onClick={() => setOverlaps(null)}>
                Back to drawing
              </Button>
            </div>
          </div>
        ) : (
          <>
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" onClick={() => save()} disabled={!ok || busy} busy={busy}>
              {isNew ? "Save part" : target.startsWith("remove:") ? "Remove this area" : "Add this area"}
            </Button>
          </>
        )
      }
    >
      <div className="draw__row">
        <label className="edlg__field">
          <span>What is it?</span>
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="new">A new part</option>
            {included.map((p) => (
              <option key={`a${p.id}`} value={`add:${p.id}`}>
                Add to “{p.name}”
              </option>
            ))}
            {included.map((p) => (
              <option key={`r${p.id}`} value={`remove:${p.id}`}>
                Remove from “{p.name}”
              </option>
            ))}
          </select>
        </label>
        {isNew && (
          <label className="edlg__field">
            <span>Name</span>
            <input value={name} maxLength={nameLimit} list={listId} onChange={(e) => setName(e.target.value)} />
            <datalist id={listId}>
              {suggestions.map((s) => (
                <option key={s} value={s} />
              ))}
            </datalist>
          </label>
        )}
      </div>
      <div className="seg" role="group" aria-label="What a tap does">
        {toolBtn("inside", "Mark the part")}
        {toolBtn("outside", "Exclude nearby area")}
        {toolBtn("draw", "Draw myself")}
        {toolBtn("adjust", "Adjust points")}
        {toolBtn("pan", "Move picture")}
      </div>
      <div className="draw__view" role="group" aria-label="Zoom">
        <Button variant="secondary" size="sm" onClick={() => zoomAt(1.6, cursor ?? undefined)} disabled={view.z >= MAX_ZOOM}>
          Zoom in
        </Button>
        <Button variant="secondary" size="sm" onClick={() => zoomAt(1 / 1.6)} disabled={view.z <= 1}>
          Zoom out
        </Button>
        <Button variant="secondary" size="sm" onClick={() => setView({ z: 1, x: 0, y: 0 })} disabled={view.z <= 1}>
          Fit
        </Button>
        <Button variant="secondary" size="sm" onClick={focusMarks} disabled={!d.pts.length && !d.inside.length && !d.outside.length}>
          Focus on marks
        </Button>
      </div>
      <svg
        ref={svgRef}
        className={`edlg__canvas draw__canvas${tool === "pan" ? " is-pan" : ""}`}
        viewBox={`${view.x} ${view.y} ${w / view.z} ${h / view.z}`}
        preserveAspectRatio="xMidYMid meet"
        onPointerDown={onPointer}
        onPointerMove={onMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onKeyDown={onKey}
        onFocus={() => setCursor((c) => c ?? [w / 2, h / 2])}
        tabIndex={0}
        role="application"
        aria-label={`Original photo${view.z > 1 ? `, zoomed ${view.z.toFixed(1)} times` : ""}. ${d.inside.length} marks on the part, ${d.outside.length} marks off it, ${d.pts.length ? `${d.pts.length} outline points${d.more.length ? ` in ${d.more.length + 1} pieces` : ""}` : "no outline yet"}. Arrow keys move the crosshair, Enter places, Backspace undoes.`}
      >
        <image href={sourceUrl} x={0} y={0} width={w} height={h} />
        {included.map((p) => (
          <path key={p.id} d={polysToPath(p.source_polygons)} fill="none" stroke="#17171C" strokeOpacity={0.25} strokeWidth={stroke / 2} pointerEvents="none" />
        ))}
        {d.more.map((pg, k) => (
          <polygon
            key={`m${k}`}
            points={pg.map((p) => p.join(",")).join(" ")}
            fill={target.startsWith("remove:") ? "rgba(229, 72, 77, 0.35)" : "rgba(255, 212, 59, 0.45)"}
            stroke="#17171C"
            strokeWidth={stroke}
            strokeDasharray={`${stroke * 3} ${stroke * 2}`}
            pointerEvents="none"
          />
        ))}
        {d.pts.length >= 2 && (
          <polygon
            points={d.pts.map((p) => p.join(",")).join(" ")}
            fill={target.startsWith("remove:") ? "rgba(229, 72, 77, 0.35)" : "rgba(255, 212, 59, 0.45)"}
            stroke="#17171C"
            strokeWidth={stroke}
            strokeDasharray={d.suggested ? `${stroke * 3} ${stroke * 2}` : undefined}
            strokeLinejoin="round"
            pointerEvents="none"
          />
        )}
        {cursor && (
          <g pointerEvents="none" stroke="#E5484D" strokeWidth={stroke}>
            <line x1={cursor[0] - stroke * 6} y1={cursor[1]} x2={cursor[0] + stroke * 6} y2={cursor[1]} />
            <line x1={cursor[0]} y1={cursor[1] - stroke * 6} x2={cursor[0]} y2={cursor[1] + stroke * 6} />
          </g>
        )}
        {(!d.suggested || tool === "adjust") &&
          d.pts.map((p, i) => (
            <circle key={i} className="edlg__pt" cx={p[0]} cy={p[1]} r={stroke * 2.2} fill={i === 0 ? "#17171C" : "#FFD43B"} stroke="#17171C" strokeWidth={stroke / 2} pointerEvents="none" />
          ))}
        {d.inside.map((p, i) => (
          <circle key={`i${i}`} cx={p[0]} cy={p[1]} r={stroke * 3} fill="#2F9E44" stroke="#fff" strokeWidth={stroke / 1.5} pointerEvents="none" />
        ))}
        {d.outside.map((p, i) => (
          <g key={`o${i}`} stroke="#E5484D" strokeWidth={stroke * 1.4} pointerEvents="none">
            <line x1={p[0] - stroke * 3} y1={p[1] - stroke * 3} x2={p[0] + stroke * 3} y2={p[1] + stroke * 3} />
            <line x1={p[0] - stroke * 3} y1={p[1] + stroke * 3} x2={p[0] + stroke * 3} y2={p[1] - stroke * 3} />
          </g>
        ))}
      </svg>
      <p className="draw__status" role="status" aria-live="polite">
        {finding
          ? "Finding the outline…"
          : note ||
            (d.pts.length >= 3
              ? d.suggested
                ? "Suggested outline — check it. Use it, refine the points, or draw it yourself."
                : `Outline with ${d.pts.length} points. Save it, or keep refining.`
              : tool === "draw"
                ? "Tap points around the part. The shape closes by itself."
                : tool === "pan"
                  ? "Drag to move the picture. Choose a marking tool to add marks."
                  : tool === "adjust"
                    ? "Drag an outline point to move it."
                    : "Tap inside the part (green). Mark anything that is not part of it with Exclude nearby area.")}
      </p>
      {d.more.length > 0 && (
        <p className="draw__pieces">
          {d.more.length + 1} pieces will be saved together.{" "}
          <Button variant="ghost" size="sm" onClick={() => commit({ ...d, more: [] }, "Kept only the largest piece.")}>
            Keep largest piece
          </Button>
        </p>
      )}
      {saveErr && (
        <p className="edlg__err" role="alert">
          {saveErr}
        </p>
      )}
      <div className="edlg__tools">
        <Button variant="primary" size="sm" onClick={find} disabled={finding || busy || (d.inside.length === 0 && d.pts.length < 3)} busy={finding}>
          Find outline
        </Button>
        <Button variant="secondary" size="sm" icon={<IconMinus size={16} />} onClick={undo} disabled={!hist.past.length || busy}>
          Undo
        </Button>
        <Button variant="secondary" size="sm" onClick={redo} disabled={!hist.future.length || busy}>
          Redo
        </Button>
        <Button variant="secondary" size="sm" icon={<IconRefresh size={16} />} onClick={() => commit(EMPTY, "Started over. Earlier marks and outline removed.")} disabled={(!d.pts.length && !d.inside.length && !d.outside.length) || busy}>
          Start over
        </Button>
      </div>
    </EditorDialog>
  );
}
