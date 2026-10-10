// TS-09 / UX02-04: the teacher outlines a structure the engine missed, on the original photo, and
// names it. Points are mapped through the SVG's inverse screen matrix, so a marker lands exactly
// where it was tapped even though the photo is letterboxed inside the canvas. Clicks in the
// letterbox padding are ignored. Keyboard: arrows move a crosshair, Enter places a point.
import { useMemo, useRef, useState } from "react";
import { Button } from "./Button";
import { EditorDialog, svgPoint } from "./EditorDialog";
import { IconMinus, IconRefresh } from "./Icons";

export function DrawPartDialog({
  sourceUrl,
  size,
  nameLimit,
  suggestions,
  initialName,
  busy,
  onAdd,
  onClose,
}: {
  sourceUrl: string;
  size: [number, number];
  nameLimit: number;
  suggestions: string[];
  initialName?: string;
  busy: boolean;
  onAdd: (name: string, polygon: [number, number][]) => void;
  onClose: () => void;
}) {
  const [pts, setPts] = useState<[number, number][]>([]);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const [name, setName] = useState(initialName ?? suggestions[0] ?? "");
  const svgRef = useRef<SVGSVGElement>(null);
  const [w, h] = size;
  const listId = useMemo(() => `draw-sugg-${Math.random().toString(36).slice(2, 8)}`, []);

  const addPoint = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const p = svgPoint(svg, e.clientX, e.clientY);
    if (!p) return;
    const [x, y] = p;
    if (x < 0 || y < 0 || x > w || y > h) return; // tap in the letterbox padding, not on the photo
    setPts((prev) => [...prev, [Math.round(x), Math.round(y)]]);
    setCursor([x, y]);
  };

  const onKey = (e: React.KeyboardEvent<SVGSVGElement>) => {
    const step = (e.shiftKey ? 0.05 : 0.01) * Math.max(w, h);
    const cur: [number, number] = cursor ?? [w / 2, h / 2];
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
      setPts((p) => [...p, [Math.round(cur[0]), Math.round(cur[1])]]);
      setCursor(cur);
    } else if (e.key === "Backspace") {
      e.preventDefault();
      setPts((p) => p.slice(0, -1));
    }
  };

  const ok = pts.length >= 3 && name.trim().length > 0 && name.length <= nameLimit;
  const stroke = Math.max(2, Math.max(w, h) / 300);

  return (
    <EditorDialog
      title="Draw a missing part"
      intro="Tap points around the part on the photo, or use the arrow keys and Enter. The shape closes by itself and becomes its own raised part with a braille key."
      onClose={onClose}
      initialFocus="input"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" onClick={() => ok && onAdd(name.trim(), pts)} disabled={!ok || busy} busy={busy}>
            Add part
          </Button>
        </>
      }
    >
      <label className="edlg__field">
        <span>Name</span>
        <input value={name} maxLength={nameLimit} list={listId} onChange={(e) => setName(e.target.value)} />
        <datalist id={listId}>
          {suggestions.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      </label>
      <svg
        ref={svgRef}
        className="edlg__canvas"
        viewBox={`0 0 ${w} ${h}`}
        preserveAspectRatio="xMidYMid meet"
        onPointerDown={addPoint}
        onKeyDown={onKey}
        onFocus={() => setCursor((c) => c ?? [w / 2, h / 2])}
        tabIndex={0}
        role="application"
        aria-label={`Original photo. ${pts.length} points placed. Arrow keys move the crosshair, Enter places a point, Backspace removes the last one.`}
      >
        <image href={sourceUrl} x={0} y={0} width={w} height={h} />
        {pts.length >= 2 && (
          <polygon
            points={pts.map((p) => p.join(",")).join(" ")}
            fill="rgba(255, 212, 59, 0.45)"
            stroke="#17171C"
            strokeWidth={stroke}
            strokeLinejoin="round"
          />
        )}
        {cursor && (
          <g pointerEvents="none" stroke="#E5484D" strokeWidth={stroke}>
            <line x1={cursor[0] - stroke * 6} y1={cursor[1]} x2={cursor[0] + stroke * 6} y2={cursor[1]} />
            <line x1={cursor[0]} y1={cursor[1] - stroke * 6} x2={cursor[0]} y2={cursor[1] + stroke * 6} />
          </g>
        )}
        {pts.map((p, i) => (
          <circle
            key={i}
            className="edlg__pt"
            cx={p[0]}
            cy={p[1]}
            r={stroke * 2.2}
            fill={i === 0 ? "#17171C" : "#FFD43B"}
            stroke="#17171C"
            strokeWidth={stroke / 2}
          />
        ))}
      </svg>
      <div className="edlg__tools">
        <span className="edlg__count" aria-live="polite">
          {pts.length} {pts.length === 1 ? "point" : "points"}
          {pts.length > 0 && pts.length < 3 ? " · add at least 3" : ""}
        </span>
        <Button variant="secondary" size="sm" icon={<IconMinus size={16} />} onClick={() => setPts((p) => p.slice(0, -1))} disabled={!pts.length || busy}>
          Undo point
        </Button>
        <Button variant="secondary" size="sm" icon={<IconRefresh size={16} />} onClick={() => setPts([])} disabled={!pts.length || busy}>
          Clear
        </Button>
      </div>
    </EditorDialog>
  );
}
