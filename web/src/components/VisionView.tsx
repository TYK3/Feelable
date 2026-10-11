// "How it read": the image pipeline made visible on the teacher's own picture.
// Four steps, each a layer you can switch on and off:
//   1 Read     — every label the OCR found (highlighter boxes); blurry ones re-read up close and
//                spelling-repaired against a science vocabulary ("Parktatbone" -> "Parietal bone")
//   2 Followed — the pointer lines traced from each label to what it names
//   3 Traced   — shapes found by colour, or carved out up close where a pointer ends (dashed)
//   4 AI found — structures the image model located when a pointer couldn't be followed (dotted)
// Line style, not colour alone, tells the shapes apart. Clicking a label selects its part.
import { useEffect, useMemo, useState } from "react";
import { getVision } from "../api";
import type { JobResult, VisionTrace } from "../types";
import { polysToPath } from "./PartOverlay";
import { sentenceCase } from "./text";

type Layer = "read" | "followed" | "traced" | "ai";

export function VisionView({
  job,
  version,
  selectedId,
  onSelect,
}: {
  job: JobResult;
  version: number;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const [v, setV] = useState<VisionTrace | null>(null);
  const [err, setErr] = useState(false);
  const [on, setOn] = useState<Record<Layer, boolean>>({ read: true, followed: true, traced: true, ai: true });

  useEffect(() => {
    const ctl = new AbortController();
    setErr(false);
    getVision(job.id, ctl.signal)
      .then(setV)
      .catch((e) => {
        if ((e as Error)?.name !== "AbortError") setErr(true);
      });
    return () => ctl.abort();
  }, [job.id, version]);

  const methodOf = useMemo(() => new Map((v?.parts ?? []).map((p) => [p.id, p.match_method])), [v]);
  const parts = job.parts.filter((p) => p.include);

  if (err) {
    return (
      <div className="vision vision--empty" role="status">
        <p>The reading steps aren't available for this lesson. The diagram, braille key and 3D views still work.</p>
      </div>
    );
  }
  if (!v) {
    return (
      <div className="vision vision--empty" aria-busy="true">
        <span className="spinner" aria-hidden="true" /> Loading what Feelable saw…
      </div>
    );
  }

  const W = v.width, H = v.height;
  const st = v.steps;
  const read = v.labels.filter((l) => l.role !== "title");
  const repaired = v.labels.filter((l) => l.raw);
  const traced = parts.filter((p) => (methodOf.get(p.id) ?? p.match_method) !== "ai");
  const aiFound = parts.filter((p) => (methodOf.get(p.id) ?? p.match_method) === "ai");
  const sw = Math.max(1.5, Math.max(W, H) / 500);
  const toggle = (k: Layer) => setOn((o) => ({ ...o, [k]: !o[k] }));

  const steps: { k: Layer; n: number; title: string; detail: string }[] = [
    { k: "read", n: 1, title: `Read ${read.length} ${read.length === 1 ? "label" : "labels"}`,
      detail: repaired.length ? `${repaired.length} blurry ${repaired.length === 1 ? "word" : "words"} re-read up close and repaired` : "Text found and read on the Feelable server" },
    { k: "followed", n: 2, title: `Followed ${st.pointers ?? v.leaders.length} pointer ${(st.pointers ?? v.leaders.length) === 1 ? "line" : "lines"}`,
      detail: "From each label to the thing it names" },
    { k: "traced", n: 3, title: `Traced ${traced.length} ${traced.length === 1 ? "shape" : "shapes"}`,
      detail: st.carved ? `${st.carved} carved out up close where a pointer ends` : "Found by colour and outline" },
    { k: "ai", n: 4, title: aiFound.length ? `AI found ${aiFound.length}` : job.ai_used ? "AI not needed" : "AI was off",
      detail: aiFound.length ? "Located by the image model where a pointer was lost"
        : job.ai_used ? "Every pointer reached its shape" : "With AI on, it finds structures whose pointer was lost" },
  ];

  return (
    <div className="vision">
      <div className="vision__stage">
        <svg viewBox={`0 0 ${W} ${H}`} className="vision__svg" role="img"
             aria-label={`What Feelable saw in ${sentenceCase(job.title) || "the diagram"}: ${read.length} labels, ${v.leaders.length} pointer lines, ${parts.length} shapes.`}>
          <image href={job.source.url} x={0} y={0} width={W} height={H} />
          <rect width={W} height={H} className="vision__dim" />
          {on.traced && traced.map((p) => (
            <path key={p.id} d={polysToPath(p.source_polygons)}
                  className={`vision__shape${(methodOf.get(p.id) ?? p.match_method) === "carved" ? " is-carved" : ""}${p.id === selectedId ? " is-sel" : ""}`}
                  strokeWidth={2} onClick={() => onSelect(p.id)} />
          ))}
          {on.ai && aiFound.map((p) => (
            <path key={p.id} d={polysToPath(p.source_polygons)} className={`vision__shape is-ai${p.id === selectedId ? " is-sel" : ""}`}
                  strokeWidth={2.5} onClick={() => onSelect(p.id)} />
          ))}
          {on.followed && v.leaders.map((L, i) => {
            const pts = L.points;
            if (pts.length < 2) return null;
            const [ex, ey] = pts[pts.length - 1];
            return (
              <g key={i} className="vision__leader">
                <polyline points={pts.map((p) => p.join(",")).join(" ")} strokeWidth={sw * 1.4} pathLength={1} />
                <circle cx={ex} cy={ey} r={sw * 2.6} />
              </g>
            );
          })}
          {on.read && read.map((l, i) => {
            const [x0, y0, x1, y1] = l.bbox;
            return (
              <g key={i} className={`vision__label${l.raw ? " is-fixed" : ""}${l.part_id ? "" : " is-loose"}`}
                 onClick={() => l.part_id && onSelect(l.part_id)}>
                <rect x={x0 - 2} y={y0 - 1} width={x1 - x0 + 4} height={y1 - y0 + 2} rx={sw * 1.5} strokeWidth={sw} />
                {l.raw && <circle cx={x1 + 2} cy={y0 - 1} r={sw * 3} />}
                <title>{l.raw ? `Read as “${l.raw}”, repaired to “${l.text}”` : l.text}</title>
              </g>
            );
          })}
        </svg>
      </div>

      <ol className="vision__steps" aria-label="How Feelable read this picture">
        {steps.map((s) => (
          <li key={s.k}>
            <button type="button" className={`vision__step is-${s.k}`} aria-pressed={on[s.k]} onClick={() => toggle(s.k)}>
              <span className="vision__num" aria-hidden="true">{s.n}</span>
              <span className="vision__txt">
                <strong>{s.title}</strong>
                <span>{s.detail}</span>
              </span>
              <span className="vision__key" aria-hidden="true" />
            </button>
          </li>
        ))}
      </ol>

      {repaired.length > 0 && (
        <details className="vision__fixes">
          <summary>Words repaired up close ({repaired.length})</summary>
          <ul>
            {repaired.map((l, i) => (
              <li key={i}>
                <s>{l.raw}</s> <span aria-hidden="true">→</span><span className="sr-only">became</span> <strong>{l.text}</strong>
              </li>
            ))}
          </ul>
        </details>
      )}
      <p className="viewer__caption">Tap a step to show or hide it. Tap a label or a shape to select that part.</p>
    </div>
  );
}
