// SVG overlay of part polygons, aligned to an image via viewBox = image pixel size.
import type { Part, Polygon } from "../types";

export function polysToPath(polys: Polygon[]): string {
  return polys
    .filter((p) => p.length >= 3)
    .map((p) => "M" + p.map(([x, y]) => `${x} ${y}`).join("L") + "Z")
    .join("");
}

export function rgb(c: [number, number, number], a = 1) {
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
}

export function PartOverlay({
  parts,
  width,
  height,
  mode,
  activeId,
  onHover,
}: {
  parts: Part[];
  width: number;
  height: number;
  mode: "source" | "tactile";
  activeId: string | null;
  onHover?: (id: string | null) => void;
}) {
  if (mode === "tactile") {
    const p = parts.find((x) => x.include && x.id === activeId);
    const d = p ? polysToPath(p.tactile_polygons) : "";
    return (
      <svg className="overlay overlay--tactile" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true" focusable="false">
        {d && <path key={p!.id} d={d} className="overlay__hl" vectorEffect="non-scaling-stroke" />}
      </svg>
    );
  }
  const active = parts.find((x) => x.id === activeId);
  const activeD = active ? polysToPath(active.source_polygons) : "";
  return (
    <svg
      className={`overlay overlay--source${activeD ? " has-active" : ""}`}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      focusable="false"
    >
      {parts.map((p) => {
        const d = polysToPath(p.source_polygons);
        if (!d) return null;
        return (
          <path
            key={p.id}
            d={d}
            fillRule="evenodd"
            className={["overlay__part", !p.include && "is-excluded"].filter(Boolean).join(" ")}
            style={{ fill: rgb(p.color, 0.35), stroke: rgb(darken(p.color), 1) }}
            vectorEffect="non-scaling-stroke"
            onMouseEnter={() => onHover?.(p.id)}
            onMouseLeave={() => onHover?.(null)}
          />
        );
      })}
      {activeD && (
        <g className="overlay__spot" pointerEvents="none">
          <path d={`M0 0H${width}V${height}H0Z${activeD}`} fillRule="evenodd" className="overlay__veil" />
          <path d={activeD} fillRule="evenodd" className="overlay__glow" vectorEffect="non-scaling-stroke" />
          <path d={activeD} fillRule="evenodd" className="overlay__edge" vectorEffect="non-scaling-stroke" />
        </g>
      )}
    </svg>
  );
}

function darken(c: [number, number, number], k = 0.55): [number, number, number] {
  return [Math.round(c[0] * k), Math.round(c[1] * k), Math.round(c[2] * k)];
}
