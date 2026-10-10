import { useId } from "react";

// Tiny preview of a tactile texture + its friendly name.
export const TEXTURE_NAMES: Record<string, string> = {
  none: "Outline only",
  dots: "Dots",
  diag: "Diagonal lines",
  horiz: "Horizontal lines",
  vert: "Vertical lines",
  cross: "Crosshatch",
  solid: "Solid",
  grid: "Grid",
};

export function textureName(t: string) {
  return TEXTURE_NAMES[t] ?? t;
}

export function TextureSwatch({ texture }: { texture: string }) {
  const clip = "tx" + useId().replace(/:/g, "");
  const lines: JSX.Element[] = [];
  const S = 16;
  if (texture === "dots") {
    for (let y = 3; y < S; y += 5) for (let x = 3; x < S; x += 5) lines.push(<circle key={`${x}-${y}`} cx={x} cy={y} r={1.1} fill="currentColor" />);
  } else if (texture === "horiz" || texture === "grid") {
    for (let y = 3; y < S; y += 4) lines.push(<line key={`h${y}`} x1={0} y1={y} x2={S} y2={y} />);
  }
  if (texture === "vert" || texture === "grid") {
    for (let x = 3; x < S; x += 4) lines.push(<line key={`v${x}`} x1={x} y1={0} x2={x} y2={S} />);
  }
  if (texture === "diag" || texture === "cross") {
    for (let k = -S; k < S * 2; k += 4) lines.push(<line key={`d${k}`} x1={k} y1={S} x2={k + S} y2={0} />);
  }
  if (texture === "cross") {
    for (let k = -S; k < S * 2; k += 4) lines.push(<line key={`c${k}`} x1={k} y1={0} x2={k + S} y2={S} />);
  }
  return (
    <svg className="texture" width={S} height={S} viewBox={`0 0 ${S} ${S}`} aria-hidden="true" focusable="false">
      <defs>
        <clipPath id={clip}>
          <rect x={0} y={0} width={S} height={S} rx={4} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clip})`} stroke="currentColor" strokeWidth={1.3}>
        {texture === "solid" && <rect x={0} y={0} width={S} height={S} fill="currentColor" />}
        {lines}
      </g>
      <rect x={0.75} y={0.75} width={S - 1.5} height={S - 1.5} rx={3.5} fill="none" stroke="currentColor" strokeWidth={1.5} />
    </svg>
  );
}
