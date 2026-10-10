// "Make a close-up lesson" (UX02/03/04): drag a box over a crowded area of the original photo;
// the crop is enlarged and converted as a linked detail lesson. The drag is mapped through the
// SVG's inverse screen matrix (no letterbox drift) and the exact same box is rasterised.
// Keyboard: focus the photo, arrows move the box, Shift+arrows resize it.
import { useRef, useState } from "react";
import { Button } from "./Button";
import { EditorDialog, svgPoint } from "./EditorDialog";

export type Box = [number, number, number, number];

export function CloseupDialog({
  sourceUrl,
  size,
  title,
  initialBox,
  onMake,
  onClose,
}: {
  sourceUrl: string;
  size: [number, number];
  title: string;
  initialBox?: Box | null;
  onMake: (file: File, box: Box) => void;
  onClose: () => void;
}) {
  const [w, h] = size;
  const [box, setBox] = useState<Box | null>(initialBox ?? null);
  const [start, setStart] = useState<[number, number] | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const clamp = (p: [number, number]): [number, number] => [Math.min(w, Math.max(0, p[0])), Math.min(h, Math.max(0, p[1]))];
  const toSrc = (e: React.PointerEvent): [number, number] | null => {
    const p = svgRef.current && svgPoint(svgRef.current, e.clientX, e.clientY);
    return p ? clamp(p) : null;
  };
  const norm = (a: [number, number], b: [number, number]): Box => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];

  const onKey = (e: React.KeyboardEvent<SVGSVGElement>) => {
    const keys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"];
    if (!keys.includes(e.key)) return;
    e.preventDefault();
    const step = 0.02 * Math.max(w, h);
    const b: Box = box ?? [w * 0.3, h * 0.3, w * 0.7, h * 0.7];
    const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
    const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
    if (e.shiftKey) {
      // resize from the bottom-right corner
      setBox([b[0], b[1], Math.min(w, Math.max(b[0] + 40, b[2] + dx)), Math.min(h, Math.max(b[1] + 40, b[3] + dy))]);
    } else {
      const bw = b[2] - b[0];
      const bh = b[3] - b[1];
      const x0 = Math.min(w - bw, Math.max(0, b[0] + dx));
      const y0 = Math.min(h - bh, Math.max(0, b[1] + dy));
      setBox([x0, y0, x0 + bw, y0 + bh]);
    }
  };

  const make = async () => {
    if (!box) return;
    const [x0, y0, x1, y1] = box;
    if (x1 - x0 < 40 || y1 - y0 < 40) {
      setErr("Choose a bigger area around the crowded part.");
      return;
    }
    setBusy(true);
    try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = sourceUrl;
      await img.decode();
      const sx = img.naturalWidth / w;
      const sy = img.naturalHeight / h;
      const cw = (x1 - x0) * sx;
      const ch = (y1 - y0) * sy;
      const scale = Math.max(1, Math.min(3, 1600 / Math.max(cw, ch))); // enlarge so small parts become feelable
      const c = document.createElement("canvas");
      c.width = Math.round(cw * scale);
      c.height = Math.round(ch * scale);
      const g = c.getContext("2d")!;
      g.imageSmoothingQuality = "high";
      g.drawImage(img, x0 * sx, y0 * sy, cw, ch, 0, 0, c.width, c.height);
      const blob: Blob = await new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("crop failed"))), "image/png"));
      onMake(new File([blob], `${title || "diagram"} close-up.png`, { type: "image/png" }), box);
    } catch {
      setErr("Couldn't make the close-up in this browser. Try again, or photograph that area up close.");
      setBusy(false);
    }
  };

  const sw = Math.max(2, Math.max(w, h) / 300);
  return (
    <EditorDialog
      title="Make a close-up lesson"
      intro="Drag a box over a crowded area (or focus the photo and use the arrow keys; Shift+arrows resize). It becomes a linked detail lesson, enlarged so small parts are big enough to feel."
      onClose={onClose}
      initialFocus=".edlg__canvas"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" onClick={make} disabled={!box || busy} busy={busy}>
            Make close-up
          </Button>
        </>
      }
    >
      <svg
        ref={svgRef}
        className="edlg__canvas"
        viewBox={`0 0 ${w} ${h}`}
        preserveAspectRatio="xMidYMid meet"
        tabIndex={0}
        role="application"
        aria-label={
          box
            ? `Original photo. Selected area ${Math.round(box[2] - box[0])} by ${Math.round(box[3] - box[1])} pixels. Arrow keys move it, Shift+arrows resize it.`
            : "Original photo. Drag to choose an area, or press an arrow key to start a box."
        }
        onKeyDown={onKey}
        onPointerDown={(e) => {
          const p = toSrc(e);
          if (!p) return;
          setStart(p);
          setBox([p[0], p[1], p[0], p[1]]);
          (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!start) return;
          const p = toSrc(e);
          if (p) setBox(norm(start, p));
        }}
        onPointerUp={() => setStart(null)}
      >
        <image href={sourceUrl} x={0} y={0} width={w} height={h} />
        {box && (
          <>
            <path
              d={`M0 0H${w}V${h}H0Z M${box[0]} ${box[1]}V${box[3]}H${box[2]}V${box[1]}Z`}
              fill="rgba(23, 23, 28, 0.35)"
              fillRule="evenodd"
              pointerEvents="none"
            />
            <rect
              className="edlg__crop"
              x={box[0]}
              y={box[1]}
              width={box[2] - box[0]}
              height={box[3] - box[1]}
              fill="none"
              stroke="#FFD43B"
              strokeWidth={sw * 1.5}
              pointerEvents="none"
            />
          </>
        )}
      </svg>
      {err && (
        <p role="alert" className="edlg__err">
          {err}
        </p>
      )}
    </EditorDialog>
  );
}
