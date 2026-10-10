import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from "react";
import {
  type BBox,
  type PartShape,
  type ViewLayout,
  childShapes,
  clientToTactile,
  distanceToShape,
  fitLayout,
  hitTest,
  isOnView,
  nearestEdgeDistance,
  tactileToClient,
} from "./geometry";

export type Where = "part" | "page" | "off";

export interface ExploreCanvasProps {
  imageUrl: string;
  shapes: PartShape[];
  view: BBox;
  pxPerMm: number;
  highlightId: string | null;
  showHighlights: boolean;
  findTargetId: string | null;
  label: string;
  /** Finger entered a part (should be announced). */
  onEnter(id: string): void;
  /** Finger left all parts (to blank page or off the page). */
  onLeave(where: Exclude<Where, "part">): void;
  onEdge(): void;
  onDescribe(id: string | null): void;
  /** Two-finger tap. `since` = performance.now() when the gesture began (ignore speech it caused). */
  onRepeat(since: number): void;
  onThreeFingerTap(): void;
  onFindDistance(mm: number | null): void;
  onFound(): void;
  onKeyDown?(e: React.KeyboardEvent<HTMLDivElement>): void;
}

export interface ExploreCanvasHandle {
  focus(): void;
}

interface Ptr {
  x: number;
  y: number;
  sx: number;
  sy: number;
  t: number;
  type: string;
}

const TAP_MAX_MS = 300;
const TAP_SLOP = 30;
const DOUBLE_TAP_MS = 350;
const LONG_PRESS_MS = 600;
const LONG_PRESS_SLOP = 12;
const SAME_PART_GRACE_MS = 1500;
const MULTI_TAP_MS = 500;

const SURROUND = "#0E0E12";
const HILITE_FILL = "rgba(255, 210, 63, 0.45)";
const HILITE_LINE = "#17171C";

const ExploreCanvas = forwardRef<ExploreCanvasHandle, ExploreCanvasProps>(function ExploreCanvas(props, ref) {
  const { imageUrl, shapes, view, pxPerMm, highlightId, showHighlights } = props;
  const p = useRef(props);
  p.current = props;

  const wrapRef = useRef<HTMLDivElement>(null);
  const baseRef = useRef<HTMLCanvasElement>(null);
  const overRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0, dpr: 1 });
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [imgFailed, setImgFailed] = useState(false);
  const layoutRef = useRef<ViewLayout | null>(null);

  // interaction state (refs: updated at pointer rate, no re-render)
  const pointers = useRef(new Map<number, Ptr>());
  const gesture = useRef({ max: 0, start: 0, moved: false });
  const cur = useRef<{ id: string | null; where: Where | null }>({ id: null, where: null });
  const onEdgeBand = useRef(false);
  const lastLift = useRef(0);
  const lastTap = useRef<{ t: number; x: number; y: number } | null>(null);
  const lp = useRef<{ timer: number | null; ax: number; ay: number; firedFor: string | null }>({
    timer: null,
    ax: 0,
    ay: 0,
    firedFor: null,
  });
  const finger = useRef<[number, number] | null>(null);
  const raf = useRef<number | null>(null);

  useImperativeHandle(ref, () => ({ focus: () => wrapRef.current?.focus() }), []);

  // ------------------------------------------------------------ image
  useEffect(() => {
    let alive = true;
    setImg(null);
    setImgFailed(false);
    const im = new Image();
    im.decoding = "async";
    im.onload = () => alive && setImg(im);
    im.onerror = () => alive && setImgFailed(true);
    im.src = imageUrl;
    return () => {
      alive = false;
    };
  }, [imageUrl]);

  // ------------------------------------------------------------ size (DPR aware, follows rotation)
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const dpr = Math.min(3, window.devicePixelRatio || 1);
      setSize((s) => (s.w === r.width && s.h === r.height && s.dpr === dpr ? s : { w: r.width, h: r.height, dpr }));
    };
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    window.addEventListener("resize", measure);
    window.addEventListener("orientationchange", measure);
    return () => {
      ro?.disconnect();
      window.removeEventListener("resize", measure);
      window.removeEventListener("orientationchange", measure);
    };
  }, []);

  const layout = size.w > 0 && size.h > 0 ? fitLayout(size.w, size.h, view, Math.min(16, size.w * 0.03)) : null;
  layoutRef.current = layout;

  // Test hook: tactile -> page (viewport) coordinates. Harmless in production.
  useEffect(() => {
    const w = window as unknown as { __feelableMap?: (x: number, y: number) => [number, number] | null };
    w.__feelableMap = (x, y) => {
      const l = layoutRef.current;
      const el = wrapRef.current;
      if (!l || !el) return null;
      const r = el.getBoundingClientRect();
      const [cx, cy] = tactileToClient(l, x, y);
      return [r.left + cx, r.top + cy];
    };
    return () => {
      delete w.__feelableMap;
    };
  }, []);

  // ------------------------------------------------------------ base drawing
  useEffect(() => {
    const c = baseRef.current;
    if (!c || !layout) return;
    const { w, h, dpr } = size;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    const g = c.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = SURROUND;
    g.fillRect(0, 0, w, h);
    g.fillStyle = "#FFFFFF";
    g.fillRect(layout.dx, layout.dy, layout.dw, layout.dh);
    if (img) {
      g.imageSmoothingEnabled = true;
      g.imageSmoothingQuality = "high";
      g.drawImage(img, layout.srcX, layout.srcY, layout.srcW, layout.srcH, layout.dx, layout.dy, layout.dw, layout.dh);
      // The view may be cropped to the drawing: blank a thin frame on cropped edges so braille
      // cells cut by the crop don't show as stray dots. The view keeps >= 8 mm around every part.
      const fr = 5 * pxPerMm * layout.scale;
      g.fillStyle = "#FFFFFF";
      if (view.x0 > 0) g.fillRect(layout.dx, layout.dy, fr, layout.dh);
      if (view.y0 > 0) g.fillRect(layout.dx, layout.dy, layout.dw, fr);
      if (view.x1 < img.naturalWidth) g.fillRect(layout.dx + layout.dw - fr, layout.dy, fr, layout.dh);
      if (view.y1 < img.naturalHeight) g.fillRect(layout.dx, layout.dy + layout.dh - fr, layout.dw, fr);
    } else if (imgFailed) {
      // Fallback: draw the part outlines so the page is still explorable.
      g.strokeStyle = "#000";
      g.lineWidth = Math.max(2, 1.5 * pxPerMm * layout.scale);
      for (const s of shapes) for (const poly of s.polys) tracePoly(g, layout, poly), g.stroke();
    }
    drawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, img, imgFailed, view, shapes, layout?.scale]);

  // ------------------------------------------------------------ overlay
  const drawOverlay = useCallback(() => {
    const c = overRef.current;
    const l = layoutRef.current;
    if (!c || !l) return;
    const { w, h, dpr } = sizeRef.current;
    if (c.width !== Math.round(w * dpr)) c.width = Math.round(w * dpr);
    if (c.height !== Math.round(h * dpr)) c.height = Math.round(h * dpr);
    const g = c.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const { highlightId: hid, showHighlights: show, shapes: shp } = p.current;
    if (!show) return;
    const s = hid ? shp.find((x) => x.id === hid) : null;
    if (s) {
      g.save();
      g.fillStyle = HILITE_FILL;
      g.beginPath();
      for (const poly of s.polys) addPoly(g, l, poly);
      g.fill("evenodd");
      // Cut nested parts out of a container's highlight so it reads as "this region".
      const kids = childShapes(shp, s);
      if (kids.length) {
        g.globalCompositeOperation = "destination-out";
        g.fillStyle = "#000";
        g.beginPath();
        for (const k of kids) for (const poly of k.polys) addPoly(g, l, poly);
        g.fill("nonzero");
        g.globalCompositeOperation = "source-over";
      }
      // Thick banana halo + dark core line: visible on white paper and on solid black parts.
      g.lineJoin = "round";
      g.beginPath();
      for (const poly of s.polys) addPoly(g, l, poly);
      g.strokeStyle = HILITE_LINE;
      g.lineWidth = 10;
      g.stroke();
      g.strokeStyle = "#FFD23F";
      g.lineWidth = 7;
      g.stroke();
      g.strokeStyle = HILITE_LINE;
      g.lineWidth = 2;
      g.stroke();
      g.restore();
    }
    const f = finger.current;
    if (f) {
      g.beginPath();
      g.arc(f[0], f[1], 22, 0, Math.PI * 2);
      g.lineWidth = 5;
      g.strokeStyle = "rgba(14,14,18,0.85)";
      g.stroke();
      g.lineWidth = 2.5;
      g.strokeStyle = "#FFD23F";
      g.stroke();
    }
  }, []);
  const sizeRef = useRef(size);
  sizeRef.current = size;

  const requestOverlay = useCallback(() => {
    if (raf.current != null) return;
    raf.current = requestAnimationFrame(() => {
      raf.current = null;
      drawOverlay();
    });
  }, [drawOverlay]);

  useEffect(() => {
    drawOverlay();
  }, [highlightId, showHighlights, drawOverlay, size]);

  useEffect(
    () => () => {
      if (raf.current != null) cancelAnimationFrame(raf.current);
      if (lp.current.timer != null) clearTimeout(lp.current.timer);
    },
    [],
  );

  // ------------------------------------------------------------ exploring
  const tolerances = () => {
    const l = layoutRef.current!;
    const ppm = p.current.pxPerMm || 5;
    return {
      hit: Math.max(2 * ppm, 6 / l.scale), // ~2 mm, and never less than 6 screen px
      edge: Math.max(1.5 * ppm, 5 / l.scale),
    };
  };

  const explore = (px: number, py: number, newTouch: boolean) => {
    const l = layoutRef.current;
    if (!l) return;
    const P = p.current;
    const now = performance.now();
    const { hit, edge } = tolerances();
    const [tx, ty] = clientToTactile(l, px, py);
    let id: string | null = null;
    let where: Where = "off";
    if (isOnView(l, px, py)) {
      id = hitTest(P.shapes, tx, ty, hit);
      where = id ? "part" : "page";
    }

    // Edge band: vibrate when the finger moves onto an outline.
    const nearEdge = where !== "off" && Number.isFinite(nearestEdgeDistance(P.shapes, tx, ty, edge));
    if (nearEdge && !onEdgeBand.current) P.onEdge();
    onEdgeBand.current = nearEdge;

    const prev = cur.current;
    if (id) {
      const reentry = newTouch && id === prev.id && now - lastLift.current > SAME_PART_GRACE_MS;
      if (id !== prev.id || reentry) {
        lp.current.firedFor = null;
        P.onEnter(id);
      }
    } else if (where !== prev.where) {
      lp.current.firedFor = null;
      P.onLeave(where as Exclude<Where, "part">);
    }
    cur.current = { id, where };

    // Find game
    if (P.findTargetId) {
      const target = P.shapes.find((s) => s.id === P.findTargetId);
      if (target) {
        const d = distanceToShape(target, tx, ty);
        if (d <= hit) {
          P.onFindDistance(0);
          P.onFound();
        } else {
          P.onFindDistance((d - hit) / (P.pxPerMm || 5));
        }
      }
    }
  };

  const local = (e: React.PointerEvent): [number, number] => {
    const r = wrapRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  const clearLongPress = () => {
    if (lp.current.timer != null) clearTimeout(lp.current.timer);
    lp.current.timer = null;
  };
  const armLongPress = (x: number, y: number) => {
    clearLongPress();
    lp.current.ax = x;
    lp.current.ay = y;
    lp.current.timer = window.setTimeout(() => {
      lp.current.timer = null;
      const id = cur.current.id;
      if (pointers.current.size === 1 && id && lp.current.firedFor !== id) {
        lp.current.firedFor = id;
        p.current.onDescribe(id);
      }
    }, LONG_PRESS_MS);
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    const [x, y] = local(e);
    const now = performance.now();
    if (pointers.current.size === 0) gesture.current = { max: 0, start: now, moved: false };
    pointers.current.set(e.pointerId, { x, y, sx: x, sy: y, t: now, type: e.pointerType });
    gesture.current.max = Math.max(gesture.current.max, pointers.current.size);
    try {
      wrapRef.current?.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    if (pointers.current.size > 1) {
      clearLongPress();
      return;
    }
    finger.current = [x, y];
    explore(x, y, e.pointerType !== "mouse");
    armLongPress(x, y);
    requestOverlay();
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const [x, y] = local(e);
    const ptr = pointers.current.get(e.pointerId);
    if (ptr) {
      ptr.x = x;
      ptr.y = y;
      if (Math.hypot(x - ptr.sx, y - ptr.sy) > 40) gesture.current.moved = true;
    }
    if (pointers.current.size > 1) return;
    const hover = !ptr && e.pointerType === "mouse";
    if (!ptr && !hover) return;
    if (ptr && Math.hypot(x - lp.current.ax, y - lp.current.ay) > LONG_PRESS_SLOP) armLongPress(x, y);
    finger.current = [x, y];
    explore(x, y, false);
    requestOverlay();
  };

  const endPointer = (e: React.PointerEvent<HTMLDivElement>, cancelled: boolean) => {
    const ptr = pointers.current.get(e.pointerId);
    if (!ptr) return;
    pointers.current.delete(e.pointerId);
    const now = performance.now();
    const [x, y] = local(e);
    clearLongPress();
    if (pointers.current.size > 0) return;

    const g = gesture.current;
    const P = p.current;
    if (!cancelled && g.max >= 2) {
      if (!g.moved && now - g.start < MULTI_TAP_MS) {
        if (g.max >= 3) P.onThreeFingerTap();
        else P.onRepeat(g.start);
      }
      lastTap.current = null;
    } else if (!cancelled && now - ptr.t < TAP_MAX_MS && Math.hypot(x - ptr.sx, y - ptr.sy) < TAP_SLOP) {
      const lt = lastTap.current;
      if (lt && ptr.t - lt.t < DOUBLE_TAP_MS && Math.hypot(x - lt.x, y - lt.y) < TAP_SLOP) {
        lastTap.current = null;
        const l = layoutRef.current;
        let id: string | null = null;
        if (l && isOnView(l, x, y)) {
          const [tx, ty] = clientToTactile(l, x, y);
          id = hitTest(P.shapes, tx, ty, tolerances().hit);
        }
        lp.current.firedFor = id;
        P.onDescribe(id);
      } else {
        lastTap.current = { t: now, x, y };
      }
    }

    if (ptr.type !== "mouse") {
      lastLift.current = now;
      finger.current = null;
      onEdgeBand.current = false;
      if (P.findTargetId) P.onFindDistance(null);
      requestOverlay();
    }
  };

  const onPointerLeave = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerType !== "mouse" || pointers.current.size) return;
    finger.current = null;
    onEdgeBand.current = false;
    if (cur.current.where !== "off" && cur.current.where !== null) {
      cur.current = { id: null, where: "off" };
      p.current.onLeave("off");
    }
    if (p.current.findTargetId) p.current.onFindDistance(null);
    requestOverlay();
  };

  return (
    <div
      ref={wrapRef}
      className="ts-t-explore"
      tabIndex={0}
      role="application"
      aria-roledescription="touch diagram"
      aria-label={props.label}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={(e) => endPointer(e, false)}
      onPointerCancel={(e) => endPointer(e, true)}
      onPointerLeave={onPointerLeave}
      onContextMenu={(e) => e.preventDefault()}
      onKeyDown={props.onKeyDown}
    >
      <canvas ref={baseRef} className="ts-t-canvas" aria-hidden="true" />
      <canvas ref={overRef} className="ts-t-canvas" aria-hidden="true" />
    </div>
  );
});

function addPoly(g: CanvasRenderingContext2D, l: ViewLayout, poly: ReadonlyArray<readonly [number, number]>) {
  poly.forEach(([x, y], i) => {
    const [cx, cy] = tactileToClient(l, x, y);
    if (i === 0) g.moveTo(cx, cy);
    else g.lineTo(cx, cy);
  });
  g.closePath();
}

function tracePoly(g: CanvasRenderingContext2D, l: ViewLayout, poly: ReadonlyArray<readonly [number, number]>) {
  g.beginPath();
  addPoly(g, l, poly);
}

export default ExploreCanvas;
