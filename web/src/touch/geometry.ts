// Pure geometry for touch mode. No DOM access here, so it is easy to unit-test.
// All "tactile" coordinates are pixels of the tactile canvas (JobResult.tactile.width/height).

export type Pt = readonly [number, number];
export type Poly = ReadonlyArray<Pt>;

export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface PartShape {
  id: string;
  polys: Poly[];
  bboxes: BBox[];
  bbox: BBox;
  area: number; // sum of polygon areas, tactile px^2
}

export interface ShapeInput {
  id: string;
  include: boolean;
  tactile_polygons: ReadonlyArray<ReadonlyArray<readonly [number, number]>>;
}

// ---------------------------------------------------------------- basics

export function polygonArea(poly: Poly): number {
  let a = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    a += (poly[j][0] + poly[i][0]) * (poly[j][1] - poly[i][1]);
  }
  return Math.abs(a) / 2;
}

export function polygonBBox(poly: Poly): BBox {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of poly) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  return { x0, y0, x1, y1 };
}

export function unionBBox(boxes: BBox[]): BBox | null {
  if (!boxes.length) return null;
  const b = { ...boxes[0] };
  for (const o of boxes) {
    b.x0 = Math.min(b.x0, o.x0);
    b.y0 = Math.min(b.y0, o.y0);
    b.x1 = Math.max(b.x1, o.x1);
    b.y1 = Math.max(b.y1, o.y1);
  }
  return b;
}

/** Distance from a point to a box (0 when inside). */
export function distToBBox(x: number, y: number, b: BBox): number {
  const dx = x < b.x0 ? b.x0 - x : x > b.x1 ? x - b.x1 : 0;
  const dy = y < b.y0 ? b.y0 - y : y > b.y1 ? y - b.y1 : 0;
  return Math.hypot(dx, dy);
}

/** Ray casting (even-odd). Points exactly on an edge may go either way; callers use a tolerance. */
export function pointInPolygon(x: number, y: number, poly: Poly): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const vx = bx - ax, vy = by - ay;
  const len2 = vx * vx + vy * vy;
  let t = len2 > 0 ? ((px - ax) * vx + (py - ay) * vy) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
}

/** Distance from a point to the outline of a polygon (always >= 0, inside or outside). */
export function distToPolygonEdge(x: number, y: number, poly: Poly): number {
  let best = Infinity;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const d = distToSegment(x, y, poly[j][0], poly[j][1], poly[i][0], poly[i][1]);
    if (d < best) best = d;
  }
  return best;
}

// ---------------------------------------------------------------- parts

export function buildShapes(parts: ReadonlyArray<ShapeInput>): PartShape[] {
  const out: PartShape[] = [];
  for (const p of parts) {
    if (!p.include) continue;
    const polys = (p.tactile_polygons || []).filter((q) => q && q.length >= 3) as Poly[];
    if (!polys.length) continue;
    const bboxes = polys.map(polygonBBox);
    out.push({
      id: p.id,
      polys,
      bboxes,
      bbox: unionBBox(bboxes)!,
      area: polys.reduce((s, q) => s + polygonArea(q), 0),
    });
  }
  return out;
}

/** 0 if the point is inside any polygon of the part, otherwise distance to its nearest outline. */
export function distanceToShape(shape: PartShape, x: number, y: number): number {
  let best = Infinity;
  for (let i = 0; i < shape.polys.length; i++) {
    const bb = shape.bboxes[i];
    if (distToBBox(x, y, bb) >= best) continue;
    const poly = shape.polys[i];
    if (pointInPolygon(x, y, poly)) return 0;
    const d = distToPolygonEdge(x, y, poly);
    if (d < best) best = d;
  }
  return best;
}

/**
 * Which part is under the point? A part matches when the point is inside it or within `tolerance`
 * of its outline. Among matches the smallest-area part wins (nested parts beat their container).
 */
export function hitTest(shapes: ReadonlyArray<PartShape>, x: number, y: number, tolerance = 0): string | null {
  let bestId: string | null = null;
  let bestArea = Infinity;
  let bestDist = Infinity;
  for (const s of shapes) {
    const b = s.bbox;
    if (x < b.x0 - tolerance || x > b.x1 + tolerance || y < b.y0 - tolerance || y > b.y1 + tolerance) continue;
    const d = distanceToShape(s, x, y);
    if (d > tolerance) continue;
    if (s.area < bestArea || (s.area === bestArea && d < bestDist)) {
      bestId = s.id;
      bestArea = s.area;
      bestDist = d;
    }
  }
  return bestId;
}

/** Distance to the nearest outline of any part, or Infinity if none closer than maxDist. */
export function nearestEdgeDistance(shapes: ReadonlyArray<PartShape>, x: number, y: number, maxDist: number): number {
  let best = Infinity;
  for (const s of shapes) {
    for (let i = 0; i < s.polys.length; i++) {
      const bb = s.bboxes[i];
      if (distToBBox(x, y, bb) > Math.min(maxDist, best)) continue;
      const d = distToPolygonEdge(x, y, s.polys[i]);
      if (d < best) best = d;
    }
  }
  return best <= maxDist ? best : Infinity;
}

/** Included parts that sit inside `shape` (smaller and with a vertex inside it). Used to cut holes in highlights. */
export function childShapes(shapes: ReadonlyArray<PartShape>, shape: PartShape): PartShape[] {
  return shapes.filter((o) => {
    if (o.id === shape.id || o.area >= shape.area) return false;
    const [x, y] = o.polys[0][0];
    return shape.polys.some((p) => pointInPolygon(x, y, p));
  });
}

// ---------------------------------------------------------------- layout / mapping

/** Maps a tactile-space rectangle (the "view") letterboxed into a container (CSS px). */
export interface ViewLayout {
  srcX: number;
  srcY: number;
  srcW: number;
  srcH: number;
  dx: number; // container px of the view's top-left
  dy: number;
  dw: number;
  dh: number;
  scale: number; // container px per tactile px
}

export function fitLayout(containerW: number, containerH: number, view: BBox, pad = 0): ViewLayout {
  const srcW = Math.max(1, view.x1 - view.x0);
  const srcH = Math.max(1, view.y1 - view.y0);
  const aw = Math.max(1, containerW - 2 * pad);
  const ah = Math.max(1, containerH - 2 * pad);
  const scale = Math.min(aw / srcW, ah / srcH);
  const dw = srcW * scale;
  const dh = srcH * scale;
  return {
    srcX: view.x0,
    srcY: view.y0,
    srcW,
    srcH,
    dx: (containerW - dw) / 2,
    dy: (containerH - dh) / 2,
    dw,
    dh,
    scale,
  };
}

export function clientToTactile(l: ViewLayout, px: number, py: number): [number, number] {
  return [l.srcX + (px - l.dx) / l.scale, l.srcY + (py - l.dy) / l.scale];
}

export function tactileToClient(l: ViewLayout, tx: number, ty: number): [number, number] {
  return [l.dx + (tx - l.srcX) * l.scale, l.dy + (ty - l.srcY) * l.scale];
}

/** True when a container point lands on the drawn page area. */
export function isOnView(l: ViewLayout, px: number, py: number): boolean {
  return px >= l.dx && px <= l.dx + l.dw && py >= l.dy && py <= l.dy + l.dh;
}

/**
 * The part of the page worth showing: union of the parts (and their braille keys) plus a margin,
 * clamped to the page. Falls back to the whole page. Keeps the drawing big on small phones.
 */
export function focusView(
  shapes: ReadonlyArray<PartShape>,
  keyPoints: ReadonlyArray<Pt>,
  pageW: number,
  pageH: number,
  marginPx: number,
  keyExtentPx: number,
): BBox {
  const boxes = shapes.map((s) => s.bbox);
  for (const [x, y] of keyPoints) boxes.push({ x0: x - keyExtentPx, y0: y - keyExtentPx, x1: x + keyExtentPx, y1: y + keyExtentPx });
  const u = unionBBox(boxes);
  if (!u) return { x0: 0, y0: 0, x1: pageW, y1: pageH };
  return {
    x0: Math.max(0, u.x0 - marginPx),
    y0: Math.max(0, u.y0 - marginPx),
    x1: Math.min(pageW, u.x1 + marginPx),
    y1: Math.min(pageH, u.y1 + marginPx),
  };
}
