// A photo handed from one page to another (e.g. "Make a close-up lesson" on the review page
// sends a cropped part of the diagram to the upload flow on the home page), plus the context
// needed to carry the parent lesson's names over to the close-up.
import type { JobResult, Part } from "../types";

export interface CloseupContext {
  parent: JobResult;
  box: [number, number, number, number]; // crop in the parent's source coords
}

/** UX05: the close-up keeps the parent's settings (AI off stays off; same paper). */
export function closeupOptions(parent: JobResult): { useAi: boolean; paper: "a4" | "letter" } {
  return { useAi: !!parent.ai_used, paper: paperOf(parent) };
}

/** Paper size from the tactile page's real size (A4 297 mm vs Letter 279 mm long side). */
export function paperOf(job: JobResult): "a4" | "letter" {
  const t = job.tactile;
  if (!t?.px_per_mm) return "a4";
  const longMm = Math.max(t.width, t.height) / t.px_per_mm;
  return Math.abs(longMm - 279.4) < Math.abs(longMm - 297) ? "letter" : "a4";
}

// UX05: durable "Back to full diagram" link, kept in this browser (like edit tokens).
const PARENTS_KEY = "ts.parents";
export interface ParentLink {
  id: string;
  title: string;
}
function readParents(): Record<string, ParentLink> {
  try {
    return JSON.parse(localStorage.getItem(PARENTS_KEY) || "{}") || {};
  } catch {
    return {};
  }
}
export function linkParent(childId: string, parent: ParentLink): void {
  try {
    const all = readParents();
    all[childId] = parent;
    const keys = Object.keys(all);
    for (const k of keys.slice(0, Math.max(0, keys.length - 50))) delete all[k];
    localStorage.setItem(PARENTS_KEY, JSON.stringify(all));
  } catch {
    /* private mode: the link lives only for this page */
  }
}
export function parentOf(childId: string): ParentLink | null {
  return readParents()[childId] ?? memParents.get(childId) ?? null;
}
const memParents = new Map<string, ParentLink>();

let pending: { file: File; ctx?: CloseupContext } | null = null;
let ctxForNextJob: CloseupContext | null = null;
const ctxByJob = new Map<string, CloseupContext>();

export function setPendingUpload(file: File, ctx?: CloseupContext): void {
  pending = { file, ctx };
}

export function takePendingUpload(): { file: File; ctx?: CloseupContext } | null {
  const p = pending;
  pending = null;
  ctxForNextJob = p?.ctx ?? null;
  return p;
}

/** Called when the upload finished: links the close-up context to the new job id. */
export function attachCloseup(jobId: string): void {
  if (ctxForNextJob) {
    ctxByJob.set(jobId, ctxForNextJob);
    const link = { id: ctxForNextJob.parent.id, title: ctxForNextJob.parent.title };
    memParents.set(jobId, link);
    linkParent(jobId, link);
  }
  ctxForNextJob = null;
}

export function takeCloseup(jobId: string): CloseupContext | null {
  const c = ctxByJob.get(jobId) ?? null;
  ctxByJob.delete(jobId);
  return c;
}

function inside(pt: [number, number], poly: [number, number][]): boolean {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi + 1e-9) + xi) c = !c;
  }
  return c;
}

/** Names for the close-up's parts, taken from the parent part under each part's centre. */
export function namesFromParent(child: JobResult, ctx: CloseupContext): { id: string; name: string; include: boolean }[] {
  const [x0, y0, x1, y1] = ctx.box;
  const out: { id: string; name: string; include: boolean }[] = [];
  const used = new Set<string>();
  const named = ctx.parent.parts.filter((p: Part) => p.include && !/^part \d+$/i.test(p.name));
  for (const p of child.parts) {
    const unnamed = /^(part \d+|background)$/i.test(p.name.trim()) || !p.include;
    if (!unnamed) continue;
    const [bx0, by0, bx1, by1] = p.source_bbox;
    const cx = x0 + (((bx0 + bx1) / 2) / child.source.width) * (x1 - x0);
    const cy = y0 + (((by0 + by1) / 2) / child.source.height) * (y1 - y0);
    const hits = named
      .filter((q) => q.source_polygons.some((poly) => inside([cx, cy], poly)))
      .sort((a, b) => a.area - b.area);
    const hit = hits[0];
    if (!hit || used.has(hit.name)) continue;
    used.add(hit.name);
    out.push({ id: p.id, name: hit.name, include: true });
  }
  return out;
}
