// Shared API types. OWNER: Lead. Mirrors CONTRACT.md §4 + CONTRACT-v2.md §2.7.

export type RGB = [number, number, number];
export type Point = [number, number];
export type Polygon = Point[];

export interface Part {
  id: string;                    // "p1", "p2", ... stable across edits
  key: string;                   // braille key letters, e.g. "n"
  name: string;                  // "nucleus"
  description: string;           // one spoken sentence
  include: boolean;
  color: RGB;                    // source colour
  texture: string;               // "none" | "dots" | "diag" | "horiz" | "vert" | "cross" | "solid" | "grid"
  count: number;                 // connected components
  area: number;                  // source px
  source_bbox: [number, number, number, number];   // SOURCE coords x0,y0,x1,y1
  source_polygons: Polygon[];    // SOURCE coords
  tactile_polygons: Polygon[];   // TACTILE coords (empty if include=false)
  key_position: Point | null;    // TACTILE coords
  label_text?: string | null;    // v2: source label this part matched
  match_method?: string;         // v2: "leader" | "inside" | "nearest" | "carved" | "ai" | "teacher" | "none"
  description_source?: "auto" | "ai" | "teacher";  // v2
}

export type InputKind = "diagram" | "chart" | "photo_object" | "photo_scene" | "text_page" | "blank";

export interface ReadinessIssue {
  code: string;
  severity: "block" | "warn" | "info";
  message: string;
  part_id?: string;
  action?: "rename" | "include" | "split" | "merge" | "retake" | "reduce" | "draw" | "shorten";
  /** UX07: the label text of a possibly-missing part (action "draw"). */
  target?: string;
  /** UX07: source-image bbox [x0,y0,x1,y1] of that label, for a detail view. */
  bbox?: [number, number, number, number];
  /** F08: source-image box around where the label's pointer ends (the structure itself). */
  target_bbox?: [number, number, number, number] | null;
}

export interface Readiness {
  status: "blocked" | "needs_review" | "ready";
  approved: boolean;
  geometry: { passed: number; total: number };
  issues: ReadinessIssue[];
  unmatched_labels: string[];
}

export interface Check {
  rule: string;
  passed: boolean;
  detail: string;
}

export interface JobResult {
  id: string;
  created_at: string;            // ISO 8601
  title: string;
  diagram_type: string;
  ai_used: boolean;
  page_flattened: boolean;
  source: { width: number; height: number; url: string };            // working image (original.png)
  tactile: {
    width: number; height: number; px_per_mm: number;
    url: string;                 // tactile.png  (raised = black, white background, page 1)
    print_url: string;           // tactile_print.png (raised black + grey print labels for teachers)
    legend_url: string;          // legend.png (page 2)
    stl_size_mm?: [number, number, number] | null;   // v2: real model footprint
  };
  parts: Part[];
  dropped: string[];
  warnings: string[];
  checks: Check[];
  timings: Record<string, number>;   // seconds: analyze, render, export, total
  files: { pdf: string; stl: string; png: string; print_png: string; original: string; legend_png: string };
  // v2 (optional so older jobs still parse)
  version?: number;
  input_kind?: InputKind;
  notes?: string[];
  retention_hours?: number;
  expires_at?: string;
  readiness?: Readiness;
  edit_token?: string;           // only present in create responses
}

export type JobStage = "queued" | "read" | "find" | "ai" | "layout" | "files" | "done";

export interface JobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  stage: JobStage;
  stages: JobStage[];
  elapsed: number;
  error: { code: string; message: string } | null;
}

export interface AsyncJobCreated {
  id: string;
  edit_token: string;
  status: "queued";
}

export interface PartEdit {
  id: string;
  name?: string;
  description?: string;
  include?: boolean;
}

export interface JobEdit {
  title?: string;
  parts?: PartEdit[];
  split?: string;                // v2: split a multi-piece part
  merge?: string[];              // v2: merge parts (first id kept)
  approved?: boolean;            // v2: teacher approval
  base_version?: number;         // v2: 409 on conflict
  add_part?: {
    name: string;
    polygon: [number, number][];
    /** cut (default): cut it out of what it covers; keep: overlap (nested detail); replace: also switch fully covered parts off */
    overlap?: "cut" | "keep" | "replace";
    /** add the area to (or with mode "remove", take it from) an existing part instead of making a new one */
    into?: string | null;
    mode?: "add" | "remove";
  };   // TS-09: teacher-drawn part (source coords)
}

export interface Sample {
  id: string;
  title: string;
  thumb_url: string;
  kind: string;
  credit?: string;   // e.g. "OpenStax Anatomy & Physiology, CC BY 4.0"
  group?: "simple" | "complex";
  objective?: string;
  /** reviewed corrections replayed when the example was prepared (0 = automatic result as is) */
  corrections?: number;
}

export interface Health {
  status: "ok";
  version: string;
  ai: { provider: string | null; model: string | null; sends?: string };
  limits?: { name: number; description: number; title: number };
  retention_hours?: number;
}

/** GET /api/jobs/{id}/vision — the image pipeline made visible ("What Feelable saw"). */
export interface VisionTrace {
  width: number;
  height: number;
  input_kind: string;
  labels: { text: string; raw: string | null; bbox: [number, number, number, number]; part_id: string | null; role: string; confidence: number }[];
  leaders: { label: string; points: [number, number][] }[];
  parts: { id: string; name: string; match_method: string; include: boolean }[];
  steps: { read?: number; repaired?: number; pointers?: number; carved?: number; ai_located?: number; shapes?: number; methods: Record<string, number> };
}
