// API client. OWNER: Lead (v2 additions per CONTRACT-v2.md; Frontend may extend to match the contract).
import type { AsyncJobCreated, Health, JobEdit, JobResult, JobStatus, Sample, VisionTrace } from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string, public code: string | null = null) {
    super(message);
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    let code: string | null = null;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      if (typeof body?.code === "string") code = body.code;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, msg, code);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<Health> {
  return handle(await fetch("/api/health"));
}

export async function listSamples(): Promise<Sample[]> {
  return handle(await fetch("/api/samples"));
}

type JobOpts = { useAi?: boolean; paper?: "a4" | "letter"; signal?: AbortSignal; cancelToken?: string };

export async function createJob(file: File | Blob, opts: JobOpts = {}): Promise<JobResult> {
  const fd = new FormData();
  fd.append("file", file, (file as File).name ?? "upload.png");
  fd.append("use_ai", String(opts.useAi ?? true));
  fd.append("paper", opts.paper ?? "a4");
  if (opts.cancelToken) fd.append("cancel_token", opts.cancelToken);
  return keepToken(await handle<JobResult>(await fetch("/api/jobs", { method: "POST", body: fd, signal: opts.signal })));
}

/** Create responses carry the teacher edit token (CONTRACT-v2 §2.2): remember it for this browser. */
function keepToken(job: JobResult): JobResult {
  if (job.edit_token) saveEditToken(job.id, job.edit_token);
  return job;
}

export async function createJobFromSample(sampleId: string, opts: JobOpts = {}): Promise<JobResult> {
  const q = new URLSearchParams({ use_ai: String(opts.useAi ?? true), paper: opts.paper ?? "a4" });
  return keepToken(
    await handle<JobResult>(await fetch(`/api/jobs/from-sample/${encodeURIComponent(sampleId)}?${q}`, { method: "POST", signal: opts.signal })),
  );
}

/** Ask the server to stop a job started with this token (best effort; errors are ignored). */
export function cancelJob(cancelToken: string): void {
  fetch(`/api/jobs/cancel/${encodeURIComponent(cancelToken)}`, { method: "POST", keepalive: true }).catch(() => {});
}

export function newCancelToken(): string {
  const a = new Uint8Array(12);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** The teacher-approved snapshot students see (404 code "not_published" until approved). */
export async function getPublished(id: string): Promise<JobResult> {
  return handle(await fetch(`/api/jobs/${encodeURIComponent(id)}/published`, { cache: "no-store" }));
}

export async function getVision(id: string, signal?: AbortSignal): Promise<VisionTrace> {
  return handle(await fetch(`/api/jobs/${encodeURIComponent(id)}/vision`, { signal, cache: "no-store" }));
}

export async function getJob(id: string, signal?: AbortSignal): Promise<JobResult> {
  return handle(await fetch(`/api/jobs/${encodeURIComponent(id)}`, { signal, cache: "no-store" }));
}

// ---------------------------------------------------------------- v2: teacher edit tokens
const TOKENS_KEY = "ts.tokens";

function readTokens(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(TOKENS_KEY) || "{}") || {};
  } catch {
    return {};
  }
}

export function getEditToken(jobId: string): string | null {
  return readTokens()[jobId] ?? null;
}

export function saveEditToken(jobId: string, token: string): void {
  try {
    const t = readTokens();
    t[jobId] = token;
    localStorage.setItem(TOKENS_KEY, JSON.stringify(t));
  } catch {
    /* private mode: token lives only in memory for this page */
  }
}

function tokenHeaders(token: string | null | undefined): Record<string, string> {
  return token ? { "X-Edit-Token": token } : {};
}

export async function updateJob(id: string, edit: JobEdit, token?: string | null): Promise<JobResult> {
  return handle(
    await fetch(`/api/jobs/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", ...tokenHeaders(token ?? getEditToken(id)) },
      body: JSON.stringify(edit),
    }),
  );
}

// ---------------------------------------------------------------- v2: async jobs + progress
export async function createJobAsync(file: File | Blob, opts: JobOpts = {}): Promise<AsyncJobCreated> {
  const fd = new FormData();
  fd.append("file", file, (file as File).name ?? "upload.jpg");
  fd.append("use_ai", String(opts.useAi ?? true));
  fd.append("paper", opts.paper ?? "a4");
  fd.append("async", "1");
  const created = await handle<AsyncJobCreated>(await fetch("/api/jobs", { method: "POST", body: fd, signal: opts.signal }));
  saveEditToken(created.id, created.edit_token);
  return created;
}

export async function getJobStatus(id: string, signal?: AbortSignal): Promise<JobStatus> {
  return handle(await fetch(`/api/jobs/${encodeURIComponent(id)}/status`, { signal, cache: "no-store" }));
}

export function cancelJobById(id: string): void {
  fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
    headers: tokenHeaders(getEditToken(id)),
    keepalive: true,
  }).catch(() => {});
}

export async function aiDescribe(id: string, opts: { overwriteTeacher?: boolean } = {}): Promise<JobResult> {
  return handle(
    await fetch(`/api/jobs/${encodeURIComponent(id)}/ai-describe`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...tokenHeaders(getEditToken(id)) },
      body: JSON.stringify({ overwrite_teacher: !!opts.overwriteTeacher }),
    }),
  );
}
