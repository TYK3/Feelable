// Hand-off cache so a freshly created job renders instantly on #/job/{id}
// (the review page still re-fetches with getJob on refresh / shared links).
import type { JobResult } from "../types";

const cache = new Map<string, JobResult>();

export function rememberJob(job: JobResult) {
  cache.set(job.id, job);
}

export function takeCachedJob(id: string): JobResult | undefined {
  return cache.get(id);
}

/** Append a cache-busting version so images refresh after edits even if the URL is unchanged. */
export function bust(url: string, version: number) {
  if (!url) return url;
  return url + (url.includes("?") ? "&" : "?") + "v=" + version;
}
