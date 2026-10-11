// Readiness (CONTRACT-v2 §2.7). Uses job.readiness from the API; older jobs without it get a
// conservative fallback computed here so the panel never claims "ready" without teacher approval.
import type { JobResult, Readiness, ReadinessIssue } from "../types";

export function readinessOf(job: JobResult): Readiness {
  if (job.readiness) return job.readiness;
  const inc = job.parts.filter((p) => p.include);
  const issues: ReadinessIssue[] = [];
  if (inc.length === 0) issues.push({ code: "no_parts", severity: "block", message: "No parts are switched on, so there is nothing to feel." });
  for (const p of inc) {
    if (/^part \d+$/i.test(p.name.trim())) {
      issues.push({ code: "unnamed", severity: "warn", action: "rename", part_id: p.id, message: `“${p.name}” has no name yet. Give it the name a student should hear.` });
    }
  }
  if (inc.length > 8) {
    issues.push({ code: "too_many", severity: "warn", action: "reduce", message: `${inc.length} parts is a lot to read by touch. Switch off the least important ones (8 or fewer is best).` });
  }
  for (const c of job.checks) {
    if (!c.passed && !/^8 parts/i.test(c.rule)) issues.push({ code: "geometry", severity: "warn", message: `${c.rule}: ${c.detail}` });
  }
  for (const w of job.warnings) issues.push({ code: "warning", severity: "warn", message: w });
  const blocked = issues.some((i) => i.severity === "block");
  return {
    status: blocked ? "blocked" : "needs_review",
    approved: false,
    geometry: { passed: job.checks.filter((c) => c.passed).length, total: job.checks.length },
    issues,
    unmatched_labels: [],
  };
}

export const STATUS_COPY: Record<Readiness["status"], { label: string; short: string }> = {
  blocked: { label: "Can't be used yet", short: "Blocked" },
  needs_review: { label: "Check before teaching", short: "Needs review" },
  ready: { label: "Ready to teach", short: "Ready" },
};

/** "Matched to label 'nucleus' by its pointer line" — F18. */
export function whyText(method: string | undefined, label: string | null | undefined): string {
  const l = label ? `“${label}”` : "";
  switch (method) {
    case "leader":
      return l ? `Matched to the label ${l} by its pointer line.` : "Matched to a label by its pointer line.";
    case "inside":
      return l ? `The label ${l} is written inside this shape.` : "Its label is written inside this shape.";
    case "nearest":
      return l ? `Matched to the nearest label, ${l}. Worth a check.` : "Matched to the nearest label. Worth a check.";
    case "carved":
      return l ? `Its pointer line for ${l} ends here: Feelable zoomed in and traced the shape around it.` : "Traced around where a pointer line ends.";
    case "ai":
      return l ? `The image AI found ${l} in the picture and Feelable traced it. Check the outline.` : "Named by AI from the picture. Check the name is right.";
    case "legend":
      return l ? `Named from the chart legend entry ${l}, whose marker has the same colour. Check it.` : "Named from the chart legend by colour. Check it.";
    case "landmark":
      return l
        ? `The pointer for ${l} ends on a spot rather than a whole shape, so Feelable marked that spot. Check it is in the right place.`
        : "A marked spot where a pointer ends. Check it is in the right place.";
    case "teacher":
      return "Named by you.";
    default:
      return "No label pointed to this shape, so it needs a name.";
  }
}

export function isUncertain(method: string | undefined) {
  return method === "nearest" || method === "ai" || method === "landmark" || method === "legend" || method === "none" || !method;
}
