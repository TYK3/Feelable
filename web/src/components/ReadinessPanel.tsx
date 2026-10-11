// Readiness (F02/F07/F11): meaning review + teacher approval, kept apart from the tactile layout checks.
import { useState } from "react";
import type { JobResult, Readiness, ReadinessIssue } from "../types";
import { STATUS_COPY } from "../lib/readiness";
import { Button } from "./Button";
import { IconArrowRight, IconCheck, IconChevronDown, IconWarn, IconZoom } from "./Icons";

const ACTION_LABEL: Record<string, string> = {
  rename: "Name it",
  include: "Show part",
  split: "Split it",
  merge: "Merge parts",
  retake: "New photo",
  reduce: "Choose parts",
  draw: "Add it",
  shorten: "Edit title",
  retitle: "Edit title",
  outline: "Suggest outline",
  retry: "Try again",
};

/** UX07: say what actually needs looking at, e.g. "Review 2 names and the missing pancreas." */
export function reviewLead(actionable: ReadinessIssue[]): string {
  const missing = actionable.filter((i) => i.action === "draw" && i.target).map((i) => i.target as string);
  const names = actionable.filter((i) => i.action === "rename").length;
  const other = actionable.length - missing.length - names;
  const bits: string[] = [];
  if (names) bits.push(`${names} ${names === 1 ? "name" : "names"}`);
  if (missing.length === 1) bits.push(`the possibly missing ${missing[0]}`);
  else if (missing.length > 1) bits.push(`${missing.length} possibly missing parts`);
  if (other) bits.push(`${other} layout ${other === 1 ? "note" : "notes"}`);
  if (!bits.length) return "Look over the names, then approve and publish.";
  const list = bits.length === 1 ? bits[0] : `${bits.slice(0, -1).join(", ")} and ${bits[bits.length - 1]}`;
  return `Review ${list}, then approve and publish.`;
}

export function ReadinessPanel({
  job,
  readiness,
  canEdit,
  dirty,
  approving,
  compact,
  onAction,
  onDetail,
  onApprove,
  onExpand,
}: {
  job: JobResult;
  readiness: Readiness;
  canEdit: boolean;
  dirty: boolean;
  approving: boolean;
  compact?: boolean;
  onAction: (i: ReadinessIssue) => void;
  onDetail?: (i: ReadinessIssue) => void;
  onApprove: () => void;
  onExpand?: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const [showChecks, setShowChecks] = useState(false);
  const { status, issues, geometry } = readiness;
  const actionable = issues.filter((i) => i.severity !== "info");
  const info = issues.filter((i) => i.severity === "info");
  // F04: "teacher approved" and "all checks passed" are different things; say which applies
  const openChecks = actionable.filter((i) => i.severity === "warn").length;
  const approvedOpen = readiness.approved && status === "needs_review";
  const copy = approvedOpen
    ? { label: `Approved — ${openChecks} ${openChecks === 1 ? "check" : "checks"} still open`, short: "Approved, open checks" }
    : STATUS_COPY[status];

  const lead =
    status === "blocked"
      ? "Fix the problems below before you print or share."
      : status === "ready"
        ? "You approved this version and nothing is flagged. Print it or share it with your student."
        : approvedOpen
          ? "You approved this version, so students can use it, but these checks are still open. Fix them and approve again, or keep them as known limitations."
        : actionable.length
          ? reviewLead(actionable)
          : "Draft — nothing flagged. Feelable can't judge meaning, so look over the names before you publish.";

  if (compact) {
    return (
      <div className={`ready ready--compact is-${status}`}>
        <StatusIcon status={status} />
        <span className="ready__compact-text">
          <strong>{copy.label}</strong>
          {status !== "ready" && actionable.length > 0 && <span> · {actionable.length} to check</span>}
        </span>
        {onExpand && (
          <button type="button" className="ready__more" onClick={onExpand}>
            Details
          </button>
        )}
      </div>
    );
  }

  const narrow = typeof window !== "undefined" && window.matchMedia?.("(max-width: 999px)").matches;
  const limit = status === "ready" ? 0 : narrow ? 2 : 3;
  const shown = showAll ? actionable : actionable.slice(0, limit);
  return (
    <section className={`ready is-${status}`} aria-labelledby="ready-title">
      <div className="ready__head">
        <StatusIcon status={status} />
        <div>
          <h2 id="ready-title" className="ready__title">{copy.label}</h2>
          <p className="ready__lead">{dirty ? "You have unsaved changes. Save them to re-check." : lead}</p>
        </div>
      </div>

      {actionable.length > 0 && (
        <ul className="issues">
          {shown.map((i, n) => (
            <li key={n} className={`issue issue--${i.severity}`}>
              <span className="issue__dot" aria-hidden="true" />
              <span className="issue__msg">
                <span className="sr-only">{i.severity === "block" ? "Problem: " : "Check: "}</span>
                {i.message}
                {i.detail && (
                  <details className="issue__detail">
                    <summary>Diagnostics</summary>
                    <code>{i.detail}</code>
                  </details>
                )}
              </span>
              {i.action && (canEdit || i.action === "retake") && (
                <button type="button" className="issue__act" onClick={() => onAction(i)}>
                  {ACTION_LABEL[i.action] ?? "Fix"}
                  <IconArrowRight size={16} />
                </button>
              )}
              {i.action === "draw" && (i.target_bbox || i.bbox) && canEdit && onDetail && (
                <button type="button" className="issue__act issue__act--alt" onClick={() => onDetail(i)}>
                  <IconZoom size={16} />
                  Make a close-up
                </button>
              )}
            </li>
          ))}
          {actionable.length > limit && (
            <li>
              <button type="button" className="issues__more" aria-expanded={showAll} onClick={() => setShowAll((v) => !v)}>
                {showAll ? "Show fewer" : limit === 0 ? `Show ${actionable.length} ${actionable.length === 1 ? "note" : "notes"}` : `Show ${actionable.length - limit} more`}
              </button>
            </li>
          )}
        </ul>
      )}
      {info.map((i, n) => (
        <p key={n} className="ready__info">{i.message}</p>
      ))}

      {canEdit && status !== "blocked" && (
        <div className="ready__approve">
          {readiness.approved ? (
            <p className="ready__approved">
              <IconCheck size={18} strokeWidth={3} /> Approved for teaching (version {job.version ?? 1})
              {openChecks > 0 && <span> · {openChecks} {openChecks === 1 ? "check" : "checks"} still open</span>}
            </p>
          ) : (
            <Button variant="ink" size="lg" className="btn--block" onClick={onApprove} busy={approving} disabled={dirty || approving}>
              Approve and publish
            </Button>
          )}
          {dirty && !readiness.approved && <p className="ready__hint">Save your changes first.</p>}
        </div>
      )}

      <div className="geo">
        <button type="button" className="geo__toggle" aria-expanded={showChecks} onClick={() => setShowChecks((v) => !v)}>
          <span>
            Print layout checks{" "}
            {geometry.total === 0 ? (
              <strong className="geo__n is-warn">No diagram to check</strong>
            ) : (
              <strong className={geometry.passed === geometry.total ? "geo__n is-ok" : "geo__n is-warn"}>
                {geometry.passed}/{geometry.total}
              </strong>
            )}
          </span>
          <IconChevronDown size={18} className={showChecks ? "geo__chev is-open" : "geo__chev"} />
        </button>
        {showChecks && (
          <div className="geo__body">
            <p className="geo__note">
              Line widths, spacing and textures on the print. These are not a source-match review: they don't check that a
              name sits on the right structure or that every labelled feature was found.
            </p>
            <ul className="checks">
              {job.checks.map((c, i) => (
                <li key={i} className={c.passed ? "check is-pass" : "check is-fail"}>
                  <span className="check__icon" aria-hidden="true">
                    {c.passed ? <IconCheck size={14} strokeWidth={3} /> : <IconWarn size={14} strokeWidth={2.5} />}
                  </span>
                  <span>
                    <span className="sr-only">{c.passed ? "Passed: " : "Needs attention: "}</span>
                    {c.rule}
                    {c.detail && <span className="check__detail"> — {c.detail}</span>}
                  </span>
                </li>
              ))}
            </ul>
            {job.dropped.length > 0 && <p className="geo__note">Removed as clutter: {job.dropped.join("; ")}.</p>}
          </div>
        )}
      </div>
    </section>
  );
}

export function StatusIcon({ status }: { status: Readiness["status"] }) {
  return (
    <span className={`status-ico is-${status}`} aria-hidden="true">
      {status === "ready" ? <IconCheck size={18} strokeWidth={3} /> : status === "blocked" ? <IconWarn size={18} strokeWidth={2.5} /> : <span className="status-ico__dot">!</span>}
    </span>
  );
}
