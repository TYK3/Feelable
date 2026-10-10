// Processing screen driven by the REAL job stages from GET /api/jobs/{id}/status (F13).
import { useEffect, useRef, useState } from "react";
import type { JobStage } from "../types";
import { Button } from "./Button";
import { IconCheck, IconImage, IconRefresh, IconWarn, IconX } from "./Icons";

export const STAGE_LABEL: Record<string, string> = {
  queued: "Waiting for the server",
  upload: "Sending your photo",
  read: "Reading labels & colours",
  find: "Finding the parts",
  ai: "AI is describing each part",
  layout: "Applying tactile rules",
  files: "Braille, PDF & 3D",
  sample: "Opening the example",
};

export type Recovery = "choose" | "retry" | "home";
export type ProcError = { code: string; message: string; recovery: Recovery };

export function ProcessingView({
  imageUrl,
  label,
  stages,
  current,
  done,
  error,
  serverElapsed,
  onCancel,
  onRetry,
  onChoose,
  onHome,
}: {
  imageUrl: string | null;
  label: string;
  stages: string[]; // ordered stage ids incl. "upload"/"queued"
  current: string; // current stage id
  done: boolean;
  error: ProcError | null;
  serverElapsed: number | null;
  onCancel: () => void;
  onRetry: () => void;
  onChoose: () => void;
  onHome: () => void;
}) {
  const [elapsed, setElapsed] = useState(0);
  const t0 = useRef(performance.now());
  useEffect(() => {
    if (done || error) return;
    const h = window.setInterval(() => setElapsed(performance.now() - t0.current), 100);
    return () => window.clearInterval(h);
  }, [done, error]);
  const secs = Math.max(elapsed / 1000, serverElapsed ?? 0);

  const idx = done ? stages.length : Math.max(0, stages.indexOf(current));
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (error) headingRef.current?.focus();
  }, [error]);

  return (
    <section className={"proc" + (error ? " is-error" : "") + (done ? " is-done" : "")} aria-labelledby="proc-title">
      <div className="proc__preview">
        {imageUrl ? <img src={imageUrl} alt={`Your diagram: ${label}`} /> : <div className="proc__blank"><IconImage size={40} /></div>}
        {!error && !done && <span className="proc__scan" aria-hidden="true" />}
        <span className="proc__dots" aria-hidden="true" />
      </div>

      <div className="proc__side">
        {error ? (
          <>
            <p className="proc__kicker proc__kicker--err">
              <IconWarn size={18} /> Not made
            </p>
            <h1 id="proc-title" className="proc__title" ref={headingRef} tabIndex={-1}>
              {error.recovery === "choose" ? "This photo didn't work." : "We couldn't finish this one."}
            </h1>
            <p className="proc__msg" role="alert">{error.message}</p>
            <div className="proc__actions">
              {error.recovery === "choose" && (
                <Button variant="primary" size="lg" icon={<IconImage size={20} />} onClick={onChoose}>
                  Choose another photo
                </Button>
              )}
              {error.recovery === "retry" && (
                <Button variant="primary" size="lg" icon={<IconRefresh size={20} />} onClick={onRetry}>
                  Retry
                </Button>
              )}
              <Button variant="secondary" size="lg" onClick={onHome}>
                Back to start
              </Button>
            </div>
          </>
        ) : (
          <>
            <p className="proc__kicker">
              <span className="tabular" aria-hidden="true">{secs.toFixed(1)} s</span>
              <span className="sr-only">{Math.round(secs)} seconds</span>
            </p>
            <h1 id="proc-title" className="proc__title">
              {done ? "Ready. Opening your lesson…" : "Making it feelable…"}
            </h1>
            <ol className="stages" aria-live="polite">
              {stages.map((s, i) => {
                const state = i < idx ? "done" : i === idx ? "active" : "todo";
                return (
                  <li key={s} className={`stage stage--${state}`} aria-current={state === "active" ? "step" : undefined}>
                    <span className="stage__mark" aria-hidden="true">
                      {state === "done" ? <IconCheck size={15} strokeWidth={3} /> : state === "active" ? <span className="stage__pulse" /> : null}
                    </span>
                    <span className="stage__title">{STAGE_LABEL[s] ?? s}</span>
                    <span className="sr-only">{state === "done" ? ", done" : state === "active" ? ", in progress" : ""}</span>
                  </li>
                );
              })}
            </ol>
            {!done && (
              <div className="proc__actions">
                <Button variant="secondary" icon={<IconX size={18} />} onClick={onCancel}>
                  Cancel
                </Button>
              </div>
            )}
            {!done && secs > 12 && (
              <p className="proc__note" role="status">
                Still working. On the free demo server a new photo can take up to a minute. The examples open instantly.
              </p>
            )}
          </>
        )}
      </div>
    </section>
  );
}

export type { JobStage };
