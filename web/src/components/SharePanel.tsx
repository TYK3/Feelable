// Share step (F04/F08/F09/F14/F16): downloads, the student link + QR, the private teacher link.
import { useEffect, useState } from "react";
import QRCode from "qrcode";
import type { JobResult, Readiness } from "../types";
import { absoluteUrl, navigate } from "../router";
import { Button, LinkButton } from "./Button";
import { stlSize } from "./DiagramViewer";
import { IconClock, IconCopy, IconCube, IconDownload, IconHand, IconImage, IconLock, IconShare } from "./Icons";
import { bust } from "./jobStore";
import { toast } from "./Toast";
import { sentenceCase } from "./text";

export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through */
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  ta.remove();
  return ok;
}

function expiresText(job: JobResult, retention: number) {
  if (job.expires_at) {
    const d = new Date(job.expires_at);
    if (!isNaN(d.getTime())) {
      const hrs = Math.max(0, Math.round((d.getTime() - Date.now()) / 36e5));
      const when = d.toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" });
      return `Links work until ${when} (about ${hrs} h) on the free demo server. Download the PDF to keep it.`;
    }
  }
  return `Links last ${retention} h on the free demo server — download the PDF to keep it.`;
}

export function SharePanel({
  job,
  version,
  readiness,
  token,
  retentionHours,
}: {
  job: JobResult;
  version: number;
  readiness: Readiness;
  token: string | null;
  retentionHours: number;
}) {
  const studentUrl = absoluteUrl(`#/touch/${job.id}`);
  const teacherUrl = token ? absoluteUrl(`#/job/${job.id}?t=${token}`) : null;
  const [qr, setQr] = useState<string | null>(null);
  const [showTeacher, setShowTeacher] = useState(false);
  const blocked = readiness.status === "blocked";
  const pageMm = `${Math.round(job.tactile.width / job.tactile.px_per_mm)} × ${Math.round(job.tactile.height / job.tactile.px_per_mm)} mm`;
  const stl = stlSize(job);
  const isLocal = /^(localhost|127\.|\[::1\])/.test(window.location.hostname);
  const canShare = typeof navigator !== "undefined" && typeof navigator.share === "function";
  const title = sentenceCase(job.title) || "Diagram";

  useEffect(() => {
    let alive = true;
    QRCode.toDataURL(studentUrl, { margin: 1, width: 360, errorCorrectionLevel: "M", color: { dark: "#17171C", light: "#FFFFFF" } })
      .then((u) => alive && setQr(u))
      .catch(() => alive && setQr(null));
    return () => {
      alive = false;
    };
  }, [studentUrl]);

  const copy = async (text: string, what: string) => {
    const ok = await copyText(text);
    toast(ok ? `${what} copied` : "Couldn't copy. Select the link and copy it by hand.", ok ? "ok" : "warn");
  };

  return (
    <div className="share">
      {readiness.status === "needs_review" && (
        <p className="share__caution">
          Not approved yet. You can still print it, but check the parts first.
        </p>
      )}

      <section className="scard" aria-labelledby="print-title">
        <h2 id="print-title" className="scard__title">Print</h2>
        {blocked ? (
          <p className="share__blocked">Downloads unlock once the problems above are fixed.</p>
        ) : (
          <>
            <LinkButton href={bust(job.files.pdf, version)} download variant="primary" size="lg" className="btn--block" icon={<IconDownload size={20} />}>
              Download PDF for swell paper
            </LinkButton>
            <p className="scard__note">2 pages: the raised drawing and the braille key · {pageMm}. Print at 100% scale.</p>
            <div className="share__row">
              <LinkButton href={bust(job.files.stl, version)} download variant="secondary" icon={<IconCube size={18} />}>
                3D model (STL)
              </LinkButton>
              <LinkButton href={bust(job.files.png, version)} download variant="secondary" icon={<IconImage size={18} />}>
                Image (PNG)
              </LinkButton>
            </div>
            <p className="scard__note">{stl ? `STL plate: ${stl}.` : "STL plate size is shown in your slicer."} Braille: UEB grade 1 (uncontracted), English.</p>
          </>
        )}
      </section>

      <section className="scard" aria-labelledby="student-title">
        <h2 id="student-title" className="scard__title">Student's phone</h2>
        <p className="scard__lead">The student touches the screen and hears each part. This link is view-only: it can't change the lesson.</p>
        {readiness.status !== "ready" && (
          <p className="scard__note" role="status">
            <strong>Students see the last approved version.</strong>{" "}
            {readiness.approved ? "" : "Approve this lesson first; until then the link says “not ready yet”."}
          </p>
        )}
        <div className="student">
          <div className="student__qr">
            {qr ? <img src={qr} width={168} height={168} alt={`QR code for the student link to ${title}`} /> : <div className="student__qr-empty" />}
          </div>
          <div className="student__acts">
            <div className="linkbox">
              <input className="linkbox__url" readOnly value={studentUrl} aria-label="Student link" onFocus={(e) => e.currentTarget.select()} />
              <button type="button" className="linkbox__copy" onClick={() => copy(studentUrl, "Student link")} disabled={blocked}>
                <IconCopy size={18} />
                <span>Copy</span>
              </button>
            </div>
            {canShare && (
              <Button
                variant="secondary"
                icon={<IconShare size={18} />}
                disabled={blocked}
                onClick={() => navigator.share({ title: `${title} · Feelable`, text: `Explore “${title}” by touch`, url: studentUrl }).catch(() => {})}
              >
                Share…
              </Button>
            )}
            <Button variant="ink" icon={<IconHand size={18} />} onClick={() => navigate(`#/touch/${job.id}`)} disabled={blocked}>
              Try touch mode here
            </Button>
          </div>
        </div>
        {isLocal && (
          <p className="scard__note">
            Phones can't open <code>localhost</code>. Open Feelable with this computer's network address first, then share.
          </p>
        )}
      </section>

      {teacherUrl && (
        <section className="scard scard--private" aria-labelledby="teacher-title">
          <h2 id="teacher-title" className="scard__title">
            <IconLock size={18} /> Your teacher link
          </h2>
          <p className="scard__lead">
            Opens this page with editing switched on, on any device. <strong>Keep it private</strong> — anyone with it can change the lesson.
          </p>
          {showTeacher ? (
            <div className="linkbox">
              <input className="linkbox__url" readOnly value={teacherUrl} aria-label="Teacher link (private)" onFocus={(e) => e.currentTarget.select()} />
              <button type="button" className="linkbox__copy" onClick={() => copy(teacherUrl, "Teacher link")}>
                <IconCopy size={18} />
                <span>Copy</span>
              </button>
            </div>
          ) : (
            <Button variant="secondary" size="sm" onClick={() => setShowTeacher(true)}>
              Show teacher link
            </Button>
          )}
        </section>
      )}

      <p className="share__retention">
        <IconClock size={16} />
        <span>{expiresText(job, retentionHours)} Students receive changes after you approve them. An open lesson offers to update.</span>
      </p>
    </div>
  );
}
