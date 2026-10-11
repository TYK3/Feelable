// Teacher review workspace: Preview · Fix · Share, readiness on top, sticky status bar.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { aiDescribe, ApiError, getEditToken, getJob, saveEditToken, updateJob } from "../api";
import { navigate, proceed, replaceHash, setNavGuard } from "../router";
import type { JobEdit, JobResult, PartEdit, ReadinessIssue } from "../types";
import { AppBar } from "../components/AppBar";
import { BrailleWord } from "../components/BrailleWord";
import { Button, LinkButton } from "../components/Button";
import { Dialog } from "../components/Dialog";
import { DiagramViewer } from "../components/DiagramViewer";
import { IconArrowLeft, IconArrowRight, IconEye, IconLock, IconRefresh, IconSparkle, IconUpload } from "../components/Icons";
import { AiBadge } from "../components/PartRow";
import { PartsPanel } from "../components/PartsPanel";
import { ReadinessPanel, StatusIcon } from "../components/ReadinessPanel";
import { SharePanel } from "../components/SharePanel";
import { toast } from "../components/Toast";
import { rememberJob, takeCachedJob } from "../components/jobStore";
import { sentenceCase } from "../components/text";
import { loadDraft, saveDraft, type PartDraft } from "../lib/drafts";
import { aiAvailable, limitsOf, useHealth } from "../lib/health";
import { readinessOf, STATUS_COPY, whyText } from "../lib/readiness";
import { DrawPartDialog, type DrawOptions } from "../components/DrawPartDialog";
import { CloseupDialog } from "../components/CloseupDialog";
import { namesFromParent, parentOf, setPendingUpload, takeCloseup } from "../lib/pending";

type Step = "preview" | "fix" | "share";
type DraftState = { title?: string; parts: Record<string, PartDraft> };
type LoadError = { status: number; code: string | null; message: string };

const STEP_KEY = (id: string) => `ts.step.${id}`;

export default function ReviewPage({ id, urlToken }: { id: string; urlToken: string | null }) {
  const health = useHealth();
  const limits = limitsOf(health);
  const aiOk = aiAvailable(health);

  // Teacher link: remember the token, then take it out of the address bar.
  const [token, setToken] = useState<string | null>(() => {
    if (urlToken) saveEditToken(id, urlToken);
    return urlToken ?? getEditToken(id);
  });
  useEffect(() => {
    if (urlToken) replaceHash(`#/job/${id}`);
  }, [id, urlToken]);
  const canEdit = !!token;

  const [job, setJob] = useState<JobResult | null>(() => takeCachedJob(id) ?? null);
  const [loadError, setLoadError] = useState<LoadError | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [bump, setBump] = useState(0);
  const [step, setStepState] = useState<Step>(() => {
    try {
      return (sessionStorage.getItem(STEP_KEY(id)) as Step) || "preview";
    } catch {
      return "preview";
    }
  });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [draft, setDraft] = useState<DraftState>({ parts: {} });
  const hydrated = useRef(false);
  const [saving, setSaving] = useState(false);
  const [structural, setStructural] = useState(false);
  const [approving, setApproving] = useState(false);
  const [aiRunning, setAiRunning] = useState(false);
  const [aiFailed, setAiFailed] = useState<string | null>(null);
  const [mergeMode, setMergeMode] = useState(false);
  const [leave, setLeave] = useState<string | null>(null);

  const setStep = useCallback(
    (s: Step) => {
      setStepState(s);
      try {
        sessionStorage.setItem(STEP_KEY(id), s);
      } catch {
        /* ignore */
      }
      const nav = document.getElementById("rv-steps");
      if (nav && nav.getBoundingClientRect().top < 0) window.scrollTo({ top: window.scrollY + nav.getBoundingClientRect().top - 64 });
    },
    [id],
  );

  // ---- load ------------------------------------------------------------------
  useEffect(() => {
    let alive = true;
    setLoadError(null);
    getJob(id)
      .then((j) => {
        if (!alive) return;
        rememberJob(j);
        setJob(j);
      })
      .catch((e) => {
        if (!alive) return;
        setLoadError(
          e instanceof ApiError ? { status: e.status, code: e.code, message: e.message } : { status: 0, code: "network", message: "We couldn't reach the server." },
        );
      });
    return () => {
      alive = false;
    };
  }, [id, reloadKey]);

  // Restore a saved draft once the job is known (F05).
  useEffect(() => {
    if (!job || hydrated.current) return;
    hydrated.current = true;
    const d = loadDraft(id);
    if (d && canEdit) {
      const parts: Record<string, PartDraft> = {};
      for (const p of job.parts) if (d.parts[p.id]) parts[p.id] = d.parts[p.id];
      if (d.title !== undefined || Object.keys(parts).length) {
        setDraft({ title: d.title, parts });
        toast(d.baseVersion !== undefined && job.version !== undefined && d.baseVersion !== job.version ? "Restored your unsaved changes (the lesson changed since — check them)." : "Restored your unsaved changes.");
      }
    }
  }, [job, id, canEdit]);

  useEffect(() => {
    if (job) document.title = `${sentenceCase(job.title) || "Untitled diagram"} · Feelable`;
  }, [job]);

  // ---- edits -----------------------------------------------------------------
  const titleNow = draft.title ?? sentenceCase(job?.title ?? "");
  const edit: JobEdit = useMemo(() => {
    const e: JobEdit = {};
    if (!job) return e;
    if (draft.title !== undefined && draft.title.trim() !== sentenceCase(job.title)) e.title = draft.title.trim();
    const parts: PartEdit[] = [];
    for (const p of job.parts) {
      const d = draft.parts[p.id];
      if (!d) continue;
      const pe: PartEdit = { id: p.id };
      if (d.name.trim() !== p.name) pe.name = d.name.trim();
      if (d.description.trim() !== p.description) pe.description = d.description.trim();
      if (d.include !== p.include) pe.include = d.include;
      if (Object.keys(pe).length > 1) parts.push(pe);
    }
    if (parts.length) e.parts = parts;
    return e;
  }, [job, draft]);

  const changeCount = (edit.title !== undefined ? 1 : 0) + (edit.parts ?? []).reduce((n, pe) => n + Object.keys(pe).length - 1, 0);
  const dirty = changeCount > 0 || (draft.title !== undefined && draft.title.trim() === "");
  const problem = !job
    ? null
    : draft.title !== undefined && draft.title.trim() === ""
      ? "Give the lesson a title."
      : job.parts.some((p) => draft.parts[p.id] && draft.parts[p.id].name.trim() === "")
        ? "Every part needs a name."
        : job.parts.length > 0 && job.parts.every((p) => !(draft.parts[p.id] ?? p).include)
          ? "Keep at least one part switched on."
          : (edit.title?.length ?? 0) > limits.title || (edit.parts ?? []).some((pe) => (pe.name?.length ?? 0) > limits.name || (pe.description?.length ?? 0) > limits.description)
            ? "Some text is over the length limit."
            : null;

  // persist draft
  useEffect(() => {
    if (!job || !hydrated.current || !canEdit) return;
    const t = window.setTimeout(() => {
      saveDraft(id, dirty ? { title: draft.title, parts: draft.parts, baseVersion: job.version, savedAt: Date.now() } : null);
    }, 250);
    return () => window.clearTimeout(t);
  }, [draft, dirty, id, job, canEdit]);

  // guard internal navigation + Back while there are unsaved edits
  useEffect(() => {
    if (!dirty) return;
    setNavGuard((target) => {
      setLeave(target);
      return false;
    });
    return () => setNavGuard(null);
  }, [dirty]);

  const onPartChange = useCallback(
    (pid: string, patch: Partial<PartDraft>) => {
      if (!job) return;
      setDraft((ds) => {
        const p = job.parts.find((x) => x.id === pid)!;
        const cur = ds.parts[pid] ?? { name: p.name, description: p.description, include: p.include };
        return { ...ds, parts: { ...ds.parts, [pid]: { ...cur, ...patch } } };
      });
    },
    [job],
  );

  const discard = () => {
    setDraft({ parts: {} });
    saveDraft(id, null);
  };

  const accept = (res: JobResult, msg: string) => {
    rememberJob(res);
    setJob(res);
    setBump((b) => b + 1);
    toast(msg);
  };

  const handlePatchError = async (e: unknown) => {
    if (e instanceof ApiError && (e.status === 403 || e.code === "view_only")) {
      setToken(null);
      toast("This link is view-only, so changes can't be saved. Open the teacher link to edit.", "warn");
    } else if (e instanceof ApiError && (e.status === 409 || e.code === "conflict")) {
      toast("This lesson changed in another tab. We loaded the latest version — check your edits and save again.", "warn");
      try {
        const fresh = await getJob(id);
        rememberJob(fresh);
        setJob(fresh);
        setBump((b) => b + 1);
      } catch {
        /* keep current */
      }
    } else if (e instanceof ApiError && e.status === 404) {
      setLoadError({ status: 404, code: e.code, message: e.message });
    } else {
      toast(e instanceof ApiError ? `Couldn't save: ${e.message}` : "Couldn't reach the server. Your draft is kept in this browser.", "warn");
    }
  };

  const apply = async (): Promise<boolean> => {
    if (!job || changeCount === 0 || problem) return false;
    setSaving(true);
    try {
      const res = await updateJob(job.id, { ...edit, ...(job.version !== undefined ? { base_version: job.version } : {}) }, token);
      setDraft({ parts: {} });
      saveDraft(id, null);
      accept(res, "Saved. The print, braille key and 3D model are updated.");
      return true;
    } catch (e) {
      await handlePatchError(e);
      return false;
    } finally {
      setSaving(false);
    }
  };

  const structuralPatch = async (e: JobEdit, msg: string, setBusy = setStructural) => {
    if (!job) return;
    setBusy(true);
    try {
      const res = await updateJob(job.id, { ...e, ...(job.version !== undefined ? { base_version: job.version } : {}) }, token);
      accept(res, msg);
      return res;
    } catch (err) {
      await handlePatchError(err);
    } finally {
      setBusy(false);
    }
  };

  const approve = () => structuralPatch({ approved: true }, "Approved. Students now get this version.", setApproving);
  const split = async (pid: string) => {
    const res = await structuralPatch({ split: pid }, "Split into separate parts. Give each one a name.");
    if (res) setSelectedId(pid);
  };
  const merge = async (ids: string[]) => {
    const res = await structuralPatch({ merge: ids }, "Merged into one part.");
    if (res) {
      setMergeMode(false);
      setSelectedId(ids[0]);
    }
  };
  // UX07: a "Possible missing part" warning opens these prefilled with its label / area.
  const [drawing, setDrawing] = useState<false | { name?: string; into?: string }>(false);
  const [closeup, setCloseup] = useState<false | { box?: [number, number, number, number]; hint?: string; target?: [number, number, number, number] }>(false);
  const closeupBox = useRef<[number, number, number, number] | null>(null);
  // A close-up lesson just created from another lesson: carry that lesson's names over once.
  const closeupDone = useRef<string | null>(null);
  useEffect(() => {
    if (!job || !canEdit || closeupDone.current === job.id) return;
    const ctx = takeCloseup(job.id);
    closeupDone.current = job.id;
    if (!ctx) return;
    const renames = namesFromParent(job, ctx);
    if (renames.length) structuralPatch({ title: `${ctx.parent.title} (close-up)`.slice(0, 80), parts: renames }, `Named ${renames.length} parts from the full diagram. Check them.`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id, canEdit]);
  // Returns an error message for the dialog (which keeps the teacher's outline), or null when saved.
  const addDrawn = async (name: string, polygon: [number, number][], o: DrawOptions): Promise<string | null> => {
    if (!job) return null;
    setStructural(true);
    try {
      const edit: JobEdit = {
        add_part: { name, polygon, overlap: o.overlap, into: o.into, mode: o.mode, ...(o.more?.length ? { more_polygons: o.more } : {}) },
      };
      const res = await updateJob(job.id, { ...edit, ...(job.version !== undefined ? { base_version: job.version } : {}) }, token);
      const intoName = o.into ? job.parts.find((p) => p.id === o.into)?.name : null;
      accept(
        res,
        o.into
          ? o.mode === "remove"
            ? `Removed that area from “${intoName}”. Check it on the tactile view.`
            : `Added that area to “${intoName}”. Check it on the tactile view.`
          : `Added “${name}”. Check it on the tactile view.`,
      );
      setDrawing(false);
      const p = o.into ? res.parts.find((x) => x.id === o.into) : res.parts.find((x) => x.name === name);
      if (p) setSelectedId(p.id);
      return null;
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        await handlePatchError(err);
        return "This lesson was changed somewhere else. Your outline is kept here: check the updated lesson, then save again.";
      }
      return err instanceof ApiError ? err.message : "Couldn't save the part. Your outline is kept; try again.";
    } finally {
      setStructural(false);
    }
  };
  const runAi = async () => {
    if (!job) return;
    setAiRunning(true);
    setAiFailed(null);
    try {
      const res = await aiDescribe(job.id);
      accept(res, "AI wrote new descriptions. They're marked AI — check them.");
      const failed = res.readiness?.issues.find((i) => i.code === "ai_failed");
      if (failed) setAiFailed(failed.message);
    } catch (e) {
      if (e instanceof ApiError && (e.code === "ai_off" || e.status === 503)) setAiFailed("AI isn't set up on this server, so descriptions can't be written automatically. You can type your own.");
      else if (e instanceof ApiError && e.status === 403) await handlePatchError(e);
      else setAiFailed(e instanceof ApiError ? `AI couldn't describe the parts: ${e.message}` : "AI couldn't be reached. Try again in a minute.");
    } finally {
      setAiRunning(false);
    }
  };

  const titleRef = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = titleRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  });
  // UX05: a close-up made in this browser links back to its full diagram.
  const parentLink = job ? parentOf(job.id) : null;
  // UX07: "Make detail view" — open the close-up tool around a crowded label (3× its box, clamped).
  const onDetail = (i: ReadinessIssue) => {
    // F08: open on the structure the pointer reaches; only fall back to the label when unknown
    const tb = i.target_bbox ?? null;
    const base = tb ?? i.bbox;
    if (!job || !base) return;
    const [x0, y0, x1, y1] = base;
    const W = job.source.width;
    const H = job.source.height;
    const side = Math.max(x1 - x0, y1 - y0, 0.18 * Math.max(W, H)) * 1.6;
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const bx0 = Math.max(0, Math.min(W - side, cx - side / 2));
    const by0 = Math.max(0, Math.min(H - side, cy - side / 2));
    setStep("fix");
    setCloseup({
      box: [bx0, by0, Math.min(W, bx0 + side), Math.min(H, by0 + side)],
      target: tb ?? undefined,
      hint: tb
        ? `The box starts where the pointer for “${i.target ?? "this label"}” ends (dashed). Check it covers the structure, then make the close-up.`
        : `Choose the structure “${i.target ?? "this label"}” refers to. The box starts on the label because its pointer wasn't traced.`,
    });
  };
  const onIssue = (i: ReadinessIssue) => {
    switch (i.action) {
      case "retake":
        navigate("#/");
        return;
      case "rename":
      case "include":
        setStep("fix");
        if (i.part_id) {
          setSelectedId(i.part_id);
          window.setTimeout(() => document.getElementById(i.action === "rename" ? `name-${i.part_id}` : `pbody-${i.part_id}`)?.focus(), 120);
        }
        return;
      case "outline":
        setStep("fix");
        if (i.part_id) {
          setSelectedId(i.part_id);
          setDrawing({ into: i.part_id });
        }
        return;
      case "retry":
        navigate("#/");
        return;
      case "retitle":
      case "shorten":
        setStep("fix");
        window.setTimeout(() => titleRef.current?.focus(), 120);
        return;
      case "split":
        if (i.part_id) split(i.part_id);
        return;
      case "merge":
        setStep("fix");
        setMergeMode(true);
        return;
      case "draw":
        setStep("fix");
        setDrawing({ name: i.target });
        return;
      case "reduce":
        setStep("fix");
        setSelectedId(null);
        toast("Switch off the parts your student needs least.");
        window.setTimeout(() => document.getElementById("parts-title")?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
        return;
    }
  };

  // ---- render states ---------------------------------------------------------
  const newDiagram = (
    <LinkButton href="#/" variant="secondary" size="sm" icon={<IconUpload size={18} />}>
      New diagram
    </LinkButton>
  );

  if (loadError && loadError.status === 404) {
    const expired = loadError.code === "expired";
    return (
      <>
        <AppBar right={newDiagram} />
        <main className="page gone" id="main" tabIndex={-1}>
          <div className="gone__card">
            <BrailleWord word="gone" className="gone__braille" />
            <h1 className="gone__title">This lesson is no longer on the demo server.</h1>
            <p className="gone__msg">
              {expired
                ? "Lessons are kept for 24 hours on the free demo server, and this one has been removed."
                : "It may have expired, or the demo server was restarted."}{" "}
              If you downloaded the PDF, it still works. Making a new one takes seconds.
            </p>
            <LinkButton href="#/" variant="primary" size="lg" icon={<IconUpload size={20} />}>
              New diagram
            </LinkButton>
          </div>
        </main>
      </>
    );
  }

  if (loadError && !job) {
    return (
      <>
        <AppBar right={newDiagram} />
        <main className="page gone" id="main" tabIndex={-1}>
          <div className="gone__card">
            <h1 className="gone__title">This lesson didn't load.</h1>
            <p className="gone__msg">{loadError.message} Try again in a moment.</p>
            <Button variant="primary" size="lg" icon={<IconRefresh size={20} />} onClick={() => setReloadKey((k) => k + 1)}>
              Try again
            </Button>
          </div>
        </main>
      </>
    );
  }

  if (!job) {
    return (
      <>
        <AppBar right={newDiagram} />
        <main className="page review" id="main" tabIndex={-1} aria-busy="true">
          <div className="skel skel--title" />
          <div className="rv-grid">
            <div className="skel skel--view" />
            <div className="skel skel--side" />
          </div>
          <p className="sr-only" role="status">Loading lesson…</p>
        </main>
      </>
    );
  }

  const readiness = readinessOf(job);
  const v = (job.version ?? 0) * 1000 + bump;
  const total = job.timings?.total;
  const actionable = readiness.issues.filter((i) => i.severity !== "info").length;
  const names = Object.fromEntries(job.parts.map((p) => [p.id, draft.parts[p.id]?.name ?? p.name]));
  const selected = job.parts.find((p) => p.id === selectedId) ?? null;
  const busy = saving || structural || aiRunning || approving;
  const status = readiness.status;

  const stepsDef: { id: Step; label: string; badge?: string }[] = [
    { id: "preview", label: "Preview" },
    { id: "fix", label: canEdit ? "Fix" : "Parts", badge: canEdit && actionable && status !== "ready" ? String(actionable) : undefined },
    { id: "share", label: "Share" },
  ];

  return (
    <>
      <AppBar right={newDiagram} />
      {!canEdit && (
        <div className="viewonly" role="note">
          <IconEye size={18} />
          <span>
            <strong>View-only.</strong> You can look and download. To edit, open the teacher link.
          </span>
        </div>
      )}
      <main className={`page review step-${step}`} id="main" tabIndex={-1}>
        <header className="rv-head">
          {parentLink && (
            <a className="rv-parent" href={`#/job/${encodeURIComponent(parentLink.id)}`}>
              <IconArrowLeft size={16} />
              Back to full diagram
              {parentLink.title && <span className="rv-parent__title">: {sentenceCase(parentLink.title)}</span>}
            </a>
          )}
          {canEdit ? (
            <div className="rv-title">
              <label htmlFor="job-title" className="sr-only">Lesson title</label>
              {/* UX08: a one-line textarea so long titles wrap on phones instead of being cut off */}
              <textarea
                id="job-title"
                ref={titleRef}
                rows={1}
                className="rv-title__input"
                value={titleNow}
                maxLength={limits.title}
                placeholder="Untitled diagram"
                disabled={busy}
                onChange={(e) => setDraft((d) => ({ ...d, title: e.target.value.replace(/\s*\n+\s*/g, " ") }))}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    (e.target as HTMLTextAreaElement).blur();
                  }
                  if (e.key === "Escape") setDraft((d) => ({ ...d, title: undefined }));
                }}
              />
              <span className={titleNow.length > limits.title * 0.85 ? "rv-title__count is-near" : "rv-title__count"}>
                {titleNow.length}/{limits.title}
              </span>
            </div>
          ) : (
            <h1 className="rv-title__text">{job.title && job.title.trim().toLowerCase() !== "diagram" ? sentenceCase(job.title) : "Untitled diagram"}</h1>
          )}
          <ul className="rv-meta" aria-label="About this lesson">
            {(() => {
              const inc = job.parts.filter((p) => p.include).length;
              const off = job.parts.length - inc;
              return (
                <li>
                  {inc} included {inc === 1 ? "part" : "parts"}
                  {off > 0 ? ` · ${off} omitted ${off === 1 ? "candidate" : "candidates"} (see Fix parts)` : ""}
                </li>
              );
            })()}
            {job.ai_used && (
              <li className="rv-meta__ai">
                <IconSparkle size={14} /> Descriptions added
              </li>
            )}
            {job.page_flattened && <li>Page straightened</li>}
            {typeof total === "number" && total > 0 && <li>Made in {total.toFixed(1)} s</li>}
            {job.readiness && (
              <li>{job.readiness.status === "ready" ? "Approved for students" : job.readiness.status === "blocked" ? "Not ready" : "Draft — not shared with students yet"}</li>
            )}
          </ul>
        </header>

        <nav className="steps" id="rv-steps" aria-label="Review steps">
          <div className="steps__inner" role="tablist">
            {stepsDef.map((s, i) => (
              <button
                key={s.id}
                type="button"
                role="tab"
                id={`step-${s.id}`}
                aria-selected={step === s.id}
                aria-controls="rv-body"
                className={step === s.id ? "steps__tab is-on" : "steps__tab"}
                onClick={() => setStep(s.id)}
                onKeyDown={(e) => {
                  const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
                  if (!d) return;
                  e.preventDefault();
                  const n = stepsDef[(i + d + stepsDef.length) % stepsDef.length];
                  setStep(n.id);
                  document.getElementById(`step-${n.id}`)?.focus();
                }}
                tabIndex={step === s.id ? 0 : -1}
              >
                <span className="steps__n" aria-hidden="true">{i + 1}</span>
                {s.label}
                {s.badge && (
                  <span className="steps__badge">
                    {s.badge}
                    <span className="sr-only"> to check</span>
                  </span>
                )}
                {s.id === "share" && status === "blocked" && <IconLock size={14} />}
              </button>
            ))}
          </div>
        </nav>

        <div className="rv-grid" id="rv-body" role="tabpanel" aria-labelledby={`step-${step}`}>
          <div className="rv-ready">
            {step === "preview" ? (
              <ReadinessPanel job={job} readiness={readiness} canEdit={canEdit} dirty={dirty} approving={approving} onAction={onIssue} onDetail={onDetail} onApprove={approve} />
            ) : (
              <ReadinessPanel job={job} readiness={readiness} canEdit={canEdit} dirty={dirty} approving={approving} compact onAction={onIssue} onApprove={approve} onExpand={() => setStep("preview")} />
            )}
          </div>

          <div className="rv-view">
            <DiagramViewer
              job={job}
              version={v}
              selectedId={selectedId}
              hoverId={hoverId}
              onSelect={setSelectedId}
              onHover={setHoverId}
              names={names}
              compact={step === "fix"}
            />
          </div>

          <div className="rv-content">
            {step === "preview" && (
              <>
                <section className="peek" aria-live="polite" aria-label="Selected part">
                  {selected ? (
                    <>
                      <div className="peek__top">
                        {selected.key && <BrailleWord word={selected.key.replace(/[^a-z]/gi, "").slice(0, 2)} className="peek__braille" />}
                        <h2 className="peek__name">{sentenceCase(names[selected.id])}</h2>
                        {!(draft.parts[selected.id]?.include ?? selected.include) && <span className="tag tag--off">Removed</span>}
                      </div>
                      <p className="peek__desc">
                        {selected.description_source === "ai" && <AiBadge />} {selected.description || "No description yet."}
                      </p>
                      <p className="peek__why">{whyText(selected.match_method, selected.label_text)}</p>
                      {canEdit && (
                        <div className="peek__actions">
                          <Button variant="secondary" size="sm" onClick={() => setStep("fix")}>
                            Edit this part
                          </Button>
                          {(draft.parts[selected.id]?.include ?? selected.include) ? (
                            <Button variant="ghost" size="sm" onClick={() => onPartChange(selected.id, { include: false })}>
                              Remove part
                            </Button>
                          ) : (
                            <Button variant="ghost" size="sm" onClick={() => onPartChange(selected.id, { include: true })}>
                              {selected.include ? "Undo remove" : "Restore part"}
                            </Button>
                          )}
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="peek__hint">Tap a shape or a part below to see what the student will hear.</p>
                  )}
                </section>
                <section className="glance" aria-labelledby="glance-title">
                  <h2 id="glance-title" className="glance__title">What the student can explore</h2>
                  <ul className="glance__list">
                    {job.parts.filter((p) => p.include).map((p) => (
                      <li key={p.id}>
                        <button
                          type="button"
                          className={["glance__chip", p.id === selectedId && "is-on", !p.include && "is-off"].filter(Boolean).join(" ")}
                          aria-pressed={p.id === selectedId}
                          onClick={() => setSelectedId(p.id === selectedId ? null : p.id)}
                          onMouseEnter={() => setHoverId(p.id)}
                          onMouseLeave={() => setHoverId(null)}
                        >
                          <span className="glance__key">{p.key || "–"}</span>
                          {sentenceCase(names[p.id])}
                        </button>
                      </li>
                    ))}
                  </ul>
                </section>
                <div className="rv-next">
                  <Button variant="secondary" size="lg" onClick={() => setStep(canEdit ? "fix" : "share")}>
                    {canEdit ? "Next: fix parts" : "Next: share"} <IconArrowRight size={18} />
                  </Button>
                </div>
              </>
            )}

            {step === "fix" && (
              <>
                <PartsPanel
                  parts={job.parts}
                  drafts={draft.parts}
                  selectedId={selectedId}
                  hoverId={hoverId}
                  canEdit={canEdit}
                  limits={limits}
                  readiness={readiness}
                  sourceUrl={job.source.url}
                  sourceSize={[job.source.width, job.source.height]}
                  busy={busy}
                  dirty={dirty}
                  ai={{ available: aiOk, running: aiRunning, failed: aiFailed ?? readiness.issues.find((i) => i.code === "ai_failed")?.message ?? null }}
                  onSelect={setSelectedId}
                  onHover={setHoverId}
                  onChange={onPartChange}
                  onSplit={split}
                  onMerge={merge}
                  onDraw={canEdit ? () => setDrawing({}) : undefined}
                  onFixOutline={canEdit ? (pid) => setDrawing({ into: pid }) : undefined}
                  onCloseup={canEdit ? () => setCloseup({}) : undefined}
                  onAiDescribe={runAi}
                  onNewPhoto={() => navigate("#/")}
                  mergeMode={mergeMode}
                  setMergeMode={setMergeMode}
                />
                {closeup && job && (
                  <CloseupDialog
                    sourceUrl={job.source.url}
                    size={[job.source.width, job.source.height]}
                    title={job.title}
                    initialBox={closeup.box}
                    hint={closeup.hint}
                    target={closeup.target}
                    onMake={(file, box) => {
                      closeupBox.current = box;
                      setPendingUpload(file, { parent: job, box: closeupBox.current! });
                      setCloseup(false);
                      navigate("#/");
                    }}
                    onClose={() => setCloseup(false)}
                  />
                )}
                {drawing && job && (
                  <DrawPartDialog
                    jobId={job.id}
                    parts={job.parts}
                    initialInto={drawing.into ?? null}
                    sourceUrl={job.source.url}
                    size={[job.source.width, job.source.height]}
                    nameLimit={limits.name}
                    initialName={drawing.name}
                    suggestions={(readiness.issues || [])
                      .filter((x) => x.action === "draw" && x.target)
                      .map((x) => x.target as string)}
                    busy={structural}
                    onAdd={addDrawn}
                    onClose={() => setDrawing(false)}
                  />
                )}
                <div className="rv-next">
                  <Button variant="secondary" size="lg" onClick={() => setStep("share")}>
                    Next: share <IconArrowRight size={18} />
                  </Button>
                </div>
              </>
            )}

            {step === "share" && <SharePanel job={job} version={v} readiness={readiness} token={token} retentionHours={job.retention_hours ?? health?.retention_hours ?? 24} />}
          </div>
        </div>

        <div className={`statusbar is-${dirty ? "dirty" : status}`} role="region" aria-label="Lesson status">
          <div className="statusbar__inner">
            <p className="statusbar__text" aria-live="polite">
              {dirty ? (
                <>
                  <span className="statusbar__pip" aria-hidden="true" />
                  {saving ? (
                    "Saving and redrawing…"
                  ) : problem ? (
                    problem
                  ) : (
                    <span>
                      <strong>{changeCount}</strong> unsaved {changeCount === 1 ? "change" : "changes"}
                      <span className="statusbar__sub"> · draft kept in this browser</span>
                    </span>
                  )}
                </>
              ) : (
                <>
                  <StatusIcon status={status} />
                  <span>
                    <strong>{STATUS_COPY[status].label}</strong>
                    {status !== "ready" && actionable > 0 && <span className="statusbar__sub"> · {actionable} to check</span>}
                  </span>
                </>
              )}
            </p>
            <div className="statusbar__actions">
              {dirty ? (
                <>
                  <Button variant="ghost" onClick={discard} disabled={saving}>
                    Discard
                  </Button>
                  <Button variant="primary" onClick={apply} busy={saving} disabled={saving || changeCount === 0 || !!problem}>
                    {saving ? "Saving…" : "Save changes"}
                  </Button>
                </>
              ) : status === "blocked" ? (
                step !== "preview" && (
                  <Button variant="secondary" onClick={() => setStep("preview")}>
                    See problems
                  </Button>
                )
              ) : status === "needs_review" && canEdit ? (
                <Button variant="primary" onClick={approve} busy={approving} disabled={busy}>
                  <span className="only-wide">Approve and publish</span>
                  <span className="only-narrow">Approve<span className="sr-only"> and publish</span></span>
                </Button>
              ) : step !== "share" ? (
                <Button variant="primary" onClick={() => setStep("share")}>
                  Share <IconArrowRight size={18} />
                </Button>
              ) : null}
            </div>
          </div>
        </div>
      </main>

      {leave && (
        <Dialog title="Keep your changes?" onClose={() => setLeave(null)}>
          <p className="dlg__msg">
            You have {changeCount} unsaved {changeCount === 1 ? "change" : "changes"} to this lesson.
          </p>
          <div className="dlg__actions">
            <Button
              variant="primary"
              size="lg"
              busy={saving}
              disabled={!!problem || saving}
              onClick={async () => {
                const target = leave;
                if (await apply()) {
                  setLeave(null);
                  proceed(target);
                }
              }}
            >
              Save and continue
            </Button>
            <Button variant="secondary" size="lg" onClick={() => { const t = leave; setLeave(null); proceed(t); }}>
              Keep draft for later
            </Button>
            <Button variant="ghost" size="lg" className="btn--danger" onClick={() => { const t = leave; discard(); setLeave(null); proceed(t); }}>
              Discard changes
            </Button>
          </div>
          {problem && <p className="dlg__note">{problem} Fix it to save, or keep the draft.</p>}
          <button type="button" className="dlg__stay" onClick={() => setLeave(null)}>Stay on this page</button>
        </Dialog>
      )}
    </>
  );
}
