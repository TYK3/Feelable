import { useCallback, useEffect, useId, useRef, useState } from "react";
import { ApiError, cancelJobById, createJobAsync, createJobFromSample, getJob, getJobStatus, listSamples } from "../api";
import { navigate } from "../router";
import type { JobResult, Sample } from "../types";
import { AppBar } from "../components/AppBar";
import { Button } from "../components/Button";
import { Segmented, Switch } from "../components/Controls";
import { IconCamera, IconChevronDown, IconHand, IconImage, IconWarn } from "../components/Icons";
import { MiniDemo } from "../components/MiniDemo";
import { ProcessingView, type ProcError } from "../components/ProcessingView";
import { SampleGallery } from "../components/SampleGallery";
import { rememberJob } from "../components/jobStore";
import { AI_DATA_NOTE, aiAvailable as aiAvail, useHealth } from "../lib/health";
import { HEIC_MSG, IntakeError, isHeic, prepareUpload } from "../lib/resize";
import { attachCloseup, closeupOptions, takePendingUpload } from "../lib/pending";

export type Paper = "a4" | "letter";

type Prepared = { blob: Blob; name: string; preview: string | null; heic: boolean };
type Source = { kind: "file"; file: Prepared } | { kind: "sample"; sample: Sample };
type Busy = {
  run: number;
  source: Source;
  jobId: string | null;
  stages: string[];
  current: string;
  serverElapsed: number | null;
  done: boolean;
  error: ProcError | null;
};

const POLL_MS = 1000;

function errorFor(err: unknown, heic = false): ProcError {
  const code = err instanceof ApiError ? err.code ?? (err.status === 413 ? "too_big" : err.status >= 500 ? "server" : "bad") : "network";
  switch (code) {
    case "decode":
      return { code, recovery: "choose", message: heic ? HEIC_MSG : "We couldn't read that image. The file may be damaged or in a format we can't open. Try another photo or a screenshot." };
    case "not_image":
      return { code, recovery: "choose", message: "That file isn't an image. Choose a photo or screenshot of a diagram." };
    case "too_big":
      return { code, recovery: "choose", message: "That photo is too large for the server. Choose a smaller photo or take a screenshot of it." };
    case "no_diagram":
      return { code, recovery: "choose", message: "We couldn't find a diagram in this photo. Photograph just the diagram, filling the frame, in good light." };
    case "busy":
      return { code, recovery: "retry", message: "The server is busy making other diagrams. Retry in a moment — your photo is still here." };
    case "network":
      return { code, recovery: "retry", message: "We couldn't reach the Feelable server. Check your connection, then retry." };
    case "unavailable":
      return { code, recovery: "retry", message: "The service is temporarily unavailable. Your diagram may still be processing. Try again." };
    case "server":
      return { code, recovery: "retry", message: "The server hit a problem while making this diagram. Retry, or try a clearer photo." };
    default:
      return { code, recovery: "retry", message: err instanceof Error && err.message ? err.message : "Something went wrong." };
  }
}

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

const READ_STEPS = [
  { img: "1-read", title: "Read every label.",
    text: "On-device OCR finds the text; blurry words are re-read up close and checked against 880 science terms.",
    alt: "The skull figure with every label highlighted in yellow." },
  { img: "2-follow", title: "Follow each pointer.",
    text: "Every pointer line is traced from its label to the spot it names.",
    alt: "Pointer lines traced in black from each label to a yellow dot on the bone it names." },
  { img: "3-trace", title: "Trace the shapes.",
    text: "Where neighbours share a colour, it zooms in and carves along the real edge. With AI on, Gemini finds what a lost pointer meant.",
    alt: "Each bone outlined; the ones carved out up close are shaded yellow." },
  { img: "4-feel", title: "Make it feelable.",
    text: "A swell-paper page with textures and braille keys, a 3D model, and a talking phone mode.",
    alt: "The finished raised-line page: bones in different textures with braille key letters." },
];

export default function HomePage() {
  const health = useHealth();
  const aiOk = aiAvail(health);
  const [paper, setPaper] = useState<Paper>("a4");
  const [useAi, setUseAi] = useState(true);
  const forcedOpts = useRef<{ useAi: boolean; paper: Paper } | null>(null);
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [samplesFailed, setSamplesFailed] = useState(false);
  const skull = samples?.find((x) => x.id === "skull-lateral") ?? null;
  const [clientError, setClientError] = useState<string | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [busy, setBusy] = useState<Busy | null>(null);
  const [dragging, setDragging] = useState(false);
  const run = useRef(0);
  const pickRef = useRef<HTMLInputElement>(null);
  const camRef = useRef<HTMLInputElement>(null);
  const errId = useId();
  const aiNoteId = useId();

  useEffect(() => {
    document.title = "Feelable · Every diagram, feelable.";
    listSamples()
      .then((s) => setSamples(Array.isArray(s) ? s : []))
      .catch(() => setSamplesFailed(true));
  }, []);

  const finish = useCallback((my: number, job: JobResult) => {
    if (run.current !== my) return;
    rememberJob(job);
        attachCloseup(job.id);
    setBusy((b) => (b ? { ...b, done: true, current: "done" } : b));
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    window.setTimeout(() => run.current === my && navigate(`#/job/${job.id}`), reduce ? 100 : 600);
  }, []);

  const fail = useCallback((my: number, e: ProcError) => {
    if (run.current !== my) return;
    if (e.code === "cancelled") {
      setBusy(null);
      return;
    }
    setBusy((b) => (b ? { ...b, error: e } : b));
  }, []);

  const start = useCallback(
    async (source: Source) => {
      const my = ++run.current;
      setClientError(null);
      // Send the teacher's choice unless health positively says AI is off; the server skips AI when no key is set.
      // A close-up keeps its parent lesson's settings (UX05).
      const forced = forcedOpts.current;
      forcedOpts.current = null;
      const opts = forced ? { useAi: aiOk !== false && forced.useAi, paper: forced.paper } : { useAi: aiOk !== false && useAi, paper };
      const isSample = source.kind === "sample";
      setBusy({
        run: my,
        source,
        jobId: null,
        stages: isSample ? ["sample"] : ["upload", "queued", "read", "find", ...(opts.useAi ? ["ai"] : []), "layout", "files"],
        current: isSample ? "sample" : "upload",
        serverElapsed: null,
        done: false,
        error: null,
      });
      window.scrollTo({ top: 0 });

      if (source.kind === "sample") {
        try {
          finish(my, await createJobFromSample(source.sample.id, opts));
        } catch (e) {
          fail(my, errorFor(e));
        }
        return;
      }

      const heic = source.file.heic;
      let id: string;
      try {
        const file = new File([source.file.blob], source.file.name, { type: source.file.blob.type || "image/jpeg" });
        const created = (await createJobAsync(file, opts)) as unknown as Partial<JobResult> & { id: string };
        if (run.current !== my) return;
        if (Array.isArray(created.parts)) {
          finish(my, created as JobResult); // server without async support answered synchronously
          return;
        }
        id = created.id;
        setBusy((b) => (b && b.run === my ? { ...b, jobId: id, current: "queued" } : b));
      } catch (e) {
        fail(my, errorFor(e, heic));
        return;
      }

      let misses = 0;
      while (run.current === my) {
        await sleep(POLL_MS);
        if (run.current !== my) return;
        try {
          const st = await getJobStatus(id);
          misses = 0;
          if (run.current !== my) return;
          if (st.status === "done") {
            finish(my, await getJob(id));
            return;
          }
          if (st.status === "error" || st.status === "cancelled") {
            const code = st.error?.code ?? (st.status === "cancelled" ? "cancelled" : "server");
            fail(my, errorFor(new ApiError(500, st.error?.message ?? "", code), heic));
            return;
          }
          setBusy((b) =>
            b && b.run === my
              ? {
                  ...b,
                  stages: ["upload", "queued", ...st.stages.filter((s) => s !== "queued" && s !== "done")],
                  current: st.stage === "done" ? "files" : st.stage,
                  serverElapsed: st.elapsed,
                }
              : b,
          );
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) {
            fail(my, errorFor(new ApiError(500, "", "server")));
            return;
          }
          if (++misses >= 4) {
            fail(my, errorFor(e));
            return;
          }
        }
      }
    },
    [aiOk, useAi, paper, finish, fail],
  );

  const onFile = useCallback(
    async (file: File) => {
      setClientError(null);
      setPreparing(true);
      try {
        const prep = await prepareUpload(file);
        const preview = prep.width ? URL.createObjectURL(prep.blob) : null;
        await start({ kind: "file", file: { blob: prep.blob, name: prep.name, preview, heic: isHeic(file) } });
      } catch (e) {
        setBusy(null);
        setClientError(e instanceof IntakeError ? e.message : "We couldn't open that photo. Try another one.");
      } finally {
        setPreparing(false);
      }
    },
    [start],
  );

  // A close-up handed over from the review page ("Make a close-up lesson")
  useEffect(() => {
    const p = takePendingUpload();
    if (p?.ctx) forcedOpts.current = closeupOptions(p.ctx.parent);
    if (p) onFile(p.file);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cancel = () => {
    const id = busy?.jobId;
    run.current++;
    if (id) cancelJobById(id);
    setBusy(null);
  };

  // Paste an image anywhere on the page; drop it anywhere too.
  useEffect(() => {
    if (busy) return;
    const onPaste = (e: ClipboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.kind === "file" && i.type.startsWith("image/"));
      const blob = item?.getAsFile();
      if (!blob) return;
      e.preventDefault();
      onFile(new File([blob], `pasted.${blob.type.split("/")[1] || "png"}`, { type: blob.type }));
    };
    const onOver = (e: DragEvent) => {
      if (!e.dataTransfer?.types.includes("Files")) return;
      e.preventDefault();
      setDragging(true);
    };
    const onLeave = (e: DragEvent) => {
      if (!e.relatedTarget) setDragging(false);
    };
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer?.files?.[0];
      if (f) onFile(f);
    };
    window.addEventListener("paste", onPaste);
    window.addEventListener("dragover", onOver);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("paste", onPaste);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [busy, onFile]);

  // Release object URLs.
  const previewRef = useRef<string | null>(null);
  useEffect(() => {
    const now = busy?.source.kind === "file" ? busy.source.file.preview : null;
    if (previewRef.current && previewRef.current !== now) URL.revokeObjectURL(previewRef.current);
    previewRef.current = now;
  }, [busy]);
  useEffect(() => () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
  }, []);

  const heart = samples?.find((s) => s.id === "heart") ?? samples?.[0] ?? null;
  const aiOn = aiOk === true && useAi;

  const inputs = (
    <>
      <input
        ref={pickRef}
        type="file"
        accept="image/*,.heic,.heif"
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) onFile(f);
        }}
      />
      <input
        ref={camRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) onFile(f);
        }}
      />
    </>
  );

  if (busy) {
    const src = busy.source;
    return (
      <>
        <AppBar />
        <main className="page page--proc" id="main" tabIndex={-1}>
          {inputs}
          <ProcessingView
            imageUrl={src.kind === "file" ? src.file.preview : src.sample.thumb_url}
            label={src.kind === "file" ? src.file.name : src.sample.title}
            stages={busy.stages}
            current={busy.current}
            done={busy.done}
            error={busy.error}
            serverElapsed={busy.serverElapsed}
            onCancel={cancel}
            onRetry={() => start(src)}
            onChoose={() => pickRef.current?.click()}
            onHome={() => {
              run.current++;
              setBusy(null);
            }}
          />
        </main>
      </>
    );
  }

  return (
    <>
      <AppBar right={<a className="appbar__link" href="#samples" onClick={(e) => { e.preventDefault(); document.getElementById("samples")?.scrollIntoView({ behavior: "smooth" }); }}>Examples</a>} />
      <main className={dragging ? "page home is-dragging" : "page home"} id="main" tabIndex={-1}>
        {inputs}
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero__copy">
            <h1 id="hero-title" className="hero__title">Every diagram, feelable.</h1>
            <p className="hero__lead">
              Photograph a textbook diagram. Get a raised-line print with braille keys, a 3D model, and a talking phone
              mode for blind students.
            </p>

            <div className="hero__actions">
              <Button variant="primary" size="lg" className="only-touch" icon={<IconCamera size={22} />} onClick={() => camRef.current?.click()} busy={preparing}>
                Take photo
              </Button>
              <Button
                variant="ink"
                size="lg"
                icon={<IconImage size={22} />}
                onClick={() => pickRef.current?.click()}
                aria-describedby={clientError ? errId : undefined}
              >
                Choose photo
              </Button>
              <Button variant="secondary" size="lg" className="hero__example" icon={<IconHand size={22} />} disabled={!heart} onClick={() => heart && start({ kind: "sample", sample: heart })}>
                Try an example
              </Button>
            </div>
            <p className="hero__hint only-pointer">…or drop or paste an image anywhere on this page.</p>

            {clientError && (
              <p className="alert alert--err" id={errId} role="alert">
                <IconWarn size={18} />
                {clientError}
              </p>
            )}

            <details className="opts">
              <summary className="opts__summary">
                <span>Options</span>
                <span className="opts__state">
                  {paper === "a4" ? "A4" : "Letter"}, AI descriptions {aiOn ? "on" : "off"}
                </span>
                <IconChevronDown size={18} className="opts__chev" />
              </summary>
              <div className="opts__body">
                <Segmented<Paper>
                  label="Paper size"
                  value={paper}
                  onChange={setPaper}
                  options={[
                    { value: "a4", label: "A4" },
                    { value: "letter", label: "US Letter" },
                  ]}
                />
                <div className="field">
                  <Switch
                    label="AI names & descriptions"
                    checked={aiOn}
                    disabled={aiOk !== true}
                    onChange={setUseAi}
                    describedBy={aiNoteId}
                  />
                  <p className="opts__note" id={aiNoteId}>
                    {aiOk === true
                      ? AI_DATA_NOTE
                      : aiOk === false
                        ? "AI is not set up on this server, so names come from the labels in your photo and you write the descriptions. Nothing leaves our server."
                        : "Checking whether AI is available…"}
                  </p>
                </div>
              </div>
            </details>
          </div>

          <div className="hero__demo">
            <MiniDemo />
            <p className="hero__demo-cap">Touch a part. This is how a student explores on their phone.</p>
          </div>
        </section>

        <section className="how" aria-labelledby="how-title">
          <h2 id="how-title" className="sr-only">How it works</h2>
          <ol className="how__steps">
            <li>
              <span className="how__n" aria-hidden="true">1</span>
              <span><strong>Snap.</strong> Photograph a diagram, or pick an example.</span>
            </li>
            <li>
              <span className="how__n" aria-hidden="true">2</span>
              <span><strong>Check.</strong> Fix any names, then approve it for teaching.</span>
            </li>
            <li>
              <span className="how__n" aria-hidden="true">3</span>
              <span><strong>Share.</strong> Print on swell paper, 3D-print, or send the phone link.</span>
            </li>
          </ol>
        </section>

        <section className="reads" aria-labelledby="reads-title">
          <div className="reads__head">
            <h2 id="reads-title" className="h2">How Feelable reads a picture</h2>
            <p className="reads__lead">
              Labels in small print, pointer lines that cross, bones of almost the same colour. Making a tactile
              graphic of that by hand is slow, specialist work, so most diagrams reach blind students as a paragraph
              of description. Feelable reads it the way a person would: up close, one label at a time.
            </p>
          </div>
          <ol className="reads__steps">
            {READ_STEPS.map((st, i) => (
              <li key={st.img} className="reads__step">
                <figure>
                  <img src={`/how/${st.img}.webp`} alt={st.alt} loading="lazy" width={720} height={494} />
                  <figcaption>
                    <span className="reads__n" aria-hidden="true">{i + 1}</span>
                    <span><strong>{st.title}</strong> {st.text}</span>
                  </figcaption>
                </figure>
              </li>
            ))}
          </ol>
          <div className="reads__foot">
            {skull && (
              <Button variant="secondary" onClick={() => start({ kind: "sample", sample: skull })}>
                Open the skull and see “How it read”
              </Button>
            )}
            <p className="reads__credit">Figure: OpenStax Anatomy &amp; Physiology, CC BY 4.0.</p>
          </div>
        </section>

        <section className="samples-sec" id="samples" aria-labelledby="samples-title">
          <h2 id="samples-title" className="h2">Examples</h2>
          <p className="muted">Ready-made textbook diagrams. Each opens in a second.</p>
          <SampleGallery samples={samples} failed={samplesFailed} onPick={(s) => start({ kind: "sample", sample: s })} />
        </section>

        <footer className="footer">
          <p>
            Made for Banana Hacks 2026. Feelable processes your image on its server. Optional automatic descriptions send a
            smaller image and detected labels to Google Gemini.
            Lessons stay on the free demo server for {health?.retention_hours ?? 24} hours.
          </p>
        </footer>

        {dragging && (
          <div className="dropveil" aria-hidden="true">
            <span>Drop the diagram to make it feelable</span>
          </div>
        )}
      </main>
    </>
  );
}
