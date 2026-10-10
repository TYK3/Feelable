import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getEditToken, getJob, getPublished } from "../api";
import type { JobResult, Part } from "../types";
import ExploreCanvas, { type ExploreCanvasHandle } from "./ExploreCanvas";
import Menu, { cap, type TouchSettings } from "./Menu";
import PartList from "./PartList";
import { Finder } from "./finder";
import { buildShapes, focusView } from "./geometry";
import { PATTERN_ENTER, PATTERN_FOUND, canVibrate, edgeTick, setVibrationEnabled, vibrate } from "./haptics";
import { initSpeech, isSpeechSupported, prefetchNatural, speak, stopSpeech } from "./speech";
import "./touch.css";

type Phase = { kind: "loading" } | { kind: "error"; heading: string; message: string; retry: boolean } | { kind: "start" } | { kind: "explore" } | { kind: "finished" };
type Overlay = null | "menu" | "find" | "list";

const SETTINGS_KEY = "feelable.touch.settings";
const NOTICE_KEY = "feelable.touch.noSpeechNoticeShown";
const ROTATE_KEY = "feelable.touch.rotateHintShown";
const ROTATE_HINT = "Turn your phone sideways for a bigger picture.";

/** True the first time it is called for `key` on this device (and true when storage is unavailable). */
function firstTime(key: string): boolean {
  try {
    if (localStorage.getItem(key) === "1") return false;
    localStorage.setItem(key, "1");
  } catch {
    /* storage unavailable: show it */
  }
  return true;
}

interface Notice {
  id: "speech" | "rotate";
  text: string;
}
const DEFAULT_SETTINGS: TouchSettings = { rate: 1, showHighlights: true, vibration: true };

function loadSettings(): TouchSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) {
      const s = { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } as TouchSettings;
      s.rate = Math.min(1.5, Math.max(0.7, Number(s.rate) || 1));
      return s;
    }
  } catch {
    /* storage unavailable */
  }
  return DEFAULT_SETTINGS;
}

function introText(job: JobResult, n: number): string {
  return `${cap(job.title || "Diagram")}. ${n} ${n === 1 ? "part" : "parts"}. Explore with one finger. Double tap to hear more. Two-finger tap to repeat.`;
}

export default function TouchMode(props: { jobId: string; onExit?: () => void }) {
  const { jobId, onExit } = props;
  const [job, setJob] = useState<JobResult | null>(null);
  const [phase, setPhase] = useState<Phase>({ kind: "loading" });
  const [reload, setReload] = useState(0);
  const [overlay, setOverlay] = useState<Overlay>(null);
  const [settings, setSettings] = useState<TouchSettings>(loadSettings);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [caption, setCaption] = useState<{ main: string; sub: string; on: boolean }>({
    main: "Touch the picture",
    sub: "Move one finger slowly to explore",
    on: false,
  });
  const [live, setLive] = useState("");
  const [finding, setFinding] = useState<Part | null>(null);
  const [notices, setNotices] = useState<Notice[]>([]);
  const dismissNotice = useCallback((id: Notice["id"]) => setNotices((n) => n.filter((x) => x.id !== id)), []);

  const canvasRef = useRef<ExploreCanvasHandle>(null);
  const menuBtnRef = useRef<HTMLButtonElement>(null);
  const finderRef = useRef<Finder | null>(null);
  // Recent remembered utterances with timestamps, so a two-finger tap can skip what its own first finger triggered.
  const history = useRef<{ text: string; t: number }[]>([]);
  const findingRef = useRef<Part | null>(null);
  findingRef.current = finding;
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  // ------------------------------------------------------------ data
  // Students get the teacher-approved snapshot only (TS-06). A teacher (holding the edit token)
  // can preview an unapproved draft, clearly labelled.
  const isTeacher = !!getEditToken(jobId);
  const [preview, setPreview] = useState(false);
  const loadLesson = useCallback(async (): Promise<{ job: JobResult; preview: boolean }> => {
    try {
      return { job: await getPublished(jobId), preview: false };
    } catch (err) {
      if (err instanceof ApiError && err.code === "not_published" && isTeacher) {
        return { job: await getJob(jobId), preview: true };
      }
      throw err;
    }
  }, [jobId, isTeacher]);
  useEffect(() => {
    let alive = true;
    setPhase({ kind: "loading" });
    loadLesson()
      .then(({ job: j, preview: pv }) => {
        if (!alive) return;
        const usable = j.parts.some((p) => p.include && p.tactile_polygons.length > 0);
        if (!usable) {
          setPhase({ kind: "error", heading: "Lesson not ready", message: "This lesson has no parts to explore yet. Ask your teacher to check it.", retry: true });
          return;
        }
        setJob(j);
        setPreview(pv);
        setPhase({ kind: "start" });
      })
      .catch((err: unknown) => {
        if (!alive) return;
        // UX09: name the state and keep the specific, actionable cause
        if (err instanceof ApiError && err.code === "not_published")
          setPhase({ kind: "error", heading: "Lesson not ready", message: "Your teacher hasn't approved this lesson yet. Ask them to approve it, then try again.", retry: true });
        else if (err instanceof ApiError && err.status === 404)
          setPhase({ kind: "error", heading: "Lesson no longer available", message: "This lesson is no longer on the demo server. Ask your teacher for a new link.", retry: false });
        else
          setPhase({ kind: "error", heading: "We couldn't open this lesson", message: "Check your connection, then try again or go home.", retry: true });
      });
    return () => {
      alive = false;
    };
  }, [jobId, reload]);

  // F09: notice when the teacher saves a newer version (every 20 s and when the page regains focus).
  const [update, setUpdate] = useState<JobResult | null>(null);
  const jobRef = useRef<JobResult | null>(null);
  jobRef.current = job;
  useEffect(() => {
    if (!job) return;
    let alive = true;
    const sig = (j: JobResult) => (j.version !== undefined ? String(j.version) : JSON.stringify([j.title, j.parts.map((p) => [p.name, p.description, p.include])]));
    const check = () => {
      if (document.visibilityState === "hidden") return;
      loadLesson()
        .then(({ job: j }) => {
          const cur = jobRef.current;
          if (alive && cur && sig(j) !== sig(cur) && (j.version === undefined || cur.version === undefined || j.version > cur.version)) setUpdate(j);
        })
        .catch(() => {});
    };
    const t = window.setInterval(check, 20000);
    window.addEventListener("focus", check);
    document.addEventListener("visibilitychange", check);
    return () => {
      alive = false;
      window.clearInterval(t);
      window.removeEventListener("focus", check);
      document.removeEventListener("visibilitychange", check);
    };
  }, [jobId, !!job]); // eslint-disable-line react-hooks/exhaustive-deps

  const applyUpdate = useCallback(() => {
    if (!update) return;
    setJob(update);
    setUpdate(null);
    setCurrentId(null);
    speak("Lesson updated.", { rate: settingsRef.current.rate });
    setLive("Lesson updated.");
  }, [update]);

  useEffect(() => {
    if (phase.kind === "loading") document.title = "Loading diagram… · Feelable";
    else if (phase.kind === "error") document.title = `${phase.heading} · Feelable`;
    else if (job) document.title = `${cap(job.title || "Diagram")} · Touch mode · Feelable`;
  }, [phase.kind, job]);

  const parts = useMemo(() => (job ? job.parts.filter((p) => p.include && p.tactile_polygons.length > 0) : []), [job]);

  // Natural voice: fetch clips for every part's name and description in the background.
  useEffect(() => {
    if (!job) return;
    const ctrl = new AbortController();
    const texts = job.parts
      .filter((p) => p.include && p.tactile_polygons.length > 0)
      .flatMap((p) => [cap(p.name), p.description?.trim() || ""]);
    prefetchNatural(job.id, texts, ctrl.signal);
    return () => ctrl.abort();
  }, [job]);
  const partById = useMemo(() => new Map(parts.map((p) => [p.id, p])), [parts]);
  const shapes = useMemo(() => buildShapes(parts), [parts]);
  const view = useMemo(() => {
    if (!job) return { x0: 0, y0: 0, x1: 1, y1: 1 };
    const ppm = job.tactile.px_per_mm || 5;
    const keys = parts.flatMap((p) => (p.key_position ? [p.key_position] : []));
    return focusView(shapes, keys, job.tactile.width, job.tactile.height, 8 * ppm, 6 * ppm);
  }, [job, parts, shapes]);

  // ------------------------------------------------------------ settings side effects
  useEffect(() => {
    setVibrationEnabled(settings.vibration);
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      /* ignore */
    }
  }, [settings]);

  useEffect(
    () => () => {
      stopSpeech();
      finderRef.current?.dispose();
      finderRef.current = null;
    },
    [],
  );

  // ------------------------------------------------------------ speech helpers
  const say = useCallback((text: string, opts: { remember?: boolean } = {}) => {
    if (opts.remember !== false) {
      history.current.push({ text, t: performance.now() });
      if (history.current.length > 10) history.current.shift();
    }
    speak(text, { rate: settingsRef.current.rate });
    setLive(text);
  }, []);

  const describe = useCallback(
    (part: Part) => {
      const text = part.description?.trim() || `${cap(part.name)}.`;
      say(text);
      setCaption({ main: cap(part.name), sub: text, on: true });
    },
    [say],
  );

  const finder = () => (finderRef.current ??= new Finder());

  // ------------------------------------------------------------ start
  const start = useCallback(
    (thenList = false) => {
      if (!job) return;
      initSpeech();
      finder().unlock(); // inside the user gesture
      const next: Notice[] = [];
      if (!isSpeechSupported() && firstTime(NOTICE_KEY)) {
        next.push({ id: "speech", text: "Speech isn't available in this browser. The caption bar shows each part instead." });
      }
      // Landscape diagram on a portrait screen: suggest rotating (once per device).
      const viewAspect = (view.x1 - view.x0) / Math.max(1, view.y1 - view.y0);
      const portrait = window.innerHeight > window.innerWidth * 1.1;
      const rotate = !thenList && portrait && viewAspect > 1.15 && firstTime(ROTATE_KEY);
      if (rotate) next.push({ id: "rotate", text: ROTATE_HINT });
      setNotices(next);
      say(introText(job, parts.length) + (rotate ? ` ${ROTATE_HINT}` : ""));
      setPhase({ kind: "explore" });
      setOverlay(thenList ? "list" : null);
      if (!thenList) requestAnimationFrame(() => canvasRef.current?.focus());
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [job, parts.length, say, view],
  );

  // The rotate hint goes away by itself once the phone is sideways, or after 12 s.
  const hasRotateNotice = notices.some((n) => n.id === "rotate");
  useEffect(() => {
    if (!hasRotateNotice) return;
    const check = () => {
      if (window.innerWidth > window.innerHeight) dismissNotice("rotate");
    };
    const t = window.setTimeout(() => dismissNotice("rotate"), 12000);
    window.addEventListener("resize", check);
    window.addEventListener("orientationchange", check);
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("resize", check);
      window.removeEventListener("orientationchange", check);
    };
  }, [hasRotateNotice, dismissNotice]);

  // ------------------------------------------------------------ explore events
  const enterPart = useCallback(
    (id: string) => {
      const part = partById.get(id);
      if (!part) return;
      setCurrentId(id);
      vibrate(PATTERN_ENTER);
      if (findingRef.current) {
        setCaption({ main: cap(part.name), sub: `Finding the ${findingRef.current.name}. Follow the beeps.`, on: true });
        return; // keep the audio channel for the beeps
      }
      say(cap(part.name));
      setCaption({ main: cap(part.name), sub: "Double tap or hold to hear more", on: true });
    },
    [partById, say],
  );

  const leave = useCallback((where: "page" | "off") => {
    setCurrentId(null);
    const f = findingRef.current;
    setCaption({
      main: where === "off" ? "Off the page" : "…",
      sub: f ? `Finding the ${f.name}. Follow the beeps.` : where === "off" ? "Move back onto the picture" : "Blank paper",
      on: false,
    });
  }, []);

  const onDescribe = useCallback(
    (id: string | null) => {
      const part = id ? partById.get(id) : null;
      if (part) describe(part);
      else say("Blank paper. No part here.", { remember: false });
    },
    [partById, describe, say],
  );

  const repeat = useCallback(
    (since?: number) => {
      const h = history.current;
      const cutoff = since ?? Infinity;
      for (let i = h.length - 1; i >= 0; i--) {
        if (h[i].t < cutoff - 5) {
          say(h[i].text);
          return;
        }
      }
      if (job) say(introText(job, parts.length));
    },
    [job, parts.length, say],
  );

  const stopFind = useCallback(
    (announce = true) => {
      finderRef.current?.stop();
      if (!findingRef.current) return;
      findingRef.current = null;
      setFinding(null);
      if (announce) say("Find stopped.", { remember: false });
      setCaption({ main: "…", sub: "Find stopped. Keep exploring.", on: false });
    },
    [say],
  );

  const startFind = useCallback(
    (part: Part) => {
      setOverlay(null);
      findingRef.current = part;
      setFinding(part);
      setCurrentId(null);
      finder().start();
      say(`Find the ${part.name}. Follow the beeps. They get faster as you get closer.`);
      setCaption({ main: `Find: ${cap(part.name)}`, sub: "Follow the beeps. Three-finger tap to stop.", on: false });
      requestAnimationFrame(() => canvasRef.current?.focus());
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [say],
  );

  const onFound = useCallback(() => {
    const part = findingRef.current;
    if (!part) return;
    findingRef.current = null;
    setFinding(null);
    finderRef.current?.stop();
    finderRef.current?.chime();
    vibrate(PATTERN_FOUND);
    setCurrentId(part.id);
    say(`Found the ${part.name}!`);
    setCaption({ main: `Found the ${part.name}!`, sub: "Double tap to hear more", on: true });
  }, [say]);

  const onFindDistance = useCallback((mm: number | null) => {
    if (findingRef.current) finderRef.current?.update(mm);
  }, []);

  // ------------------------------------------------------------ menu / exit
  const exit = useCallback(() => {
    stopSpeech();
    finderRef.current?.stop();
    // Only a browser holding the teacher's edit token goes back to the editor (F04).
    if (onExit) onExit();
    else {
      setOverlay(null);
      setPhase({ kind: "finished" });
    }
  }, [onExit]);

  const closeOverlay = useCallback(() => {
    setOverlay(null);
    requestAnimationFrame(() => canvasRef.current?.focus());
  }, []);

  // keyboard cycling through parts. TS-07: reaching the Find target by keyboard completes Find
  // exactly like the pointer path does.
  const enterOrFind = useCallback(
    (id: string) => {
      if (findingRef.current && findingRef.current.id === id) onFound();
      else enterPart(id);
    },
    [enterPart, onFound],
  );
  const cycle = useCallback(
    (dir: 1 | -1): boolean => {
      if (!parts.length) return false;
      const i = currentId ? parts.findIndex((p) => p.id === currentId) : -1;
      const next = i + dir;
      if (next < 0 || next >= parts.length) {
        if (i === -1) {
          enterOrFind(parts[dir === 1 ? 0 : parts.length - 1].id);
          return true;
        }
        return false;
      }
      enterOrFind(parts[next].id);
      return true;
    },
    [parts, currentId, enterOrFind],
  );

  const onExploreKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
        e.preventDefault();
        cycle(1);
        break;
      case "ArrowLeft":
      case "ArrowUp":
        e.preventDefault();
        cycle(-1);
        break;
      case "Tab":
        // Tab walks the parts; at either end focus moves on normally.
        if (cycle(e.shiftKey ? -1 : 1)) e.preventDefault();
        break;
      case "Enter":
        e.preventDefault();
        onDescribe(currentId);
        break;
      case " ":
        e.preventDefault();
        repeat();
        break;
    }
  };

  // global shortcuts: M / Esc open the menu
  useEffect(() => {
    if (phase.kind !== "explore") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      if (overlay) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "m" || e.key === "M" || e.key === "Escape") {
        e.preventDefault();
        setOverlay("menu");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase.kind, overlay]);

  // ------------------------------------------------------------ render
  if (phase.kind === "loading") {
    return (
      <div className="ts-touch ts-t-center">
        <div className="ts-t-spinner" aria-hidden="true" />
        <p className="ts-t-big" role="status" aria-live="polite">
          Loading diagram…
        </p>
      </div>
    );
  }

  if (phase.kind === "error" || !job) {
    // UX09: Exit must work without lesson data — teacher (holds the edit token) goes back to
    // the editor, everyone else goes to the Feelable home page.
    const leave = () => {
      stopSpeech();
      if (onExit && isTeacher) onExit();
      else window.location.hash = "#/";
    };
    const heading = phase.kind === "error" ? phase.heading : "We couldn't open this lesson";
    const message = phase.kind === "error" ? phase.message : "Try again or go home.";
    const canRetry = phase.kind !== "error" || phase.retry;
    return (
      <div className="ts-touch ts-t-center">
        <p className="ts-t-brand">Feelable</p>
        <div role="alert" className="ts-t-error">
          <h1 className="ts-t-big">{heading}</h1>
          <p className="ts-t-lead">{message}</p>
        </div>
        <div className="ts-t-row">
          {canRetry && (
            <button className="ts-t-btn ts-t-btn-primary" onClick={() => setReload((n) => n + 1)}>
              Try again
            </button>
          )}
          <button className={canRetry ? "ts-t-btn" : "ts-t-btn ts-t-btn-primary"} onClick={leave}>
            {onExit && isTeacher ? "Back to the editor" : "Go to Feelable home"}
          </button>
        </div>
      </div>
    );
  }

  const title = cap(job.title || "Diagram");

  if (phase.kind === "finished") {
    return (
      <div className="ts-touch ts-t-center">
        <p className="ts-t-brand">Feelable</p>
        <h1 className="ts-t-big">Lesson finished</h1>
        <p className="ts-t-lead">{title}. Well done.</p>
        <div className="ts-t-row">
          <button className="ts-t-btn ts-t-btn-primary" autoFocus onClick={() => setPhase({ kind: "start" })}>
            Explore again
          </button>
          <a className="ts-t-btn" href="#/">
            Feelable home
          </a>
        </div>
      </div>
    );
  }

  const speechOk = isSpeechSupported();
  const vibOk = canVibrate();
  const caps = (
    <ul className="ts-t-caps" aria-label="What works on this device">
      <li className={speechOk ? "is-on" : "is-off"}>{speechOk ? "Speech on" : "No speech here — captions instead"}</li>
      <li className={vibOk ? "is-on" : "is-off"}>{vibOk ? "Vibration available (if your phone allows it)" : "No vibration on this device"}</li>
      <li className="is-on">Captions always on</li>
    </ul>
  );

  if (phase.kind === "start") {
    return (
      <div className="ts-touch ts-t-start" onClick={() => start(false)}>
        <p className="ts-t-brand">Feelable</p>
        {preview && (
          <p className="ts-t-start-net" role="status" style={{ color: "#FFD43B", fontWeight: 700 }}>
            Teacher preview: students won't see this until you approve it.
          </p>
        )}
        <h1 className="ts-t-start-title">{title}</h1>
        <button
          className="ts-t-start-btn"
          autoFocus
          onClick={(e) => {
            e.stopPropagation();
            start(false);
          }}
        >
          Tap anywhere to start
        </button>
        <p className="ts-t-start-intro">
          {parts.length} {parts.length === 1 ? "part" : "parts"}. Explore with one finger. Double tap to hear more.
          Two-finger tap to repeat.
        </p>
        <button
          className="ts-t-btn ts-t-start-list"
          onClick={(e) => {
            e.stopPropagation();
            start(true);
          }}
        >
          Use the parts list instead
        </button>
        {caps}
        <p className="ts-t-start-net">Needs an internet connection to open. Downloaded PDFs work without one.</p>
      </div>
    );
  }

  const highlightId = currentId;

  return (
    <div className="ts-touch ts-t-explore-shell">
      <header className="ts-t-top">
        <h1 className="ts-t-title">{title}</h1>
        <button className="ts-t-btn ts-t-top-btn" onClick={() => setOverlay("list")}>
          Parts
        </button>
        <button
          ref={menuBtnRef}
          className="ts-t-btn ts-t-btn-primary ts-t-top-btn"
          aria-haspopup="dialog"
          aria-keyshortcuts="M"
          onClick={() => setOverlay("menu")}
        >
          Menu
        </button>
      </header>

      <div className="ts-t-stage">
        <ExploreCanvas
          ref={canvasRef}
          imageUrl={job.tactile.url}
          shapes={shapes}
          view={view}
          pxPerMm={job.tactile.px_per_mm || 5}
          highlightId={highlightId}
          showHighlights={settings.showHighlights}
          findTargetId={finding?.id ?? null}
          label={`${title} diagram, ${parts.length} parts. Touch to explore, or use arrow keys to step through parts, Enter to hear more, M for menu. Screen reader users: open the Parts list.`}
          onEnter={enterPart}
          onLeave={leave}
          onEdge={edgeTick}
          onDescribe={onDescribe}
          onRepeat={repeat}
          onThreeFingerTap={() => stopFind(true)}
          onFindDistance={onFindDistance}
          onFound={onFound}
          onKeyDown={onExploreKey}
        />
        {update && (
          <div className="ts-t-notice ts-t-update" role="status">
            <span>Your teacher updated this lesson.</span>
            <button className="ts-t-btn ts-t-btn-small ts-t-btn-primary" onClick={applyUpdate}>
              Update
            </button>
          </div>
        )}
        {notices.length > 0 && (
          <div className="ts-t-notice" role="note">
            <span>{notices.map((n) => n.text).join(" ")}</span>
            <button className="ts-t-btn ts-t-btn-small" onClick={() => setNotices([])}>
              OK
            </button>
          </div>
        )}
      </div>

      <div className={"ts-t-caption" + (caption.on ? " is-on" : "") + (finding ? " is-finding" : "")} aria-hidden="true">
        <div className="ts-t-caption-main">{caption.main}</div>
        <div className="ts-t-caption-sub">{caption.sub}</div>
      </div>
      <div className="ts-t-sr" aria-live="polite" aria-atomic="true">
        {live}
      </div>

      {overlay === "menu" || overlay === "find" ? (
        <Menu
          parts={parts}
          settings={settings}
          canVibrate={canVibrate()}
          finding={finding}
          startOnFind={overlay === "find"}
          onChange={setSettings}
          onClose={() => {
            setOverlay(null);
            requestAnimationFrame(() => menuBtnRef.current?.focus());
          }}
          onExit={exit}
          onList={() => setOverlay("list")}
          onFind={startFind}
          onStopFind={() => {
            stopFind(true);
            closeOverlay();
          }}
        />
      ) : null}
      {overlay === "list" && (
        <PartList
          title={job.title || "Diagram"}
          parts={parts}
          selectedId={currentId}
          onSelect={(part) => {
            setCurrentId(part.id);
            describe(part);
          }}
          onClose={closeOverlay}
        />
      )}
    </div>
  );
}
