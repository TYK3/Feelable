// Thin wrapper around the Web Speech API with a "most human voice first" picker.
// Every utterance is also recorded on window.__feelableLastSpoken / window.__feelableSpoken
// (test hook, harmless in production).
//
// Browsers ship very different voices. The old picker preferred *local* en-US voices, which on
// Windows means the robotic "Microsoft David/Zira Desktop". We now rank voices so natural/neural
// ones win: Edge "Microsoft … Online (Natural)", Chrome "Google UK/US English", Apple
// "Enhanced/Premium/Siri" voices, Android "Google" network voices. The teacher/student can also
// pick a voice by name (saved in localStorage).

declare global {
  interface Window {
    __feelableLastSpoken?: string;
    __feelableSpoken?: string[];
  }
}

const VOICE_KEY = "ts.voice";
const DEFAULT_RATE = 0.95; // a touch slower than default sounds calmer and clearer

let voice: SpeechSynthesisVoice | null = null;
let voicesHooked = false;
const listeners = new Set<() => void>();

export function isSpeechSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window && typeof SpeechSynthesisUtterance !== "undefined";
}

/** Higher = more natural-sounding. Exported for tests and the voice picker. */
export function voiceScore(v: Pick<SpeechSynthesisVoice, "name" | "lang" | "localService" | "default">): number {
  const n = v.name.toLowerCase();
  let s = 0;
  if (/^en([-_]|$)/i.test(v.lang)) s += 100;
  else return -1000; // descriptions are English
  if (/en[-_](gb|us|au|ie|ca|nz)/i.test(v.lang)) s += 5;
  if (/natural|neural|wavenet|journey|studio/.test(n)) s += 60; // Edge/Azure & Google neural voices
  if (/premium|enhanced|siri/.test(n)) s += 50; // Apple high-quality voices
  if (/online/.test(n)) s += 25;
  if (/^google/.test(n)) s += 35; // Chrome's Google voices are far smoother than OS defaults
  if (/(samantha|daniel|karen|moira|serena|aria|jenny|libby|sonia|ava|allison|susan)/.test(n)) s += 15;
  if (/desktop|espeak|compact|robot|novelty|zarvox|whisper|bad news|bells|boing|bubbles|cellos|trinoids|albert|fred|junior|ralph/.test(n)) s -= 80;
  if (/(david|zira|mark)\b/.test(n) && /microsoft/.test(n) && !/online|natural/.test(n)) s -= 40; // Windows SAPI voices
  return s;
}

export function englishVoices(): SpeechSynthesisVoice[] {
  if (!isSpeechSupported()) return [];
  return window.speechSynthesis
    .getVoices()
    .filter((v) => voiceScore(v) > -1000)
    .sort((a, b) => voiceScore(b) - voiceScore(a));
}

function savedVoiceName(): string | null {
  try {
    return localStorage.getItem(VOICE_KEY);
  } catch {
    return null;
  }
}

function pickVoice(): void {
  if (!isSpeechSupported()) return;
  const ranked = englishVoices();
  const all = window.speechSynthesis.getVoices();
  if (!all.length) return;
  const wanted = savedVoiceName();
  voice = (wanted && all.find((v) => v.name === wanted)) || ranked[0] || all.find((v) => v.default) || all[0] || null;
  listeners.forEach((fn) => fn());
}

export function initSpeech(): void {
  if (!isSpeechSupported() || voicesHooked) return;
  voicesHooked = true;
  pickVoice();
  window.speechSynthesis.addEventListener?.("voiceschanged", pickVoice);
}

/** Name of the voice in use (for the picker), or null. */
export function currentVoiceName(): string | null {
  if (!voice) pickVoice();
  return voice?.name ?? null;
}

export function setVoiceByName(name: string | null): void {
  try {
    if (name) localStorage.setItem(VOICE_KEY, name);
    else localStorage.removeItem(VOICE_KEY);
  } catch {
    /* private mode: choice lasts for this page only */
  }
  pickVoice();
}

/** Re-render hook for components that show the voice list (voices load asynchronously). */
export function onVoicesChanged(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function record(text: string): void {
  if (typeof window === "undefined") return;
  window.__feelableLastSpoken = text;
  const log = (window.__feelableSpoken ??= []);
  log.push(text);
  if (log.length > 100) log.splice(0, log.length - 100);
}

export interface SpeakOptions {
  rate?: number;
  /** Cancel whatever is being said first (default true). */
  interrupt?: boolean;
}

/** Speak like a person: tidy punctuation so the voice pauses naturally instead of running on. */
export function humanize(text: string): string {
  return text
    .replace(/\s+/g, " ")
    .replace(/\s*—\s*/g, ", ")
    .replace(/\s*;\s*/g, ", ")
    .replace(/\bp(\d+)\b/gi, "part $1")
    .trim();
}

// ---------------------------------------------------------------- natural (neural) voice clips
// When the server has a Gemini key, touch mode prefetches a natural-voice recording of each part's
// name and description. speak() plays the clip if it's ready, otherwise the browser voice speaks
// instantly — so exploration never waits on the network.
const NATURAL_KEY = "ts.naturalVoice";
const clips = new Map<string, string>(); // normalised text -> object URL
const clipTexts: string[] = []; // original texts, in fetch order (for "Test voice" in natural mode)
let audioEl: HTMLAudioElement | null = null;
let naturalOff = false;

const norm = (t: string) => t.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

export function naturalVoiceEnabled(): boolean {
  try {
    return localStorage.getItem(NATURAL_KEY) !== "0";
  } catch {
    return true;
  }
}

export function setNaturalVoice(on: boolean): void {
  try {
    localStorage.setItem(NATURAL_KEY, on ? "1" : "0");
  } catch {
    /* ignore */
  }
  listeners.forEach((fn) => fn());
}

/** A text that has a natural-voice clip ready, so "Test voice" plays what students will really hear. */
export function naturalSample(): string | null {
  return clipTexts[0] ?? null;
}

export function naturalClipCount(): number {
  return clips.size;
}

/** Fetch natural-voice clips for these texts (one at a time, politely). Stops if the server has none. */
export async function prefetchNatural(jobId: string, texts: string[], signal?: AbortSignal): Promise<void> {
  if (naturalOff || typeof fetch === "undefined") return;
  for (const raw of texts) {
    const t = raw.trim();
    if (!t || clips.has(norm(t))) continue;
    if (signal?.aborted) return;
    try {
      const r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t }),
        signal,
      });
      if (r.status === 503) {
        naturalOff = r.headers.get("content-type")?.includes("json") ? (await r.json())?.code === "ai_off" : false;
        if (naturalOff) return;
        continue;
      }
      if (!r.ok) continue;
      const blob = await r.blob();
      clips.set(norm(t), URL.createObjectURL(blob));
      clipTexts.push(t);
      listeners.forEach((fn) => fn());
    } catch {
      return; // offline / aborted: browser voice keeps working
    }
  }
}

function playClip(t: string, rate: number): boolean {
  if (!naturalVoiceEnabled()) return false;
  const url = clips.get(norm(t));
  if (!url) return false;
  if (isSpeechSupported()) window.speechSynthesis.cancel();
  audioEl?.pause();
  audioEl = new Audio(url);
  audioEl.playbackRate = Math.max(0.7, Math.min(1.5, rate));
  audioEl.play().catch(() => {
    /* autoplay blocked: nothing else to do; captions still show the text */
  });
  return true;
}

export function speak(text: string, opts: SpeakOptions = {}): void {
  const t = text.trim();
  if (!t) return;
  record(t);
  if (opts.interrupt !== false) audioEl?.pause();
  if (playClip(t, opts.rate ?? 1)) return;
  if (!isSpeechSupported()) return;
  const synth = window.speechSynthesis;
  if (opts.interrupt !== false) synth.cancel();
  const u = new SpeechSynthesisUtterance(humanize(t));
  if (!voice) pickVoice();
  if (voice) u.voice = voice;
  u.lang = voice?.lang || "en-GB";
  // The UI's 1.0× maps to a calm default; the slider still scales it.
  u.rate = (opts.rate ?? 1) * DEFAULT_RATE;
  u.pitch = 1;
  u.volume = 1;
  // Chrome sometimes leaves the queue paused after a cancel(); resume defensively.
  synth.resume();
  synth.speak(u);
}

export function stopSpeech(): void {
  audioEl?.pause();
  if (isSpeechSupported()) window.speechSynthesis.cancel();
}
