// "Find a part" audio guide: a short beep that gets faster and higher as the finger
// gets closer to the target. Built on WebAudio; create it inside a user gesture.

declare global {
  interface Window {
    __feelableBeeps?: number;
    webkitAudioContext?: typeof AudioContext;
  }
}

const FAR_MM = 70; // distances beyond this all sound "far"

export class Finder {
  private ctx: AudioContext | null = null;
  private distMm: number | null = null;
  private timer: number | null = null;
  private running = false;

  static supported(): boolean {
    return typeof window !== "undefined" && !!(window.AudioContext || window.webkitAudioContext);
  }

  /** Must be called from a user gesture the first time (unlocks audio on iOS/Chrome). */
  unlock(): void {
    if (!Finder.supported()) return;
    if (!this.ctx) {
      const Ctor = window.AudioContext || window.webkitAudioContext!;
      try {
        this.ctx = new Ctor();
      } catch {
        this.ctx = null;
      }
    }
    if (this.ctx && this.ctx.state === "suspended") void this.ctx.resume();
  }

  start(): void {
    this.unlock();
    this.running = true;
    this.distMm = null;
    this.schedule(0);
  }

  stop(): void {
    this.running = false;
    this.distMm = null;
    if (this.timer != null) window.clearTimeout(this.timer);
    this.timer = null;
  }

  /** Distance from finger to target in mm (0 = on it), or null when no finger is down. */
  update(distMm: number | null): void {
    const wasIdle = this.distMm == null;
    this.distMm = distMm;
    // Respond immediately when a finger lands instead of waiting for the idle tick.
    if (this.running && wasIdle && distMm != null) this.schedule(0);
  }

  private closeness(): number {
    const d = this.distMm ?? FAR_MM;
    return 1 - Math.min(d, FAR_MM) / FAR_MM; // 0 far .. 1 on target
  }

  private schedule(delay: number): void {
    if (this.timer != null) window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => this.tick(), delay);
  }

  private tick(): void {
    this.timer = null;
    if (!this.running) return;
    if (this.distMm == null) {
      this.schedule(150); // idle: wait for a finger
      return;
    }
    const c = this.closeness();
    this.beep(320 + 680 * c * c, 0.045);
    const interval = 70 + (1 - c) * 560; // 630 ms far -> 70 ms close
    this.schedule(interval);
  }

  private beep(freq: number, dur: number, at = 0, vol = 0.18): void {
    if (typeof window !== "undefined") window.__feelableBeeps = (window.__feelableBeeps ?? 0) + 1;
    const ctx = this.ctx;
    if (!ctx) return;
    const t0 = ctx.currentTime + at;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, t0);
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(vol, t0 + 0.008);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
  }

  /** Rising three-note chime (C-E-G). */
  chime(): void {
    this.unlock();
    [523.25, 659.25, 783.99].forEach((f, i) => this.beep(f, 0.22, i * 0.11, 0.22));
  }

  dispose(): void {
    this.stop();
    if (this.ctx) void this.ctx.close().catch(() => undefined);
    this.ctx = null;
  }
}
