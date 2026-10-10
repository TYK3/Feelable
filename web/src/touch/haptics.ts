// Vibration helpers. Silently do nothing where navigator.vibrate is missing (iOS Safari, desktop).

declare global {
  interface Window {
    __feelableVibrations?: (number | number[])[];
  }
}

export function canVibrate(): boolean {
  return typeof navigator !== "undefined" && typeof navigator.vibrate === "function";
}

let enabled = true;
let lastEdge = 0;

export function setVibrationEnabled(on: boolean): void {
  enabled = on;
  if (!on && canVibrate()) navigator.vibrate(0);
}

export function vibrate(pattern: number | number[]): void {
  if (!enabled) return;
  if (typeof window !== "undefined") {
    const log = (window.__feelableVibrations ??= []);
    log.push(pattern);
    if (log.length > 100) log.splice(0, log.length - 100);
  }
  if (!canVibrate()) return;
  try {
    navigator.vibrate(pattern);
  } catch {
    /* some browsers throw without a user gesture */
  }
}

/** Short tick when the finger crosses an outline; throttled to one per 80 ms. */
export function edgeTick(now = performance.now()): void {
  if (now - lastEdge < 80) return;
  lastEdge = now;
  vibrate(12);
}

export const PATTERN_ENTER = [25];
export const PATTERN_FOUND = [60, 40, 60, 40, 140];
