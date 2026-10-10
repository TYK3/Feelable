// Cached /api/health (limits, retention, AI provider) shared by every page.
import { useEffect, useState } from "react";
import { getHealth } from "../api";
import type { Health } from "../types";

export const DEFAULT_LIMITS = { name: 60, description: 400, title: 80 };
export const AI_DATA_NOTE =
  "When AI descriptions are on, a smaller copy of your photo and the label text we read from it are sent to Google Gemini to name the parts and write the spoken descriptions. Turn this off to keep everything on our server.";

let pending: Promise<Health | null> | null = null;
let cached: Health | null | undefined;

export function loadHealth(): Promise<Health | null> {
  if (!pending) {
    pending = getHealth()
      .then((h) => (cached = h))
      .catch(() => {
        pending = null; // allow a later retry
        return (cached = null);
      });
  }
  return pending;
}

/** undefined = still checking, null = server unreachable */
export function useHealth(): Health | null | undefined {
  const [h, setH] = useState<Health | null | undefined>(cached);
  useEffect(() => {
    let alive = true;
    loadHealth().then((v) => alive && setH(v));
    return () => {
      alive = false;
    };
  }, []);
  return h;
}

export function limitsOf(h: Health | null | undefined) {
  return { ...DEFAULT_LIMITS, ...(h?.limits ?? {}) };
}

export function aiAvailable(h: Health | null | undefined): boolean | null {
  if (h === undefined) return null;
  return !!h?.ai?.provider;
}
