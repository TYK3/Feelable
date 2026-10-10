// Tiny hash router with a navigation guard. OWNER: Frontend.
// Routes: #/ (home) · #/job/{id}[?t=TOKEN] (teacher review) · #/touch/{id} (student touch mode)
//
// The guard (F05) intercepts every route change — links, navigate(), and the browser Back/Forward
// buttons — while a page has unsaved work. A blocked change is undone in the address bar and handed to
// the guard, which can later call proceed() to continue.
import { useSyncExternalStore } from "react";

export type Route =
  | { name: "home" }
  | { name: "job"; id: string; token: string | null }
  | { name: "touch"; id: string }
  | { name: "notfound"; path: string };

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#/, "");
  const [pathPart, query = ""] = raw.split("?");
  const path = pathPart.replace(/\/+$/, "") || "/";
  if (path === "/" || path === "") return { name: "home" };
  const m = path.match(/^\/(job|touch)\/([^/?#]+)$/);
  if (m) {
    const id = decodeURIComponent(m[2]);
    if (m[1] === "touch") return { name: "touch", id };
    const t = new URLSearchParams(query).get("t");
    return { name: "job", id, token: t && /^[0-9a-f]{16,64}$/i.test(t) ? t : null };
  }
  return { name: "notfound", path };
}

/** Same route ignoring the ?t= token (so stripping the token doesn't remount a page). */
export function routeKey(r: Route): string {
  return r.name === "job" || r.name === "touch" ? `${r.name}:${r.id}` : r.name;
}

// ---------------------------------------------------------------- guard
type Guard = (target: string) => boolean; // return true to allow
let guard: Guard | null = null;
let current = typeof window !== "undefined" ? window.location.hash : "";
const listeners = new Set<() => void>();

export function setNavGuard(g: Guard | null) {
  guard = g;
}

function onHashChange() {
  const next = window.location.hash;
  if (next === current) return;
  if (guard && parseHash(next).name !== "notfound" && routeKey(parseHash(next)) !== routeKey(parseHash(current))) {
    if (!guard(next)) {
      // Undo the change in the address bar without a hashchange event (pushState restores a forward entry
      // after Back, so the history stack stays sensible).
      history.pushState(null, "", current || "#/");
      return;
    }
  }
  current = next;
  listeners.forEach((l) => l());
}

if (typeof window !== "undefined") window.addEventListener("hashchange", onHashChange);

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
const getHash = () => current;

export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribe, getHash, () => "");
  return parseHash(hash);
}

function norm(path: string) {
  return path.startsWith("#") ? path : "#" + (path.startsWith("/") ? path : "/" + path);
}

/** navigate("#/job/abc") or navigate("/job/abc") — goes through the guard. */
export function navigate(path: string) {
  const target = norm(path);
  if (window.location.hash !== target) window.location.hash = target;
}

/** Leave without asking the guard again (after the user chose Save / Keep draft / Discard). */
export function proceed(path: string) {
  guard = null;
  navigate(path);
}

/** Replace the current URL (e.g. strip ?t=) without a route change. */
export function replaceHash(path: string) {
  const target = norm(path);
  history.replaceState(null, "", target);
  current = target;
}

/** Absolute URL for a hash route, e.g. for sharing with a student's phone. */
export function absoluteUrl(path: string) {
  return `${window.location.origin}${window.location.pathname}${norm(path)}`;
}
