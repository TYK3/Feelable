// Teacher drafts persisted per job in localStorage (F05): they survive refresh, Back and New diagram.
export type PartDraft = { name: string; description: string; include: boolean };
export type Draft = { title?: string; parts: Record<string, PartDraft>; baseVersion?: number; savedAt: number };

const key = (id: string) => `ts.draft.${id}`;

export function loadDraft(id: string): Draft | null {
  try {
    const raw = localStorage.getItem(key(id));
    if (!raw) return null;
    const d = JSON.parse(raw) as Draft;
    if (!d || typeof d !== "object" || typeof d.parts !== "object") return null;
    return d;
  } catch {
    return null;
  }
}

export function saveDraft(id: string, d: Draft | null) {
  try {
    if (!d || (d.title === undefined && Object.keys(d.parts).length === 0)) localStorage.removeItem(key(id));
    else localStorage.setItem(key(id), JSON.stringify(d));
  } catch {
    /* storage full / private mode: the draft lives in memory only */
  }
}
