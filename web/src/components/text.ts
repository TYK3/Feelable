/** "human heart" -> "Human heart". Leaves already-capitalised text alone. */
export function sentenceCase(s: string): string {
  const t = (s ?? "").trim();
  return t ? t[0].toUpperCase() + t.slice(1) : t;
}
