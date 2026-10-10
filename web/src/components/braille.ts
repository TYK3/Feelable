// Uncontracted (grade 1) braille letters a-z. Dots are numbered 1-6:
//   1 4
//   2 5
//   3 6
// Unicode braille = U+2800 + bitmask (bit i = dot i+1).

export const LETTER_DOTS: Record<string, number[]> = {
  a: [1], b: [1, 2], c: [1, 4], d: [1, 4, 5], e: [1, 5], f: [1, 2, 4], g: [1, 2, 4, 5],
  h: [1, 2, 5], i: [2, 4], j: [2, 4, 5], k: [1, 3], l: [1, 2, 3], m: [1, 3, 4],
  n: [1, 3, 4, 5], o: [1, 3, 5], p: [1, 2, 3, 4], q: [1, 2, 3, 4, 5], r: [1, 2, 3, 5],
  s: [2, 3, 4], t: [2, 3, 4, 5], u: [1, 3, 6], v: [1, 2, 3, 6], w: [2, 4, 5, 6],
  x: [1, 3, 4, 6], y: [1, 3, 4, 5, 6], z: [1, 3, 5, 6],
};

export function dotsOf(ch: string): number[] {
  return LETTER_DOTS[ch.toLowerCase()] ?? [];
}

export function toUnicodeBraille(text: string): string {
  let out = "";
  for (const ch of text.toLowerCase()) {
    if (ch === " ") {
      out += "⠀";
      continue;
    }
    const dots = LETTER_DOTS[ch];
    if (!dots) continue;
    const mask = dots.reduce((m, d) => m | (1 << (d - 1)), 0);
    out += String.fromCharCode(0x2800 + mask);
  }
  return out;
}
