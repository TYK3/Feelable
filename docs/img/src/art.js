// Shared hand-drawn SVG pieces for Feelable README art. Plain JS, returns SVG strings.
const INK = "#17171C", BANANA = "#FFD23F", CREAM = "#FFFBF0";
const BR = { a:[1], b:[1,2], c:[1,4], h:[1,2,5], m:[1,3,4], n:[1,3,4,5], v:[1,2,3,6], w:[2,4,5,6], l:[1,2,3] };

// Braille cells; x,y = top-left, d = dot pitch
function braille(word, x, y, d = 7, r = 2.4, fill = INK) {
  let s = "";
  [...word].forEach((ch, i) => {
    const on = BR[ch] || [];
    for (let k = 1; k <= 6; k++) {
      const col = k <= 3 ? 0 : 1, row = (k - 1) % 3;
      const cx = x + i * d * 2.6 + col * d, cy = y + row * d;
      s += on.includes(k)
        ? `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}"/>`
        : `<circle cx="${cx}" cy="${cy}" r="${r * 0.45}" fill="${fill}" opacity=".12"/>`;
    }
  });
  return s;
}

// The logo mark at (x,y) with size s
function mark(x, y, s) {
  return `<g transform="translate(${x} ${y}) scale(${s / 64})">
    <rect x="2" y="2" width="60" height="60" rx="15" fill="${BANANA}"/>
    <path d="M19 18C27 14.5 38 20 49 15.5" fill="none" stroke="${INK}" stroke-width="7.5" stroke-linecap="round"/><path d="M29 35.5C34 33.5 38.5 36.5 44 34" fill="none" stroke="${INK}" stroke-width="7" stroke-linecap="round"/>
    <circle cx="19.5" cy="29" r="5" fill="${INK}"/><circle cx="19.5" cy="40" r="5" fill="${INK}"/><circle cx="19.5" cy="51" r="5" fill="${INK}"/></g>`;
}

// Shared <defs>: textures + raised-line shadow
function defs() {
  return `<defs>
    <pattern id="tx-dots" width="9" height="9" patternUnits="userSpaceOnUse"><circle cx="4.5" cy="4.5" r="1.9" fill="${INK}"/></pattern>
    <pattern id="tx-lines" width="11" height="11" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="2.4" height="11" fill="${INK}"/></pattern>
    <pattern id="tx-grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M0 5H10M5 0V10" stroke="${INK}" stroke-width="1.4"/></pattern>
    <filter id="raised" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="1.6" stdDeviation="0.9" flood-color="#5A4300" flood-opacity=".35"/>
    </filter>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="14" stdDeviation="16" flood-color="#8A6A00" flood-opacity=".18"/>
    </filter>
    <filter id="soft-sm" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="6" stdDeviation="7" flood-color="#8A6A00" flood-opacity=".22"/>
    </filter>
    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="#FFE27A" stop-opacity=".85"/>
      <stop offset=".55" stop-color="#FFE9A0" stop-opacity=".35"/>
      <stop offset="1" stop-color="#FFF1BF" stop-opacity="0"/>
    </radialGradient>
  </defs>`;
}

// A tactile plant cell (as on swell paper), drawn in a 400x270 box at origin.
// opts.highlight: "nucleus" fills it yellow (touch mode). opts.labels: braille keys.
function plantCell(opts = {}) {
  const hl = opts.highlight;
  const stroke = opts.stroke || INK;
  const raised = opts.flat ? "" : `filter="url(#raised)"`;
  let s = `<g ${raised}>`;
  // cell wall (double outline)
  s += `<rect x="6" y="6" width="388" height="258" rx="46" fill="none" stroke="${stroke}" stroke-width="7"/>`;
  s += `<rect x="20" y="20" width="360" height="230" rx="36" fill="none" stroke="${stroke}" stroke-width="3.5"/>`;
  // vacuole (line texture)
  s += `<path d="M150 70 C 210 52, 300 60, 318 108 C 334 150, 300 200, 236 204 C 170 208, 128 178, 126 136 C 124 100, 130 78, 150 70Z" fill="url(#tx-lines)" stroke="${stroke}" stroke-width="4" opacity="${hl ? .35 : 1}"/>`;
  // nucleus (dot texture) + nucleolus
  const nf = hl === "nucleus" ? BANANA : "url(#tx-dots)";
  s += `<circle cx="84" cy="96" r="40" fill="${CREAM}"/>`;
  s += `<circle cx="84" cy="96" r="40" fill="${nf}" stroke="${stroke}" stroke-width="5"/>`;
  s += `<circle cx="84" cy="96" r="12" fill="${stroke}"/>`;
  // chloroplasts
  const chl = [[74,196,-20],[338,72,30],[352,188,-35],[196,232,8],[292,232,-10]];
  for (const [x,y,a] of chl) {
    s += `<g transform="translate(${x} ${y}) rotate(${a})" opacity="${hl ? .35 : 1}"><ellipse rx="25" ry="12" fill="url(#tx-grid)" stroke="${stroke}" stroke-width="3.5"/></g>`;
  }
  s += `</g>`;
  if (opts.labels) {
    const pad = (x, y, w) => `<rect x="${x - 7}" y="${y - 7}" width="${w}" height="27" rx="6" fill="#FFFFFF"/>`;
    s += pad(64, 152, 22) + braille("n", 68, 155, 6.5, 2.3);
    s += pad(206, 118, 22) + braille("v", 210, 121, 6.5, 2.3);
    s += pad(296, 98, 39) + braille("ch", 300, 101, 6.5, 2.3);
  }
  return s;
}

// The same cell as a colourful textbook illustration with printed labels.
function textbookCell() {
  let s = `<rect x="6" y="6" width="388" height="258" rx="46" fill="#CFEFB8" stroke="#5E9E3A" stroke-width="7"/>`;
  s += `<rect x="20" y="20" width="360" height="230" rx="36" fill="#E7F7D6" stroke="#8CC26A" stroke-width="3"/>`;
  s += `<path d="M150 70 C 210 52, 300 60, 318 108 C 334 150, 300 200, 236 204 C 170 208, 128 178, 126 136 C 124 100, 130 78, 150 70Z" fill="#BFE0F7" stroke="#5DA3D6" stroke-width="3"/>`;
  s += `<circle cx="84" cy="96" r="40" fill="#D9C2F0" stroke="#8E62C4" stroke-width="3.5"/><circle cx="84" cy="96" r="12" fill="#8E62C4"/>`;
  for (const [x,y,a] of [[74,196,-20],[338,72,30],[352,188,-35],[196,232,8],[292,232,-10]])
    s += `<g transform="translate(${x} ${y}) rotate(${a})"><ellipse rx="25" ry="12" fill="#5BB04A" stroke="#3B8A30" stroke-width="2.5"/><path d="M-14 0H14M-8 -6V6M0 -7V7M8 -6V6" stroke="#2F7326" stroke-width="1.6"/></g>`;
  return s;
}

// Wireframe-ish 3D relief tile of the cell (for the "3D print" card). Origin = tile centre.
function reliefTile() {
  const slab = "#FFF4CC", side = "#E2C35A", top = "#FFD23F", wall = "#B88E0A";
  const tf = "matrix(0.82 0 0.34 0.5 0 0)"; // oblique top view
  const rect = (fill) => `<rect x="-230" y="-155" width="460" height="310" rx="18" fill="${fill}"/>`;
  let s = "";
  for (let k = 18; k > 0; k -= 1.5) s += `<g transform="translate(0 ${k})"><g transform="${tf}">${rect(side)}</g></g>`;
  s += `<g transform="${tf}">${rect(slab)}</g>`;
  const cell = (stroke) => `<g transform="translate(-200 -135)">${plantCell({ flat: true, stroke })
    .replace(/url\(#tx-[a-z]+\)/g, "none").replace(/fill="#FFFBF0"/g, 'fill="none"').replace(/fill="#17171C"|fill="${INK}"/g, `fill="${stroke}"`)}</g>`;
  for (let k = 8; k >= 1; k -= 1) s += `<g transform="translate(0 ${-k * 1.2 + 9.6})"><g transform="${tf}">${cell(wall)}</g></g>`;
  s += `<g transform="${tf}">${cell(top)}</g>`;
  return s;
}

// Friendly flat child's hand, index finger pointing UP, fingertip at (0,0).
function hand(skin = "#C98B62", shade = "#B07450", sleeve = "#6A9BF4", cuff = "#4F7FE0") {
  return `<g>
    <!-- sleeve -->
    <path d="M-34 150 C -36 200, -40 280, -48 700 L 104 700 C 96 280, 90 200, 88 150 Z" fill="${sleeve}"/>
    <rect x="-38" y="138" width="130" height="30" rx="14" fill="${cuff}"/>
    <!-- palm with curled fingers -->
    <path d="M-14 62 C -24 80, -30 104, -26 128 C -22 148, 0 156, 30 156 C 62 156, 82 146, 84 120 C 86 100, 84 84, 76 74 C 70 66, 58 64, 50 70 C 46 60, 36 56, 26 60 C 22 56, 14 56, 10 60 Z" fill="${skin}"/>
    <!-- curled finger knuckle lines -->
    <path d="M14 66 C 18 74, 18 82, 14 90" fill="none" stroke="${shade}" stroke-width="3.5" stroke-linecap="round"/>
    <path d="M30 64 C 36 72, 38 84, 32 94" fill="none" stroke="${shade}" stroke-width="3.5" stroke-linecap="round"/>
    <path d="M52 72 C 58 82, 58 92, 52 100" fill="none" stroke="${shade}" stroke-width="3.5" stroke-linecap="round"/>
    <!-- index finger -->
    <rect x="-13" y="0" width="26" height="92" rx="13" fill="${skin}"/>
    <path d="M-7 8 C -7 3, 7 3, 7 8 L 7 16 C 7 20, -7 20, -7 16 Z" fill="#F3D2BC" opacity=".9"/>
    <path d="M-8 52 H 4" stroke="${shade}" stroke-width="3" stroke-linecap="round" opacity=".7"/>
    <!-- thumb across the front -->
    <path d="M-22 120 L 22 104" stroke="${shade}" stroke-width="27" stroke-linecap="round"/>
    <path d="M-22 120 L 22 104" stroke="${skin}" stroke-width="22" stroke-linecap="round"/>
    <path d="M14 102 C 20 99, 26 101, 28 106" fill="none" stroke="#F3D2BC" stroke-width="3" stroke-linecap="round" opacity=".8"/>
  </g>`;
}

// Phone in touch mode. Box 170x340 at origin.
function phone(opts = {}) {
  const w = 170, h = 340;
  return `<g>
    <rect x="0" y="0" width="${w}" height="${h}" rx="28" fill="${INK}"/>
    <rect x="8" y="8" width="${w - 16}" height="${h - 16}" rx="22" fill="#24242B"/>
    <rect x="${w / 2 - 22}" y="15" width="44" height="9" rx="4.5" fill="${INK}"/>
    <text x="22" y="56" font-family="Inter" font-weight="700" font-size="11" fill="#9B98A6" letter-spacing=".04em">TOUCH MODE</text>
    <g transform="translate(17 72) scale(0.34)">${plantCell({ highlight: "nucleus", stroke: "#F4EFE0", flat: true })}</g>
    <rect x="18" y="186" width="${w - 36}" height="56" rx="14" fill="${BANANA}"/>
    <text x="32" y="210" font-family="Inter" font-weight="800" font-size="17" fill="${INK}">Nucleus</text>
    <text x="32" y="229" font-family="Inter" font-weight="500" font-size="9.6" fill="#3A2E00">the cell's control centre</text>
    <g transform="translate(30 262)">
      ${[10,18,26,14,22,30,16,24,12,20,28,14,8].map((v,i)=>`<rect x="${i*8.6}" y="${15 - v/2}" width="4.2" height="${v}" rx="2.1" fill="${BANANA}" opacity="${0.55 + (i%3)*0.15}"/>`).join("")}
    </g>
    <rect x="${w / 2 - 30}" y="${h - 22}" width="60" height="4" rx="2" fill="#55545E"/>
  </g>`;
}

function waves(x, y, color = INK, flip = false, n = 3) {
  let s = "";
  for (let i = 0; i < n; i++) {
    const r = 18 + i * 15;
    const a0 = flip ? 140 : -40, a1 = flip ? 220 : 40;
    const p = (a) => [x + r * Math.cos(a * Math.PI / 180), y + r * Math.sin(a * Math.PI / 180)];
    const [x0, y0] = p(a0), [x1, y1] = p(a1);
    s += `<path d="M${x0.toFixed(1)} ${y0.toFixed(1)} A ${r} ${r} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}" fill="none" stroke="${color}" stroke-width="5" stroke-linecap="round" opacity="${1 - i * 0.22}"/>`;
  }
  return s;
}

function heartShape(x, y, s, fill) {
  return `<path transform="translate(${x} ${y}) scale(${s})" d="M0 6 C -2 -1, -12 -1, -12 6 C -12 12, -4 16, 0 20 C 4 16, 12 12, 12 6 C 12 -1, 2 -1, 0 6Z" fill="${fill}"/>`;
}
