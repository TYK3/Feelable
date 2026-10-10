"""Realistic `Analysis` fixtures for the Tactile team's tests (no dependency on the Vision team).

OWNER: Tactile team.

* plant_cell_analysis()  - textbook plant cell (adapted from docs/prototype_pipeline.py): cell wall
                           (filled cell), cytoplasm container, vacuole, nucleus, 5 chloroplasts,
                           3 tiny mitochondria, an excluded "ribosomes" part, + a "light" arrow.
* grid_analysis(n=10)    - n touching-ish rectangles (tests >8 warning + texture colouring).
* single_part_analysis() - one ellipse.
* lineart_analysis()     - no parts, only linework (a triangle with an axis and an arrow).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.types import Analysis, Label, Part, WORKING_LONG_SIDE

_FONT = "/usr/share/fonts/opentype/inter/Inter-%s.otf"


def _font(size, weight="Regular"):
    try:
        return ImageFont.truetype(_FONT % weight, size)
    except OSError:
        return ImageFont.load_default()


def make_part(pid: str, name: str, mask: np.ndarray, color, description: str = "", include: bool = True) -> Part:
    mask = (mask > 0).astype(np.uint8) * 255
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cs, -1, 255, -1)
    polys = [[(int(x), int(y)) for x, y in cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)] for c in cs]
    ys, xs = np.nonzero(filled)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, 0, 0)
    return Part(id=pid, name=name, description=description or f"{name.capitalize()}.", key="",
                color=tuple(int(c) for c in color), mask=filled, polygons=polys, bbox=bbox,
                area=int((filled > 0).sum()), count=len(cs), include=include)


# --------------------------------------------------------------------------- plant cell
PAL = {
    "wall": (40, 100, 40), "cyto": (200, 235, 180), "vacuole": (150, 200, 240),
    "nucleus": (170, 120, 200), "nucleolus": (120, 70, 160),
    "chloro": (90, 190, 80), "grana": (120, 210, 105),
    "mito": (240, 150, 60), "ribo": (90, 90, 90),
}
CHLORO = [((245, 235), (40, 22), 20), ((300, 540), (42, 22), -10), ((520, 545), (40, 21), 8),
          ((626, 395), (22, 40), 0), ((214, 430), (21, 38), 8)]
# deliberately tiny (≈ 4-6 mm on the page) so the enlargement rule fires
MITO = [((400, 212), (9, 5), 0), ((640, 482), (8, 5), 30), ((362, 262), (8, 5), -20)]
_W, _H = 1100, 720
ARROW = [[(60, 100), (152, 150)], [(166, 158), (140, 155), (151, 135)]]   # "light" arrow, ends outside the cell


def _plant_cell_image(labels: bool = True):
    rng = np.random.default_rng(3)
    im = Image.new("RGB", (_W, _H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 28), "Fig. 3.2   A typical plant cell", font=_font(26, "SemiBold"), fill=(30, 30, 30))
    d.rounded_rectangle((176, 166, 696, 586), 64, fill=(225, 225, 225))
    d.rounded_rectangle((170, 160, 690, 580), 64, fill=PAL["wall"])
    d.rounded_rectangle((184, 174, 676, 566), 52, fill=PAL["cyto"])
    t = np.linspace(0, 2 * np.pi, 120)
    r = 1 + 0.06 * np.sin(3 * t) + 0.04 * np.cos(5 * t)
    d.polygon([(420 + 170 * r[i] * np.cos(t[i]), 400 + 105 * r[i] * np.sin(t[i])) for i in range(len(t))],
              fill=PAL["vacuole"])
    d.ellipse((535, 185, 635, 285), fill=PAL["nucleus"])
    d.ellipse((568, 218, 604, 254), fill=PAL["nucleolus"])
    for (cx, cy), (a, b), ang in CHLORO:
        lay = Image.new("L", (_W, _H), 0)
        ImageDraw.Draw(lay).ellipse((cx - a, cy - b, cx + a, cy + b), fill=255)
        im.paste(PAL["chloro"], mask=lay.rotate(ang, center=(cx, cy)))
    for (cx, cy), (a, b), ang in MITO:
        lay = Image.new("L", (_W, _H), 0)
        ImageDraw.Draw(lay).ellipse((cx - a, cy - b, cx + a, cy + b), fill=255)
        im.paste(PAL["mito"], mask=lay.rotate(ang, center=(cx, cy)))
    arr = np.array(im)
    for _ in range(140):
        x, y = rng.integers(195, 665), rng.integers(185, 555)
        if tuple(arr[y, x]) == PAL["cyto"]:
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=PAL["ribo"])
    # "light" arrow pointing at the top-left chloroplast (meaningful linework)
    for poly in ARROW:
        d.line(poly[:2], fill=(20, 20, 20), width=3) if len(poly) == 2 else d.polygon(poly, fill=(20, 20, 20))
    if not labels:
        return im
    d.text((40, 70), "light", font=_font(18, "Medium"), fill=(20, 20, 20))
    lab = _font(19, "Medium")
    for text, tx, ty, px, py in [
        ("Cell wall", 735, 168, 688, 200), ("Nucleus", 735, 228, 630, 238),
        ("Mitochondrion", 735, 300, 655, 478), ("Chloroplast", 735, 380, 648, 395),
        ("Cytoplasm", 735, 520, 650, 548), ("Vacuole", 735, 450, 560, 420)]:
        d.line((tx - 6, ty + 12, px, py), fill=(40, 40, 40), width=1)
        d.text((tx, ty), text, font=lab, fill=(20, 20, 20))
    return im


def _arrow_mask(shape, scale):
    m = np.zeros(shape, np.uint8)
    p = lambda x, y: (int(round(x * scale)), int(round(y * scale)))
    cv2.line(m, p(*ARROW[0][0]), p(*ARROW[0][1]), 255, max(1, int(round(3 * scale))))
    cv2.fillPoly(m, [np.array([p(*q) for q in ARROW[1]], np.int32)], 255)
    return m


def plant_cell_analysis() -> Analysis:
    s = WORKING_LONG_SIDE / _W
    W, H = WORKING_LONG_SIDE, int(round(_H * s))
    img = np.array(_plant_cell_image(True).resize((W, H), Image.NEAREST))
    # segment the label-free rendering (Vision removes text + leader lines before finding parts)
    a = np.array(_plant_cell_image(False).resize((W, H), Image.NEAREST)).astype(np.int32)
    keys = ["wall", "cyto", "vacuole", "nucleus", "nucleolus", "chloro", "grana", "mito"]
    cols = np.array([PAL[k] for k in keys])
    dist = np.linalg.norm(a[:, :, None, :] - cols[None, None], axis=3)
    lab = dist.argmin(2)
    ok = dist.min(2) < 45
    m = lambda *ks: (np.isin(lab, [keys.index(k) for k in ks]) & ok).astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def clean(mask, min_area):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
        n, cc, st, _ = cv2.connectedComponentsWithStats(mask)
        out = np.zeros_like(mask)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_area:
                out[cc == i] = 255
        cs, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, 255, -1)
        return out

    cell = clean(m(*keys), 5000)
    cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cs, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cell = np.zeros_like(cell)
    cv2.drawContours(cell, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    cyto = clean(m("cyto", "vacuole", "nucleus", "nucleolus", "chloro", "grana", "mito"), 5000)
    vac = clean(m("vacuole"), 3000)
    nuc = clean(m("nucleus", "nucleolus"), 1500)
    chl = clean(m("chloro", "grana"), 500)
    mito = (m("mito") > 0).astype(np.uint8) * 255
    ribo = np.zeros_like(cell)
    rib = (np.linalg.norm(a - np.array(PAL["ribo"]), axis=2) < 30).astype(np.uint8) * 255
    ribo[:] = rib
    parts = [
        make_part("p1", "cell wall", cell, PAL["wall"], "Cell wall. A tough outer layer that gives the cell its shape."),
        make_part("p2", "cytoplasm", cyto, PAL["cyto"], "Cytoplasm. Jelly that fills the cell."),
        make_part("p3", "vacuole", vac, PAL["vacuole"], "Vacuole. A large sac of water that keeps the cell firm."),
        make_part("p4", "nucleus", nuc, PAL["nucleus"], "Nucleus. Controls the cell and holds its DNA."),
        make_part("p5", "chloroplast", chl, PAL["chloro"], "Chloroplast. Uses sunlight to make food. There are five."),
        make_part("p6", "mitochondrion", mito, PAL["mito"], "Mitochondrion. Releases energy for the cell."),
        make_part("p7", "ribosomes", ribo, PAL["ribo"], "Ribosomes. Tiny dots that make proteins.", include=False),
    ]
    parts.sort(key=lambda p: -p.area)
    labels = [Label("Fig. 3.2 A typical plant cell", (37, 30, 400, 62), 0.95, None, "title")]
    for p in parts:
        labels.append(Label(p.name.capitalize(), (802, 180, 950, 205), 0.9, p.id, "part_label"))
    return Analysis(width=W, height=H, image=img, title="plant cell", diagram_type="biology",
                    parts=parts, labels=labels, linework=_arrow_mask((H, W), s),
                    dropped=["140 small dots (too small to feel)", "drop shadow"], timings={"analyze": 0.0})


# --------------------------------------------------------------------------- grid of rectangles
GRID_NAMES = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "moon", "sun",
              "ceres", "pluto"]


def grid_analysis(n: int = 10) -> Analysis:
    W, H = 1200, 700
    img = np.full((H, W, 3), 255, np.uint8)
    cols = 5
    rows = (n + cols - 1) // cols
    cw, ch, gap = 200, 260, 8
    x0 = (W - cols * cw - (cols - 1) * gap) // 2
    y0 = (H - rows * ch - (rows - 1) * gap) // 2
    rng = np.random.default_rng(7)
    parts = []
    for i in range(n):
        r, c = divmod(i, cols)
        m = np.zeros((H, W), np.uint8)
        x, y = x0 + c * (cw + gap), y0 + r * (ch + gap)
        cv2.rectangle(m, (x, y), (x + cw, y + ch), 255, -1)
        col = tuple(int(v) for v in rng.integers(60, 230, 3))
        img[m > 0] = col
        parts.append(make_part(f"p{i + 1}", GRID_NAMES[i % len(GRID_NAMES)], m, col))
    return Analysis(width=W, height=H, image=img, title="planets", diagram_type="other", parts=parts,
                    labels=[], linework=np.zeros((H, W), np.uint8))


# --------------------------------------------------------------------------- single part
def single_part_analysis() -> Analysis:
    W, H = 1200, 800
    img = np.full((H, W, 3), 255, np.uint8)
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (600, 400), (300, 200), 15, 0, 360, 255, -1)
    img[m > 0] = (230, 120, 90)
    return Analysis(width=W, height=H, image=img, title="Red blood cell 2", diagram_type="biology",
                    parts=[make_part("p1", "red blood cell", m, (230, 120, 90))], labels=[],
                    linework=np.zeros((H, W), np.uint8))


# --------------------------------------------------------------------------- line art only
def lineart_analysis() -> Analysis:
    W, H = 1200, 900
    lw = np.zeros((H, W), np.uint8)
    cv2.line(lw, (100, 800), (1100, 800), 255, 2)            # x axis
    cv2.line(lw, (100, 800), (100, 100), 255, 2)             # y axis
    cv2.polylines(lw, [np.array([(250, 700), (650, 200), (1000, 700)], np.int32)], True, 255, 3)
    cv2.arrowedLine(lw, (700, 450), (900, 300), 255, 2, tipLength=0.15)
    cv2.circle(lw, (1150, 860), 1, 255, -1)                   # speck (must be removed)
    img = np.full((H, W, 3), 255, np.uint8)
    img[lw > 0] = 0
    return Analysis(width=W, height=H, image=img, title="triangle", diagram_type="other", parts=[],
                    labels=[], linework=lw)


ALL_FIXTURES = {
    "plant_cell": plant_cell_analysis,
    "grid": grid_analysis,
    "single": single_part_analysis,
    "lineart": lineart_analysis,
}
