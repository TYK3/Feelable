#!/usr/bin/env python3
"""Feelable sample diagrams (Content team).

Deterministic generator for every image in ``samples/`` plus ``samples/manifest.json``.

    python3 samples/make_samples.py            # regenerate everything
    python3 samples/make_samples.py heart      # regenerate only some ids

Style: flat textbook illustrations - crisp flat fills with a thin darker outline per part,
Inter Medium labels with 1-2 px leader lines, "Fig. X.Y" title top-left, one grey caption line
at the bottom, plus a little realistic clutter (ribosome dots, highlights, drop shadows) so the
pipeline's clutter removal is exercised.  Everything is drawn at 2x and downsampled (anti-aliased
edges, like a real scan/screenshot).  No randomness without a fixed seed.
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/opentype/inter/Inter-%s.otf"
S = 2  # supersampling factor

INK = (28, 28, 34)        # label text
LEADER = (45, 45, 52)     # leader lines
CAPTION = (118, 118, 126)
PAGE = (255, 255, 255)
OFFWHITE = (253, 252, 248)

_font_cache: dict = {}


def font(size: float, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    key = (round(size * S), weight)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT % weight, key[0])
    return _font_cache[key]


def darker(c, f=0.62):
    return tuple(int(v * f) for v in c)


def lighter(c, f=0.45):
    return tuple(int(v + (255 - v) * f) for v in c)


def ellipse_pts(cx, cy, a, b, ang=0.0, n=96, wobble=None):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones_like(t) if wobble is None else wobble(t)
    x, y = a * r * np.cos(t), b * r * np.sin(t)
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    return list(zip(cx + x * ca - y * sa, cy + x * sa + y * ca))


def geom_polys(g):
    """Yield exterior/interior coordinate lists of a shapely (Multi)Polygon."""
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]


class Canvas:
    """A PIL canvas in final-pixel coordinates, rendered at S x and downsampled."""

    def __init__(self, w, h, bg=PAGE):
        self.w, self.h = w, h
        self.im = Image.new("RGB", (w * S, h * S), bg)
        self.d = ImageDraw.Draw(self.im)

    # -- primitives (coordinates in final px) ------------------------------------------
    def _p(self, pts):
        return [(x * S, y * S) for x, y in pts]

    def poly(self, pts, fill=None, outline=None, width=2):
        pts = self._p(pts)
        if fill is not None:
            self.d.polygon(pts, fill=fill)
        if outline is not None:
            self.d.line(pts + [pts[0]], fill=outline, width=int(width * S), joint="curve")

    def shape(self, g, fill=None, outline=None, width=2):
        """Draw a shapely polygon (holes respected) with an optional outline."""
        for p in geom_polys(g):
            mask = Image.new("L", self.im.size, 0)
            md = ImageDraw.Draw(mask)
            md.polygon(self._p(p.exterior.coords), fill=255)
            for hole in p.interiors:
                md.polygon(self._p(hole.coords), fill=0)
            if fill is not None:
                self.im.paste(fill, mask=mask)
            if outline is not None:
                for ring in [p.exterior, *p.interiors]:
                    self.d.line(self._p(ring.coords), fill=outline, width=int(width * S), joint="curve")

    def ellipse(self, cx, cy, a, b, ang=0.0, fill=None, outline=None, width=2, wobble=None):
        self.poly(ellipse_pts(cx, cy, a, b, ang, wobble=wobble), fill, outline, width)

    def line(self, pts, fill=LEADER, width=1.5):
        self.d.line(self._p(pts), fill=fill, width=max(1, int(round(width * S))), joint="curve")

    def dot(self, x, y, r, fill):
        self.d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=fill)

    def text(self, x, y, s, size=22, weight="Medium", fill=INK, anchor="la"):
        self.d.text((x * S, y * S), s, font=font(size, weight), fill=fill, anchor=anchor)

    def textlen(self, s, size=22, weight="Medium"):
        return self.d.textlength(s, font=font(size, weight)) / S

    # -- textbook furniture -----------------------------------------------------------
    def title(self, s, x=60, y=44):
        self.text(x, y, s, size=30, weight="SemiBold", fill=(24, 24, 30))

    def caption(self, s, y=None):
        self.text(60, y if y is not None else self.h - 58, s, size=18, weight="Regular", fill=CAPTION)

    def label(self, s, tx, ty, px, py, side="right", size=22, elbow=None):
        """Label text at (tx, ty) [left edge, vertical middle]; leader to point (px, py).

        side='right' -> label sits right of the drawing, leader leaves from the text's left edge.
        side='left'  -> tx is the RIGHT edge of the text, leader leaves from its right edge.
        side='below'/'above' -> text centred at tx, leader leaves from the top/bottom edge.
        elbow: optional intermediate point for a bent leader.
        """
        w = self.spaced_len(s, size)
        if side == "right":
            start, anchor = (tx - 8, ty), "lm"
        elif side == "left":
            start, anchor = (tx + 8, ty), "rm"
        elif side == "below":   # label below the point: leader leaves the text's top edge
            start, anchor = (tx, ty - size * 0.75), "mm"
        else:                   # 'above': label above the point: leader leaves the bottom edge
            start, anchor = (tx, ty + size * 0.75), "mm"
        pts = [start] + ([elbow] if elbow else []) + [(px, py)]
        self.line(pts, LEADER, 1.5)
        self.dot(px, py, 3.2, LEADER)
        x0 = {"lm": tx, "rm": tx - w, "mm": tx - w / 2}[anchor]
        self.spaced_text(x0, ty, s, size)
        return w

    # multi-word labels get a slightly wider word gap so OCR keeps the words apart
    WORD_GAP = 1.6  # x the normal space width

    def spaced_len(self, s, size, weight="Medium"):
        words = s.split(" ")
        gap = self.textlen(" ", size, weight) * self.WORD_GAP
        return sum(self.textlen(w_, size, weight) for w_ in words) + gap * (len(words) - 1)

    def spaced_text(self, x, y, s, size, weight="Medium", fill=INK):
        gap = self.textlen(" ", size, weight) * self.WORD_GAP
        for w_ in s.split(" "):
            self.text(x, y, w_, size=size, weight=weight, fill=fill, anchor="lm")
            x += self.textlen(w_, size, weight) + gap

    def shadow(self, g, dx=8, dy=10, blur=7, colour=(0, 0, 0), alpha=40):
        """Soft drop shadow under shapely geometry g (clutter)."""
        mask = Image.new("L", self.im.size, 0)
        md = ImageDraw.Draw(mask)
        for p in geom_polys(g):
            md.polygon([((x + dx) * S, (y + dy) * S) for x, y in p.exterior.coords], fill=alpha)
        mask = mask.filter(ImageFilter.GaussianBlur(blur * S))
        self.im.paste(colour, mask=mask)

    def final(self) -> Image.Image:
        return self.im.resize((self.w, self.h), Image.LANCZOS)


def blob(cx, cy, a, b, ang=0.0, wob=((3, 0.05, 0.0), (5, 0.03, 1.0)), n=160):
    def f(t):
        r = np.ones_like(t)
        for k, amp, ph in wob:
            r = r + amp * np.sin(k * t + ph)
        return r
    return Polygon(ellipse_pts(cx, cy, a, b, ang, n=n, wobble=f))


# =====================================================================================
# 1. Plant cell
# =====================================================================================
PC = {
    "wall": (46, 112, 58), "cyto": (214, 238, 196), "vacuole": (156, 204, 238),
    "nucleus": (176, 132, 206), "nucleolus": (120, 78, 160), "chloro": (86, 176, 74),
    "grana": (52, 128, 50), "mito": (238, 150, 66), "ribo": (96, 104, 96),
}


def draw_plant_cell(c: Canvas, ox=0.0, oy=0.0, k=1.0, seed=7):
    """Draw the plant cell. Returns dict of named anchor points (canvas coords)."""
    T = lambda x, y: (ox + x * k, oy + y * k)
    rng = np.random.default_rng(seed)

    def rr(x0, y0, x1, y1, r):
        return box(*T(x0 + r, y0 + r), *T(x1 - r, y1 - r)).buffer(r * k, resolution=32)

    outer = rr(330, 170, 1030, 790, 70)
    inner = rr(352, 192, 1008, 768, 52)
    c.shadow(outer, dx=10 * k, dy=12 * k, blur=8 * k)
    c.shape(outer, PC["wall"], darker(PC["wall"], 0.6), 2)
    c.shape(inner, PC["cyto"], darker(PC["wall"], 0.85), 2)

    vac = blob(*T(660, 520), 215 * k, 140 * k, wob=((3, 0.06, 0.3), (5, 0.035, 1.2), (2, 0.03, 0.0)))
    c.shape(vac, PC["vacuole"], darker(PC["vacuole"], 0.7), 2)
    c.ellipse(*T(560, 455), 46 * k, 15 * k, -18, fill=lighter(PC["vacuole"], 0.55))  # highlight
    c.ellipse(*T(618, 432), 10 * k, 6 * k, -18, fill=lighter(PC["vacuole"], 0.55))

    nuc = Point(*T(845, 300)).buffer(68 * k, resolution=48)
    c.shape(nuc, PC["nucleus"], darker(PC["nucleus"], 0.6), 2)
    c.ellipse(*T(858, 292), 24 * k, 22 * k, fill=PC["nucleolus"])
    c.ellipse(*T(822, 270), 16 * k, 8 * k, -30, fill=lighter(PC["nucleus"], 0.4))  # highlight

    chloros = [((470, 285), 50, 24, 18), ((420, 520), 48, 23, 80), ((520, 700), 50, 24, -8),
               ((880, 690), 50, 24, 12), ((935, 470), 48, 23, 95)]
    for (x, y), a, b, ang in chloros:
        cx, cy = T(x, y)
        c.ellipse(cx, cy, a * k, b * k, ang, fill=PC["chloro"], outline=darker(PC["chloro"], 0.6), width=2)
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        for j in range(-3, 4):  # grana stacks
            gx, gy = cx + j * 12 * k * ca, cy + j * 12 * k * sa
            nx, ny = -sa * 9 * k, ca * 9 * k
            c.line([(gx - nx, gy - ny), (gx + nx, gy + ny)], PC["grana"], 3.2 * k)

    mitos = [((640, 250), 36, 15, 8), ((700, 720), 34, 14, -20), ((960, 590), 34, 14, 60)]
    for (x, y), a, b, ang in mitos:
        cx, cy = T(x, y)
        c.ellipse(cx, cy, a * k, b * k, ang, fill=PC["mito"], outline=darker(PC["mito"], 0.6), width=2)
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        pts = []
        for j, tt in enumerate(np.linspace(-a * 0.75, a * 0.75, 9)):  # cristae zig-zag
            off = (b * 0.55 if j % 2 else -b * 0.55) * k
            pts.append((cx + tt * k * ca - off * sa, cy + tt * k * sa + off * ca))
        c.line(pts, darker(PC["mito"], 0.75), 1.6 * k)

    # ribosomes: tiny dots in free cytoplasm only
    taken = unary_union([vac.buffer(8 * k), nuc.buffer(8 * k)]
                        + [Point(*T(x, y)).buffer((a + 10) * k) for (x, y), a, b, ang in chloros]
                        + [Point(*T(x, y)).buffer((a + 8) * k) for (x, y), a, b, ang in mitos])
    free = inner.buffer(-10 * k).difference(taken)
    placed = 0
    minx, miny, maxx, maxy = free.bounds
    while placed < 170:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if free.contains(Point(x, y)):
            c.dot(x, y, 2.6 * k, PC["ribo"])
            placed += 1
    return {
        "wall": T(1022, 230), "cyto": T(780, 750), "vacuole": T(470, 560), "nucleus": T(905, 335),
        "chloro": T(905, 470), "mito": T(640, 262), "ribo": T(385, 400), "chloro_l": T(487, 290),
        "mito_r": T(972, 600),
    }


def plant_cell_labels(c: Canvas, a, dx=0.0, dy=0.0):
    L, R = 1080 + dx, 280 + dx  # right column (left edge) / left column (right edge)
    c.label("Cell wall", L, 214 + dy, *a["wall"])
    c.label("Nucleus", L, 330 + dy, *a["nucleus"])
    c.label("Chloroplast", L, 446 + dy, *a["chloro"])
    c.label("Mitochondrion", L, 600 + dy, *a["mito_r"])
    c.label("Cytoplasm", L, 760 + dy, *a["cyto"])
    c.label("Ribosomes", R, 400 + dy, *a["ribo"], side="left")
    c.label("Vacuole", R, 560 + dy, *a["vacuole"], side="left")


def make_plant_cell():
    c = Canvas(1400, 950)
    c.title("Fig. 3.2   A typical plant cell")
    a = draw_plant_cell(c)
    plant_cell_labels(c, a)
    c.caption("The large central vacuole is filled with cell sap. Ribosomes are shown as dots. Not to scale.")
    return c.final()


# =====================================================================================
# 2. Animal cell
# =====================================================================================
AC = {
    "membrane": (204, 96, 120), "cyto": (252, 226, 214), "nucleus": (120, 150, 214),
    "nucleolus": (60, 86, 160), "mito": (232, 160, 60), "golgi": (90, 170, 150),
    "ribo": (110, 90, 96),
}


def make_animal_cell():
    c = Canvas(1400, 950)
    c.title("Fig. 3.1   A typical animal cell")
    cell = blob(690, 495, 330, 285, ang=-8, wob=((3, 0.045, 0.4), (4, 0.03, 2.0), (2, 0.04, 1.1), (7, 0.012, 0.0)), n=240)
    c.shadow(cell, 10, 12, 8)
    c.shape(cell, AC["membrane"], darker(AC["membrane"], 0.65), 2)
    cyto = cell.buffer(-11)
    c.shape(cyto, AC["cyto"], darker(AC["membrane"], 0.85), 1.5)

    nuc = blob(640, 470, 108, 92, ang=15, wob=((3, 0.03, 0.0), (2, 0.025, 1.0)))
    c.shape(nuc, AC["nucleus"], darker(AC["nucleus"], 0.6), 2)
    c.ellipse(660, 480, 34, 30, fill=AC["nucleolus"], outline=darker(AC["nucleolus"], 0.7), width=1.5)
    c.ellipse(606, 432, 26, 10, -25, fill=lighter(AC["nucleus"], 0.45))  # highlight
    rng = np.random.default_rng(11)
    for _ in range(26):  # chromatin flecks (clutter)
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(45, 85)
        x, y = 640 + rad * math.cos(ang), 470 + rad * 0.85 * math.sin(ang)
        c.dot(x, y, 2.2, darker(AC["nucleus"], 0.75))

    mitos = [((500, 300), 48, 20, 25), ((880, 650), 46, 19, -30), ((480, 650), 44, 19, 70)]
    for (cx, cy), a, b, ang in mitos:
        c.ellipse(cx, cy, a, b, ang, fill=AC["mito"], outline=darker(AC["mito"], 0.6), width=2)
        c.ellipse(cx, cy, a - 7, b - 7, ang, fill=None, outline=darker(AC["mito"], 0.8), width=1.2)
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        pts = []
        for j, tt in enumerate(np.linspace(-a * 0.7, a * 0.7, 9)):
            off = b * 0.5 if j % 2 else -b * 0.5
            pts.append((cx + tt * ca - off * sa, cy + tt * sa + off * ca))
        c.line(pts, darker(AC["mito"], 0.7), 1.6)

    # golgi body: stack of curved cisternae
    gcx, gcy = 860, 400
    golgi_parts = []
    for i in range(4):
        r0 = 70 + i * 17
        arc = [(gcx + r0 * math.cos(t), gcy + r0 * math.sin(t)) for t in np.linspace(math.radians(-150), math.radians(-30), 40)]
        golgi_parts.append(LineString(arc).buffer(6.0, cap_style=1))
    gol = unary_union(golgi_parts)
    # rotate + shift the stack so it sits inside the cytoplasm
    gol = affinity.translate(affinity.rotate(gol, 20, origin=(gcx, gcy)), 30, 120)
    for p in geom_polys(gol):
        c.shape(p, AC["golgi"], darker(AC["golgi"], 0.6), 1.5)
    gb = gol.bounds
    for (vx, vy) in [(gb[0] + 10, gb[3] + 18), (gb[0] + 34, gb[3] + 30), (gb[2] - 6, gb[1] - 18)]:  # vesicles
        c.dot(vx, vy, 6, AC["golgi"])
    golgi_pt = max(geom_polys(gol), key=lambda q: q.area).representative_point().coords[0]
    ext = cell.exterior
    mem_pt = ext.interpolate(ext.project(Point(1000, 240)))
    mem_pt = (mem_pt.x - 4, mem_pt.y + 4)

    taken = unary_union([nuc.buffer(10), gol.buffer(16), Point(700, 640).buffer(10)]
                        + [Point(x, y).buffer(a + 8) for (x, y), a, b, ang in mitos])
    free = cyto.buffer(-10).difference(taken)
    minx, miny, maxx, maxy = free.bounds
    placed = 0
    while placed < 160:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if free.contains(Point(x, y)):
            c.dot(x, y, 2.6, AC["ribo"])
            placed += 1

    L = 1100
    c.label("Cell membrane", L, 250, *mem_pt)
    c.label("Golgi body", L, 380, *golgi_pt)
    c.label("Mitochondrion", L, 640, 905, 640)
    c.label("Cytoplasm", L, 760, 700, 640, elbow=(800, 760))
    R = 290
    c.label("Nucleus", R, 330, 570, 405, side="left")
    c.label("Nucleolus", R, 470, 645, 482, side="left", elbow=(540, 470))
    c.label("Ribosomes", R, 600, 420, 560, side="left")
    c.caption("Animal cells have no cell wall and no chloroplasts. Not to scale.")
    return c.final()


# =====================================================================================
# 3. Heart
# =====================================================================================
HC = {
    "ra": (140, 186, 236), "rv": (64, 118, 196), "vc": (44, 66, 140), "pa": (110, 104, 196),
    "la": (246, 164, 150), "lv": (214, 76, 70), "ao": (160, 36, 48), "muscle": (240, 210, 186), "septum": (186, 132, 96),
}


def chaikin(pts, it=4, closed=True):
    P = np.array(pts, float)
    for _ in range(it):
        Q = np.roll(P, -1, axis=0) if closed else P[1:]
        A = P if closed else P[:-1]
        R = np.empty((len(A) * 2, 2))
        R[0::2] = 0.75 * A + 0.25 * Q
        R[1::2] = 0.25 * A + 0.75 * Q
        P = R if closed else np.vstack([P[:1], R, P[-1:]])
    return [tuple(p) for p in P]


def tube(pts, r, cap=2):
    return LineString(chaikin(pts, 4, closed=False)).buffer(r, cap_style=cap, join_style=1, resolution=24)


def make_heart():
    c = Canvas(1400, 950)
    c.title("Fig. 7.4   The human heart (front view)")
    heart = Polygon(chaikin([
        (468, 470), (478, 360), (540, 305), (615, 312), (668, 352), (705, 372), (745, 345),
        (835, 305), (915, 318), (962, 375), (968, 455), (978, 545), (958, 645), (900, 740),
        (820, 815), (762, 850), (712, 826), (610, 742), (528, 645), (482, 560)], 5))

    # vessels behind the heart: pulmonary artery (back), aorta, vena cava
    pa = unary_union([tube([(655, 420), (650, 300), (640, 262)], 25, cap=1),
                      tube([(642, 268), (600, 240), (520, 228), (440, 226)], 20),
                      tube([(650, 268), (700, 250), (760, 248)], 18)])
    aorta = unary_union([tube([(752, 420), (752, 260), (770, 190), (830, 160), (895, 175), (935, 230), (938, 330), (925, 480)], 29),
                         tube([(790, 172), (778, 112)], 11), tube([(840, 160), (842, 100)], 11), tube([(890, 172), (908, 112)], 11)])
    svc = tube([(545, 118), (546, 260), (552, 360)], 27)
    c.shadow(heart, 10, 12, 9)
    c.shape(pa, HC["pa"], darker(HC["pa"], 0.6), 2)
    c.shape(aorta, HC["ao"], darker(HC["ao"], 0.6), 2)
    c.shape(svc, HC["vc"], darker(HC["vc"], 0.6), 2)
    c.shape(heart, HC["muscle"], darker(HC["muscle"], 0.6), 2)

    # chambers: inset heart split by septum + AV groove
    inner = heart.buffer(-20)
    septum = LineString(chaikin([(712, 330), (712, 480), (722, 640), (752, 860)], 4, closed=False))
    av = LineString(chaikin([(400, 505), (560, 490), (712, 486), (860, 478), (1020, 470)], 4, closed=False))
    walls = unary_union([septum.buffer(13), av.buffer(13)])
    named = {}
    for p in geom_polys(inner.difference(walls)):
        x, y = p.representative_point().coords[0]
        key = ("r" if x < 715 else "l") + ("a" if y < 486 else "v")
        named[key] = unary_union([named[key], p]) if key in named else p
    for key in ("ra", "la", "rv", "lv"):
        g = named[key].buffer(-24).buffer(24)
        c.shape(g, HC[key], darker(HC[key], 0.62), 2)
    # septum: a distinct darker band down the middle, outlined so it is separate from the wall
    sep_band = septum.buffer(8, cap_style=2).intersection(inner.buffer(-3))
    c.shape(sep_band, HC["septum"], darker(HC["septum"], 0.5), 2)
    sp = septum.interpolate(septum.project(Point(730, 700)))
    sep_pt = (sp.x, sp.y)
    c.ellipse(598, 590, 46, 12, 62, fill=lighter(HC["rv"], 0.3))   # highlights (clutter)
    c.ellipse(830, 585, 44, 12, -58, fill=lighter(HC["lv"], 0.3))
    c.ellipse(560, 380, 30, 9, -30, fill=lighter(HC["ra"], 0.4))
    for x, y in [(660, 640), (630, 700), (790, 700), (770, 650)]:  # trabeculae flecks (clutter)
        c.dot(x, y, 2.5, darker(HC["rv"] if x < 715 else HC["lv"], 0.75))

    R = 330
    c.label("Vena cava", R, 150, 545, 150, side="left")
    c.label("Pulmonary artery", R, 240, 470, 227, side="left")
    c.label("Right atrium", R, 410, 560, 415, side="left")
    c.label("Right ventricle", R, 620, 625, 625, side="left")
    L = 1060
    c.label("Aorta", L, 190, 930, 210)
    c.label("Left atrium", L, 380, 860, 395)
    c.label("Left ventricle", L, 620, 845, 625)
    c.label("Heart wall", L, 500, 961, 500)
    c.label("Septum", L, 800, *sep_pt, elbow=(900, 800))
    c.caption("Blue: deoxygenated blood (right side).  Red: oxygenated blood (left side).  Simplified.")
    return c.final()


# =====================================================================================
# 4. Water cycle
# =====================================================================================
def arrow(c: Canvas, pts, width=4.0, head=20, col=(22, 22, 28)):
    """Curved arrow through pts (smoothed with a Catmull-Rom spline), head at the end."""
    P = np.array(pts, float)
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        for t in np.linspace(0, 1, 20, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    out = np.array(out)
    end = out[-1]
    dvec = out[-1] - out[-4]
    dvec /= np.linalg.norm(dvec)
    nvec = np.array([-dvec[1], dvec[0]])
    base = end - dvec * head
    shaft = [tuple(p) for p in out if np.dot(p - base, dvec) < 0] + [tuple(base + dvec * 2)]
    c.line(shaft, col, width)
    c.poly([tuple(end), tuple(base + nvec * head * 0.55), tuple(base - nvec * head * 0.55)], fill=col)


def cloud_shape(cx, cy, s=1.0):
    circles = [(-70, 10, 42), (-25, -18, 52), (30, -25, 48), (75, 5, 40), (10, 18, 46), (-45, 25, 36), (55, 25, 34)]
    return unary_union([Point(cx + dx * s, cy + dy * s).buffer(r * s, resolution=32) for dx, dy, r in circles])


def make_water_cycle():
    c = Canvas(1400, 950, PAGE)
    c.title("Fig. 9.1   The water cycle")
    sky = box(60, 100, 1340, 860)
    c.shape(sky, (226, 240, 251), (150, 170, 190), 2)
    # sea (left) and land with mountain (right)
    sea = Polygon([(60, 700), (500, 700)] + chaikin([(500, 700), (540, 740), (520, 800), (580, 860)], 3, closed=False) + [(580, 860), (60, 860)])
    land = Polygon([(470, 700)] + [(560 + i * 10, 690 - 18 * math.sin(i / 9)) for i in range(0, 79)]
                   + [(1340, 640), (1340, 860), (470, 860)])
    mountain = Polygon(chaikin([(700, 860), (880, 560), (960, 470), (1050, 398), (1060, 398),
                                (1150, 490), (1240, 590), (1420, 860)], 2)).intersection(box(0, 0, 2000, 716))
    snow = Polygon(chaikin([(1000, 438), (1050, 398), (1060, 398), (1108, 446), (1085, 462), (1062, 446),
                            (1042, 468), (1022, 452)], 1))
    c.shape(sea, (70, 140, 210), darker((70, 140, 210), 0.65), 2)
    for i in range(6):  # wave squiggles (clutter)
        y = 740 + i * 20
        x0 = 100 + (i % 2) * 40
        pts = [(x0 + t, y + 4 * math.sin(t / 9)) for t in range(0, 70, 3)]
        c.line(pts, lighter((70, 140, 210), 0.35), 2)
        pts = [(x0 + 220 + t, y + 4 * math.sin(t / 9)) for t in range(0, 60, 3)]
        c.line(pts, lighter((70, 140, 210), 0.35), 2)
    c.shape(land.difference(sea), (130, 186, 92), darker((130, 186, 92), 0.6), 2)
    c.shape(mountain, (150, 128, 108), darker((150, 128, 108), 0.6), 2)
    c.shape(snow, (250, 250, 250), darker((150, 128, 108), 0.6), 1.5)
    # sun
    c.ellipse(220, 230, 62, 62, fill=(250, 200, 50), outline=(205, 150, 20), width=2)
    for k in range(12):
        ang = k * math.pi / 6
        c.line([(220 + 78 * math.cos(ang), 230 + 78 * math.sin(ang)), (220 + 100 * math.cos(ang), 230 + 100 * math.sin(ang))],
               (225, 165, 25), 4)
    c.ellipse(200, 210, 20, 9, -30, fill=(253, 228, 140))  # highlight
    # clouds (white cloud + grey rain cloud)
    cl1 = cloud_shape(620, 230, 1.0)
    cl2 = cloud_shape(1020, 240, 1.15)
    for g, col in ((cl1, (250, 251, 253)), (cl2, (208, 214, 224))):
        c.shadow(g, 6, 8, 5, alpha=30)
        c.shape(g, col, (110, 120, 136), 2)
    c.ellipse(560, 205, 28, 9, -10, fill=(255, 255, 255))
    # rain
    rng = np.random.default_rng(5)
    for i in range(22):
        x = 935 + (i % 11) * 18 + rng.uniform(-4, 4)
        y = 322 + (i // 11) * 40 + rng.uniform(0, 14)
        c.line([(x, y), (x - 6, y + 20)], (60, 120, 200), 3)
    # process arrows (linework)
    arrow(c, [(300, 690), (340, 560), (420, 410), (520, 300)])                   # evaporation
    arrow(c, [(735, 190), (800, 160), (880, 168)])                              # condensation
    arrow(c, [(905, 320), (892, 400), (900, 470)])                              # precipitation
    arrow(c, [(1180, 690), (1060, 755), (820, 800), (600, 780)], width=4)        # collection
    # labels: process names (no leader) + object labels with leaders
    c.text(405, 545, "Evaporation", 24, "SemiBold", anchor="lm")
    c.text(810, 128, "Condensation", 24, "SemiBold", anchor="mm")
    c.text(872, 400, "Precipitation", 24, "SemiBold", anchor="rm")
    c.text(900, 832, "Collection", 24, "SemiBold", anchor="mm")
    c.label("Sun", 252, 372, 233, 280, side="below")
    c.label("Cloud", 640, 345, 640, 270, side="below")
    c.label("Mountain", 1240, 452, 1165, 560, side="above")
    c.label("Sea", 140, 650, 200, 725, side="above")
    c.caption("Heat from the Sun drives the cycle. Water vapour condenses into clouds and falls as rain.", y=890)
    return c.final()


# =====================================================================================
# 5. Bar chart
# =====================================================================================
def make_bar_chart():
    c = Canvas(1400, 950, PAGE)
    c.title("Fig. 5.1   Favourite fruit in class 7B")
    data = [("Apple", 8, (213, 94, 0)), ("Banana", 11, (240, 214, 40)), ("Orange", 6, (230, 159, 0)),
            ("Grapes", 4, (150, 100, 190)), ("Kiwi", 7, (0, 158, 115))]
    x0, y0, x1, y1 = 230, 170, 1260, 760  # plot area (y0 top)
    ymax = 12
    sy = (y1 - y0) / ymax
    for v in range(0, ymax + 1, 2):  # gridlines (clutter)
        y = y1 - v * sy
        if v:
            c.line([(x0, y), (x1, y)], (222, 222, 228), 1.2)
        c.line([(x0 - 10, y), (x0, y)], INK, 2)
        c.text(x0 - 20, y, str(v), 20, "Regular", anchor="rm")
    n = len(data)
    slot = (x1 - x0) / n
    bw = slot * 0.56
    for i, (name, v, col) in enumerate(data):
        cx = x0 + slot * (i + 0.5)
        top = y1 - v * sy
        g = box(cx - bw / 2, top, cx + bw / 2, y1 - 1)
        c.shadow(g, 6, 0, 3, alpha=35)
        c.shape(g, col, darker(col, 0.6), 2)
        c.text(cx, top - 22, str(v), 22, "SemiBold", anchor="mm")
        c.line([(cx, y1), (cx, y1 + 10)], INK, 2)
        c.text(cx, y1 + 34, name, 22, "Medium", anchor="mm")
    c.line([(x0, y0 - 10), (x0, y1), (x1 + 10, y1)], INK, 2.5)
    c.text((x0 + x1) / 2, y1 + 84, "Fruit", 22, "SemiBold", anchor="mm")
    # rotated y-axis title
    tmp = Image.new("RGBA", (420 * S, 40 * S), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((210 * S, 20 * S), "Number of students", font=font(22, "SemiBold"), fill=INK, anchor="mm")
    tmp = tmp.rotate(90, expand=True)
    c.im.paste(tmp, (int(110 * S), int(((y0 + y1) / 2 - 210) * S)), tmp)
    c.caption("Survey of 36 students, September. Each student chose one fruit.", y=890)
    return c.final()


# =====================================================================================
# 6. Series circuit (black line art)
# =====================================================================================
def make_circuit():
    c = Canvas(1400, 950, PAGE)
    K = (0, 0, 0)
    c.title("Fig. 12.3   A simple series circuit")
    W = 4.5
    xl, xr, yt, yb = 330, 1070, 250, 720
    # battery on top wire (centre 700), switch on left wire, bulb on right wire, resistor bottom
    bx = 700
    c.line([(xl, yt), (bx - 22, yt)], K, W)
    c.line([(bx + 22, yt), (xr, yt)], K, W)
    c.line([(bx - 22, yt - 55), (bx - 22, yt + 55)], K, 4.5)       # long plate (+)
    c.poly([(bx + 14, yt - 28), (bx + 30, yt - 28), (bx + 30, yt + 28), (bx + 14, yt + 28)], fill=K)  # short thick (-)
    c.text(bx - 52, yt - 50, "+", 26, "SemiBold", anchor="mm")
    c.text(bx + 58, yt - 50, "−", 26, "SemiBold", anchor="mm")
    # right wire with bulb (circle with cross)
    by, br = 485, 50
    c.line([(xr, yt), (xr, by - br)], K, W)
    c.line([(xr, by + br), (xr, yb)], K, W)
    c.ellipse(xr, by, br, br, fill=None, outline=K, width=W)
    d = br * 0.707
    c.line([(xr - d, by - d), (xr + d, by + d)], K, W)
    c.line([(xr - d, by + d), (xr + d, by - d)], K, W)
    # bottom wire with resistor (rectangle)
    rx0, rx1, rh = 610, 790, 30
    c.line([(xr, yb), (rx1, yb)], K, W)
    c.line([(rx0, yb), (xl, yb)], K, W)
    c.poly([(rx0, yb - rh), (rx1, yb - rh), (rx1, yb + rh), (rx0, yb + rh)], fill=None, outline=K, width=W)
    # left wire with closed switch
    sy0, sy1 = 430, 560
    c.line([(xl, yb), (xl, sy1)], K, W)
    c.line([(xl, sy0), (xl, yt)], K, W)
    c.ellipse(xl, sy1, 9, 9, fill=(255, 255, 255), outline=K, width=3.5)
    c.ellipse(xl, sy0, 9, 9, fill=(255, 255, 255), outline=K, width=3.5)
    c.line([(xl, sy1 - 9), (xl - 34, sy0 + 6)], K, W)  # lever resting on top contact
    # labels with leaders (outside the loop where possible)
    c.label("Battery", 700, 130, 700, 190, side="above")
    c.label("Switch", 210, 495, xl - 30, 495, side="left")
    c.label("Bulb", 1170, 485, xr + br + 6, 485, side="right")
    c.label("Resistor", 700, 820, 700, yb + rh + 4, side="below")
    c.label("Wire", 1170, 640, xr + 3, 640, side="right")
    c.caption("When the switch is closed, current flows through the bulb and the resistor in turn.", y=890)
    return c.final()


# =====================================================================================
# 7. Earth layers (globe with a wedge cut out)
# =====================================================================================
EC = {"ocean": (88, 150, 214), "land": (128, 180, 96), "crust": (140, 96, 64), "mantle": (214, 104, 64),
      "outer": (240, 166, 52), "inner": (252, 222, 92)}


def make_earth_layers():
    c = Canvas(1400, 950, PAGE)
    c.title("Fig. 10.2   Inside the Earth")
    cx, cy, R = 600, 510, 330
    globe = Point(cx, cy).buffer(R, resolution=96)

    def wedge(r, a0=-50, a1=40):
        pts = [(cx, cy)] + [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
                            for a in np.linspace(a0, a1, 80)]
        return Polygon(pts)

    cut = wedge(R + 2)
    c.shadow(globe, 12, 14, 10)
    shell = globe.difference(cut)
    c.shape(shell, EC["ocean"], darker(EC["ocean"], 0.6), 2)
    c.ellipse(420, 300, 80, 24, -40, fill=lighter(EC["ocean"], 0.3))  # highlight (clutter)
    # continents (clipped to the shell)
    for (x, y, a, b, ang, seed) in [(470, 380, 120, 80, 20, 0.3), (430, 630, 80, 130, -15, 1.4),
                                    (640, 750, 110, 50, 10, 2.2), (680, 230, 60, 30, 0, 0.9)]:
        g = blob(x, y, a, b, ang, wob=((3, 0.12, seed), (5, 0.08, seed * 2), (7, 0.04, 0.0))).intersection(shell.buffer(-4))
        c.shape(g, EC["land"], darker(EC["land"], 0.6), 1.5)
    # layers inside the cut
    radii = [("crust", R), ("mantle", R - 26), ("outer", 175), ("inner", 95)]
    for name, r in radii:
        c.shape(wedge(r), EC[name], darker(EC[name], 0.6), 2)
    # lava speckles in mantle (clutter)
    rng = np.random.default_rng(4)
    for _ in range(40):
        a = math.radians(rng.uniform(-45, 35))
        r = rng.uniform(190, 290)
        c.dot(cx + r * math.cos(a), cy + r * math.sin(a), 2.5, darker(EC["mantle"], 0.8))
    p = lambda r, a: (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
    L = 1060
    c.label("Crust", L, 250, *p(R - 12, -40))
    c.label("Mantle", L, 400, *p(250, -15))
    c.label("Outer core", L, 540, *p(135, 8))
    c.label("Inner core", L, 660, *p(55, 30))
    c.caption("Not to scale: the crust is really much thinner than shown here.", y=890)
    return c.final()


# =====================================================================================
# 8. Plant cell photographed on a textbook page
# =====================================================================================
BODY = [
    "All living things are made of cells. Plant cells have a cell wall made of cellulose,",
    "which keeps the cell rigid, and chloroplasts, which absorb light for photosynthesis.",
]
BODY2 = [
    "The vacuole stores water and dissolved substances. When it is full, it pushes on the cell",
    "wall and keeps the plant upright. Mitochondria release energy from glucose by respiration.",
]


def make_page():
    """A textbook page: heading, body text, the plant-cell figure, caption, more body text."""
    pw, ph = 1260, 1130
    c = Canvas(pw, ph, (250, 248, 240))
    c.text(70, 56, "3.2  Plant cells", 34, "Bold", fill=(30, 60, 110))
    for i, s in enumerate(BODY):
        c.text(70, 118 + i * 32, s, 21, "Regular", fill=(70, 70, 74))
    c.text(70, 214, "Fig. 3.2   A typical plant cell", 24, "SemiBold", fill=(24, 24, 30))
    dx, dy = -100, 90      # figure shifted so its left labels start at the page margin
    a = draw_plant_cell(c, ox=dx, oy=dy)
    plant_cell_labels(c, a, dx, dy)
    y = 790 + dy + 50
    c.text(70, y, "The large central vacuole is filled with cell sap. Not to scale.", 19, "Italic", fill=(110, 110, 116))
    for i, s in enumerate(BODY2):
        c.text(70, y + 56 + i * 32, s, 21, "Regular", fill=(70, 70, 74))
    c.line([(70, ph - 52), (pw - 70, ph - 52)], (200, 196, 186), 1)
    c.text(pw - 70, ph - 30, "47", 19, "Regular", fill=(110, 110, 116), anchor="rm")
    return c.final(), a


def wood(w, h, seed=21):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    n = cv2.resize(rng.normal(0, 1, (h // 30 + 2, w // 90 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    grain = np.sin((y + 22 * n + 6 * np.sin(x / 300)) / 9.0) * 0.5 + 0.5
    fine = cv2.resize(rng.normal(0, 1, (h // 2, w // 70)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    t = 0.65 * grain + 0.12 * fine
    base = np.array([150, 104, 66], np.float32)
    dark = np.array([112, 74, 44], np.float32)
    img = base[None, None] * (1 - t[..., None] * 0.6) + dark[None, None] * (t[..., None] * 0.6)
    # plank seams
    for sy in (230, 640, 1060):
        img[max(0, sy - 2):sy + 2] *= 0.55
    return np.clip(img, 0, 255).astype(np.uint8)


def make_plant_cell_photo():
    W, H = 1600, 1200
    page, _ = make_page()
    pg = np.array(page)
    ph, pw = pg.shape[:2]
    bg = wood(W, H)
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    dst = np.float32([[205, 62], [1445, 118], [1520, 1158], [92, 1120]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(pg, M, (W, H), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (W, H), flags=cv2.INTER_LINEAR)
    # page shadow on the table
    sh = cv2.GaussianBlur(cv2.warpAffine(mask, np.float32([[1, 0, 14], [0, 1, 18]]), (W, H)), (0, 0), 14)
    out = bg.astype(np.float32) * (1 - 0.45 * sh[..., None] / 255.0)
    m = mask[..., None].astype(np.float32) / 255.0
    out = out * (1 - m) + warped.astype(np.float32) * m
    # uneven lighting: warm light from top-left, darker bottom-right, vignette
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1.06 - 0.22 * ((x / W) * 0.6 + (y / H) * 0.8)
    vign = 1 - 0.35 * (((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2) / 2
    out *= (light * vign)[..., None]
    out *= np.array([1.02, 1.0, 0.94])[None, None]  # warm white balance
    out = cv2.GaussianBlur(out, (0, 0), 1.1)
    rng = np.random.default_rng(8)
    out += rng.normal(0, 4.0, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)  # saved as JPEG quality 85 by main()


# =====================================================================================
# 9. Parts of a flower
# =====================================================================================
FL = {"stem": (78, 140, 64), "leaf": (128, 196, 96), "sepal": (176, 204, 84), "petal": (238, 128, 168),
      "carpel": (150, 112, 196), "stigma": (96, 62, 150), "anther": (246, 196, 54), "filament": (214, 150, 60)}


def make_flower():
    c = Canvas(1400, 950, PAGE)
    c.title("Fig. 4.6   Parts of a flower")
    cx = 560
    # stem and leaves
    stem = Polygon([(cx - 13, 860), (cx + 13, 860), (cx + 11, 560), (cx - 11, 560)])
    c.shape(stem, FL["stem"], darker(FL["stem"], 0.6), 2)
    for (lx, ly, a, b, ang) in [(cx - 120, 760, 120, 38, 28), (cx + 115, 680, 115, 36, -30)]:
        leaf = blob(lx, ly, a, b, ang, wob=((2, 0.08, 0.4), (4, 0.03, 1.0)))
        c.shape(leaf.difference(stem.buffer(10)), FL["leaf"], darker(FL["leaf"], 0.6), 2)
        # leaf vein (clutter)
        c.line([(lx - a * 0.7 * math.cos(math.radians(ang)), ly - a * 0.7 * math.sin(math.radians(ang))),
                (lx + a * 0.7 * math.cos(math.radians(ang)), ly + a * 0.7 * math.sin(math.radians(ang)))],
               darker(FL["leaf"], 0.75), 1.2)
    # sepals (below the petals)
    for (sx, sang) in [(cx - 70, 35), (cx + 70, -35)]:
        c.ellipse(sx, 548, 70, 20, sang, fill=FL["sepal"], outline=darker(FL["sepal"], 0.6), width=2)
    # petals (fanned cup)
    for ang in (-62, -32, 32, 62):
        r = math.radians(ang)
        px, py = cx + 175 * math.sin(r), 400 - 40 * math.cos(r)
        c.ellipse(px, py, 150, 62, 90 - ang, fill=FL["petal"], outline=darker(FL["petal"], 0.6), width=2)
        c.ellipse(px - 20 * math.sin(r), py - 30, 40, 10, 90 - ang, fill=lighter(FL["petal"], 0.35))  # sheen (clutter)
    # carpel (ovary + style) and stigma
    carpel = Polygon(chaikin([(cx - 52, 520), (cx - 46, 470), (cx - 16, 430), (cx - 12, 290), (cx + 12, 290),
                              (cx + 16, 430), (cx + 46, 470), (cx + 52, 520)], it=3, closed=True))
    c.shape(carpel, FL["carpel"], darker(FL["carpel"], 0.6), 2)
    c.ellipse(cx, 268, 34, 22, fill=FL["stigma"], outline=darker(FL["stigma"], 0.6), width=2)
    # stamens: filament + anther
    for (fx, top) in [(cx - 110, 300), (cx - 62, 250), (cx + 62, 250), (cx + 110, 300)]:
        c.line([(cx + (fx - cx) * 0.25, 470), (fx, top)], darker(FL["filament"], 0.7), 3)
        c.ellipse(fx, top - 14, 15, 24, 0, fill=FL["anther"], outline=darker(FL["anther"], 0.6), width=2)
    L, Lx = 1010, 120
    c.label("Stigma", L, 200, cx + 20, 262)
    c.label("Stamen", L, 290, cx + 110, 272)
    c.label("Petal", L, 400, cx + 250, 420)
    c.label("Carpel", L, 500, cx + 30, 480)
    c.label("Sepal", L, 590, cx + 110, 562)
    c.label("Leaf", L, 690, cx + 160, 660)
    c.label("Stem", Lx + 180, 800, cx - 10, 800, side="left")
    c.caption("Insects carry pollen from the stamens to the stigma of another flower.", y=890)
    return c.final()


# =====================================================================================
# 10. Volcano cross-section
# =====================================================================================
VO = {"crust": (206, 178, 136), "cone": (138, 98, 72), "chamber": (214, 66, 38), "vent": (246, 150, 52),
      "lava": (186, 34, 30), "ash": (150, 150, 160)}


def make_volcano():
    c = Canvas(1400, 950, PAGE)
    c.title("Fig. 8.3   Inside a volcano")
    cx = 560
    crust = Polygon([(120, 640), (1000, 640), (1000, 860), (120, 860)])
    cone = Polygon(chaikin([(170, 640), (430, 330), (500, 268), (620, 268), (690, 330), (950, 640)], it=2, closed=False)
                   + [(950, 640), (170, 640)])
    chamber = blob(cx, 770, 170, 58, 0, wob=((3, 0.06, 0.5), (5, 0.03, 1.2)))
    vent = Polygon([(cx - 16, 730), (cx + 16, 730), (cx + 14, 280), (cx - 14, 280)])
    c.shape(crust.difference(chamber.buffer(8)).difference(vent.buffer(8)), VO["crust"], darker(VO["crust"], 0.6), 2)
    c.shape(cone.difference(vent.buffer(8)), VO["cone"], darker(VO["cone"], 0.6), 2)
    # rock layers in the cone (clutter)
    for k in range(1, 4):
        y = 640 - k * 85
        c.line([(220 + k * 60, y), (cx - 30, y - 10)], darker(VO["cone"], 0.8), 1.5)
        c.line([(cx + 30, y - 10), (900 - k * 60, y)], darker(VO["cone"], 0.8), 1.5)
    c.shape(chamber, VO["chamber"], darker(VO["chamber"], 0.6), 2)
    c.shape(vent, VO["vent"], darker(VO["vent"], 0.6), 2)
    # lava flows down the right flank
    lava = Polygon(chaikin([(612, 276), (650, 300), (700, 380), (742, 470), (770, 560), (740, 566),
                            (706, 480), (664, 392), (622, 320)], it=3, closed=True))
    c.shape(lava, VO["lava"], darker(VO["lava"], 0.6), 2)
    # ash cloud
    ash = cloud_shape(cx + 20, 150, 1.25)
    c.shape(ash, VO["ash"], darker(VO["ash"], 0.6), 2)
    rng = np.random.default_rng(11)
    for _ in range(30):   # ash specks (clutter)
        c.dot(cx + rng.uniform(-120, 160), rng.uniform(200, 250), 2.2, darker(VO["ash"], 0.7))
    L = 1080
    c.label("Ash cloud", L, 150, cx + 160, 150)
    c.label("Lava", L, 330, 690, 370)
    c.label("Vent", L, 450, cx + 6, 450)
    c.label("Cone", L, 560, 850, 600)
    c.label("Crust", L, 690, 960, 700)
    c.label("Magma chamber", L, 800, cx + 120, 790)
    c.caption("Magma rises from the chamber through the vent and erupts as lava and ash.", y=905)
    return c.final()


# =====================================================================================
# manifest + main
# =====================================================================================
SAMPLES = [
    dict(id="plant-cell", title="Plant cell", file="plant-cell.png", kind="colour", fn=make_plant_cell,
         expected_parts=["cell wall", "cytoplasm", "vacuole", "nucleus", "chloroplast", "mitochondrion"],
         description="Classic textbook plant cell with five chloroplasts, three mitochondria and ribosome clutter."),
    dict(id="heart", title="Human heart", file="heart.png", kind="colour", fn=make_heart,
         expected_parts=["right atrium", "left atrium", "right ventricle", "left ventricle", "aorta",
                         "pulmonary artery", "vena cava", "septum", "heart wall"],
         description="Front view of the heart with four colour-coded chambers, great vessels and septum."),
    dict(id="water-cycle", title="Water cycle", file="water-cycle.png", kind="colour", fn=make_water_cycle,
         expected_parts=["sun", "cloud", "mountain", "sea"],
         arrow_labels=["evaporation", "condensation", "precipitation", "collection"],
         description="Landscape scene with sun, clouds, rain, mountain and sea, and curved process arrows."),
    dict(id="plant-cell-photo", title="Plant cell (phone photo)", file="plant-cell-photo.jpg", kind="photo",
         fn=make_plant_cell_photo,
         expected_parts=["cell wall", "cytoplasm", "vacuole", "nucleus", "chloroplast", "mitochondrion"],
         description="The plant cell printed in a textbook, photographed at an angle on a wooden table."),
    dict(id="animal-cell", title="Animal cell", file="animal-cell.png", kind="colour", fn=make_animal_cell,
         expected_parts=["cell membrane", "cytoplasm", "nucleus", "nucleolus", "mitochondrion", "golgi body"],
         description="Irregular animal cell with nucleus, nucleolus, mitochondria, golgi body and ribosomes."),
    dict(id="earth-layers", title="Layers of the Earth", file="earth-layers.png", kind="colour", fn=make_earth_layers,
         expected_parts=["crust", "mantle", "outer core", "inner core"],
         description="Globe with a wedge cut away to show the crust, mantle, outer core and inner core."),
    dict(id="bar-chart", title="Favourite fruit (bar chart)", file="bar-chart.png", kind="chart", fn=make_bar_chart,
         expected_parts=["apple", "banana", "orange", "grapes", "kiwi"],
         description="Five coloured bars with axes, ticks, gridlines, value labels and axis titles."),
    dict(id="circuit", title="Series circuit", file="circuit.png", kind="lineart", fn=make_circuit,
         expected_parts=["battery", "switch", "bulb", "resistor"], min_parts=2,
         description="Black line-art series circuit: cell, closed switch, bulb and resistor in one loop."),
    dict(id="flower", title="Parts of a flower", file="flower.png", kind="colour", fn=make_flower,
         expected_parts=["petal", "sepal", "stamen", "carpel", "stigma", "stem", "leaf"],
         description="Side view of a flower with petals, sepals, stamens, carpel, stigma, stem and leaves."),
    dict(id="volcano", title="Inside a volcano", file="volcano.png", kind="colour", fn=make_volcano,
         expected_parts=["ash cloud", "lava", "vent", "cone", "crust", "magma chamber"],
         description="Cross-section of a volcano: magma chamber, vent, cone, lava flow, ash cloud and crust."),
]


def manifest():
    out = []
    for s in SAMPLES:
        e = {k: s[k] for k in ("id", "title", "file", "kind", "expected_parts")}
        e["min_parts"] = s.get("min_parts", len(s["expected_parts"]) - 1)
        e["photo"] = s["kind"] == "photo"
        e["description"] = s["description"]
        if "arrow_labels" in s:
            e["arrow_labels"] = s["arrow_labels"]
        out.append(e)
    return out


def main(argv):
    only = set(argv[1:])
    for s in SAMPLES:
        if only and s["id"] not in only:
            continue
        im = s["fn"]()
        path = os.path.join(HERE, s["file"])
        if path.endswith(".jpg"):
            im.convert("RGB").save(path, "JPEG", quality=85)
        else:
            im.save(path, optimize=True)
        print(f"wrote {s['file']:24s} {im.size[0]}x{im.size[1]}")
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest(), f, indent=2)
        f.write("\n")
    print("wrote manifest.json")


if __name__ == "__main__":
    main(sys.argv)
