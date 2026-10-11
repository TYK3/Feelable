"""Tactile rendering: Analysis -> raised-ink page + braille legend, following tactile design rules.

OWNER: Tactile team.

Coordinates: everything here is in TACTILE px (PX_PER_MM px per mm) unless named `src_*`.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine import braille
from engine.textures import TEXTURE_ORDER, fill_region
from engine.types import (
    MAX_RECOMMENDED_PARTS,
    MIN_LINE_MM,
    MIN_PART_MM,
    OUTLINE_MM,
    PAGE_MARGIN_MM,
    PAPER_MM,
    PX_PER_MM,
    Analysis,
    Check,
    Part,
    TactileOptions,
    TactileResult,
)

import os as _os
_BUNDLED_FONT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fonts", "Inter-Regular.otf")
FONT_PATH = _BUNDLED_FONT if _os.path.exists(_BUNDLED_FONT) else "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
PRINT_GREY = (0x8C, 0x8A, 0x94)

TITLE_ROW_MM = 11.0         # braille title (6.5 mm) + print title underneath
INSET_MM = 8.0              # extra inset of the drawing area so keys fit outside parts
GAP_MM = 2.0                # white gap between a texture and any outline
KEY_PAD_MM = 1.5            # clear pad around a braille key
KEY_MAX_DIST_MM = 15.0      # farther than this a key is no longer "next to" its part
SMALL_PART_MM2 = 150.0      # parts smaller than this are filled solid
ADJ_MM = 3.0                # parts closer than this are neighbours
MIN_LINEWORK_MM = 3.0       # shorter linework components are clutter
MAX_ENLARGE = 3.0
PRINT_CAP_MM = 3.0          # cap height of print labels
LEGEND_PITCH_MM = 10.0


# ============================================================================ keys
def _letters(name: str) -> str:
    return "".join(c for c in (name or "").lower() if "a" <= c <= "z")


def assign_keys(parts: list[Part]) -> None:
    """Set part.key for included parts (excluded parts get "").

    Keys come from the part's own name wherever possible, larger parts first:
    1. the largest part with a given first letter keeps that single letter;
    2. other parts try first letter + another letter of their own name: word initials
       ("left atrium" -> "la"), then each following letter in order ("nucleolus" -> "nu", "nc", "nl"...);
    3. only if all of those are taken, the next unused letter a..z (then aa, ab, ...).
    Deterministic; keys are lowercase a-z only and unique.
    """
    inc = [p for p in parts if p.include]
    for p in parts:
        if not p.include:
            p.key = ""
    order = sorted(inc, key=lambda p: (-int(getattr(p, "area", 0) or 0), p.id))
    used: set[str] = set()
    keys: dict[str, str] = {}
    for p in order:                                   # pass 1: single first letter, biggest part wins
        f = _letters(p.name)[:1]
        if f and f not in used:
            keys[p.id] = f
            used.add(f)
    for p in order:                                   # pass 2: two letters from the part's own name
        if p.id in keys:
            continue
        letters = _letters(p.name)
        if not letters:
            continue
        words = [_letters(w) for w in (p.name or "").split() if _letters(w)]
        cands = []
        if len(words) >= 2:
            cands.append(words[0][0] + words[1][0])
        cands += [letters[0] + c for c in letters[1:]]
        for c in cands:
            if len(c) == 2 and c not in used:
                keys[p.id] = c
                used.add(c)
                break
    pool = [chr(c) for c in range(97, 123)] + [a + b for a in "abcdefghijklmnopqrstuvwxyz"
                                                for b in "abcdefghijklmnopqrstuvwxyz"]
    it = iter(pool)
    for p in order:                                   # pass 3: next unused letter
        if p.id in keys:
            continue
        k = next(it)
        while k in used:
            k = next(it)
        keys[p.id] = k
        used.add(k)
    for p in inc:
        p.key = keys[p.id]


# ============================================================================ helpers
def _disk(d: float) -> np.ndarray:
    d = max(1, int(round(d)))
    if d % 2 == 0:
        d += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _bbox(mask: np.ndarray):
    pts = cv2.findNonZero((mask > 0).astype(np.uint8))
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    return x, y, x + w, y + h


def _dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Dilate with a disk of `radius_px`, working on the bbox ROI for speed."""
    if radius_px <= 0:
        return mask.copy()
    bb = _bbox(mask)
    if bb is None:
        return mask.copy()
    r = int(math.ceil(radius_px)) + 1
    H, W = mask.shape
    x0, y0, x1, y1 = max(0, bb[0] - r), max(0, bb[1] - r), min(W, bb[2] + r), min(H, bb[3] + r)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = cv2.dilate(mask[y0:y1, x0:x1], _disk(2 * radius_px + 1))
    return out


def _contours(mask: np.ndarray, eps: float):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        a = cv2.approxPolyDP(c, eps, True)
        if len(a) >= 3:
            out.append(a)
    return out


def _rect_sum(integral: np.ndarray, x0, y0, x1, y1):
    """Vectorised sum over [x0,x1) x [y0,y1) of the image whose cv2.integral is given."""
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


@lru_cache(maxsize=8)
def _font(px: int):
    try:
        return ImageFont.truetype(FONT_PATH, px)
    except OSError:
        return ImageFont.load_default()


def print_font(px_per_mm: float):
    # Inter cap height = 0.727 em
    return _font(max(8, int(round(PRINT_CAP_MM * px_per_mm / 0.727))))


def text_box(text: str, font) -> tuple[int, int]:
    l, t, r, b = font.getbbox(text, anchor="lt")
    return int(r) + 1, int(b) + 1


BRAILLE_WARNING_PREFIX = "Braille ("
# F03: text that could not be printed in full. Lines starting with this prefix are surfaced by
# readiness as a blocking "braille_overflow" issue, so a cut key can never publish quietly.
BRAILLE_OVERFLOW_PREFIX = braille.OVERFLOW_PREFIX
TITLE_MAX_ROWS = 3          # braille title rows before the title is reported as overflowing
TITLE_ROW_GAP_MM = 2.0      # space between wrapped braille title rows


def _wrap_cells(text: str, maxc: int) -> list[list[int]]:
    """Split `text` into lines of at most `maxc` cells, breaking at spaces where possible.
    A single word longer than a line is split by cell (never dropped)."""
    lines: list[list[int]] = []
    cur: list[int] = []
    for word in (text or "").split():
        wc = braille.to_cells(word)
        if not wc:
            continue
        sep = [0] if cur else []
        if len(cur) + len(sep) + len(wc) <= maxc:
            cur = cur + sep + wc
            continue
        if cur:
            lines.append(cur)
            cur = []
        while len(wc) > maxc:
            lines.append(wc[:maxc])
            wc = wc[maxc:]
        cur = wc
    if cur:
        lines.append(cur)
    return lines


def _title_block_mm(rows: int) -> float:
    """Height (mm) of `rows` stacked braille title rows."""
    return rows * braille.cell_height_mm() + max(0, rows - 1) * TITLE_ROW_GAP_MM


def _braille_warning(analysis: Analysis, title: str, included: list[Part]) -> None:
    """F15: say which characters were left out of the braille, and which code we use.
    Replaces any earlier braille warning (re-renders after edits must not pile them up)."""
    bad: set[str] = set(braille.unsupported_chars(title))
    for p in included:
        bad |= braille.unsupported_chars(p.name)
    warns = [w for w in (analysis.warnings or []) if not w.startswith(BRAILLE_WARNING_PREFIX)]
    if bad:
        shown = " ".join(sorted(bad)[:12]) + (" …" if len(bad) > 12 else "")
        # F02: a limitation of this translator, not of UEB itself
        warns.append(f"{BRAILLE_WARNING_PREFIX}{braille.BRAILLE_CODE}) Feelable cannot currently translate: {shown} — "
                     f"left out of the braille; rename these using plain letters and numbers")
    analysis.warnings = warns


# ============================================================================ main
def render_tactile(analysis: Analysis, options: TactileOptions = TactileOptions()) -> TactileResult:
    ppm = float(PX_PER_MM)
    mm = lambda v: v * ppm
    pw, ph = PAPER_MM.get(options.paper, PAPER_MM["a4"])
    W, H = int(round(pw * ppm)), int(round(ph * ppm))
    margin = int(round(mm(PAGE_MARGIN_MM)))
    sh, sw = analysis.height, analysis.width

    included = [p for p in analysis.parts
                if p.include and p.mask is not None and p.mask.shape[:2] == (sh, sw) and p.mask.any()]
    if any(not p.key for p in included) or len({p.key for p in included}) != len(included):
        assign_keys(analysis.parts)

    # ---------------------------------------------------------------- title
    title = (analysis.title or "").strip()
    max_cells = int((W - 2 * margin - mm(braille.BRAILLE_DOT_SPACING_MM + braille.BRAILLE_DOT_DIAMETER_MM))
                    // mm(braille.BRAILLE_CELL_SPACING_MM)) + 1
    # F03: wrap the title onto more braille rows instead of cutting it at one row's width
    title_lines = _wrap_cells(title, max_cells) if options.title_in_braille else []
    if len(title_lines) > TITLE_MAX_ROWS:
        title_lines = title_lines[:TITLE_MAX_ROWS]
        title_cut = True
    else:
        title_cut = False
    title_cells = title_lines
    title_h = int(round(mm(_title_block_mm(len(title_lines)) + 4.5))) if title_cells else (
        int(round(mm(TITLE_ROW_MM))) if (title and options.title_in_braille) else 0)
    _braille_warning(analysis, (title if options.title_in_braille else ""), included)
    _overflow_warnings(analysis, title_cut, title_cells, title, options.title_in_braille)

    # ---------------------------------------------------------------- layout
    inset = mm(INSET_MM)
    ax0, ay0 = margin + inset, margin + title_h + inset
    ax1, ay1 = W - margin - inset, H - margin - inset
    lw_src = analysis.linework if (analysis.linework is not None and analysis.linework.shape[:2] == (sh, sw)) \
        else np.zeros((sh, sw), np.uint8)
    lw_src = (lw_src > 0).astype(np.uint8) * 255

    def union_bbox(lw):
        boxes = [_bbox(p.mask > 0) for p in included]
        if lw.any():
            boxes.append(_bbox(lw))
        boxes = [b for b in boxes if b]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))

    def fit(bb):
        bw, bh = max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
        s = min((ax1 - ax0) / bw, (ay1 - ay0) / bh)
        ox = ax0 + ((ax1 - ax0) - bw * s) / 2 - bb[0] * s
        oy = ay0 + ((ay1 - ay0) - bh * s) / 2 - bb[1] * s
        return s, ox, oy

    bb = union_bbox(lw_src)
    if bb is None:
        scale, ox, oy = 1.0, 0.0, 0.0
    else:
        scale, _, _ = fit(bb)
        # drop linework specks (< 3 mm at this scale) before final fit
        if lw_src.any():
            n, cc, st, _ = cv2.connectedComponentsWithStats(lw_src)
            ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) * scale
            keep = ext >= mm(MIN_LINEWORK_MM)
            keep[0] = False
            lw_src = np.where(keep[cc], 255, 0).astype(np.uint8)
            bb = union_bbox(lw_src) or bb
        scale, ox, oy = fit(bb)
    M = np.array([[scale, 0, ox], [0, scale, oy]], np.float64)

    def warp(mask):
        out = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        if scale > 1.5:    # upscaled source pixels show as stairs: smooth them
            out = cv2.GaussianBlur(out, (0, 0), 0.45 * scale)
        return (out > 127).astype(np.uint8) * 255

    # ---------------------------------------------------------------- map + clean + enlarge parts
    min_part_px = mm(MIN_PART_MM)
    tiny_px = mm(1.5) ** 2 * 2          # ~2 mm^2: a component this small is clutter
    base: dict[str, np.ndarray] = {}
    for p in included:
        m = warp((p.mask > 0).astype(np.uint8) * 255)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 2:
            areas = st[1:, cv2.CC_STAT_AREA]
            keep = np.concatenate([[False], (areas >= tiny_px) | (areas == areas.max())])
            m = np.where(keep[cc], 255, 0).astype(np.uint8)
        # holes must be filled (types.py contract); enforce it so solid fills have no gaps
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(m, cs, -1, 255, -1)
        base[p.id] = m
    # a ring-shaped part (e.g. a cell membrane) whose outline is broken where another part overlaps it
    # cannot have its hole filled; close small gaps and use the filled shape if it then encloses others
    for p in included:
        m = base[p.id]
        if not m.any():
            continue
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(mm(6.0)))
        cs, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cs, -1, 255, -1)
        if np.count_nonzero(filled) >= 2.5 * np.count_nonzero(m) and any(
                q.id != p.id and _mostly_inside(base[q.id], filled, 0.6) for q in included):
            base[p.id] = filled

    tm: dict[str, np.ndarray] = {}
    enlarged: list[str] = []
    enlarge_ok = True
    for p in included:
        m = base[p.id]
        if not m.any():
            tm[p.id] = m
            continue
        others = np.zeros_like(m)
        for q in included:
            if q.id != p.id and not _mostly_inside(m, base[q.id]):
                others |= base[q.id]
        n, cc, st, cen = cv2.connectedComponentsWithStats(m)
        out = np.zeros_like(m)
        fmax = 1.0
        for i in range(1, n):
            comp = (cc == i).astype(np.uint8) * 255
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            (_, _), (rw, rh), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
            mind = max(1.0, min(rw, rh))
            if mind >= min_part_px:
                out |= comp
                continue
            f = min(MAX_ENLARGE, min_part_px / mind)
            if min_part_px / mind > MAX_ENLARGE + 1e-6:
                enlarge_ok = False
            fmax = max(fmax, f)
            out |= _enlarge_component(comp, cen[i], f, others, ppm, smooth=0.45 * f * max(scale, 1.0))
        if fmax > 1.0:
            enlarged.append(f"{p.name} ×{fmax:.1f}")
        tm[p.id] = out

    ids = [p.id for p in included]
    byid = {p.id: p for p in included}
    area = {i: int(np.count_nonzero(tm[i])) for i in ids}

    # ---------------------------------------------------------------- nesting
    ancestors: dict[str, set[str]] = {i: set() for i in ids}
    for b in ids:
        for a in ids:
            if a != b and area[a] > area[b] and area[b] > 0 and _mostly_inside(tm[b], tm[a]):
                ancestors[b].add(a)
    children = {a: {b for b in ids if a in ancestors[b]} for a in ids}

    # ---------------------------------------------------------------- adjacency
    near = {i: _dilate(tm[i], mm(ADJ_MM)) for i in ids}
    adj: dict[str, set[str]] = {i: set() for i in ids}
    for ai, a in enumerate(ids):
        for b in ids[ai + 1:]:
            if b in ancestors[a] or a in ancestors[b] or np.any(near[a] & tm[b]):
                adj[a].add(b)
                adj[b].add(a)

    # ---------------------------------------------------------------- textures
    textures: dict[str, str] = {}
    centre = {}
    for i in ids:
        mo = cv2.moments(tm[i], binaryImage=True)
        centre[i] = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]) if mo["m00"] else (0.0, 0.0)
    comp_area_mm2 = {}
    for i in ids:
        n, cc, st, _ = cv2.connectedComponentsWithStats(tm[i])
        # the LARGEST piece decides: a big region with a few ragged crumbs must never go solid
        # (a large solid area swells into one slab on swell paper and is hard to read)
        comp_area_mm2[i] = float(np.max(st[1:, cv2.CC_STAT_AREA])) / ppm ** 2 if n > 1 else 0.0
    for i in ids:
        if children[i]:
            textures[i] = "none"
    for i in ids:
        if i in textures:
            continue
        if comp_area_mm2[i] < SMALL_PART_MM2 and not any(textures.get(j) == "solid" for j in adj[i]):
            textures[i] = "solid"
    used_global: list[str] = [t for t in textures.values()]
    for i in ids:
        if i in textures:
            continue
        nb = {textures.get(j) for j in adj[i]}
        free = [t for t in TEXTURE_ORDER if t not in nb]
        fresh = [t for t in free if t not in used_global]
        if fresh:
            t = fresh[0]
        elif free:   # reuse the texture whose nearest same-texture part is farthest away
            def sep(t):
                ds = [math.dist(centre[i], centre[j]) for j in textures if textures[j] == t]
                return -min(ds) if ds else 0.0
            t = min(free, key=lambda x: (sep(x), TEXTURE_ORDER.index(x)))
        else:   # more than 6 textured neighbours: least-used texture among neighbours
            t = min(TEXTURE_ORDER, key=lambda x: (sum(textures.get(j) == x for j in adj[i]), TEXTURE_ORDER.index(x)))
        textures[i] = t
        used_global.append(t)

    # ---------------------------------------------------------------- linework
    lw = warp(lw_src) if lw_src.any() else np.zeros((H, W), np.uint8)
    if lw_src.any():
        # thin source strokes: warp with a low threshold so they survive downscaling
        soft = cv2.warpAffine(lw_src, M, (W, H), flags=cv2.INTER_LINEAR)
        lw = (soft > 60).astype(np.uint8) * 255
        n, cc, st, _ = cv2.connectedComponentsWithStats(lw)
        ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
        keep = ext >= mm(MIN_LINEWORK_MM)
        keep[0] = False
        lw = np.where(keep[cc], 255, 0).astype(np.uint8)
        lw = _thicken_lines(lw, ppm, n_min=MIN_LINE_MM)
        lw = _hollow_slabs(lw, ppm)

    # ---------------------------------------------------------------- outlines
    eps = mm(0.4)
    oth = mm(OUTLINE_MM)
    part_polygons: dict[str, list[list[tuple[int, int]]]] = {}
    outline_layer = np.zeros((H, W), np.uint8)
    outline_len = 0.0
    outline_area = 0        # summed per part, so shared borders don't hide the true width
    largest_contour: dict[str, np.ndarray] = {}
    for i in ids:
        cs = _contours(tm[i], eps)
        part_polygons[i] = [[(int(x), int(y)) for x, y in c.reshape(-1, 2)] for c in cs]
        if cs:
            largest_contour[i] = max(cs, key=cv2.contourArea)
        outline_area += _draw_poly_outline(outline_layer, cs, oth)
        outline_len += sum(cv2.arcLength(c, True) for c in cs)

    # ---------------------------------------------------------------- texture regions
    gap_px = mm(GAP_MM) + 1 + eps    # +1 px rasterisation, + polygon simplification tolerance
    tex_layer = np.zeros((H, W), np.uint8)
    solid_layer = np.zeros((H, W), np.uint8)
    lw_block = _dilate(lw, gap_px + 0.5) if lw.any() else None
    for i in ids:
        t = textures[i]
        if t == "none":
            continue
        if t == "solid":
            solid_layer |= tm[i]
            continue
        obst = np.zeros((H, W), np.uint8)
        for j in ids:
            if j != i and j not in ancestors[i]:
                obst |= tm[j]
        region = cv2.bitwise_and(tm[i], cv2.bitwise_not(_dilate(obst, oth / 2 + gap_px)))
        if lw_block is not None:
            region = cv2.bitwise_and(region, cv2.bitwise_not(lw_block))
        region = cv2.erode(region, _disk(2 * (oth / 2 + gap_px) + 1))
        filled_tex = fill_region(t, region, ppm)
        if np.count_nonzero(region) < 0.15 * max(1, area[i]) or not filled_tex.any():
            # too narrow for a readable texture: small parts go solid, big ones outline only
            if comp_area_mm2[i] < 3 * SMALL_PART_MM2 and not any(textures.get(j) == "solid" for j in adj[i]):
                textures[i] = "solid"
                solid_layer |= tm[i]
            else:
                textures[i] = "none"
            continue
        tex_layer |= filled_tex

    ink = tex_layer | solid_layer | outline_layer | lw
    drawing = ink.copy()       # page-1 drawing before braille

    # ---------------------------------------------------------------- title
    title_box = None
    if title_cells:
        tw = 0
        for r_i, line in enumerate(title_cells):
            y_row = margin + int(round(r_i * (braille.cell_height_mm() + TITLE_ROW_GAP_MM) * ppm))
            braille.draw_cells(ink, line, margin, y_row, ppm)
            tw = max(tw, int(round(mm(braille.cells_width_mm(len(line))))))
        title_box = (margin, margin, margin + max(tw, 1), margin + title_h)

    # ---------------------------------------------------------------- braille keys
    key_positions, key_boxes, key_fallback = _place_keys(
        ink, included, tm, largest_contour, ancestors, children, title_box, W, H, margin, ppm)

    # ---------------------------------------------------------------- print layer
    print_layer, label_info = _print_layer(ink, included, key_boxes, title, title_box, title_cells, W, H,
                                           margin, ppm)

    # ---------------------------------------------------------------- legend page
    legend_ink, legend, legend_more = _legend(included, W, H, margin, ppm)
    cut = [e for e in legend if e.get("cut")]
    if cut:
        analysis.warnings = list(analysis.warnings or []) + [
            f"{BRAILLE_OVERFLOW_PREFIX}{len(cut)} key line(s) are too long for the braille-only key "
            f"and were cut ({', '.join(str(e['key']) for e in cut[:6])}); use the print key or shorten the names"]

    # ---------------------------------------------------------------- checks
    checks = _checks(drawing, outline_layer, outline_area, outline_len, tex_layer, solid_layer, lw, ids, byid, textures, adj,
                     enlarged, enlarge_ok, tm, key_fallback, ink, key_boxes, ppm, included)

    return TactileResult(
        width=W, height=H, px_per_mm=ppm, ink=ink, legend_ink=legend_ink, print_layer=print_layer,
        part_polygons=part_polygons, key_positions=key_positions, textures=textures, legend=legend,
        checks=checks, transform=(float(scale), float(ox), float(oy)), legend_pages=legend_more,
    )


# ============================================================================ pieces
def _mostly_inside(b: np.ndarray, a: np.ndarray, frac: float = 0.9) -> bool:
    nb = np.count_nonzero(b)
    if nb == 0:
        return False
    return np.count_nonzero(cv2.bitwise_and(a, b)) >= frac * nb


def _enlarge_component(comp, centre, f, others, ppm, smooth=1.0):
    """Scale `comp` by f about its centroid (smoothed); if it then collides with other parts,
    nudge it (up to 4 mm) to the position with the least overlap."""
    H, W = comp.shape
    cx, cy = float(centre[0]), float(centre[1])
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(comp))
    r = int(max(w, h) * f / 2 + 5 * ppm)
    x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
    x1, y1 = min(W, int(cx) + r + 1), min(H, int(cy) + r + 1)
    sub = comp[y0:y1, x0:x1]
    guard = _dilate(others, 1.5 * ppm)[y0:y1, x0:x1] if others.any() else None
    shifts = [(0.0, 0.0)] + [(rr * math.cos(a), rr * math.sin(a))
                             for rr in (1.0, 2.0, 3.0, 4.0) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    best, best_ov = None, None
    for dx, dy in shifts:
        Mx = cv2.getRotationMatrix2D((cx - x0, cy - y0), 0, f)
        Mx[0, 2] += dx * ppm
        Mx[1, 2] += dy * ppm
        e = cv2.warpAffine(sub, Mx, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        e = (cv2.GaussianBlur(e, (0, 0), max(0.8, smooth)) > 127).astype(np.uint8) * 255
        ov = int(np.count_nonzero(e & guard)) if guard is not None else 0
        if best_ov is None or ov < best_ov:
            best, best_ov = e, ov
        if ov == 0:
            break
    if guard is not None and best_ov:
        # TS-04: never let an enlarged part spread over a neighbour. Keep the part's own original
        # pixels, drop only the *new* growth that lands in another part (plus its clearance gap).
        best = ((best > 0) & ~(guard > 0) | (sub > 0)).astype(np.uint8) * 255
    out = np.zeros_like(comp)
    out[y0:y1, x0:x1] = best
    return out


def _draw_poly_outline(layer, contours, thickness_px):
    """Draw closed polylines exactly `thickness_px` wide: 2x supersampled AA, area-downsampled, thresholded."""
    if not contours:
        return 0
    H, W = layer.shape
    allp = np.concatenate([c.reshape(-1, 2) for c in contours], 0)
    r = int(math.ceil(thickness_px)) + 2
    x0, y0 = max(0, int(allp[:, 0].min()) - r), max(0, int(allp[:, 1].min()) - r)
    x1, y1 = min(W, int(allp[:, 0].max()) + r + 1), min(H, int(allp[:, 1].max()) + r + 1)
    ss, sh = 2, 3
    big = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    pts = [np.round(((c.reshape(-1, 2).astype(np.float64) - (x0, y0)) * ss + (ss - 1) / 2) * (1 << sh)).astype(np.int32)
           for c in contours]
    cv2.polylines(big, pts, True, 255, max(1, int(round(thickness_px * ss)) - 1), cv2.LINE_AA, sh)
    small = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    drawn = (small > 127).astype(np.uint8) * 255
    layer[y0:y1, x0:x1] |= drawn
    return int(np.count_nonzero(drawn))


def _hollow_slabs(lw, ppm, slab_mm=4.0, rim_mm=1.5):
    """Big solid dark areas (a black silhouette, a filled logo, a dark country on a map) would be
    raised as one hard slab that tells a finger nothing. Keep only a rim around anything wider
    than `slab_mm`, so it reads as an outline; thin lines are untouched."""
    if not lw.any():
        return lw
    core = cv2.morphologyEx(lw, cv2.MORPH_OPEN, _disk(max(3, int(round(slab_mm * ppm)) | 1)))
    if not core.any():
        return lw
    inner = cv2.erode(lw, _disk(max(3, int(round(2 * rim_mm * ppm)) | 1)))
    hollow = inner & cv2.dilate(core, _disk(3))
    return (lw & ~hollow).astype(np.uint8)


def _thicken_lines(lw, ppm, n_min=MIN_LINE_MM):
    """Make every linework stroke at least n_min mm wide and smooth its edges.

    Stroke width per component is estimated as 2*area/perimeter (exact for long strokes); thin
    components are dilated just enough, then everything is blurred + thresholded (smooth edges)."""
    out = np.zeros_like(lw)
    target = n_min * ppm * 1.25
    n, cc, st, _ = cv2.connectedComponentsWithStats(lw)
    cs, _ = cv2.findContours(lw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    per = np.zeros(n)
    for c in cs:
        x, y = c[0, 0]
        per[cc[y, x]] += cv2.arcLength(c, True)
    by_r: dict[int, list[int]] = {}
    for i in range(1, n):
        w = 2.0 * st[i, cv2.CC_STAT_AREA] / max(per[i], 1.0)
        r = int(math.ceil(max(0.0, (target - w) / 2)))
        by_r.setdefault(r, []).append(i)
    for r, comps in by_r.items():
        sel = np.zeros(n, bool)
        sel[comps] = True
        m = np.where(sel[cc], 255, 0).astype(np.uint8)
        out |= cv2.dilate(m, _disk(2 * r + 1)) if r > 0 else m
    return (cv2.GaussianBlur(out, (0, 0), 1.2) > 127).astype(np.uint8) * 255


def _place_keys(ink, included, tm, largest_contour, ancestors, children, title_box, W, H, margin, ppm,
                outline_px=OUTLINE_MM * PX_PER_MM):
    """Put each part's braille key in the closest free spot next to its largest component.

    A spot is valid when the key box + KEY_PAD_MM pad holds no raised ink and no other key,
    the box lies inside the page margins, and it is not inside another (non-ancestor) part.
    Score = distance to own outline, penalised when another part's outline is closer (ambiguity).
    Returns (key_positions, key_boxes, names_that_needed_the_fallback)."""
    pad = int(round(KEY_PAD_MM * ppm))
    key_positions: dict[str, tuple[int, int]] = {}
    key_boxes: dict[str, tuple[int, int, int, int]] = {}
    fallback: list[str] = []
    blocked = np.zeros((H, W), np.uint8)
    if title_box:
        x0, y0, x1, y1 = title_box
        blocked[y0:y1, x0:x1] = 255
    ids = [p.id for p in included]
    all_outlines = np.zeros((H, W), np.uint8)
    for i in ids:
        cv2.polylines(all_outlines, _contours(tm[i], 0.4 * ppm), True, 255, 1)
    reach = int(round(KEY_MAX_DIST_MM * ppm))
    keys_mask = np.zeros((H, W), np.uint8)

    order = sorted(included, key=lambda p: (int(np.count_nonzero(tm[p.id])), p.id))   # small parts first
    for p in order:
        i = p.id
        if i not in largest_contour:
            continue
        kw, kh = braille.text_size_px(p.key, ppm)
        if kw == 0:
            continue
        cx0, cy0, cw, ch = cv2.boundingRect(largest_contour[i])
        rx0 = max(0, cx0 - reach - kw - pad)
        ry0 = max(0, cy0 - reach - kh - pad)
        rx1 = min(W, cx0 + cw + reach + kw + pad)
        ry1 = min(H, cy0 + ch + reach + kh + pad)
        sl = (slice(ry0, ry1), slice(rx0, rx1))
        occ = cv2.bitwise_or(ink[sl], blocked[sl])
        forb = np.zeros_like(occ)
        for j in ids:
            if j != i and j not in ancestors[i]:
                forb |= tm[j][sl]
        if not children[i]:
            forb |= tm[i][sl]
        own = np.zeros_like(occ)
        cv2.polylines(own, [largest_contour[i] - (rx0, ry0)], True, 255, 1)
        oth = cv2.bitwise_and(all_outlines[sl], cv2.bitwise_not(cv2.dilate(own, _disk(3))))
        # other components of the same part do not make a spot ambiguous
        same = cv2.dilate(tm[i][sl], _disk(5))
        oth = cv2.bitwise_and(oth, cv2.bitwise_not(same))
        d_own = cv2.distanceTransform(cv2.bitwise_not(own), cv2.DIST_L2, 3)
        d_oth = cv2.distanceTransform(cv2.bitwise_not(oth), cv2.DIST_L2, 3) if oth.any() \
            else np.full(occ.shape, 1e4, np.float32)
        kern = np.ones((kh, kw), np.uint8)
        dmin_own = cv2.erode(d_own, kern, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
        dmin_oth = cv2.erode(d_oth, kern, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
        I_occ = cv2.integral((occ > 0).astype(np.uint8))
        I_forb = cv2.integral((forb > 0).astype(np.uint8))
        rh, rw = occ.shape
        # candidate top-left corners (ROI-local), box inside the page margins
        xs = np.arange(max(margin - rx0, 0), min(W - margin - kw - rx0, rw - kw) + 1)
        ys = np.arange(max(margin - ry0, 0), min(H - margin - kh - ry0, rh - kh) + 1)
        if len(xs) == 0 or len(ys) == 0:
            continue
        X, Y = np.meshgrid(xs, ys)
        occ_sum = _rect_sum(I_occ, np.maximum(X - pad, 0), np.maximum(Y - pad, 0),
                            np.minimum(X + kw + pad, rw), np.minimum(Y + kh + pad, rh))
        forb_sum = _rect_sum(I_forb, X, Y, X + kw, Y + kh)
        do = dmin_own[Y, X]
        dt = dmin_oth[Y, X]
        # pad clipped by the ROI border is only safe where the ROI border is the page border
        inner = ((X - pad >= 0) | (rx0 == 0)) & ((Y - pad >= 0) | (ry0 == 0)) & \
                ((X + kw + pad <= rw) | (rx1 == W)) & ((Y + kh + pad <= rh) | (ry1 == H))
        valid = (occ_sum == 0) & (forb_sum == 0) & inner & (do <= reach)
        cyc = cy0 + ch / 2 - ry0
        score = do + 1.5 * np.maximum(0.0, do - dt + ppm) + 0.05 * np.abs(Y + kh / 2 - cyc)
        if keys_mask.any():   # keep keys apart so each clearly belongs to one part
            kd = cv2.distanceTransform(cv2.bitwise_not(keys_mask[sl]), cv2.DIST_L2, 3)
            kd = cv2.erode(kd, kern, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)[Y, X]
            score = score + 1.5 * np.maximum(0.0, 10 * ppm - kd)
        if valid.any():
            k = int(np.argmin(np.where(valid, score, np.inf)))
        else:
            fallback.append(p.name)
            # fallback 1: inside the part itself, in a window cleared of texture (unambiguous)
            inside = cv2.erode(tm[i][sl], _disk(2 * (outline_px / 2 + pad + 1) + 1))
            I_in = cv2.integral((inside > 0).astype(np.uint8))
            I_blk = cv2.integral((blocked[sl] > 0).astype(np.uint8))
            in_ok = (_rect_sum(I_in, X, Y, X + kw, Y + kh) == kw * kh) & inner & \
                (_rect_sum(I_blk, np.maximum(X - pad, 0), np.maximum(Y - pad, 0),
                           np.minimum(X + kw + pad, rw), np.minimum(Y + kh + pad, rh)) == 0)
            if in_ok.any():
                # nearest to the part's centre of mass
                mo = cv2.moments(tm[i][sl], binaryImage=True)
                mx, my = mo["m10"] / max(mo["m00"], 1), mo["m01"] / max(mo["m00"], 1)
                dc = np.hypot(X + kw / 2 - mx, Y + kh / 2 - my)
                k = int(np.argmin(np.where(in_ok, dc, np.inf)))
            else:
                # fallback 2: least raised ink in the padded box, 2..8 mm from the part; clear its pad
                ring = (do >= 2 * ppm) & (do <= 8 * ppm) & (forb_sum == 0) & inner
                if not ring.any():
                    ring = (do <= 8 * ppm) & inner
                if not ring.any():
                    ring = np.ones_like(valid)
                k = int(np.argmin(np.where(ring, occ_sum + 0.01 * do, np.inf)))
        yy, xx = divmod(k, X.shape[1])
        x, y = int(X[yy, xx]) + rx0, int(Y[yy, xx]) + ry0
        if not valid.any():
            ink[max(0, y - pad):y + kh + pad, max(0, x - pad):x + kw + pad] = 0
        braille.draw_braille(ink, p.key, x, y, ppm)
        key_positions[i] = (x, y)
        key_boxes[i] = (x, y, x + kw, y + kh)
        blocked[max(0, y - pad):y + kh + pad, max(0, x - pad):x + kw + pad] = 255
        keys_mask[y:y + kh, x:x + kw] = 255
    return key_positions, key_boxes, fallback


def _print_layer(ink, included, key_boxes, title, title_box, title_cells, W, H, margin, ppm):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    font = print_font(ppm)
    small = _font(max(8, int(round(2.6 * ppm / 0.727))))
    occ = (cv2.dilate(ink, _disk(2 * 0.8 * ppm + 1)) > 0).astype(np.uint8) * 255
    for (x0, y0, x1, y1) in key_boxes.values():
        occ[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3] = 255
    labels = {}
    if title:
        ty = (margin + int(round(_title_block_mm(len(title_cells)) * ppm + 1.2 * ppm))) if title_cells else margin
        tw, th = text_box(title, small)
        if ty + th <= H and not occ[ty:ty + th, margin:margin + tw].any():
            d.text((margin, ty), title, font=small, fill=PRINT_GREY, anchor="lt")
            occ[ty:ty + th, margin:margin + tw] = 255
    gap = int(round(1.5 * ppm))
    for p in included:
        if p.id not in key_boxes:
            continue
        x0, y0, x1, y1 = key_boxes[p.id]
        name = p.name.strip() or p.key
        tw, th = text_box(name, font)
        kh = y1 - y0
        placed = _find_label_spot(occ, (x0, y0, x1, y1), tw, th, W, H, margin, ppm)
        if placed:
            x, y = placed
            d.text((x, y), name, font=font, fill=PRINT_GREY, anchor="lt")
            mg = int(round(1.0 * ppm))
            occ[max(0, y - mg):y + th + mg, max(0, x - mg):x + tw + mg] = 255
            labels[p.id] = (x, y, x + tw, y + th)
    arr = np.array(img)
    arr[ink > 0] = (0, 0, 0)
    return arr, labels


def _find_label_spot(occ, key_box, tw, th, W, H, margin, ppm, reach_mm=12.0):
    """Closest spot for a tw x th print label near key_box where `occ` is empty.
    Prefers right of the key, vertically centred; then below/above/left."""
    x0, y0, x1, y1 = key_box
    reach = int(reach_mm * ppm)
    lo = margin // 2
    rx0, ry0 = max(lo, x0 - reach - tw), max(lo, y0 - reach - th)
    rx1, ry1 = min(W - lo, x1 + reach + tw), min(H - lo, y1 + reach + th)
    if rx1 - rx0 < tw or ry1 - ry0 < th:
        return None
    I = cv2.integral((occ[ry0:ry1, rx0:rx1] > 0).astype(np.uint8))
    xs = np.arange(0, rx1 - rx0 - tw + 1)
    ys = np.arange(0, ry1 - ry0 - th + 1)
    X, Y = np.meshgrid(xs, ys)
    free = _rect_sum(I, X, Y, X + tw, Y + th) == 0
    if not free.any():
        return None
    LX, LY = X + rx0, Y + ry0
    # rectangle gap between label and key box (0 when overlapping -> forbidden)
    gx = np.maximum(0, np.maximum(x0 - (LX + tw), LX - x1))
    gy = np.maximum(0, np.maximum(y0 - (LY + th), LY - y1))
    overlap = (gx == 0) & (gy == 0)
    dist = np.hypot(gx, gy)
    centred = np.abs((LY + th / 2) - (y0 + y1) / 2)
    right = (LX >= x1)
    score = dist + 0.3 * centred + np.where(right, 0.0, 0.6 * ppm) + np.where(dist < 1.0 * ppm, 3 * ppm, 0.0)
    score = np.where(free & ~overlap & (dist <= reach), score, np.inf)
    k = int(np.argmin(score))
    if not np.isfinite(score.flat[k]):
        return None
    yy, xx = divmod(k, X.shape[1])
    return int(LX[yy, xx]), int(LY[yy, xx])


def _overflow_warnings(analysis: Analysis, title_cut: bool, title_cells, title: str, in_braille: bool) -> None:
    """Replace earlier overflow warnings with the current ones (re-renders must not pile up)."""
    warns = [w for w in (analysis.warnings or []) if not w.startswith(BRAILLE_OVERFLOW_PREFIX)]
    if title_cut and in_braille:
        warns.append(f"{BRAILLE_OVERFLOW_PREFIX}the title is too long to print in full; shorten it")
    analysis.warnings = warns


def _legend(included, W, H, margin, ppm):
    ink = np.zeros((H, W), np.uint8)
    mm = lambda v: v * ppm
    braille.draw_braille(ink, "key", margin, margin, ppm)
    pitch = mm(LEGEND_PITCH_MM)
    y_start = margin + 2 * pitch
    cell_h = mm(braille.cell_height_mm())
    rows_fit = max(1, int((H - margin - cell_h - y_start) // pitch) + 1)
    font = print_font(ppm)
    rows = []
    kmax = max((len(braille.to_cells(p.key)) for p in included), default=1)
    for p in included:
        kc = braille.to_cells(p.key)
        nc = braille.to_cells(p.name) or braille.to_cells(p.key)
        # key, then 2 blank cells; names aligned in one column
        rows.append({"p": p, "cells": kc + [0] * (kmax - len(kc) + 2) + nc, "print": f"{p.key} = {p.name}"})
    n = len(rows)
    avail = W - 2 * margin
    gutter = mm(12)
    layout = None
    for ncols in (1, 2, 3, 4):
        per = math.ceil(n / ncols) if n else 0
        if per > rows_fit:
            continue
        cols = [rows[c * per:(c + 1) * per] for c in range(ncols)]
        widths = []
        for col in cols:
            if not col:
                widths.append(0)
                continue
            widths.append(max(mm(braille.cells_width_mm(len(r["cells"]))) for r in col) + mm(5)
                          + max(text_box(r["print"], font)[0] for r in col))
        if sum(widths) + gutter * (ncols - 1) <= avail:
            layout = (per, cols, widths, True)
            break
    if layout is None:   # no room for print beside braille: braille only, in columns
        ncols = max(1, math.ceil(n / rows_fit))
        per = math.ceil(n / ncols) if n else 0
        cols = [rows[c * per:(c + 1) * per] for c in range(ncols)]
        colw = (avail - gutter * (ncols - 1)) / ncols
        maxc = max(3, int((colw - mm(4.0)) // mm(braille.BRAILLE_CELL_SPACING_MM)) + 1)
        if any(len(r["cells"]) > maxc for col in cols for r in col):
            # F03: a name would have to be cut. Never cut braille: wrap long names onto
            # continuation lines under the name column and add key pages as needed.
            return _legend_paged(included, W, H, margin, ppm, kmax)
        layout = (per, cols, [colw] * ncols, False)
    per, cols, widths, with_print = layout
    legend = []
    x = margin
    for col, w in zip(cols, widths):
        bmax = max((mm(braille.cells_width_mm(len(r["cells"]))) for r in col), default=0)
        for r_i, r in enumerate(col):
            y = int(round(y_start + r_i * pitch))
            bw, bh = braille.draw_cells(ink, r["cells"], int(x), y, ppm)
            entry = {"part_id": r["p"].id, "key": r["p"].key, "name": r["p"].name, "x": int(x), "y": y}
            if r.get("cut"):
                entry["cut"] = True
            if with_print:
                th = text_box("Ag", font)[1]     # constant: rows line up regardless of descenders
                entry["print_x"] = int(x + bmax + mm(5))
                entry["print_y"] = int(y + (bh - th) / 2)
                entry["print"] = r["print"]
            legend.append(entry)
        x += w + gutter
    return ink, legend, []


def _wrap_words(text: str, font, max_w: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and text_box(trial, font)[0] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def _legend_paged(included, W, H, margin, ppm, kmax):
    """F03: single-column key over as many pages as needed. Each row: key cells, two blank cells,
    the name in braille wrapped at word boundaries onto continuation lines that stay indented
    under the name (so every line still belongs to its key); grey print for the teacher beside it."""
    mm = lambda v: v * ppm
    pitch = mm(LEGEND_PITCH_MM)
    cell = mm(braille.BRAILLE_CELL_SPACING_MM)
    cell_h = mm(braille.cell_height_mm())
    y_start = margin + 2 * pitch
    lines_fit = max(1, int((H - margin - cell_h - y_start) // pitch) + 1)
    font = print_font(ppm)
    avail = W - 2 * margin
    braille_w = 0.62 * avail
    maxc = max(kmax + 6, int((braille_w - mm(braille.BRAILLE_DOT_SPACING_MM + braille.BRAILLE_DOT_DIAMETER_MM))
                             // cell) + 1)
    indent = kmax + 2
    name_c = maxc - indent
    print_x_off = mm(braille.cells_width_mm(maxc)) + mm(5)
    print_w = max(mm(30), avail - print_x_off)
    pages: list[np.ndarray] = []
    legend: list[dict] = []
    ink = None
    line = lines_fit                      # force a new page for the first row
    for p in included:
        kc = braille.to_cells(p.key)
        name_lines = _wrap_cells(p.name, name_c) or [braille.to_cells(p.key)]
        b_lines = [kc + [0] * (indent - len(kc)) + name_lines[0]] + [[0] * indent + nl for nl in name_lines[1:]]
        pr = f"{p.key} = {p.name}"
        p_lines = _wrap_words(pr, font, print_w)
        need = max(len(b_lines), len(p_lines))
        if line + need > lines_fit:
            ink = np.zeros((H, W), np.uint8)
            braille.draw_braille(ink, "key", margin, margin, ppm)
            pages.append(ink)
            line = 0
        y0 = y_start + line * pitch
        th = text_box("Ag", font)[1]
        for j, cells in enumerate(b_lines):
            braille.draw_cells(ink, cells, int(margin), int(round(y0 + j * pitch)), ppm)
        entry = {"part_id": p.id, "key": p.key, "name": p.name, "x": int(margin), "y": int(round(y0)),
                 "page": len(pages) - 1, "lines": len(b_lines), "print": pr,
                 "print_x": int(margin + print_x_off), "print_y": int(round(y0 + (cell_h - th) / 2)),
                 "print_lines": [[int(margin + print_x_off), int(round(y0 + j * pitch + (cell_h - th) / 2)), t]
                                 for j, t in enumerate(p_lines)]}
        legend.append(entry)
        line += need
    if not pages:
        pages = [np.zeros((H, W), np.uint8)]
        braille.draw_braille(pages[0], "key", margin, margin, ppm)
    return pages[0], legend, pages[1:]


def _thinnest_line_mm(ink, ppm):
    """Largest disk diameter (mm) that fits every raised feature, measured on the raster.

    A feature 'fails' diameter d if opening with a d-px disk removes a residue blob of at
    least ~1 mm^2 (corner nibbles of thick lines are much smaller than that)."""
    if not ink.any():
        return float("inf")
    best = 0.0
    lo = int(round(MIN_LINE_MM * ppm)) - 1
    for d in range(max(2, lo), lo + 5):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
        resid = cv2.bitwise_and(ink, cv2.bitwise_not(opened))
        resid = cv2.morphologyEx(resid, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(resid)
        big = (st[1:, cv2.CC_STAT_AREA] >= ppm * ppm).sum() if n > 1 else 0
        if big:
            break
        best = d / ppm
    return best


def _checks(drawing, outline_layer, outline_area, outline_len, tex_layer, solid_layer, lw, ids, byid, textures, adj,
            enlarged, enlarge_ok, tm, key_fallback, ink, key_boxes, ppm, included):
    checks = []
    # 1. thinnest line
    thin = _thinnest_line_mm(drawing, ppm)
    if math.isinf(thin):
        checks.append(Check("Lines at least 1 mm", True, "nothing raised yet"))
    else:
        checks.append(Check("Lines at least 1 mm", thin >= MIN_LINE_MM - 1e-6, f"thinnest line {thin:.1f} mm"))
    # 2. outline thickness = outline area / outline length
    if outline_len > 0:
        t = outline_area / outline_len / ppm
        checks.append(Check("Outlines 1.5 mm", abs(t - OUTLINE_MM) <= 0.25, f"measured {t:.1f} mm"))
    else:
        checks.append(Check("Outlines 1.5 mm", True, "no part outlines"))
    # 3. small parts
    checks.append(Check("Small parts enlarged", enlarge_ok,
                        ", ".join(enlarged) if enlarged else "none needed"
                        + ("" if enlarge_ok else " (some parts are still small: enlargement is capped at 3×)")))
    # 4. neighbours
    clashes = []
    for a in ids:
        for b in adj[a]:
            if a < b and textures[a] == textures[b] and textures[a] != "none":
                clashes.append(f"{byid[a].name} / {byid[b].name}")
    checks.append(Check("Neighbours have different textures", not clashes,
                        "clash: " + "; ".join(clashes) if clashes else
                        f"{len({t for t in textures.values() if t != 'none'})} textures, no shared borders"))
    # 5. gap between textures and any line
    if tex_layer.any():
        lines = outline_layer | lw | solid_layer
        dt = cv2.distanceTransform(cv2.bitwise_not(lines), cv2.DIST_L2, 5)
        g = (float(dt[tex_layer > 0].min()) - 1.0) / ppm
        checks.append(Check("2 mm gap around textures", g >= 2.0 - 0.5 / ppm, f"smallest gap {g:.1f} mm"))
    else:
        checks.append(Check("2 mm gap around textures", True, "no textures"))
    # 6. keys
    pad = int(round(KEY_PAD_MM * ppm))
    touching = []
    for i, (x0, y0, x1, y1) in key_boxes.items():
        # everything raised in the padded box must be this key's own dots
        bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
        box = ink[by0:y1 + pad, bx0:x1 + pad].copy()
        box[max(0, y0 - by0 - 1):y1 - by0 + 1, max(0, x0 - bx0 - 1):x1 - bx0 + 1] = 0
        if box.any():
            touching.append(byid[i].name)
    missing = [p.name for p in included if p.id not in key_boxes]
    ok = not key_fallback and not touching and not missing
    if ok:
        detail = f"{len(key_boxes)} keys, each with a {KEY_PAD_MM:g} mm clear pad"
    else:
        bad = sorted(set(key_fallback + touching + missing))
        detail = "crowded: " + ", ".join(bad) + " (key squeezed in; consider removing a part)"
    checks.append(Check("Braille keys clear of lines", ok, detail))
    # 7. part count
    n = len(included)
    checks.append(Check("8 parts or fewer", n <= MAX_RECOMMENDED_PARTS,
                        f"{n} parts" if n <= MAX_RECOMMENDED_PARTS else
                        f"{n} parts: too many to read by touch — split into two graphics"))
    return checks
