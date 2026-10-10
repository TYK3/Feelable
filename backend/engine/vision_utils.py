"""Small, pure computer-vision helpers used by engine.analyze.

OWNER: Vision team.

Also contains `save_overlay()` - a debug renderer that paints the detected parts,
labels and leader lines over the working image so results can be inspected as a PNG:

    cd backend && python -m engine.vision_utils path/to/image.png out.png [--ai]
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- colour


def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> float32 Lab (L in 0..100, a/b roughly -128..127)."""
    return cv2.cvtColor(img_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def lab_of_rgb(rgb: Iterable[float]) -> np.ndarray:
    px = np.array([[list(rgb)]], dtype=np.float32) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_RGB2Lab)[0, 0]


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 colour difference (broadcasting)."""
    return np.sqrt(np.sum((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2, axis=-1))


# --------------------------------------------------------------------------- masks


def to_mask(boolarr: np.ndarray) -> np.ndarray:
    return boolarr.astype(np.uint8) * 255


def disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every hole of every component (outer contours drawn filled)."""
    out = np.zeros_like(mask)
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        cv2.drawContours(out, cs, -1, 255, -1)
    return out


def outer_contours(mask: np.ndarray, approx: bool = False) -> list[np.ndarray]:
    mode = cv2.CHAIN_APPROX_SIMPLE if approx else cv2.CHAIN_APPROX_NONE
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, mode)
    return list(cs)


def mask_polygons(mask: np.ndarray, eps: float = 1.5) -> list[list[tuple[int, int]]]:
    """Simplified outer contour per component, largest first."""
    polys = []
    for c in sorted(outer_contours(mask), key=cv2.contourArea, reverse=True):
        if len(c) < 3:
            continue
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(a) < 3:
            continue
        polys.append([(int(x), int(y)) for x, y in a])
    return polys


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def max_thickness(mask: np.ndarray) -> float:
    """Radius of the largest inscribed disk (px)."""
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return 0.0
    m = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return float(cv2.distanceTransform(m, cv2.DIST_L2, 3).max())


def rect_distance(px: float, py: float, box: tuple[int, int, int, int]) -> float:
    """Euclidean distance from a point to an axis-aligned rectangle (0 inside)."""
    x0, y0, x1, y1 = box
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return float(np.hypot(dx, dy))


def rects_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    dx = max(b[0] - a[2], 0, a[0] - b[2])
    dy = max(b[1] - a[3], 0, a[1] - b[3])
    return float(np.hypot(dx, dy))


def expand_box(box, pad: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))


def contour_points(mask: np.ndarray) -> np.ndarray:
    cs = outer_contours(mask)
    if not cs:
        return np.zeros((0, 2), np.float32)
    return np.concatenate([c.reshape(-1, 2) for c in cs]).astype(np.float32)


def point_mask_distance(px: float, py: float, mask: np.ndarray, pts: Optional[np.ndarray] = None) -> float:
    """0 if the point is inside the mask, else distance to the nearest outline pixel."""
    h, w = mask.shape
    ix, iy = int(round(px)), int(round(py))
    if 0 <= ix < w and 0 <= iy < h and mask[iy, ix]:
        return 0.0
    if pts is None:
        pts = contour_points(mask)
    if len(pts) == 0:
        return float("inf")
    return float(np.sqrt(((pts - (px, py)) ** 2).sum(1)).min())


def box_mask_distance(box, pts: np.ndarray) -> float:
    if len(pts) == 0:
        return float("inf")
    x0, y0, x1, y1 = box
    dx = np.maximum(np.maximum(x0 - pts[:, 0], 0), pts[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - pts[:, 1], 0), pts[:, 1] - y1)
    return float(np.sqrt(dx * dx + dy * dy).min())


# --------------------------------------------------------------------------- skeleton


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning (vectorised numpy; deterministic, no opencv-contrib needed)."""
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return m
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = np.pad(m[y0:y1, x0:x1], 1)
    for _ in range(100):
        changed = False
        for step in (0, 1):
            p2 = img[:-2, 1:-1]; p3 = img[:-2, 2:]; p4 = img[1:-1, 2:]; p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]; p7 = img[2:, :-2]; p8 = img[1:-1, :-2]; p9 = img[:-2, :-2]
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                c = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            rm = (img[1:-1, 1:-1] == 1) & (b >= 2) & (b <= 6) & (a == 1) & c
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = img[1:-1, 1:-1]
    return out


def skeleton_degree(skel: np.ndarray) -> np.ndarray:
    s = (skel > 0).astype(np.float32)
    k = np.ones((3, 3), np.float32)
    deg = cv2.filter2D(s, -1, k, borderType=cv2.BORDER_CONSTANT) - s
    return (deg * s).astype(np.int32)


_NBR = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # orthogonal first


class Trace:
    def __init__(self):
        self.points: list[tuple[int, int]] = []
        self.crossings: list[tuple[int, int]] = []
        self.end_junction = False


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return (v[0] / n, v[1] / n) if n > 1e-6 else (0.0, 0.0)


def trace_stroke(skel: np.ndarray, start: tuple[int, int], *, max_gap: int = 24,
                 gap_ok: Optional[Callable[[int, int], bool]] = None, max_len: int = 4000,
                 weak: Optional[np.ndarray] = None, max_weak_gap: int = 80) -> Trace:
    """Follow a 1-px skeleton from `start` (x, y).

    * goes straight through X crossings (the stroke re-appears straight ahead),
    * stops at T-junctions (records end_junction=True),
    * where the skeleton breaks (free end or T) it may jump along the heading: up to max_gap px
      when every gap pixel satisfies gap_ok(x, y) (e.g. "not paper"), or up to max_weak_gap px
      when the stroke stays faintly visible (`weak` mask) all along the gap (a leader drawn
      over a dark region / across an outlined border).
    """
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    tr = Trace()
    x, y = start
    tr.points.append((x, y))
    visited[y, x] = True

    def on(mask, px, py):
        return mask[max(0, py - 1):py + 2, max(0, px - 1):px + 2].any()

    def nbrs(cx, cy):
        out = []
        for dx, dy in _NBR:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] and not visited[ny, nx]:
                out.append((nx, ny))
        return out

    def heading():
        k = min(len(tr.points) - 1, 12)
        if k < 3:
            return None
        a_, b_ = tr.points[-1 - k], tr.points[-1]
        return _unit((b_[0] - a_[0], b_[1] - a_[1]))

    def try_jump(cx, cy, hd):
        if hd is None or gap_ok is None:
            return None
        limit = max_weak_gap if weak is not None else max_gap
        for s_ in range(2, limit + 1):
            qx, qy = cx + hd[0] * s_, cy + hd[1] * s_
            hit = None
            for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                ix, iy = int(round(qx)) + ox, int(round(qy)) + oy
                if 0 <= ix < w and 0 <= iy < h and skel[iy, ix] and not visited[iy, ix]:
                    hit = (ix, iy)
                    break
            gx, gy = int(round(qx)), int(round(qy))
            if not (0 <= gx < w and 0 <= gy < h):
                return None
            if hit is None:
                # the gap pixel itself must be allowed
                if not gap_ok(gx, gy):
                    return None
                if weak is not None and not on(weak, gx, gy):
                    return None      # the stroke really stops here
                if weak is None and s_ > max_gap:
                    return None
                continue
            # does the stroke carry on in the same direction after the hit?
            ahead = 0
            for t in range(s_ + 2, s_ + 8):
                ax, ay = int(round(cx + hd[0] * t)), int(round(cy + hd[1] * t))
                if 0 <= ax < w and 0 <= ay < h and (skel[max(0, ay - 1):ay + 2, max(0, ax - 1):ax + 2].any()
                                                     or (weak is not None and on(weak, ax, ay))):
                    ahead += 1
            if ahead >= 3 or s_ <= 3:
                return hit, s_
            # a line crossing ours (e.g. an outline): keep looking further along the heading
        return None

    def do_jump(cx, cy, hd, res):
        (hx, hy), s_ = res
        for t in range(1, s_):
            tr.points.append((int(round(cx + hd[0] * t)), int(round(cy + hd[1] * t))))
        visited[hy, hx] = True
        tr.points.append((hx, hy))
        return hx, hy

    after_jump = 0
    while len(tr.points) < max_len:
        hd = heading()
        cand = nbrs(x, y)
        if cand and after_jump > 0 and hd is not None:
            cand = [c for c in cand if (_unit((c[0] - x, c[1] - y))[0] * hd[0]
                                        + _unit((c[0] - x, c[1] - y))[1] * hd[1]) >= 0.5]
        if not cand:
            res = try_jump(x, y, hd)
            if res is None:
                break
            x, y = do_jump(x, y, hd, res)
            after_jump = 6
            continue
        after_jump = max(0, after_jump - 1)
        groups: list[list[tuple[int, int]]] = []
        for c in cand:
            for g in groups:
                if any(abs(c[0] - o[0]) <= 1 and abs(c[1] - o[1]) <= 1 for o in g):
                    g.append(c)
                    break
            else:
                groups.append([c])
        if len(groups) == 1:
            nx, ny = groups[0][0]
        else:
            # junction: carry straight on if the stroke re-appears ahead (an X crossing)
            hit = None
            if hd is not None:
                for s_ in range(4, 11):
                    qx, qy = x + hd[0] * s_, y + hd[1] * s_
                    for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                        ix, iy = int(round(qx)) + ox, int(round(qy)) + oy
                        if 0 <= ix < w and 0 <= iy < h and skel[iy, ix] and not visited[iy, ix]:
                            hit = (ix, iy, s_)
                            break
                    if hit:
                        break
            straight = False
            if hit is not None:
                ix, iy, s_ = hit
                ahead = 0
                for s2 in range(s_ + 3, s_ + 9):
                    qx, qy = int(round(x + hd[0] * s2)), int(round(y + hd[1] * s2))
                    if skel[max(0, qy - 2):qy + 3, max(0, qx - 2):qx + 3].any():
                        ahead += 1
                straight = ahead >= 3
            if straight:
                tr.crossings.append((x, y))
                for t in range(1, s_):
                    tr.points.append((int(round(x + hd[0] * t)), int(round(y + hd[1] * t))))
                nx, ny = ix, iy
            else:
                res = try_jump(x, y, hd) if weak is not None else None
                if res is None:
                    tr.end_junction = True
                    break
                tr.crossings.append((x, y))
                x, y = do_jump(x, y, hd, res)
                after_jump = 6
                continue
        visited[ny, nx] = True
        x, y = nx, ny
        tr.points.append((x, y))
    return tr


# --------------------------------------------------------------------------- text

_FIG_RE = re.compile(r"^\s*(?:fig(?:ure)?s?)\s*\.?\s*[0-9ivx]+(?:\s*[.\-–]\s*[0-9]+)*\s*[a-z]?\b\s*[:.\-–—)]*\s*", re.I)
_FIG_ONLY_RE = re.compile(r"^\s*(?:fig(?:ure)?s?)\b\.?\s*[:.\-–—]?\s*", re.I)
_ENUM_RE = re.compile(r"^\s*(?:\(?\d{1,2}[.)]|\(?[a-hA-H][.)])\s+")
_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.I)
_SECTION_RE = re.compile(r"^\s*(?:chapter|section|unit)?\s*\d+(?:\s*[.\-]\s*\d+)*\s+", re.I)


def clean_text(text: str) -> str:
    """Strip figure prefixes, enumerators, stray punctuation and OCR junk; keep case."""
    t = text.replace("’", "'").replace("‘", "'")
    # OCR sometimes glues the figure number to the next word: "Fig. 7.4The heart" -> "Fig. 7.4 The heart"
    t = re.sub(r"^(\s*fig(?:ure)?s?\s*\.?\s*[0-9][0-9.\-–]*)(?=[A-Za-z]{2})", r"\1 ", t, flags=re.I)
    t = _FIG_RE.sub("", t)
    t = _FIG_ONLY_RE.sub("", t)
    t = _ENUM_RE.sub("", t)
    t = re.sub(r"[^A-Za-z0-9'\- ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -'")
    return t


def clean_name(text: str) -> str:
    return respace(clean_text(text).lower())


def clean_title(text: str) -> str:
    t = re.sub(r"\([^)]*\)", " ", text)            # "(front view)"
    t = clean_name(_SECTION_RE.sub("", clean_text(t)))
    t = _ARTICLE_RE.sub("", t)
    t = re.sub(r"\b(?:typical|simplified|labell?ed|diagram of(?: a| an| the)?)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _ARTICLE_RE.sub("", t).strip()
    return t


_TITLE_STOP = {
    "a", "an", "the", "inside", "typical", "simple", "simplified", "diagram", "labelled", "labeled", "parts",
    "part", "structure", "structures", "view", "front", "side", "top", "cross", "section", "cutaway", "model",
    "figure", "fig", "showing", "basic", "main", "and", "how", "works", "work", "layers", "inner", "outer",
    "favourite", "favorite",
}
_PREPS = {"in", "of", "on", "at", "for", "from", "with", "through", "inside", "showing", "by", "to"}


def main_noun(title: str) -> str:
    """'inside the earth' -> 'earth', 'parts of a flower' -> 'flower', 'favourite fruit in class 7b' -> 'fruit'."""
    words = [w for w in re.split(r"[^a-z]+", title.lower()) if w]
    chunks, cur = [], []
    for wd in words:
        if wd in _PREPS:
            chunks.append(cur)
            cur = []
        else:
            cur.append(wd)
    chunks.append(cur)
    for ch in chunks:
        good = [wd for wd in ch if wd not in _TITLE_STOP and len(wd) > 2]
        if good:
            return good[-1]
    return ""


# common words in school diagrams, used to re-space OCR output like "outercore" -> "outer core"
_VOCAB = set("""
outer inner core crust mantle cell wall membrane nucleus nucleolus cytoplasm vacuole chloroplast
mitochondrion ribosome ribosomes golgi body apparatus left right upper lower atrium ventricle vena cava
pulmonary artery vein aorta septum valve blood vessel water cycle cloud sun sea ocean mountain river lake
rain snow ice land ground soil root roots stem leaf leaves flower petal sepal stamen anther filament pistil
stigma style ovary ovule seed fruit bud branch trunk bark small large intestine stomach liver kidney lung
lungs heart brain spinal cord bone muscle skin eye lens retina cornea iris pupil optic nerve ear drum
canal tube light ray mirror prism battery switch bulb resistor wire circuit cell plate north south east
west pole magnet field line lines force air gas liquid solid particle electron proton neutron orbit
planet moon earth star layer upper surface top bottom front back side main central large food chain
web energy plant animal cell sap tissue xylem phloem guard stoma stomata epidermis cuticle palisade
spongy mesophyll lower egg sperm tail head neck foot hand arm leg wing nose mouth tooth teeth tongue
throat windpipe trachea bronchus bronchi alveoli diaphragm rib ribs cage oesophagus esophagus gall
bladder pancreas rectum anus large small red white blood cells platelets plasma
""".split())


_SHORT_WORDS = {"a", "an", "of", "in", "on", "to", "at", "by", "is", "as", "or", "and", "the"}
_VOCAB |= set("""
the and inside typical simple simplified human heart water cycle plant cell animal earth layers favourite favorite
fruit class series circuit diagram parts structure section cross front view side top bottom model labelled labeled
flower leaf root stem eye ear brain lung lungs kidney skeleton digestive system solar planet moon sun volcano
river map world body blood flow food chain web life rock atom molecule electric current magnet light sound
""".split())


def respace(name: str) -> str:
    """Split OCR tokens that lost their spaces when they are made of known words."""
    out = []
    for tok in name.split():
        if tok in _VOCAB or len(tok) < 7 or not tok.isalpha():
            out.append(tok)
            continue
        n = len(tok)
        best: list = [None] * (n + 1)
        best[0] = []
        for i in range(1, n + 1):
            for j in range(max(0, i - 14), i):
                w = tok[j:i]
                if best[j] is not None and ((len(w) >= 3 and w in _VOCAB) or w in _SHORT_WORDS):
                    cand = best[j] + [tok[j:i]]
                    if best[i] is None or len(cand) < len(best[i]):
                        best[i] = cand
        out.extend(best[n] if best[n] and len(best[n]) > 1 else [tok])
    return " ".join(out)


def squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def capitalise(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# --------------------------------------------------------------------------- debug overlay

_DEBUG_COLOURS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                  (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
                  (128, 0, 0), (0, 0, 128)]


def save_overlay(analysis, path: str, debug: Optional[dict] = None) -> np.ndarray:
    """Render parts (coloured, outlined, named), label boxes, leaders and linework to a PNG."""
    img = analysis.image.astype(np.float32)
    base = (img * 0.35 + 255 * 0.65).astype(np.uint8)
    over = base.copy()
    for i, p in enumerate(analysis.parts):
        col = _DEBUG_COLOURS[i % len(_DEBUG_COLOURS)]
        m = p.mask > 0
        over[m] = (0.45 * np.array(col) + 0.55 * over[m]).astype(np.uint8)
    over[analysis.linework > 0] = (0, 0, 0)
    for i, p in enumerate(analysis.parts):
        col = _DEBUG_COLOURS[i % len(_DEBUG_COLOURS)]
        for poly in p.polygons:
            cv2.polylines(over, [np.array(poly, np.int32)], True, col, 2)
    for lb in analysis.labels:
        c = {"title": (200, 0, 200), "caption": (120, 120, 120), "part_label": (0, 90, 220)}.get(lb.role, (150, 150, 0))
        cv2.rectangle(over, lb.bbox[:2], lb.bbox[2:], c, 1)
    if debug:
        for ld in debug.get("leaders", []):
            pts = np.array(ld["points"], np.int32)
            cv2.polylines(over, [pts], False, (255, 0, 0), 2)
            cv2.circle(over, tuple(int(v) for v in ld["end"]), 5, (255, 0, 0), -1)
    for i, p in enumerate(analysis.parts):
        col = _DEBUG_COLOURS[i % len(_DEBUG_COLOURS)]
        if not p.polygons:
            continue
        x0, y0, x1, y1 = p.bbox
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ys, xs = np.nonzero(p.mask)
        if len(xs):
            j = len(xs) // 2
            cx, cy = int(xs[j]), int(ys[j])
        txt = f"{p.id} {p.name}" + (f" x{p.count}" if p.count > 1 else "") + ("" if p.include else " [off]")
        cv2.putText(over, txt, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(over, txt, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(int(c * 0.6) for c in col), 1, cv2.LINE_AA)
    lines = [f"title: {analysis.title} | type: {analysis.diagram_type} | flattened: {analysis.page_flattened}"
             f" | {analysis.timings.get('analyze', 0):.2f}s"]
    lines += ["dropped: " + "; ".join(analysis.dropped)] if analysis.dropped else []
    lines += ["warn: " + "; ".join(analysis.warnings)] if analysis.warnings else []
    pad = 22 * len(lines) + 8
    canvas = np.full((over.shape[0] + pad, over.shape[1], 3), 255, np.uint8)
    canvas[: over.shape[0]] = over
    for k, ln in enumerate(lines):
        cv2.putText(canvas, ln[:170], (6, over.shape[0] + 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return canvas


if __name__ == "__main__":  # pragma: no cover - debug helper
    import sys

    from engine.analyze import _analyze_impl, load_image

    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        im = load_image(f.read())
    a, dbg = _analyze_impl(im, use_ai="--ai" in sys.argv)
    save_overlay(a, dst, dbg)
    for p in a.parts:
        print(p.id, repr(p.name), "count", p.count, "area", p.area, "color", p.color, "|", p.description)
    print("title:", a.title, "| type:", a.diagram_type, "| dropped:", a.dropped, "| warnings:", a.warnings,
          "| timings:", a.timings)


# ---------------------------------------------------------------- name sanity (robustness round 2)
_PROFANE_RE = re.compile(r"(fuck|shit|cunt|bitch|dick|cock|piss|slut|whore|nigg|fag|wank|twat|bollock|arse)", re.I)
_FUNCTION_WORDS = {"the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with", "by", "from", "is", "are",
                   "at", "as", "that", "which", "this", "its", "it", "into"}
_VERBISH = {"makes", "make", "allow", "allows", "allowing", "form", "forms", "contains", "contain", "produces",
            "produce", "carries", "carry", "helps", "help", "digests", "digest", "stores", "store", "controls",
            "control", "moves", "move", "flows", "flow", "breaks", "converts", "transports", "provides", "protects"}
_BRAND_RE = re.compile(r"(basics\b|\binc\b|\bltd\b|\bcorp\b|\.com|www\.|®|©|™)", re.I)
_PART_N_RE = re.compile(r"^part \d+$")


def name_problem(name: str) -> str | None:
    """Why an OCR-derived name must not be shown to a student (None = fine).

    Catches profanity-like tokens, brand text, sentence fragments ("digests food and"),
    and mixed letter/digit noise ("x7b3"). Real labels like "left atrium" pass."""
    n = (name or "").strip().lower()
    if not n or _PART_N_RE.match(n):
        return None
    if _PROFANE_RE.search(n):
        return "profanity"
    if _BRAND_RE.search(n):
        return "brand"
    toks = re.findall(r"[a-z0-9]+", n)
    if not toks:
        return "no letters"
    if any(re.search(r"[a-z]\d|\d[a-z]", t) for t in toks) and not re.fullmatch(r"[a-z]{1,3}\d{1,2}", toks[0]):
        return "noise"
    if len(toks) >= 5:
        return "sentence"
    if len(toks) >= 2 and (toks[-1] in _FUNCTION_WORDS or toks[0] in {"and", "or", "of", "to", "the", "that"}):
        return "fragment"
    if len(toks) >= 3 and any(t in _VERBISH for t in toks):
        return "sentence"
    if len(toks) == 1 and len(toks[0]) >= 4 and not re.search(r"[aeiouy]", toks[0]):
        return "noise"
    if any(re.search(r"q(?!u)", t) for t in toks):
        return "noise"                                 # OCR misread ("qhid"): English q is followed by u
    return None


def title_problem(title: str) -> str | None:
    t = (title or "").strip().lower()
    if not t:
        return None
    if _PROFANE_RE.search(t) or _BRAND_RE.search(t):
        return "unsafe"
    toks = re.findall(r"[a-z0-9]+", t)
    if len(toks) >= 7:
        return "sentence"
    if len(toks) >= 2 and toks[-1] in _FUNCTION_WORDS:
        return "fragment"
    if len(toks) >= 4 and sum(tk in _VERBISH for tk in toks) >= 1 and not t.startswith(("how ", "parts of", "inside")):
        return "sentence"
    return None


# ---------------------------------------------------------------- is this label a structure? (UX07)
_MODIFIERS = {"smooth", "rough", "intermediate", "outer", "inner", "left", "right", "upper", "lower", "small",
              "large", "central", "middle", "top", "bottom", "front", "back", "main", "major", "minor", "anterior",
              "posterior", "ascending", "descending", "transverse"}
_NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "both", "several", "many"}
_CAPTION_WORDS = {"contains", "contain", "maintains", "maintain", "connect", "connects", "against", "between",
                  "allows", "allow", "filled", "made", "used", "called", "where", "which", "that", "this", "these",
                  "site", "stores", "store", "forms", "form", "keeps", "helps", "gives", "carries", "holds", "is", "are"}


def looks_like_structure(text: str) -> bool:
    """True if a label reads like the name of a thing a student could feel ("pancreas",
    "endoplasmic reticulum"), not an explanatory phrase ("channels connect", "filled with cell
    sap", "two plant cells") or a lone modifier left over from a two-line name ("smooth")."""
    t = (text or "").strip().lower()
    if not t or name_problem(t):
        return False
    toks = re.findall(r"[a-z]+", t)
    if not toks or re.search(r"\d", t):
        return False
    if any(w in _NUMBER_WORDS for w in toks) or any(w in _CAPTION_WORDS for w in toks):
        return False
    if len(toks) == 1 and toks[0] in _MODIFIERS:
        return False
    if toks[0] in {"a", "an", "the", "as", "in", "of", "and", "or", "with"}:
        return False                                   # the tail of a caption: "an animal cell"
    if toks[0].endswith("ed") and len(toks) > 1:
        return False
    return len(toks) <= 4
