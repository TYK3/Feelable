"""Diagram understanding: image -> Analysis (parts, labels, linework, title).

OWNER: Vision team.

Pipeline (all offline, deterministic):
  normalise -> page flattening (photos) -> lighting flattening (photos) -> OCR -> text removal
  -> leader lines -> mode decision (colour / line-art) -> segmentation -> label matching
  -> naming / descriptions -> optional AI enrichment.
"""
from __future__ import annotations

import io
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from engine import ai
from engine.lexicon import is_term
from engine.ocr import MIN_CONFIDENCE as OCR_MIN_CONFIDENCE, read_crop, run_ocr
from engine.types import WORKING_LONG_SIDE, Analysis, Label, Part
from engine.vision_utils import (
    box_mask_distance, capitalise, clean_name, clean_text, clean_title, contour_points, delta_e, disk,
    expand_box, fill_holes, main_noun, mask_bbox, mask_polygons, outer_contours, rect_distance, rgb_to_lab,
    skeleton_degree,
    thin, to_mask, trace_stroke, name_problem,
)

# ----------------------------------------------------------------------------- tunables
LINEART_COLOUR_FRACTION = 0.03     # < 3 % coloured pixels -> line-art mode ...
GREY_FILL_FRACTION = 0.05          # ... unless >= 5 % of the page is flat grey fill
COLOUR_MIN_AREA_FRAC = 0.0003      # colour components smaller than this are clutter
LINEART_MIN_AREA_FRAC = 0.003      # enclosed line-art regions smaller than this are ignored
SOLID_INK_RADIUS = 6               # line-art: ink still there after opening with this disk is a solid shape
MAX_REGION_FRAC = 0.80             # a "part" covering more than this is page / frame, not content
MAX_PARTS = 16
KMEANS_K = 10
KMEANS_SAMPLES = 40000
MERGE_DELTA_E = 12.0
LEADER_START_PX = 16
LEADER_MIN_LEN = 15
LEADER_SNAP_PX = 15
LEADER_FAR_START_K = 2.5           # fallback: a leader may start up to 2.5 text heights from its label
LEADER_FAR_START_MAX = 70
LEADER_RAY_PX = 60                 # a leader stopping short of a shape: look this far ahead
NEAREST_LABEL_PX = 120
SPLIT_LABEL_PX = 150               # split_part: an unmatched label this close names a new piece
LINEWORK_MIN_LEN = 25
SEED = 1234
LOAD_MAX_PIXELS = 80_000_000       # decompression-bomb guard: refuse to decode more than 80 MP
LOAD_MAX_SIDE = 3000               # decoded images are downscaled to this long side straight away
LOAD_DRAFT_SIDE = 1600             # JPEG draft target (decoder picks scale 1/2..1/8 >= this)
LOAD_MAX_ASPECT = 25.0             # wider / taller than 25:1 is a strip, not a diagram

# input kind 
FEAT_SIDE = 400                    # features are measured on a copy this big
BLANK_EDGE = 0.002                 # Canny edge density below this: nothing drawn
BLANK_STD = 4.0                    # or grey-level spread below this
TEXT_PAGE_FRAC = 0.22              # OCR boxes cover >= 22 % of the picture (with >= 3 lines) -> text page
# photos
PHOTO_SEG_SIDE = 400               # GrabCut runs on a copy this big (cheap on a 0.15 CPU server)
GRABCUT_ITERS = 3
PHOTO_DETAIL_MIN_FRAC = 0.03       # an interior detail must cover >= 3 % of the object
PHOTO_MAX_DETAILS = 3
DETAIL_MIN_CONTRAST = 20.0          # Lab distance between a detail and the object around it
PHOTO_MAX_AI_PARTS = 8
SCENE_SIDE = 160
SCENE_MIN_FRAC = 0.06              # scene regions smaller than 6 % of the picture are dropped
PHOTO_SCENE_MAX = 5
SCENE_MERGE_DE = 12.0              # scene clusters closer than this in colour are one region

Image.MAX_IMAGE_PIXELS =LOAD_MAX_PIXELS * 2   # we check ourselves; keep Pillow's own guard as backstop


# ============================================================================ loading


class ImageTooBig(ValueError):
    """More pixels than we will ever decode (decompression-bomb guard). str() == "too big"."""
    code = "too_big"

    def __init__(self, msg: str = "too big"):
        super().__init__(msg)


class NotAnImage(ValueError):
    code = "not_image"

    def __init__(self, msg: str = "Not an image"):
        super().__init__(msg)


_HEIF_TRIED = False


def _register_heif() -> None:
    """HEIC/HEIF (iPhone photos) via pillow-heif, if installed. Lazy: only when a file needs it."""
    global _HEIF_TRIED
    if _HEIF_TRIED:
        return
    _HEIF_TRIED = True
    try:
        import pillow_heif  # type: ignore
        pillow_heif.register_heif_opener()
    except Exception:  # pragma: no cover - optional dependency
        pass


def _looks_heif(data: bytes) -> bool:
    return len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in (
        b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"avif", b"avis")


def _flatten_mode(im: Image.Image) -> Image.Image:
    """Any Pillow mode -> RGB: alpha composited over white, 16-bit / float stretched to 8-bit."""
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        a = np.asarray(im, dtype=np.float32)
        lo, hi = (float(np.percentile(a, 0.5)), float(np.percentile(a, 99.5))) if a.size else (0.0, 1.0)
        if hi <= lo:
            lo, hi = float(a.min()), float(max(a.max(), a.min() + 1))
        a = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(a).convert("RGB")
    if mode in ("RGBA", "LA", "PA", "RGBa", "La") or (mode == "P" and "transparency" in im.info) \
            or (mode in ("L", "RGB") and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return bg.convert("RGB")
    if mode == "RGB":
        return im
    return im.convert("RGB")      # L, 1, P, CMYK, YCbCr, LAB, HSV ...


_TRUNC_LOCK = threading.Lock()


def _load_truncated(data: bytes, fmt) -> Image.Image:
    """Decode a JPEG/PNG whose last bytes are missing (common for images saved from the web).

    Accepted only when the missing part is tiny: the decoded bottom rows must still hold picture
    content, not the flat fill Pillow uses for absent data. Otherwise it's NotAnImage."""
    from PIL import ImageFile
    with _TRUNC_LOCK:
        old = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            im = Image.open(io.BytesIO(data))
            if im.format == "JPEG" and max(im.size) > LOAD_MAX_SIDE:
                k = LOAD_DRAFT_SIDE / max(im.size)
                im.draft("RGB", (max(1, int(im.size[0] * k)), max(1, int(im.size[1] * k))))
            im.load()
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = old
    g = np.asarray(im.convert("L"))
    tail = g[int(g.shape[0] * 0.97):]
    if tail.size == 0 or (float(tail.std()) < 1.0 and float(np.abs(tail.astype(int) - int(tail.mean())).max()) <= 2
                          and float(g[: int(g.shape[0] * 0.9)].std()) > 3.0):
        raise NotAnImage()
    return im


def load_image(data: bytes, notes: list | None = None) -> np.ndarray:
    """Decode image bytes -> RGB uint8 with bounded memory.

    * PNG / JPEG / WebP / GIF (first frame) / TIFF / BMP / HEIC / HEIF (pillow-heif, if installed);
    * EXIF orientation applied; transparency composited over white; 16-bit, CMYK, palette and
      grey images converted;
    * JPEGs are decoded at reduced scale (``draft``) and everything is downscaled right after
      decoding so the long side is at most LOAD_MAX_SIDE px (a 48 MP phone photo peaks < 150 MB);
    * raises ImageTooBig("too big") over LOAD_MAX_PIXELS, NotAnImage("Not an image") for
      undecodable bytes and ValueError with a human message for unusable shapes.
    """
    if not data:
        raise NotAnImage()
    if _looks_heif(data):
        _register_heif()
    try:
        im = Image.open(io.BytesIO(data))
    except Exception as exc:  # PIL raises many types
        if not _HEIF_TRIED:
            _register_heif()
            try:
                im = Image.open(io.BytesIO(data))
            except Exception:
                raise NotAnImage() from exc
        else:
            raise NotAnImage() from exc
    try:
        w0, h0 = im.size
        if w0 * h0 > LOAD_MAX_PIXELS:
            raise ImageTooBig()
        if w0 < 1 or h0 < 1:
            raise NotAnImage()
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)                                   # animated GIF / WebP / multi-page TIFF
            if notes is not None:   # F10: say that only the first frame was used
                notes.append("This is an animated image. Only its first frame was used.")
        if im.format == "JPEG" and max(w0, h0) > LOAD_MAX_SIDE:
            # decode at 1/2, 1/4 or 1/8 scale: never allocates the full 48 MP frame
            k = LOAD_DRAFT_SIDE / max(w0, h0)
            im.draft("RGB", (max(1, int(w0 * k)), max(1, int(h0 * k))))
        try:
            im.load()
        except OSError:
            im = _load_truncated(data, im.format)        # a few bytes missing at the end: keep it
            if notes is not None:   # F10: a recovered file is not a normal successful scan
                notes.append("The image file was incomplete at its end. The missing edge was "
                             "recovered; check the result carefully or send a complete file.")
        if max(im.size) > LOAD_MAX_SIDE:
            f = max(im.size) / LOAD_MAX_SIDE
            if f >= 2 and im.mode not in ("P", "1", "I;16", "I;16B", "I;16L", "I;16N"):
                im = im.reduce(int(f))                   # cheap box downscale first
            if max(im.size) > LOAD_MAX_SIDE:
                if im.mode in ("P", "1"):
                    im = im.convert("RGBA" if "transparency" in im.info else "RGB")
                s = LOAD_MAX_SIDE / max(im.size)
                im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))),
                               Image.Resampling.LANCZOS if im.mode not in ("I", "F", "I;16") else Image.Resampling.NEAREST)
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:  # broken EXIF must not lose the picture
            pass
        im = _flatten_mode(im)
        arr = np.asarray(im, dtype=np.uint8)
    except (ImageTooBig, NotAnImage):
        raise
    except Image.DecompressionBombError as exc:
        raise ImageTooBig() from exc
    except MemoryError as exc:
        raise ImageTooBig() from exc
    except Exception as exc:  # truncated / corrupt files
        raise NotAnImage() from exc
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise NotAnImage()
    h, w = arr.shape[:2]
    if max(h, w) / min(h, w) > LOAD_MAX_ASPECT:
        raise ValueError(f"That picture is a very thin strip ({w}×{h} px) — take a photo of the whole diagram")
    return np.ascontiguousarray(arr)


def _to_working(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = WORKING_LONG_SIDE / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if (nw, nh) == (w, h):
        return img.copy()
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp)


# ============================================================================ photo handling


def _order_quad(q: np.ndarray) -> np.ndarray:
    q = q.reshape(4, 2).astype(np.float32)
    s = q.sum(1)
    d = np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def _find_page(img: np.ndarray) -> Optional[np.ndarray]:
    """Return the 4 page corners (tl, tr, br, bl) in img coords, or None."""
    h, w = img.shape[:2]
    sc = 600.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    sh, sw = gray.shape
    total = float(sh * sw)
    cands = []
    med = float(np.median(gray))
    edges = cv2.Canny(gray, max(10, 0.4 * med), max(30, 1.0 * med))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands += sorted(cs, key=cv2.contourArea, reverse=True)[:5]
    # bright-paper component (robust when the page edge is broken by shadows)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs2, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands += sorted(cs2, key=cv2.contourArea, reverse=True)[:2]
    for c in cands:
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.02, 0.03, 0.04, 0.05):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                quad = ap
                break
        if quad is None or not cv2.isContourConvex(quad):
            continue
        area = cv2.contourArea(quad)
        if not (0.35 * total <= area <= 0.97 * total):
            continue
        q = _order_quad(quad)
        # all corners must be at least a little away from the image corners-on-border case
        inside = np.zeros_like(gray)
        cv2.fillConvexPoly(inside, q.astype(np.int32), 255)
        inner = cv2.erode(inside, np.ones((7, 7), np.uint8)) > 0
        outer = cv2.dilate(inside, np.ones((7, 7), np.uint8)) == 0
        if outer.sum() < 0.02 * total or inner.sum() == 0:
            continue
        if float(gray[inner].mean()) < float(gray[outer].mean()) + 15:
            continue
        sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
        if min(sides) < 0.2 * max(sides):
            continue
        return q / sc
    return None


def _warp_page(img: np.ndarray, q: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = q
    wd = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    ht = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    wd, ht = int(round(wd)), int(round(ht))
    dst = np.array([[0, 0], [wd - 1, 0], [wd - 1, ht - 1], [0, ht - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    out = cv2.warpPerspective(img, M, (wd, ht), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    inset = max(2, int(0.008 * max(wd, ht)))   # trim the paper edge / table sliver
    return out[inset:ht - inset, inset:wd - inset]


def _border_mode_colour(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    b = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                             img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    q = (border // 16).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    k = vals[np.argmax(counts)]
    sel = border[keys == k]
    return np.median(sel, axis=0).astype(np.float32)


def _looks_like_photo(img: np.ndarray) -> bool:
    """Clean screenshots have large areas of exactly one colour and no sensor noise."""
    bg = _border_mode_colour(img)
    exact = np.all(np.abs(img.astype(np.int16) - bg.astype(np.int16)) <= 2, axis=2).mean()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    noise = float(np.median(np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16))))
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    uneven = float(np.percentile(small, 95) - np.percentile(small, 60))
    return exact < 0.25 and (noise >= 1.0 or uneven > 20 or exact < 0.02)


def _paper_model(img: np.ndarray) -> np.ndarray:
    """Smooth per-channel estimate of the paper brightness (polynomial fit, ink rejected)."""
    h, w = img.shape[:2]
    sc = 160.0 / max(h, w)
    sw, sh = max(8, int(w * sc)), max(8, int(h * sc))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float64)
    ys, xs = np.mgrid[0:sh, 0:sw]
    xn, yn = (xs / (sw - 1) * 2 - 1).ravel(), (ys / (sh - 1) * 2 - 1).ravel()
    A = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn,
                  xn ** 3, xn * xn * yn, xn * yn * yn, yn ** 3], 1)
    gray = small.mean(2).ravel()
    sel = np.ones_like(gray, bool)
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A[sel], gray[sel], rcond=None)
        fit = A @ coef
        res = gray - fit
        sd = max(4.0, float(res[sel].std()))
        new = res > -1.2 * sd
        if new.sum() < 0.15 * len(gray) or np.array_equal(new, sel):
            break
        sel = new
    chans = []
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A[sel], small[:, :, c].ravel()[sel], rcond=None)
        chans.append((A @ coef).reshape(sh, sw))
    bgs = np.clip(np.stack(chans, 2), 30, 255).astype(np.float32)
    return cv2.resize(bgs, (w, h), interpolation=cv2.INTER_CUBIC)


def _flatten_lighting(img: np.ndarray) -> np.ndarray:
    bg = _paper_model(img)
    out = img.astype(np.float32) / np.maximum(bg, 1.0) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.medianBlur(out, 3)


# ============================================================================ text


def _line_dist_off_border(lw: np.ndarray, region: np.ndarray, box, band: int = 9) -> float:
    """Distance (px) from text box `box` to the nearest linework pixel drawn *inside* `region`
    (the box / cell the text sits in), ignoring its own outline and arrows that end at its edge.
    inf when there is none nearby."""
    h, w = lw.shape[:2]
    x0, y0, x1, y1 = box
    m = 60
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
    sub = lw[Y0:Y1, X0:X1] > 0
    reg = (region[Y0:Y1, X0:X1] > 0).astype(np.uint8)
    if not sub.any():
        return float("inf")
    k = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    inner = cv2.erode(reg, k) > 0       # an arrow drawn over the region, not its outline or
    ys, xs = np.nonzero(sub & inner)    # an arrow arriving at its edge from outside
    if not len(xs):
        return float("inf")
    xs = xs + X0
    ys = ys + Y0
    dx = np.maximum(0, np.maximum(x0 - xs, xs - x1))
    dy = np.maximum(0, np.maximum(y0 - ys, ys - y1))
    return float(np.hypot(dx, dy).min())


def _assign_roles(labels: list[Label], w: int, h: int) -> None:
    if not labels:
        return
    heights = [lb.bbox[3] - lb.bbox[1] for lb in labels]
    tallest = max(heights)
    top_zone, bottom_zone = 0.15 * h, 0.85 * h
    title_idx = None
    best = -1.0
    for i, lb in enumerate(labels):
        words = lb.text.split()
        if not any(ch.isalpha() for ch in lb.text):
            continue
        is_fig = bool(re.match(r"^\s*fig(?:ure)?\b", lb.text, re.I)) and len(words) <= 12
        in_top = lb.bbox[1] <= top_zone and lb.bbox[3] <= top_zone + 0.05 * h
        is_tallest = heights[i] >= 0.9 * tallest
        # a "Fig. N ..." caption anywhere beats a page heading; else the tallest text at the top
        if is_fig and lb.bbox[1] < 0.6 * h:
            score = 1000 - lb.bbox[1] / h
        elif in_top and is_tallest and len(words) <= 10:
            score = heights[i] + len(words)
        else:
            continue
        if score > best:
            best, title_idx = score, i
    if title_idx is None:
        # a heading in ordinary type (a flowchart / map title): the topmost line of 3+ words,
        # at least as tall as the typical label, alone at the top of the page
        alpha = [i for i, lb in enumerate(labels) if any(ch.isalpha() for ch in lb.text)]
        if alpha:
            med = float(np.median([heights[i] for i in alpha]))
            top = min(alpha, key=lambda i: labels[i].bbox[1])
            lb = labels[top]
            others_top = min((labels[i].bbox[1] for i in alpha if i != top), default=h)
            if (lb.bbox[3] <= top_zone + 0.05 * h and 3 <= len(lb.text.split()) <= 10
                    and heights[top] >= 0.95 * med and others_top >= lb.bbox[3] + 0.5 * heights[top]
                    and not re.search(r"[=<>?]", lb.text)):
                title_idx = top
    if title_idx is not None:
        t = labels[title_idx]
        t.role = "title"
        ty0, ty1 = t.bbox[1], t.bbox[3]
        for lb in labels:  # same text line belongs to the title too
            if lb is t:
                continue
            ov = min(ty1, lb.bbox[3]) - max(ty0, lb.bbox[1])
            if ov > 0.5 * min(ty1 - ty0, lb.bbox[3] - lb.bbox[1]):
                lb.role = "title"
    for lb in labels:
        if lb.role == "title":
            continue
        words = lb.text.split()
        letters = sum(ch.isalpha() for ch in lb.text)
        if lb.bbox[1] >= bottom_zone and len(words) > 6:
            lb.role = "caption"
        elif len(words) <= 4 and letters >= 2 and clean_name(lb.text):
            lb.role = "part_label"
        else:
            lb.role = "other"


def _title_text(labels: list[Label]) -> str:
    ts = sorted([lb for lb in labels if lb.role == "title"], key=lambda lb: lb.bbox[0])
    if not ts:
        return ""
    return clean_title(" ".join(lb.text for lb in ts))


def _text_mask(img: np.ndarray, lab: np.ndarray, labels: list[Label]) -> np.ndarray:
    """Pixels of text glyphs (strokes inside OCR boxes, not lines merely passing through)."""
    h, w = img.shape[:2]
    out = np.zeros((h, w), np.uint8)
    for lb in labels:
        bx = expand_box(lb.bbox, 3, w, h)
        ox = expand_box(lb.bbox, 14, w, h)
        crop = lab[ox[1]:ox[3], ox[0]:ox[2]]
        if crop.size == 0:
            continue
        inner = np.zeros(crop.shape[:2], bool)
        inner[bx[1] - ox[1]:bx[3] - ox[1], bx[0] - ox[0]:bx[2] - ox[0]] = True
        rim = np.zeros_like(inner)
        rim[bx[1] - ox[1]:bx[3] - ox[1], bx[0] - ox[0]:bx[2] - ox[0]] = True
        rim &= ~cv2.erode(rim.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ref = np.median(crop[rim], axis=0)
        diff = delta_e(crop, ref) > 22
        n, cc, st, _ = cv2.connectedComponentsWithStats(diff.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            comp = cc == i
            tot = st[i, cv2.CC_STAT_AREA]
            ins = int((comp & inner).sum())
            if ins == 0:
                continue
            if tot - ins <= 0.3 * tot:
                keep[i] = True
        m = keep[cc] & inner
        if m.any():
            sub = out[ox[1]:ox[3], ox[0]:ox[2]]
            sub[m] = 255
    if out.any():
        out = cv2.dilate(out, disk(2))
    return out


# ============================================================================ leaders


@dataclass
class _Leader:
    label_idx: int
    points: list
    end: tuple
    crossings: list
    end_junction: bool
    dot_end: bool = False          # ends in a drawn dot: the end point is exact

    @property
    def heading(self):
        a_, b_ = self.points[-min(len(self.points), 12)], self.points[-1]
        v = np.array([b_[0] - a_[0], b_[1] - a_[1]], np.float64)
        n = float(np.hypot(*v))
        return (v / n) if n > 1e-6 else None


def _blackhat(img: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return np.max(np.stack([cv2.morphologyEx(img[:, :, c], cv2.MORPH_BLACKHAT, k) for c in range(3)], 0), 0)


def _thin_dark(img: np.ndarray, thresh: int = 35, bh: Optional[np.ndarray] = None) -> np.ndarray:
    if bh is None:
        bh = _blackhat(img)
    m = (bh > thresh).astype(np.uint8) * 255
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] < 6
    m[small[cc]] = 0
    return m


def _find_leaders(img: np.ndarray, labels: list[Label], nonbg: np.ndarray,
                  protect_crossings: bool) -> tuple[list[_Leader], np.ndarray]:
    h, w = img.shape[:2]
    leader_mask = np.zeros((h, w), np.uint8)
    idx = [i for i, lb in enumerate(labels) if lb.role == "part_label"]
    if not idx:
        return [], leader_mask
    bh = _blackhat(img)
    thin_m = _thin_dark(img, bh=bh)
    weak = bh > 8
    skel = thin(thin_m)
    dt_thin = cv2.distanceTransform((thin_m > 0).astype(np.uint8), cv2.DIST_L2, 3)
    deg = skeleton_degree(skel)
    ey, ex = np.nonzero((skel > 0) & (deg == 1))
    leaders: list[_Leader] = []
    if len(ex) == 0:
        return [], leader_mask
    boxes = {i: labels[i].bbox for i in idx}
    # each skeleton end-point belongs to the nearest label box within reach
    owner: dict[int, list[tuple[int, int]]] = {i: [] for i in idx}
    for x, y in zip(ex.tolist(), ey.tolist()):
        best, bd = None, LEADER_START_PX + 0.01
        for i in idx:
            d = rect_distance(x, y, boxes[i])
            if d < bd:
                best, bd = i, d
        if best is not None:
            owner[best].append((x, y))
    # labels with no stroke end close by: accept a stroke end further out (a gap between the text
    # and its leader is common), but only straight out from the side of the label
    for i in idx:
        if owner[i]:
            continue
        x0, y0, x1, y1 = boxes[i]
        bh = max(1, y1 - y0)
        reach = min(LEADER_FAR_START_MAX, LEADER_FAR_START_K * bh)
        best_pt, bd = None, reach + 0.01
        for x, y in zip(ex.tolist(), ey.tolist()):
            side = (y0 - 0.5 * bh <= y <= y1 + 0.5 * bh) and (x < x0 or x > x1)
            if not side:
                continue
            d = rect_distance(x, y, boxes[i])
            if d >= bd:
                continue
            # no other label may be closer to this end point
            if any(rect_distance(x, y, boxes[k]) < d for k in idx if k != i):
                continue
            best_pt, bd = (x, y), d
        if best_pt is not None:
            owner[i].append(best_pt)

    def gap_ok(x, y):
        return 0 <= x < w and 0 <= y < h and bool(nonbg[y, x])

    used = np.zeros((h, w), bool)
    for i in idx:
        box = boxes[i]
        found = []
        for (sx, sy) in owner[i]:
            if used[sy, sx]:
                continue
            tr = trace_stroke(skel, (sx, sy), gap_ok=gap_ok, weak=weak)
            if len(tr.points) < 5:
                continue
            dists = [rect_distance(px, py, box) for px, py in tr.points]
            j = int(np.argmax(dists))
            if dists[j] < LEADER_MIN_LEN:
                continue
            # a leader heads away from its label; reject strokes that come back to it
            pts = tr.points[: j + 1]
            end_junction = tr.end_junction and j >= len(tr.points) - 3
            # the far end must not sit next to another label (that is a line between labels)
            if any(rect_distance(pts[-1][0], pts[-1][1], boxes[k]) < 6 for k in idx if k != i):
                continue
            if (not end_junction or not protect_crossings) and len(pts) >= 6:
                # thinning shortens / frays stroke ends: keep walking along the dark stroke
                # a pointer often runs on across another organ to its target (the kidney arrow
                # crosses the colon): follow the straight continuation well past a crossing
                a_, b_ = pts[-min(len(pts), 20)], pts[-1]
                v = np.array([b_[0] - a_[0], b_[1] - a_[1]], np.float64)
                v /= max(1e-6, float(np.hypot(*v)))
                for t in range(1, 91):
                    qx, qy = int(round(b_[0] + v[0] * t)), int(round(b_[1] + v[1] * t))
                    if not (0 <= qx < w and 0 <= qy < h) or not thin_m[max(0, qy - 1):qy + 2, max(0, qx - 1):qx + 2].any():
                        break
                    pts.append((qx, qy))
            # a dot drawn at the end of the leader (common textbook style)?
            ex_, ey_ = pts[-1]
            win = dt_thin[max(0, ey_ - 5):ey_ + 6, max(0, ex_ - 5):ex_ + 6]
            along = [dt_thin[py, px] for px, py in pts[len(pts) // 3: 2 * len(pts) // 3]]
            dot_end = bool(win.size and win.max() >= 2.0 and (np.median(along) if along else 0) <= 1.5)
            found.append(_Leader(i, pts, pts[-1], tr.crossings, end_junction, dot_end))
            for px, py in tr.points:
                used[py, px] = True
        # de-duplicate (same stroke traced twice from both sides of a spur)
        found.sort(key=lambda L: -len(L.points))
        kept: list[_Leader] = []
        for L in found:
            if all(np.hypot(L.end[0] - K.end[0], L.end[1] - K.end[1]) > 6 for K in kept):
                kept.append(L)
        leaders.extend(kept[:3])
    for L in leaders:
        pts = np.array(L.points, np.int32).reshape(-1, 1, 2)
        cv2.polylines(leader_mask, [pts], False, 255, 7)
        # thinning shortens strokes by a few px: extend the mask past the far end
        if len(L.points) >= 6 and not (L.end_junction and protect_crossings):
            ex_, ey_ = L.end
            px_, py_ = L.points[-6]
            v = np.array([ex_ - px_, ey_ - py_], np.float64)
            v /= max(1e-6, float(np.hypot(*v)))
            tip = (int(round(ex_ + 4 * v[0])), int(round(ey_ + 4 * v[1])))
            cv2.line(leader_mask, (int(ex_), int(ey_)), tip, 255, 7)
            cv2.circle(leader_mask, (int(ex_), int(ey_)), 5, 255, -1)   # "dot" leader terminals
        if not protect_crossings:
            continue
        # line-art: never cut the outline a leader crosses or ends on
        keep_out = list(L.crossings) + ([L.end] if L.end_junction else [])
        for (cx, cy) in keep_out:
            cv2.circle(leader_mask, (int(cx), int(cy)), 6, 0, -1)
    return leaders, leader_mask


# ============================================================================ segmentation


@dataclass
class _Cand:
    filled: np.ndarray                 # uint8 0/255 (holes filled)
    raw: np.ndarray                    # bool, the part's own pixels (for colour)
    colour_lab: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))


def _kmeans_lab(samples: np.ndarray) -> np.ndarray:
    k = int(min(KMEANS_K, max(1, len(samples) // 50)))
    uniq = np.unique(np.round(samples).astype(np.int32), axis=0)
    k = max(1, min(k, len(uniq)))
    if k == 1:
        return samples.mean(0, keepdims=True).astype(np.float32)
    cv2.setRNGSeed(SEED)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.3)
    _, lbl, centres = cv2.kmeans(samples.astype(np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    counts = np.bincount(lbl, minlength=k).astype(np.float64)
    cs = [c.astype(np.float64) for c in centres]
    ws = list(counts)
    # agglomerative merge of near-identical centres
    while len(cs) > 1:
        best, bi, bj = 1e9, -1, -1
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                d = float(np.linalg.norm(cs[a] - cs[b]))
                if d < best:
                    best, bi, bj = d, a, b
        if best >= MERGE_DELTA_E:
            break
        wsum = ws[bi] + ws[bj]
        cs[bi] = (cs[bi] * ws[bi] + cs[bj] * ws[bj]) / max(wsum, 1)
        ws[bi] = wsum
        del cs[bj], ws[bj]
    keep = [c for c, wt in zip(cs, ws) if wt > 0]
    return np.array(keep, np.float32)


def _assign_clusters(lab: np.ndarray, cand: np.ndarray, centres: np.ndarray,
                     absorb: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = cand.shape
    best = np.full((h, w), np.inf, np.float32)
    lbl = np.full((h, w), -1, np.int16)
    for k, c in enumerate(centres):
        d = np.sum((lab - c) ** 2, axis=2)
        upd = d < best
        best[upd] = d[upd]
        lbl[upd] = k
    lbl[~cand] = -1
    lbl[best > 30.0 ** 2] = -1
    # 3x3 majority filter to remove anti-aliasing fringes
    K = len(centres)
    if K > 0:
        counts = np.stack([cv2.boxFilter((lbl == k).astype(np.float32), -1, (3, 3), normalize=False)
                           for k in range(K)], 0)
        arg = counts.argmax(0).astype(np.int16)
        mx = counts.max(0)
        new = np.where(mx >= 4, arg, -1).astype(np.int16)
        new[~cand] = -1
        lbl = new
    if absorb is not None and K > 0:
        # thin dark details drawn inside a region (stripes, zig-zags, outlines of holes) belong to it
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        free = (lbl == -1) & absorb
        for k in range(K):
            closed = cv2.morphologyEx((lbl == k).astype(np.uint8), cv2.MORPH_CLOSE, ker) > 0
            add = closed & free
            lbl[add] = k
            free &= ~add
    return lbl


def _compact_small(st_row) -> bool:
    a = st_row[cv2.CC_STAT_AREA]
    bw, bh = st_row[cv2.CC_STAT_WIDTH], st_row[cv2.CC_STAT_HEIGHT]
    return a >= 4 and max(bw, bh) <= 24 and a >= 0.35 * bw * bh


def _segment_colour(img_clean: np.ndarray, lab: np.ndarray, cand: np.ndarray, n_total: int,
                    dots: list, absorb: Optional[np.ndarray] = None) -> tuple[list[_Cand], int]:
    """K-means colour clusters -> candidate parts. Returns (cands, n_thin_dropped)."""
    ys, xs = np.nonzero(cand)
    if len(xs) < 0.002 * n_total:
        return [], 0
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(xs), size=min(KMEANS_SAMPLES, len(xs)), replace=False)
    samples = lab[ys[sel], xs[sel]]
    centres = _kmeans_lab(samples)
    lbl = _assign_clusters(lab, cand, centres, absorb)
    min_area = COLOUR_MIN_AREA_FRAC * n_total
    k3 = np.ones((3, 3), np.uint8)
    cands: list[_Cand] = []
    thin_dropped = 0
    for k in range(len(centres)):
        m = (lbl == k).astype(np.uint8) * 255
        if not m.any():
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
        n, cc, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < min_area:
                if _compact_small(st[i]):
                    dots.append((float(cen[i][0]), float(cen[i][1]), max(st[i, 2], st[i, 3]) / 2.0))
                continue
            x, y, bw, bh = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
            sub = (cc[y:y + bh, x:x + bw] == i).astype(np.uint8)
            sub = cv2.copyMakeBorder(sub, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            if cv2.distanceTransform(sub, cv2.DIST_L2, 3).max() < 3.0:
                thin_dropped += 1
                continue
            keep[i] = True
        if not keep.any():
            continue
        raw = keep[cc]
        # components of the same colour nested inside another one's hole (e.g. green
        # chloroplasts inside a green wall ring) are a different part: keep them apart
        raw_u8 = raw.astype(np.uint8)
        cs, hier = cv2.findContours(raw_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        top = np.zeros_like(raw_u8)
        nested = np.zeros_like(raw_u8)
        if hier is not None:
            hier = hier[0]
            for ci_, c_ in enumerate(cs):
                depth, par = 0, hier[ci_][3]
                while par >= 0:
                    depth += 1
                    par = hier[par][3]
                if depth == 0:
                    cv2.drawContours(top, [c_], -1, 1, -1)
                elif depth % 2 == 0:
                    cv2.drawContours(nested, [c_], -1, 1, -1)
        if nested.any():
            nraw = raw & (nested > 0)
            traw = raw & ~(nested > 0)
            cands.append(_Cand(filled=fill_holes(to_mask(traw)), raw=traw, colour_lab=centres[k]))
            cands.append(_Cand(filled=fill_holes(to_mask(nraw)), raw=nraw, colour_lab=centres[k]))
        else:
            cands.append(_Cand(filled=fill_holes(raw_u8 * 255), raw=raw, colour_lab=centres[k]))
    return cands, thin_dropped


def _segment_lineart(ink: np.ndarray, n_total: int) -> list[_Cand]:
    h, w = ink.shape
    free = (ink == 0).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    min_area = LINEART_MIN_AREA_FRAC * n_total
    cands = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < min_area or a > MAX_REGION_FRAC * n_total:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue  # touches the border -> background
        raw = cc == i
        filled = fill_holes(raw.astype(np.uint8) * 255)
        # grow 1 px into the outline so the region hugs the drawn line
        filled = fill_holes(cv2.dilate(filled, disk(1)))
        cands.append(_Cand(filled=filled, raw=raw))
    # solid ink shapes (black silhouettes) are regions too, not strokes
    covered = np.zeros((h, w), bool)
    for c in cands:
        covered |= c.filled > 0
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN, disk(SOLID_INK_RADIUS))
    n2, cc2, st2, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    for i in range(1, n2):
        if st2[i, cv2.CC_STAT_AREA] < min_area or st2[i, cv2.CC_STAT_AREA] > MAX_REGION_FRAC * n_total:
            continue
        raw = cc2 == i
        if (raw & covered).sum() > 0.5 * raw.sum():
            continue
        filled = fill_holes(cv2.dilate(raw.astype(np.uint8) * 255, disk(SOLID_INK_RADIUS // 2)))
        cands.append(_Cand(filled=filled, raw=raw))
    return cands


def _absorb_islands(cands: list[_Cand]) -> list[_Cand]:
    comps = _components(cands)
    if len(comps) < 2:
        return cands
    raws = [c.raw for c in cands]
    hosts: dict[int, Optional[int]] = {}
    for k, c in enumerate(comps):
        best, best_area = None, None
        ys, xs = np.nonzero(c.mask)
        for j, o in enumerate(comps):
            if o.cand == c.cand or o.area < 3 * c.area:
                continue
            if o.mask[ys, xs].mean() >= 0.85 and (best_area is None or o.area < best_area):
                best, best_area = j, o.area
        hosts[k] = best
    changed = False
    for ci in range(len(cands)):
        mine = [k for k, c in enumerate(comps) if c.cand == ci]
        if len(mine) < 2:
            continue
        main = max(mine, key=lambda k: comps[k].area)
        for k in mine:
            hk = hosts[k]
            if k == main or hk is None or hk == hosts[main] or comps[k].area >= 0.3 * comps[hk].area:
                continue
            c = comps[k]
            ring = cv2.dilate(c.mask.astype(np.uint8), disk(4)).astype(bool) & ~c.mask
            if ring.sum() == 0 or raws[comps[hk].cand][ring].mean() < 0.85:
                continue
            cands[ci].raw = cands[ci].raw & ~c.mask
            cands[ci].filled = cands[ci].filled.copy()
            cands[ci].filled[c.mask] = 0
            hc = cands[comps[hk].cand]
            hc.filled = cv2.bitwise_or(hc.filled, to_mask(c.mask))
            changed = True
    if changed:
        cands = [c for c in cands if c.raw.any() and (c.filled > 0).any()]
    return cands


def _linework(ink, tmask, lmask, masks, thin_m, mode, n_total) -> np.ndarray:
    """Meaningful strokes outside parts: arrows, axes, wires. Text, leaders, outlines removed."""
    lw = ink.copy()
    lw &= ~(tmask > 0)
    lw &= ~(lmask > 0)
    for m in masks:
        band = cv2.dilate(m, disk(6)) & ~cv2.erode(m, disk(2))
        lw &= ~(band > 0)
        inside = m > 0
        if (m > 0).sum() < 0.05 * n_total:
            lw &= ~inside               # details inside small parts (stripes, symbols) are texture
        elif mode == "colour":
            lw &= ~inside | thin_m      # inside big regions keep only genuine thin strokes
    lw_u8 = lw.astype(np.uint8) * 255
    n, cc, st, _ = cv2.connectedComponentsWithStats(lw_u8, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) >= LINEWORK_MIN_LEN:
            keep[i] = True
    return (keep[cc]).astype(np.uint8) * 255


def _pointed_extras(cands: list[_Cand], ink: np.ndarray, leaders: list, mode: str, n_total: int) -> list[_Cand]:
    h, w = ink.shape
    covered = np.zeros((h, w), np.uint8)
    for c in cands:
        covered |= c.filled
    near_cov = cv2.dilate(covered, disk(12)) > 0
    extras: list[_Cand] = []
    free_cc = None
    for L in leaders:
        ex, ey = int(round(L.end[0])), int(round(L.end[1]))
        if not (0 <= ex < w and 0 <= ey < h) or near_cov[ey, ex]:
            continue
        if any(e.filled[ey, ex] for e in extras):
            continue
        if mode == "colour":
            if free_cc is None:
                anyc = covered > 0
                free = (~ink & ~anyc).astype(np.uint8)
                n, free_cc, free_st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
            lab_ = free_cc[ey, ex]
            if lab_ == 0:
                continue
            x, y, bw, bh, a = free_st[lab_]
            if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
                continue
            if not (LINEART_MIN_AREA_FRAC * n_total <= a <= 0.15 * n_total):
                continue
            raw = free_cc == lab_
            extras.append(_Cand(filled=fill_holes(cv2.dilate(to_mask(raw), disk(1))), raw=raw))
        else:
            # line-art symbol: strokes near the end point, minus long straight wires
            r = 90
            x0, y0, x1, y1 = max(0, ex - r), max(0, ey - r), min(w, ex + r + 1), min(h, ey + r + 1)
            win = (ink[y0:y1, x0:x1] > 0).astype(np.uint8)
            L_ = 100
            hor = cv2.morphologyEx((ink > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, L_), np.uint8))[y0:y1, x0:x1]
            ver = cv2.morphologyEx((ink > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((L_, 1), np.uint8))[y0:y1, x0:x1]
            win[(cv2.dilate(hor | ver, np.ones((3, 3), np.uint8))) > 0] = 0
            yy, xx = np.mgrid[y0:y1, x0:x1]
            win[(xx - ex) ** 2 + (yy - ey) ** 2 > r * r] = 0
            if not win.any():
                continue
            n, cc, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(win, disk(7)), connectivity=8)
            lx, ly = ex - x0, ey - y0
            pick = np.zeros(n, bool)
            for i in range(1, n):
                m = cc == i
                ys_, xs_ = np.nonzero(m)
                if np.hypot(xs_ - lx, ys_ - ly).min() <= 25 and st[i, cv2.CC_STAT_AREA] >= 60:
                    pick[i] = True
            if not pick.any():
                continue
            sel = (pick[cc] & (win > 0)).astype(np.uint8)
            pts = cv2.findNonZero(sel)
            if pts is None or len(pts) < 10:
                continue
            hull = cv2.convexHull(pts)
            bx, by, bw, bh = cv2.boundingRect(hull)
            if max(bw, bh) < 20:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(m, (hull.reshape(-1, 2) + (x0, y0)).astype(np.int32), 255)
            m = cv2.dilate(m, disk(2))
            extras.append(_Cand(filled=m, raw=m > 0))
    return extras


def _merge_shading(cands: list[_Cand], referenced: set, labelled_diagram: bool) -> list[_Cand]:
    """Fold highlight / shade / texture clusters into the region that contains them."""
    order = sorted(range(len(cands)), key=lambda i: int((cands[i].filled > 0).sum()))
    alive = [True] * len(cands)
    areas = [int((c.filled > 0).sum()) for c in cands]
    for a in order:
        if a in referenced or not alive[a]:
            continue
        ra = cands[a].raw
        tot = int(ra.sum())
        if tot == 0:
            continue
        host, host_area = None, None
        for b in range(len(cands)):
            if b == a or not alive[b] or areas[b] < 2 * areas[a]:
                continue
            inside = int((ra & (cands[b].filled > 0)).sum())
            if inside >= 0.8 * tot and (host_area is None or areas[b] < host_area):
                host, host_area = b, areas[b]
        if host is None:
            continue
        de = float(np.linalg.norm(cands[a].colour_lab - cands[host].colour_lab))
        if de < 35 or labelled_diagram:
            cands[host].filled = cv2.bitwise_or(cands[host].filled, cands[a].filled)
            areas[host] = int((cands[host].filled > 0).sum())
            alive[a] = False
    return [c for c, al in zip(cands, alive) if al]


# ============================================================================ matching


@dataclass
class _Comp:
    cand: int
    mask: np.ndarray          # bool, filled component
    area: int
    pts: np.ndarray           # outline points


def _components(cands: list[_Cand]) -> list[_Comp]:
    comps = []
    for ci, c in enumerate(cands):
        n, cc, st, _ = cv2.connectedComponentsWithStats((c.filled > 0).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            m = cc == i
            comps.append(_Comp(ci, m, int(st[i, cv2.CC_STAT_AREA]), contour_points(m.astype(np.uint8))))
    return comps


def _point_target(px: float, py: float, comps: list[_Comp], heading=None,
                  exact: bool = False, trail=None) -> Optional[tuple[int, float]]:
    """Comp index a leader end-point names, with its distance."""
    h, w = comps[0].mask.shape
    ix, iy = int(round(min(max(px, 0), w - 1))), int(round(min(max(py, 0), h - 1)))
    containing = [k for k, c in enumerate(comps) if c.mask[iy, ix]]
    dists = {}
    for k, c in enumerate(comps):
        if k in containing:
            dists[k] = 0.0
        elif len(c.pts):
            dists[k] = float(np.sqrt(((c.pts - (px, py)) ** 2).sum(1)).min())
    if containing:
        s = min(containing, key=lambda k: comps[k].area)
        # leaders often stop just short of a small target sitting inside a big region
        near = [k for k, d in dists.items() if 0 < d <= 14 and comps[k].area < 0.5 * comps[s].area]
        # a leader that runs through a small part and stops just past it names that part
        crossed = [k for k in near if trail and sum(1 for qx, qy in trail if comps[k].mask[qy, qx]) >= 4]
        if exact:
            # a drawn dot marks the spot: take a smaller part only if it fills a good share of
            # the dot's surroundings (dot placed on that part's edge)
            r = 7
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            disk_pts = [(ix + int(dx), iy + int(dy)) for dx, dy in zip(xx.ravel(), yy.ravel())
                        if dx * dx + dy * dy <= r * r and 0 <= ix + dx < w and 0 <= iy + dy < h]
            shares = {}
            for qx, qy in disk_pts:
                inside = [k for k in near + [s] if comps[k].mask[qy, qx]]
                if inside:
                    kk = min(inside, key=lambda k: comps[k].area)
                    shares[kk] = shares.get(kk, 0) + 1
            near = [k for k in near if shares.get(k, 0) >= 0.22 * len(disk_pts) or k in crossed]
        elif heading is not None:
            # only snap to a small part the leader is heading for
            def ahead(k):
                for t in range(1, 15):
                    qx, qy = int(round(px + heading[0] * t)), int(round(py + heading[1] * t))
                    if comps[k].mask[max(0, qy - 3):qy + 4, max(0, qx - 3):qx + 4].any():
                        return True
                return False
            near = [k for k in near if ahead(k) or k in crossed]
        if near:
            s = min(near, key=lambda k: comps[k].area)
        return s, dists[s]
    near = [(d, k) for k, d in dists.items() if d <= LEADER_SNAP_PX]
    if not near:
        if heading is None:
            return None
        # the leader stops short of its shape: follow its heading a little further
        for t in range(1, LEADER_RAY_PX + 1):
            qx, qy = int(round(px + heading[0] * t)), int(round(py + heading[1] * t))
            if not (0 <= qx < w and 0 <= qy < h):
                break
            hit = [k for k, c in enumerate(comps) if c.mask[qy, qx]]
            if hit:
                k = min(hit, key=lambda k: comps[k].area)
                return k, float(t)
        return None
    d, k = min(near)
    return k, d


# ============================================================================ input kind
#
# Cheap features on a 400 px copy decide what was photographed :
#   flat  share of pixels whose 5x5 neighbourhood is flat (std < 3): drawings ~0.8, photos ~0.1-0.4
#   ncol  number of 4-bit colours covering 90 % of pixels: drawings < 10, photos ~100+
#   tex   share of mid-texture pixels (5x5 std 3..20): drawings < 0.1, photos 0.3-0.9
#   edge  Canny edge density: blank / smooth gradients ~0
# plus OCR: confident part-label-like words, the share of the picture covered by text, words per line.


def _image_features(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    sc = FEAT_SIDE / max(h, w)
    sm = cv2.resize(img, (max(2, int(w * sc)), max(2, int(h * sc))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (5, 5)) - mu * mu, 0))
    q = (sm // 16).reshape(-1, 3).astype(np.int32)
    _, cnt = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_counts=True)
    cnt = np.sort(cnt)[::-1]
    ncol = int(np.searchsorted(np.cumsum(cnt), 0.9 * cnt.sum()) + 1)
    return {
        "flat": round(float((sd < 3).mean()), 3),
        "tex": round(float(((sd >= 3) & (sd < 20)).mean()), 3),
        "ncol": ncol,
        "edge": round(float((cv2.Canny(sm, 50, 150) > 0).mean()), 4),
        "gstd": round(float(g.std()), 1),
    }


def _alpha(t: str) -> int:
    return sum(ch.isalpha() for ch in t)


def _good_labels(labels: list[Label]) -> list[Label]:
    return [lb for lb in labels if lb.confidence >= 0.75 and _alpha(lb.text) >= 3]


def _text_stats(labels: list[Label], w: int, h: int) -> dict:
    area = sum((lb.bbox[2] - lb.bbox[0]) * (lb.bbox[3] - lb.bbox[1]) for lb in labels)
    words = [len(lb.text.split()) for lb in labels]
    return {"n": len(labels), "good": len(_good_labels(labels)), "text_frac": round(area / max(1, w * h), 3),
            "words": round(float(np.mean(words)), 2) if words else 0.0}


def _is_blank(f: dict) -> bool:
    return f["edge"] < BLANK_EDGE or f["gstd"] < BLANK_STD


def _looks_drawn(f: dict) -> bool:
    """Flat colours like a drawing / screenshot (not a natural photo)."""
    return (f["flat"] >= 0.6 and f["ncol"] <= 40 and f["tex"] <= 0.2) or \
           (f["flat"] >= 0.5 and f["ncol"] <= 12 and f["tex"] <= 0.1)


def _looks_natural(f: dict) -> bool:
    """Strong natural-photo texture: few flat areas plus many colours, busy texture or dense edges
    (grayscale photos have few colours, printed circuit boards little mid texture)."""
    return f["flat"] < 0.3 and ((f["ncol"] > 80 and (f["tex"] > 0.3 or f["edge"] > 0.2))
                                or (f["tex"] > 0.3 and f["edge"] > 0.1))


def _classify(f: dict, ts: dict) -> str:
    """'blank' | 'text_page' | 'diagram' | 'photo' (object vs scene is decided by segmentation)."""
    if _is_blank(f) and ts["n"] == 0:
        return "blank"
    if ts["n"] >= 3 and ts["text_frac"] >= TEXT_PAGE_FRAC and ts["words"] >= 2.5:
        return "text_page"
    # handwriting / scribbled notes: text-like boxes cover a lot of a near-monochrome page, but OCR
    # can't read them confidently (diagram labels are printed, so they read well)
    if ts["n"] >= 2 and ts["good"] <= max(1, ts["n"] // 8) and ts["text_frac"] >= 0.2 and f["ncol"] <= 8:
        return "text_page"
    if ts["good"] >= 2 and not _looks_natural(f):
        return "diagram"
    if _looks_drawn(f):
        return "diagram"
    return "photo"


# ============================================================================ orientation (F10)


def _ocr_score(labels: list[Label]) -> float:
    return float(sum(lb.confidence * min(12, _alpha(lb.text)) for lb in labels if lb.confidence >= 0.5))


def _rotate_cw(img: np.ndarray, deg: int) -> np.ndarray:
    k = {0: 0, 90: 3, 180: 2, 270: 1}[deg % 360]          # np.rot90 turns anticlockwise
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def _auto_orient(img: np.ndarray, labels_all: list[Label], feats: dict) -> tuple[int, Optional[list[Label]]]:
    """Text that reads poorly upright: try 90/270 (tall text boxes), 180 (unsure reads), or all three
    for a drawing with hardly any readable label; keep the best reading if it is clearly better.
    Returns (degrees clockwise, labels of the rotated image) or (0, None)."""
    sure = [lb for lb in labels_all if lb.confidence >= 0.5]
    if len(_good_labels(sure)) >= 2 and float(np.mean([lb.confidence for lb in labels_all])) >= 0.7:
        return 0, None
    tall = [lb for lb in labels_all
            if (lb.bbox[3] - lb.bbox[1]) > 1.5 * (lb.bbox[2] - lb.bbox[0]) and (lb.bbox[3] - lb.bbox[1]) >= 24]
    if tall:
        cands = [90, 270]
    elif len(labels_all) >= 2 and float(np.mean([lb.confidence for lb in labels_all])) < 0.75:
        cands = [180]
    elif _looks_drawn(feats) and not _is_blank(feats):
        cands = [90, 270, 180]
    elif not labels_all and not _is_blank(feats):
        # F06: no text found at all - a sideways diagram's vertical text is invisible to the
        # detector, so try the two quarter turns before treating the picture as a photo
        cands = [90, 270]
    else:
        return 0, None
    best_deg, best_labels, best_score = 0, None, _ocr_score(labels_all)
    for deg in cands:
        labs = run_ocr(_rotate_cw(img, deg), min_confidence=0.0)
        sc = _ocr_score(labs)
        if sc > best_score:
            best_deg, best_labels, best_score = deg, labs, sc
    base = _ocr_score(labels_all)
    if best_deg and best_labels is not None and best_score >= 1.5 * base + 2 and (
            len(_good_labels(best_labels)) >= 2
            # electrical / maths notation ("R1 = 1.00 Ω") has few dictionary words: a turn that
            # reads several times more text with confidence is the right one
            or (best_score >= 3.0 * base + 3 and sum(lb.confidence >= 0.6 for lb in best_labels) >= 3)):
        return best_deg, best_labels
    return 0, None


# ============================================================================ photos


_COLOURS = [  # name, Lab-ish reference (L, a, b)
    ("black", (12, 0, 0)), ("dark grey", (35, 0, 0)), ("grey", (55, 0, 0)), ("light grey", (75, 0, 0)),
    ("white", (95, 0, 0)), ("red", (50, 65, 45)), ("dark red", (30, 45, 25)), ("orange", (65, 35, 65)),
    ("yellow", (88, -5, 80)), ("brown", (40, 18, 30)), ("tan", (68, 8, 28)), ("green", (55, -45, 40)),
    ("dark green", (32, -30, 20)), ("blue", (45, 10, -55)), ("light blue", (75, -10, -30)),
    ("dark blue", (22, 15, -40)), ("purple", (40, 45, -40)), ("pink", (75, 35, 0)),
]


def _colour_name(rgb) -> str:
    lab = rgb_to_lab(np.array(rgb, np.uint8).reshape(1, 1, 3))[0, 0].astype(np.float32)
    chroma = float(np.hypot(lab[1], lab[2]))
    best, bd = "grey", 1e9
    for name, ref in _COLOURS:
        ref_chroma = float(np.hypot(ref[1], ref[2]))
        if (chroma < 12) != (ref_chroma < 12):
            continue                           # don't call a grey "blue" or a red "grey"
        d = float(np.linalg.norm(lab - np.array(ref, np.float32)))
        if d < bd:
            best, bd = name, d
    return best


def _position_name(bbox, w: int, h: int, within: Optional[tuple] = None) -> str:
    """'top', 'lower left', 'middle', ... of bbox centre in the picture (or inside `within`)."""
    x0, y0, x1, y1 = within or (0, 0, w, h)
    cx = ((bbox[0] + bbox[2]) / 2 - x0) / max(1, x1 - x0)
    cy = ((bbox[1] + bbox[3]) / 2 - y0) / max(1, y1 - y0)
    vert = "top" if cy < 0.33 else ("bottom" if cy > 0.67 else "")
    hor = "left" if cx < 0.33 else ("right" if cx > 0.67 else "")
    if vert and hor:
        return f"{'upper' if vert == 'top' else 'lower'} {hor}"
    return vert or hor or "middle"


def _smooth_silhouette(mask: np.ndarray, eps_frac: float = 0.006) -> np.ndarray:
    """Largest outline of `mask`, polygon-simplified and refilled (one feelable shape, no holes)."""
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(mask.shape[:2], np.uint8)
    if not cs:
        return out
    c = max(cs, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, max(1.0, eps_frac * cv2.arcLength(c, True)), True)
    cv2.fillPoly(out, [ap], 255)
    return out


def _border_lab_centres(lab: np.ndarray, frac: float = 0.04, k: int = 6) -> np.ndarray:
    h, w = lab.shape[:2]
    b = max(2, int(frac * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)]).astype(np.float32)
    k = max(1, min(k, len(border) // 20))
    cv2.setRNGSeed(SEED)
    _, lbl, cent = cv2.kmeans(border, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                              1, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lbl.ravel(), minlength=k)
    keep = cent[cnt >= 0.03 * len(border)]
    return keep if len(keep) else cent


def _subject_mask(img: np.ndarray, box: Optional[tuple] = None) -> tuple[np.ndarray, dict]:
    """Main subject of a photo -> (mask uint8 0/255 in img coords, stats).

    GrabCut (3 iterations) on a <= PHOTO_SEG_SIDE px copy with the background flattened by an
    edge-preserving filter. Seeds: the image border is background; pixels far from the centre (or
    outside the AI box) and pixels coloured like the border are probably background; the centre
    (box centre) is probably foreground. Thin attachments are cut off with a morphological opening.
    """
    h, w = img.shape[:2]
    sc = min(1.0, PHOTO_SEG_SIDE / max(h, w))
    sw, sh = max(8, int(round(w * sc))), max(8, int(round(h * sc)))
    sm = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    smooth = cv2.bilateralFilter(sm, 9, 40, 9)
    lab = rgb_to_lab(smooth).astype(np.float32)
    cent = _border_lab_centres(lab)
    d_border = np.min(np.stack([np.linalg.norm(lab - c, axis=2) for c in cent]), 0)
    yy, xx = np.mgrid[0:sh, 0:sw]
    if box is not None:
        bx0, by0, bx1, by1 = [v * sc for v in box]
        bx0, bx1 = max(0.0, min(bx0, bx1)), min(float(sw), max(bx0, bx1))
        by0, by1 = max(0.0, min(by0, by1)), min(float(sh), max(by0, by1))
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        rx, ry = max(4.0, (bx1 - bx0) / 2), max(4.0, (by1 - by0) / 2)
    else:
        cx, cy, rx, ry = sw / 2, sh / 2, sw / 2, sh / 2
    r2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    m = np.full((sh, sw), cv2.GC_PR_FGD, np.uint8)
    if box is not None:
        m[r2 > 1.6] = cv2.GC_PR_BGD
        outside = (xx < bx0 - 0.03 * sw) | (xx > bx1 + 0.03 * sw) | (yy < by0 - 0.03 * sh) | (yy > by1 + 0.03 * sh)
        m[outside] = cv2.GC_BGD
        m[(d_border < 10) & (r2 > 0.25)] = cv2.GC_PR_BGD
        m[r2 < 0.1] = cv2.GC_PR_FGD
    else:
        m[r2 > 0.55] = cv2.GC_PR_BGD
        m[d_border < 10] = cv2.GC_PR_BGD
        m[r2 < 0.04] = cv2.GC_PR_FGD
    b = max(2, int(0.03 * min(sh, sw)))
    m[:b] = cv2.GC_BGD
    m[-b:] = cv2.GC_BGD
    m[:, :b] = cv2.GC_BGD
    m[:, -b:] = cv2.GC_BGD
    stats = {"area_frac": 0.0, "contrast": 0.0, "solidity": 0.0}
    if not (m == cv2.GC_PR_FGD).any() or not ((m == cv2.GC_BGD) | (m == cv2.GC_PR_BGD)).any():
        return np.zeros((h, w), np.uint8), stats
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(SEED)
        cv2.grabCut(smooth, m, None, bgm, fgm, GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), np.uint8), stats
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    k = max(3, int(0.06 * min(sh, sw))) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, cc, st, cen = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), np.uint8), stats
    best = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA] /
               (1.0 + np.hypot(cen[i][0] - cx, cen[i][1] - cy) / (0.25 * max(sh, sw))))
    fg = fill_holes((cc == best).astype(np.uint8) * 255)
    fg = _trim_leaks(fg, lab)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, cc, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), np.uint8), stats
    fg = fill_holes((cc == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))).astype(np.uint8) * 255)
    k2 = max(3, int(0.03 * min(sh, sw))) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
    inside = fg > 0
    ring = (cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k2 + 1, 2 * k2 + 1))) > 0) & ~inside
    if inside.any() and ring.any():
        stats["contrast"] = round(float(np.linalg.norm(np.median(lab[inside], 0) - np.median(lab[ring], 0))), 1)
    cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        c = max(cs, key=cv2.contourArea)
        stats["solidity"] = round(float(cv2.contourArea(c) / max(1.0, cv2.contourArea(cv2.convexHull(c)))), 2)
    stats["area_frac"] = round(float(inside.mean()), 3)
    big = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    return _smooth_silhouette((big > 127).astype(np.uint8) * 255), stats


def _trim_leaks(fg: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """GrabCut often keeps a patch of background touching the object (a shadow, a light patch of the
    mat). Split the foreground into 2 colour clusters; drop pieces of the minority cluster that lie
    on the outline, differ strongly from the rest and look like what is just outside the object."""
    inside = fg > 0
    n_in = int(inside.sum())
    if n_in < 300:
        return fg
    px = lab[inside].astype(np.float32)
    cv2.setRNGSeed(SEED)
    _, lbl, cent = cv2.kmeans(px, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 1,
                              cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    counts = np.bincount(lbl, minlength=2)
    minor = int(np.argmin(counts))
    if counts[minor] > 0.4 * n_in or float(np.linalg.norm(cent[0] - cent[1])) < 25:
        return fg
    cl = np.zeros(fg.shape, np.uint8)
    cl[inside] = (lbl == minor).astype(np.uint8)
    cl = cv2.morphologyEx(cl, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    side = max(3, int(0.05 * np.sqrt(n_in))) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    ring = (cv2.dilate(fg, kern) > 0) & ~inside
    outside_near = cv2.dilate((~inside).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if not ring.any():
        return fg
    ring_px = lab[ring].astype(np.float32)
    out = fg.copy()
    n, cc, st, _ = cv2.connectedComponentsWithStats(cl, connectivity=8)
    for i in range(1, n):
        comp = cc == i
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < 0.02 * n_in:
            continue
        cu8 = comp.astype(np.uint8)
        rim = comp & ~(cv2.erode(cu8, np.ones((3, 3), np.uint8)) > 0)
        on_edge = float((rim & outside_near).sum()) / max(1, int(rim.sum()))
        c = np.median(lab[comp], 0).astype(np.float32)
        like_outside = float((np.linalg.norm(ring_px - c, axis=1) < 15).mean())
        if on_edge >= 0.3 and like_outside >= 0.1:
            out[comp] = 0
    if out.sum() < 0.5 * fg.sum():
        return fg                                       # never throw away most of the object
    return out


def _subject_ok(st: dict, from_ai: bool) -> bool:
    lo = 0.02 if from_ai else 0.03
    return lo <= st["area_frac"] <= 0.85 and st["solidity"] >= 0.45 and \
        (st["contrast"] >= (5 if from_ai else 9))


def _interior_details(img: np.ndarray, subject: np.ndarray, text_boxes: list) -> list[np.ndarray]:
    """Clearly separated, big-enough patches inside the subject (e.g. a cup's coffee, a logo panel).
    Returns at most PHOTO_MAX_DETAILS masks in img coords."""
    h, w = img.shape[:2]
    sc = min(1.0, PHOTO_SEG_SIDE / max(h, w))
    sw, sh = max(8, int(round(w * sc))), max(8, int(round(h * sc)))
    sm = cv2.bilateralFilter(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA), 9, 40, 9)
    subj = cv2.resize(subject, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    s_area = int(subj.sum())
    if s_area < 200:
        return []
    for (x0, y0, x1, y1) in text_boxes:                 # printed text / logos are not parts
        subj[int(y0 * sc):int(np.ceil(y1 * sc)) + 1, int(x0 * sc):int(np.ceil(x1 * sc)) + 1] = False
    er = max(2, int(0.04 * np.sqrt(s_area)))
    core = cv2.erode(subj.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1))) > 0
    if core.sum() < 100:
        return []
    lab = rgb_to_lab(sm).astype(np.float32)
    px = lab[core]
    k = 3
    cv2.setRNGSeed(SEED)
    _, lbl, cent = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 1,
                              cv2.KMEANS_PP_CENTERS)
    labimg = np.full((sh, sw), -1, np.int16)
    labimg[core] = lbl.ravel()
    labimg = cv2.medianBlur((labimg + 1).astype(np.uint8), 5).astype(np.int16) - 1
    counts = np.bincount(lbl.ravel(), minlength=k)
    dominant = int(np.argmax(counts))
    found = []
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for ci in range(k):
        if ci == dominant:
            continue
        m = cv2.morphologyEx(((labimg == ci) & core).astype(np.uint8), cv2.MORPH_OPEN, kern)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if not (PHOTO_DETAIL_MIN_FRAC * s_area <= a <= 0.5 * s_area):
                continue
            comp = (cc == i).astype(np.uint8)
            c = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
            sol = cv2.contourArea(c) / max(1.0, cv2.contourArea(cv2.convexHull(c)))
            ring = (cv2.dilate(comp, kern, iterations=2) > 0) & (comp == 0) & subj
            if sol < 0.6 or ring.sum() < 10:
                continue
            contrast = float(np.linalg.norm(np.median(lab[comp > 0], 0) - np.median(lab[ring], 0)))
            if contrast < DETAIL_MIN_CONTRAST:          # glare / shading on one surface is subtler
                continue
            found.append((a * contrast, comp))
    found.sort(key=lambda t: -t[0])
    out = []
    for _, comp in found[:PHOTO_MAX_DETAILS]:
        big = cv2.resize(comp * 255, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(_smooth_silhouette(((big > 127) & (subject > 0)).astype(np.uint8) * 255, 0.01))
    return [m for m in out if m.any()]


def _scene_regions(img: np.ndarray) -> list[np.ndarray]:
    """At most PHOTO_SCENE_MAX big, simplified regions (sky / ground / main mass ...)."""
    h, w = img.shape[:2]
    sc = SCENE_SIDE / max(h, w)
    sw, sh = max(8, int(round(w * sc))), max(8, int(round(h * sc)))
    sm = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    sm = cv2.bilateralFilter(cv2.GaussianBlur(sm, (5, 5), 0), 7, 50, 7)
    lab = rgb_to_lab(sm).astype(np.float32)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    feat = np.concatenate([lab.reshape(-1, 3), (xx.reshape(-1, 1) / sw * 30), (yy.reshape(-1, 1) / sh * 30)], 1)
    k = 5
    cv2.setRNGSeed(SEED)
    _, lbl, _ = cv2.kmeans(feat.astype(np.float32), k, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.reshape(sh, sw).astype(np.uint8)
    # clusters split only by position (same colour) are one region: merge colour-alike clusters
    cols = [np.median(lab[lbl == ci], 0) if (lbl == ci).any() else np.full(3, 1e6, np.float32) for ci in range(k)]
    parent = list(range(k))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(k):
        for j in range(i + 1, k):
            if float(np.linalg.norm(cols[i] - cols[j])) < SCENE_MERGE_DE:
                parent[root(j)] = root(i)
    lbl = np.array([root(i) for i in range(k)], np.uint8)[lbl]
    if len(set(root(i) for i in range(k))) < 2:
        return []                                  # no colour structure (noise, plain texture)
    lbl = cv2.medianBlur(lbl, 5)
    regions = []
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for ci in range(k):
        m = cv2.morphologyEx((lbl == ci).astype(np.uint8), cv2.MORPH_OPEN, kern)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= SCENE_MIN_FRAC * sw * sh:
                regions.append((int(st[i, cv2.CC_STAT_AREA]), (cc == i).astype(np.uint8)))
    regions.sort(key=lambda t: -t[0])
    out = []
    for _, m in regions[:PHOTO_SCENE_MAX]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
        big = cv2.resize(m * 255, (w, h), interpolation=cv2.INTER_LINEAR)
        mm = _smooth_silhouette((big > 127).astype(np.uint8) * 255, 0.01)
        if mm.any():
            out.append(mm)
    return out


def _new_part(pid: str, name: str, description: str, mask: np.ndarray, img: np.ndarray, *,
              match_method: str = "none", description_source: str = "auto",
              label_text: Optional[str] = None) -> Part:
    m = mask > 0
    col = np.median(img[m], axis=0) if m.any() else np.array([255, 255, 255])
    polys = mask_polygons(mask, 1.5)
    return Part(id=pid, name=name, description=description, key="",
                color=(int(col[0]), int(col[1]), int(col[2])), mask=mask, polygons=polys,
                bbox=mask_bbox(mask), area=int(m.sum()), count=max(1, len(polys)), include=True, kind="region",
                label_text=label_text, match_method=match_method, description_source=description_source)


def _ai_ready() -> bool:
    try:
        return bool(ai.get_provider_info().get("provider"))
    except Exception:
        return False


def _ai_error_warning() -> str:
    reason = None
    try:
        le = getattr(ai, "last_error", None)
        reason = le() if callable(le) else None
    except Exception:
        reason = None
    return "AI description unavailable" + (f" ({reason})" if reason else "") + " — used labels from the image"


def _analyze_photo(img: np.ndarray, labels: list[Label], *, use_ai: bool, progress, t0: float,
                   rotation: int, feats: dict, text_boxes_ok: bool = True) -> Analysis:
    """photo_object / photo_scene: an outline you can feel, never junk part names.
    `img` is the unflattened photo; `text_boxes_ok` is False when the OCR boxes belong to a
    page-warped copy (then they are only reported, not used to mask out printed text)."""
    h, w = img.shape[:2]
    warnings: list[str] = []
    notes: list[str] = []
    for lb in labels:                  # brand names / printed text never name parts of a photo
        lb.role = "other"
        lb.part_id = None
    text_boxes = [lb.bbox for lb in labels] if text_boxes_ok else []
    if not text_boxes_ok:
        labels = []
    if labels:
        shown = ", ".join(f"“{lb.text}”" for lb in labels[:4])
        notes.append(f"Text printed in the photo was not used as part names: {shown}")

    _progress(progress, "find")
    od = None
    ai_tried = False
    describe = getattr(ai, "describe_object", None)
    if use_ai and callable(describe) and _ai_ready():
        _progress(progress, "ai")
        ai_tried = True
        try:
            od = describe(img)
        except Exception:
            od = None
        if od is None or not getattr(od, "parts", None):
            warnings.append(_ai_error_warning())
    ai_parts = list(getattr(od, "parts", None) or []) if od is not None else []
    main_box = None
    if ai_parts:
        try:
            main_box = tuple(int(v) for v in ai_parts[0].box)
        except Exception:
            main_box = None

    subject, st = _subject_mask(img, main_box)
    if main_box is not None and not _subject_ok(st, True):
        subject, st = _subject_mask(img, None)       # the AI box seed failed: try the centre seed
    parts: list[Part] = []
    kind = "photo_object" if (subject.any() and _subject_ok(st, main_box is not None)) else "photo_scene"
    if kind == "photo_object":
        if ai_parts:
            p0 = ai_parts[0]
            parts.append(_new_part("p1", clean_name(p0.name) or "main object",
                                   (p0.description or "").strip() or "The main object, simplified from the photo.",
                                   subject, img, match_method="ai", description_source="ai"))
        else:
            parts.append(_new_part("p1", "main object",
                                   "Main object. Its outline, simplified from the photo.", subject, img))
        details = _interior_details(img, subject, text_boxes)
        used = [False] * len(details)
        s_area = int((subject > 0).sum())
        sb = mask_bbox(subject)
        for ap in ai_parts[1:PHOTO_MAX_AI_PARTS]:
            try:
                x0, y0, x1, y1 = (int(v) for v in ap.box)
            except Exception:
                continue
            x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
            y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            boxm = np.zeros((h, w), np.uint8)
            boxm[y0:y1, x0:x1] = 255
            # a detected patch mostly inside the box is the part; else an ellipse in the box, clipped
            pick = None
            for di, dm in enumerate(details):
                if not used[di] and (dm[y0:y1, x0:x1] > 0).sum() >= 0.6 * (dm > 0).sum():
                    pick = di
                    break
            if pick is not None:
                used[pick] = True
                m = details[pick]
            else:
                m = np.zeros((h, w), np.uint8)
                cv2.ellipse(m, ((x0 + x1) // 2, (y0 + y1) // 2), (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)),
                            0, 0, 360, 255, -1)
                m = cv2.bitwise_and(m, subject)
            a = int((m > 0).sum())
            if a < PHOTO_DETAIL_MIN_FRAC * s_area or a > 0.9 * s_area:
                continue
            parts.append(_new_part(f"p{len(parts) + 1}", clean_name(ap.name) or f"part {len(parts) + 1}",
                                   (ap.description or "").strip() or capitalise(clean_name(ap.name)) + ".",
                                   m, img, match_method="ai", description_source="ai"))
        if not ai_parts:
            for di, dm in enumerate(details):
                pos = _position_name(mask_bbox(dm), w, h, within=sb)
                nm = f"{pos} detail" if pos != "middle" else "middle detail"
                if any(p.name == nm for p in parts):
                    nm = f"{nm} {len(parts) + 1}"
                parts.append(_new_part(f"p{len(parts) + 1}", nm,
                                       f"{capitalise(nm)}. A separate patch on the {pos} of the main object.",
                                       dm, img))
        title = clean_title(getattr(od, "title", "") or "") if od is not None else ""
        title = title or "photo of an object"
        notes.append("Simplified from a photo: the outline of the main object"
                     + (" and a few clearly separate details." if len(parts) > 1 else "."))
    else:
        regions = _scene_regions(img)
        names_seen: dict[str, int] = {}
        for m in regions:
            col = np.median(img[m > 0], axis=0)
            pos = _position_name(mask_bbox(m), w, h)
            base = f"{_colour_name(col)} area"
            nm = f"{base} {pos}" if pos != "middle" else f"{base} in the middle"
            nm = nm.replace(" upper ", " upper ").strip()
            names_seen[nm] = names_seen.get(nm, 0) + 1
            if names_seen[nm] > 1:
                nm = f"{nm} {names_seen[nm]}"
            share = int(round(100 * (m > 0).mean()))
            parts.append(_new_part(f"p{len(parts) + 1}", nm,
                                   f"{capitalise(base)}, {('at the ' + pos) if pos != 'middle' else 'in the middle'} "
                                   f"of the picture — about {share}% of it.", m, img))
        title = clean_title(getattr(od, "title", "") or "") if od is not None else ""
        title = title or "photo of a scene"
        if parts:
            warnings.append("This looks like a photo of a scene — scenes simplify a lot: only the "
                            f"{len(parts)} biggest areas are kept. A drawn diagram works much better.")
        else:
            warnings.append("No parts found — this photo has no clear main object. "
                            "Take a photo of a diagram or of one object on a plain background.")
    a = Analysis(width=w, height=h, image=img, title=title, diagram_type="photo", parts=parts, labels=labels,
                 linework=np.zeros((h, w), np.uint8), warnings=warnings, timings={},
                 ai_used=od is not None and bool(ai_parts), page_flattened=False, input_kind=kind,
                 rotation=rotation, notes=notes)
    if ai_tried and od is not None and getattr(od, "summary", None):
        a.notes.append(str(od.summary).strip())
    return a


def _progress(cb, stage: str) -> None:
    if cb is None:
        return
    try:
        cb(stage)
    except Exception:  # a broken progress reporter must never fail the job
        pass


# ============================================================================ main


def _empty_analysis(img: np.ndarray, t0: float, warning: str, input_kind: str = "diagram",
                    rotation: int = 0, labels: Optional[list] = None) -> Analysis:
    h, w = img.shape[:2]
    return Analysis(width=w, height=h, image=img, title="diagram", diagram_type="diagram", parts=[],
                    labels=labels or [], linework=np.zeros((h, w), np.uint8), warnings=[warning],
                    timings={"analyze": round(time.perf_counter() - t0, 3)}, input_kind=input_kind,
                    rotation=rotation)


def analyze(image_rgb: np.ndarray, *, use_ai: bool = True,
            progress: Optional[Callable[[str], None]] = None) -> Analysis:
    """Image -> Analysis. Never raises for an ordinary image.

    `progress` (optional) is called with "read" before OCR, "find" before segmentation and "ai"
    before any AI call (only when AI will run)."""
    t0 = time.perf_counter()
    try:
        analysis, _ = _analyze_impl(image_rgb, use_ai=use_ai, progress=progress)
        try:
            _circuit_components(analysis)
        except Exception:  # pragma: no cover
            pass
        try:
            _network_targets(analysis)
        except Exception:  # pragma: no cover
            pass
        try:
            _declutter_linework(analysis)
        except Exception:  # pragma: no cover
            pass
        try:
            _legend_names(analysis)
        except Exception:  # pragma: no cover
            pass
        try:
            _reading_issues(analysis)
        except Exception:  # pragma: no cover - a check must never lose the analysis
            pass
        return analysis
    except Exception as exc:  # never raise for an ordinary image
        img = image_rgb if (isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3) else np.full((10, 10, 3), 255, np.uint8)
        try:
            img = _to_working(np.ascontiguousarray(img[:, :, :3]).astype(np.uint8))
        except Exception:
            pass
        # F10: no exception names in user-facing text; keep the type for diagnostics only
        a = _empty_analysis(img, t0, "No parts found — try a clearer, flatter photo")
        a.trace = {**(a.trace or {}), "error": type(exc).__name__}
        return a


_DESIGNATOR_RE = re.compile(
    r"^\s*([RVCL])\s*[_,.]?\s*([0-9]|[a-zA-Z])?\s*[=:]\s*([0-9]+(?:[.,][0-9]+)?)\s*(k|K|M|m|u|µ)?", re.I)
_UNIT_WORDS = {"R": "ohms", "V": "volts", "C": "farads", "L": "henries"}
_KIND_WORDS = {"R": "resistor", "V": "battery", "C": "capacitor", "L": "inductor"}
_PREFIX_WORDS = {"k": "kilo", "K": "kilo", "M": "mega", "m": "milli", "u": "micro", "µ": "micro"}


_ROUTE_RE = re.compile(r"\b(route|line|loop|branch|path)\b", re.I)


def _network_targets(analysis: Analysis) -> None:
    """Handover P1 "map and wire topology": a transit / network map is stations (ringed dots,
    each named by the text beside it) joined by coloured routes (named by the legend entry printed
    in the route's colour). Stations become point targets, routes path targets; colour fills that
    the lines happen to enclose are not structures. Needs >= 4 ringed stations, >= 3 of them named,
    so ordinary diagrams with a few dots are untouched."""
    img = analysis.image
    if img is None:
        return
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    light = (gray > 200).astype(np.uint8)
    dark = gray < 110
    n, cc, st, cen = cv2.connectedComponentsWithStats(light, connectivity=4)
    diag = float(np.hypot(h, w))
    rmin, rmax = max(3, int(0.003 * diag)), max(8, int(0.014 * diag))
    stations = []
    for i in range(1, n):
        bw, bh, area = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA]
        r = (bw + bh) / 4.0
        if not (rmin <= r <= rmax) or abs(bw - bh) > 0.3 * max(bw, bh):
            continue
        if area < 0.6 * np.pi * r * r:
            continue
        comp = (cc == i).astype(np.uint8)
        ring = (cv2.dilate(comp, np.ones((5, 5), np.uint8)) > 0) & (comp == 0)
        if ring.sum() == 0 or dark[ring].mean() < 0.6:
            continue
        cx_, cy_ = float(cen[i][0]), float(cen[i][1])
        if any(lb.bbox[0] - 2 <= cx_ <= lb.bbox[2] + 2 and lb.bbox[1] - 2 <= cy_ <= lb.bbox[3] + 2
               for lb in (analysis.labels or [])):
            continue                                      # the hole of an "o" in a word
        stations.append((cx_, cy_, r))
    if len(stations) < 4:
        return
    labels = [lb for lb in (analysis.labels or []) if lb.role in ("part_label", "other") and clean_name(lb.text)]
    route_labels = [lb for lb in labels if _ROUTE_RE.search(lb.text)]
    stn_labels = [lb for lb in labels if lb not in route_labels]
    pairs = []
    for si, (x, y, r) in enumerate(stations):
        for li, lb in enumerate(stn_labels):
            d = rect_distance(x, y, lb.bbox)
            if d <= 3 * r + 25:
                pairs.append((d, si, li))
    pairs.sort()
    s_name, l_used = {}, set()
    for d, si, li in pairs:
        if si in s_name or li in l_used:
            continue
        s_name[si] = li
        l_used.add(li)
    if len(s_name) < 3:
        return
    names_l = {clean_name(stn_labels[li].text) for li in s_name.values()}
    # colour fills between the routes and parts that only repeat a station's name are not targets
    keep = []
    for p in analysis.parts:
        if p.name in names_l or p.name.startswith("part "):
            continue
        keep.append(p)
    new_parts: list[Part] = []
    for si, (x, y, r) in enumerate(stations):
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (int(round(x)), int(round(y))), int(round(r + 3)), 255, -1)
        lb = stn_labels[s_name[si]] if si in s_name else None
        nm = clean_name(lb.text) if lb is not None else ""
        new_parts.append(Part(
            id=f"s{si + 1}", name=nm or f"part s{si + 1}", description="", key="", color=(255, 255, 255),
            mask=m, polygons=mask_polygons(m, 1.5), bbox=mask_bbox(m), area=int((m > 0).sum()), count=1,
            include=True, kind="region", label_text=(clean_text(lb.text) or lb.text) if lb is not None else None,
            match_method="nearest" if lb is not None else "none", target_type="landmark", extent="full"))
        if lb is not None:
            lb.part_id = f"s{si + 1}"
    lab = rgb_to_lab(img).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128 if lab[:, :, 1].max() > 100 else lab[:, :, 1],
                      lab[:, :, 2] - 128 if lab[:, :, 2].max() > 100 else lab[:, :, 2])
    for ri, lb in enumerate(route_labels):
        x0, y0, x1, y1 = lb.bbox
        sub = lab[y0:y1, x0:x1].reshape(-1, 3)
        csub = chroma[y0:y1, x0:x1].reshape(-1)
        if csub.size == 0 or (csub > 20).sum() < 15:
            continue
        ref = np.median(sub[csub > 20], axis=0)
        dist = np.linalg.norm(lab - ref, axis=2)
        m = ((dist < 22) & (chroma > 20)).astype(np.uint8)
        for other in route_labels:                       # the legend text itself is not the route
            ox0, oy0, ox1, oy1 = other.bbox
            m[max(0, oy0 - 3):oy1 + 3, max(0, ox0 - 3):ox1 + 3] = 0
        nc, ccr, str_, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keepc = np.zeros(nc, bool)
        for c in range(1, nc):
            if max(str_[c, cv2.CC_STAT_WIDTH], str_[c, cv2.CC_STAT_HEIGHT]) >= 0.05 * diag:
                keepc[c] = True
        m = keepc[ccr].astype(np.uint8) * 255
        if not m.any():
            continue
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        nm = clean_name(lb.text)
        new_parts.append(Part(
            id=f"r{ri + 1}", name=nm, description="", key="",
            color=tuple(int(v) for v in np.median(img[m > 0], axis=0)), mask=m, polygons=mask_polygons(m, 1.5),
            bbox=mask_bbox(m), area=int((m > 0).sum()), count=1, include=True, kind="line",
            label_text=clean_text(lb.text) or lb.text, match_method="legend", target_type="path", extent="full"))
        lb.part_id = f"r{ri + 1}"
    # which stations each route serves (from geometry, so a teacher can check it against the map)
    routes = [p for p in new_parts if p.target_type == "path"]
    near = {rp.id: cv2.dilate(rp.mask, np.ones((7, 7), np.uint8)) > 0 for rp in routes}
    serves = {sp.id: [rp.name for rp in routes if (near[rp.id] & (sp.mask > 0)).any()]
              for sp in new_parts if sp.target_type == "landmark"}
    for rp in routes:
        served = [sp.name for sp in new_parts if sp.target_type == "landmark" and not sp.name.startswith("part ")
                  and rp.name in serves.get(sp.id, [])]
        if served:
            rp.flow = "stops at " + _join_names(served)
    for sp in new_parts:
        if sp.target_type == "landmark" and len(serves.get(sp.id, [])) >= 2:
            sp.flow = "interchange for the " + _join_names(serves[sp.id])
    analysis.parts = keep + new_parts
    analysis.diagram_type = "network"
    analysis.unmatched_labels = [u for u in (analysis.unmatched_labels or []) if u not in names_l]
    _describe(analysis.parts)
    for p in analysis.parts:
        if p.target_type == "landmark" and p.flow:
            p.description = f"{capitalise(p.name)} station, {p.flow}."
        elif p.target_type == "landmark" and p.match_method == "nearest":
            p.description = f"{capitalise(p.name)} station."
        elif p.target_type == "path" and p.flow:
            p.description = f"{capitalise(p.name)}, {p.flow}."
    from engine.tactile import assign_keys
    assign_keys(analysis.parts)


def _circuit_components(analysis: Analysis) -> None:
    """F07 / handover "mixed circuit": a circuit's labels are component designators with values
    ("R1 = 1.00 Ω", "V = 12.0 V"), not names of filled regions. When at least two such labels
    are read, find each component symbol (the drawing minus long straight wires, grouped) next to
    its label and make it the part, named with its value in words braille can carry. The filled
    interior of a wire loop is not a structure: it is switched off. Wires stay raised lines.
    Unknowns like "I1 = ?" are never parts or missing-part suggestions."""
    img = analysis.image
    if img is None:
        return
    found = []
    for lb in analysis.labels or []:
        t = (lb.text or "").replace("Ω", " ohm").strip()
        mm = _DESIGNATOR_RE.match(t)
        if not mm:
            continue
        kind = mm.group(1).upper()
        idx = mm.group(2) if (mm.group(2) or "").isdigit() else ("?" if mm.group(2) else "")
        found.append({"kind": kind, "idx": idx, "value": mm.group(3).replace(",", "."),
                      "prefix": _PREFIX_WORDS.get(mm.group(4) or "", ""), "label": lb})
    # designator and value on separate lines ("R2" above "6.00 Ω"): pair each value with the
    # nearest bare designator; a lone voltage ("12.0 V") is the battery
    taken = {id(f["label"]) for f in found}
    bare = [lb for lb in analysis.labels or [] if id(lb) not in taken
            and re.fullmatch(r"\s*[RCL]\s*[_,.]?\s*[0-9a-zA-Z]?\s*", lb.text or "")]
    vals = [lb for lb in analysis.labels or [] if id(lb) not in taken
            and re.fullmatch(r"\s*([0-9]+(?:[.,][0-9]+)?)\s*(k|K|M|m|u|µ)?\s*(\S{0,3})\s*", lb.text or "")]
    used_b = set()
    for vl in vals:
        mv = re.fullmatch(r"\s*([0-9]+(?:[.,][0-9]+)?)\s*(k|K|M|m|u|µ)?\s*(\S{0,3})\s*", vl.text)
        vx0, vy0, vx1, vy1 = vl.bbox
        th = max(10, vy1 - vy0)
        best, bd = None, 1e9
        for b in bare:
            if id(b) in used_b:
                continue
            bx0, by0, bx1, by1 = b.bbox
            d = float(np.hypot(max(0, max(vx0, bx0) - min(vx1, bx1)), max(0, max(vy0, by0) - min(vy1, by1))))
            if d < bd and d <= 2.5 * th:
                best, bd = b, d
        unit = (mv.group(3) or "").upper()
        if best is not None:
            used_b.add(id(best))
            mb = re.fullmatch(r"\s*([RCL])\s*[_,.]?\s*([0-9a-zA-Z])?\s*", best.text)
            kind = mb.group(1).upper()
            idx = mb.group(2) if (mb.group(2) or "").isdigit() else ("?" if mb.group(2) else "")
            found.append({"kind": kind, "idx": idx, "value": mv.group(1).replace(",", "."),
                          "prefix": _PREFIX_WORDS.get(mv.group(2) or "", ""), "label": best})
        elif unit == "V":
            found.append({"kind": "V", "idx": "", "value": mv.group(1).replace(",", "."),
                          "prefix": _PREFIX_WORDS.get(mv.group(2) or "", ""), "label": vl})
    if len(found) < 2:
        return
    keys = [(f["kind"], f["idx"]) for f in found if f["idx"] != "?"]
    if len(keys) != len(set(keys)):
        return                    # the same designator twice: several circuits / steps in one figure
    # an unreadable index ("Rg" for R3) is not guessed: the part is named by kind and value
    # ("resistor, 13.00 ohms") and flagged for the teacher to check
    unsure = {id(f) for f in found if f["idx"] == "?"}
    H, W = img.shape[:2]
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    ink = ((grey < 150) | (chroma > 35)).astype(np.uint8) * 255
    for lb in analysis.labels or []:                         # text is not a symbol
        x0, y0, x1, y1 = (int(v) for v in lb.bbox)
        ink[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = 0
    L = max(25, int(0.06 * max(H, W)))
    wires = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) | \
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    sym = cv2.subtract(ink, cv2.dilate(wires, np.ones((3, 3), np.uint8)))
    sym = cv2.morphologyEx(sym, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    g = max(5, int(0.012 * max(H, W)))
    grouped = cv2.dilate(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    n, cc, st, cent = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        bw, bh, area = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA]
        if area < 0.0008 * H * W or max(bw, bh) > 0.4 * max(H, W):
            continue
        comps.append((i, cent[i], (st[i, 0], st[i, 1], st[i, 0] + bw, st[i, 1] + bh)))
    if not comps:
        return
    pairs = []
    for fi, f in enumerate(found):
        x0, y0, x1, y1 = f["label"].bbox
        th = max(10, y1 - y0)
        for ci, (i, c, b) in enumerate(comps):
            dx = max(0, max(x0, b[0]) - min(x1, b[2]))
            dy = max(0, max(y0, b[1]) - min(y1, b[3]))
            d = float(np.hypot(dx, dy))
            if d <= 4 * th:
                pairs.append((d, fi, ci))
    pairs.sort()
    used_f, used_c, made = set(), set(), []
    for d, fi, ci in pairs:
        if fi in used_f or ci in used_c:
            continue
        used_f.add(fi)
        used_c.add(ci)
        made.append((found[fi], comps[ci]))
    if len(made) < 2:
        return
    from engine.vision_utils import mask_bbox, mask_polygons
    # the filled inside of a wire loop is paper, not a structure
    loops = []
    for p in analysis.parts:
        if p.mask is not None and p.include and p.mask.any() and (not p.name or p.name.startswith("part ")):
            med = np.median(img[p.mask > 0], axis=0)
            if float(med.min()) >= 225 and p.area > 0.05 * H * W:
                loops.append(p)
    for p in loops:
        analysis.parts.remove(p)
        # the wires along the loop were inside that "region": give them back to the raised lines
        if analysis.linework is not None and analysis.linework.shape[:2] == (H, W):
            near = cv2.dilate((p.mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            analysis.linework = (((analysis.linework > 0) | ((ink > 0) & near)) * 255).astype(np.uint8)
    new_parts = []
    for f, (i, c, b) in made:
        strokes = (cc == i) & (sym > 0)
        hull = cv2.convexHull(np.column_stack(np.nonzero(strokes))[:, ::-1].astype(np.int32))
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [hull], 255)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        kind = f["kind"]
        des = f"{kind}{f['idx']}" if f["idx"] else ""
        unit = f"{f['prefix']}{_UNIT_WORDS[kind]}"
        if kind == "V" and not f["idx"]:
            name = f"battery, {f['value']} {unit}"
            desc = f"Battery (voltage source), {f['value']} {unit}."
        elif f["idx"] == "?":
            name = f"{_KIND_WORDS[kind]}, {f['value']} {unit}"
            desc = f"{_KIND_WORDS[kind].capitalize()}, {f['value']} {unit}. Check its number on the picture."
        else:
            name = f"{des}, {f['value']} {unit}"
            desc = f"{_KIND_WORDS[kind].capitalize()} {des}, {f['value']} {unit}."
        # reuse a part already found on this symbol (keeps its id), otherwise make one
        host = None
        for p in analysis.parts:
            if p.mask is not None and p.include and ((p.mask > 0) & (m > 0)).sum() >= 0.3 * min(p.area, int((m > 0).sum())) \
                    and p.area < 0.05 * H * W:
                host = p
                break
        if host is None:
            col = np.median(img[m > 0], axis=0)
            host = Part(id=f"c{len(analysis.parts) + len(new_parts) + 1}", name="", description="", key="",
                        color=(int(col[0]), int(col[1]), int(col[2])), mask=m, polygons=mask_polygons(m, 1.5),
                        bbox=mask_bbox(m), area=int((m > 0).sum()), count=1, include=True, kind="region")
            new_parts.append(host)
        else:
            u = (host.mask > 0) | (m > 0)
            um = u.astype(np.uint8) * 255
            host.mask, host.polygons, host.bbox, host.area = um, mask_polygons(um, 1.5), mask_bbox(um), int(u.sum())
        host.name, host.description = name, desc
        host.label_text = f["label"].text
        host.match_method = "nearest" if id(f) in unsure else "circuit"
        f["label"].part_id = host.id
    analysis.parts.extend(new_parts)
    # renumber so ids stay p1..pN in area order (keys are assigned later)
    order = sorted(analysis.parts, key=lambda q: -q.area)
    remap = {}
    for k, q in enumerate(order):
        remap[q.id] = f"p{k + 1}"
    for q in analysis.parts:
        q.id = remap[q.id]
    analysis.parts = order
    for lb in analysis.labels or []:
        if lb.part_id is not None:
            lb.part_id = remap.get(lb.part_id)
    analysis.diagram_type = "circuit"
    from engine.vision_utils import title_problem
    words = (analysis.title or "").split()
    if not words or title_problem(analysis.title) or "=" in analysis.title or analysis.title == "diagram" \
            or all(len(wd) <= 2 for wd in words):
        analysis.title = "circuit"
    # designators were read as components: they are not "print only" arrow captions
    analysis.notes = [nt for nt in (analysis.notes or []) if not nt.startswith("Labels on arrows/lines")]
    analysis.unmatched_labels = [u for u in (analysis.unmatched_labels or []) if not _DESIGNATOR_RE.match(u)]
    analysis.warnings = [w for w in analysis.warnings if "no label" not in w]
    n_un = sum(1 for q in analysis.parts if q.include and q.name.startswith("part "))
    if n_un:
        analysis.warnings.append("1 part has no label — rename it in review" if n_un == 1
                                 else f"{n_un} parts have no label — rename them in review")
    analysis.notes = list(analysis.notes or []) + [
        "Read as a circuit: each component is named with its value; wires are raised lines. "
        "Unknowns such as “I1 = ?” are kept as print only."]


def _declutter_linework(analysis: Analysis) -> None:
    """Handover (handwritten notebook): ruled-paper remnants and a box drawn round the title are
    not part of the diagram. Drop raised-line pieces that touch no part and are either a short
    straight axis-aligned dash, or hug the title text. Arrows and outlines that touch the drawing
    stay; so do long lines (wires, axes)."""
    lw = analysis.linework
    if lw is None or not lw.any() or not analysis.parts:
        return
    H, W = lw.shape[:2]
    big = max(H, W)
    near = np.zeros((H, W), np.uint8)
    for p in analysis.parts:
        if p.mask is not None and p.include and p.mask.shape[:2] == (H, W):
            near |= (p.mask > 0).astype(np.uint8)
    near = cv2.dilate(near, np.ones((9, 9), np.uint8)) > 0
    title_boxes = []
    for lb in analysis.labels or []:
        if lb.role == "title":
            x0, y0, x1, y1 = (int(v) for v in lb.bbox)
            m = int(0.04 * big)
            title_boxes.append((x0 - m, y0 - m, x1 + m, y1 + m))
    n, cc, st, _ = cv2.connectedComponentsWithStats((lw > 0).astype(np.uint8), connectivity=8)
    drop = np.zeros(n, bool)
    thin = 0.006 * big + 3
    for i in range(1, n):
        x, y, w_, h_, a_ = (int(v) for v in st[i])
        comp = cc[y:y + h_, x:x + w_] == i
        if (near[y:y + h_, x:x + w_] & comp).any():
            continue
        dash = min(w_, h_) <= thin and max(w_, h_) <= 0.12 * big
        in_title = any(x >= bx0 and y >= by0 and x + w_ <= bx1 and y + h_ <= by1 for bx0, by0, bx1, by1 in title_boxes)
        if dash or in_title:
            drop[i] = True
    if drop.any():
        keep = (~drop[cc]) & (lw > 0)
        analysis.linework = (keep * 255).astype(np.uint8)


def _legend_names(analysis: Analysis) -> None:
    """F12: a chart legend says which series is which ("y=1" beside a blue dot). For each unnamed
    part, look for a short label with a small coloured marker just left of it whose colour matches
    the part, and name the part after that legend entry (match_method "legend")."""
    img = analysis.image
    if img is None or not analysis.parts:
        return
    unnamed = [p for p in analysis.parts if p.include and (not p.name or p.name.startswith("part "))]
    if not unnamed:
        return
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    entries = []
    for lb in analysis.labels or []:
        t = (lb.text or "").strip()
        if not t or len(t) > 30 or lb.part_id is not None:
            continue
        x0, y0, x1, y1 = (int(v) for v in lb.bbox)
        th = max(8, y1 - y0)
        bx0, bx1 = max(0, x0 - 3 * th), max(0, x0 - 2)
        by0, by1 = max(0, y0 - th // 4), min(H, y1 + th // 4)
        if bx1 - bx0 < 4 or by1 - by0 < 4:
            continue
        win = lab[by0:by1, bx0:bx1]
        chroma = np.hypot(win[:, :, 1] - 128, win[:, :, 2] - 128)
        m = chroma > 25                                   # a coloured marker, not grey text or paper
        if m.sum() < 0.02 * m.size or m.sum() < 12:
            continue
        entries.append((t, np.median(win[m], axis=0)))
    if len(entries) < 1:
        return
    used = set()
    for p in sorted(unnamed, key=lambda q: -q.area):
        pm = p.mask > 0
        if not pm.any():
            continue
        pc = np.median(lab[pm], axis=0)
        best, bd = None, 1e9
        for k, (t, c) in enumerate(entries):
            if k in used:
                continue
            d = float(np.linalg.norm(pc - c))
            if d < bd:
                best, bd = k, d
        if best is not None and bd <= 22:
            used.add(best)
            p.name = entries[best][0]
            p.label_text = entries[best][0]
            p.match_method = "legend"
            if p.description.lower().startswith("part "):
                p.description = f"{p.name}."
    if used:
        n_un = sum(1 for p in analysis.parts if p.include and p.name.startswith("part "))
        analysis.warnings = [w for w in analysis.warnings if "no label" not in w]
        if n_un:
            analysis.warnings.append("1 part has no label — rename it in review" if n_un == 1
                                     else f"{n_un} parts have no label — rename them in review")


_SCRIPTS = [("Chinese or Japanese", "\u4e00-\u9fff\u3040-\u30ff"), ("Korean", "\uac00-\ud7af"),
            ("Hebrew", "\u0590-\u05ff"), ("Arabic", "\u0600-\u06ff"), ("Cyrillic", "\u0400-\u04ff"),
            ("Greek", "\u0370-\u03ff"), ("Devanagari", "\u0900-\u097f"), ("Thai", "\u0e00-\u0e7f")]


def _reading_issues(analysis: Analysis) -> None:
    """F06 / F13: say plainly when labels exist but could not be used, instead of quietly turning a
    diagram into a generic photo outline. Stored in analysis.trace["issues"] for readiness."""
    from engine.lexicon import is_term
    from engine.vision_utils import looks_like_structure
    issues = []
    texts = [lb.text for lb in (analysis.labels or []) if lb.text]
    found = []
    for name, rng in _SCRIPTS:
        # two letters of the script together (a word), so unit signs like Ω or µ don't count
        if any(re.search(f"[{rng}]{{2,}}", t.replace("Ω", "").replace("µ", "").replace("μ", "")) for t in texts):
            found.append(name)
    named = [p for p in analysis.parts if p.include and p.name and not p.name.startswith("part ")
             and getattr(p, "match_method", "none") not in ("none", None)]
    photo = analysis.input_kind in ("photo_object", "photo_scene")
    if found:
        issues.append({
            "code": "unsupported_script", "severity": "block" if (photo or not named) else "warn",
            "action": "retake",
            "message": f"This picture has {' and '.join(found)} text. Feelable reads English labels and makes "
                       "English (UEB) braille, so this text can't become part names or braille yet. "
                       "Use a diagram with English labels, or name the parts yourself."})
    termish = [t for t in texts if is_term(t) or looks_like_structure(t)]
    if photo and len(termish) >= 3:
        issues.append({
            "code": "labels_unreadable", "severity": "warn", "action": "retake",
            "message": "This looks like a labelled diagram, but its labels couldn't be read reliably, so only a "
                       "simple outline was made. For the full lesson, retake the photo sharper and straight on, "
                       "or use a screenshot. Approve only if a simple outline is what you want."})
    if (analysis.trace or {}).get("ocr_failed"):
        issues.append({
            "code": "ocr_failed", "severity": "warn", "action": "retry",
            "message": "Feelable could not read the text in this picture (the text reader failed, which is "
                       "not the same as finding no labels). Try again; if it keeps happening, name the "
                       "parts yourself."})
    if issues:
        analysis.trace = {**(analysis.trace or {}), "issues": issues}


def _analyze_impl(image_rgb: np.ndarray, *, use_ai: bool = True,
                  progress: Optional[Callable[[str], None]] = None) -> tuple[Analysis, dict]:
    t0 = time.perf_counter()
    debug: dict = {}
    img = np.ascontiguousarray(image_rgb)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    img = img.astype(np.uint8)
    warnings: list[str] = []
    dropped: list[str] = []

    # 1. normalise -------------------------------------------------------------------------
    img = _to_working(img)
    img_raw = img
    feats = _image_features(img)
    debug["features"] = feats
    if _is_blank(feats):
        a = _empty_analysis(img, t0, "No parts found — the picture looks blank. "
                                     "Take a photo of the diagram itself.", input_kind="blank")
        return a, debug
    # 2. page + lighting flattening ---------------------------------------------------------
    page_flattened = False
    photo = _looks_like_photo(img)
    if photo:
        q = _find_page(img)
        if q is not None:
            page = _to_working(_warp_page(img, q))
            pf = _image_features(page)             # judge the page, not the table around it
            debug["features_page"] = pf
            if not _is_blank(pf):                  # an empty "page" is just a plain patch (sky, wall)
                img, feats, page_flattened = page, pf, True
        img = _flatten_lighting(img)
    h, w = img.shape[:2]
    N = h * w

    # 3. OCR + orientation + roles ----------------------------------------------------------
    _progress(progress, "read")
    from engine.ocr import take_failure
    take_failure()
    labels_all = run_ocr(img, min_confidence=0.0)
    debug["ocr_failed"] = take_failure()
    rotation, rot_labels = _auto_orient(img, labels_all, feats)
    if rotation:
        img = _rotate_cw(img, rotation)
        img_raw = _rotate_cw(img_raw, rotation)
        labels_all = rot_labels or []
        h, w = img.shape[:2]
        N = h * w
    labels = [lb for lb in labels_all if lb.confidence >= OCR_MIN_CONFIDENCE]
    debug["rotation"] = rotation
    ts = _text_stats(labels, w, h)
    kind0 = _classify(feats, ts)
    debug["kind0"] = kind0
    debug["text_stats"] = ts
    if kind0 == "text_page":
        _assign_roles(labels, w, h)
        a = _empty_analysis(img, t0, "No diagram found — this looks like a page of text. "
                                     "Take a photo of the diagram itself.", input_kind="text_page",
                            rotation=rotation, labels=labels)
        return a, debug
    if kind0 == "blank" or (kind0 == "photo" and not page_flattened and not labels and _is_blank(feats)):
        return _empty_analysis(img, t0, "No parts found — the picture looks blank. "
                                        "Take a photo of the diagram itself.", input_kind="blank",
                               rotation=rotation), debug
    if kind0 == "photo":
        a = _analyze_photo(img_raw, labels, use_ai=use_ai, progress=progress, t0=t0, rotation=rotation,
                           feats=feats, text_boxes_ok=not page_flattened)
        a.timings["analyze"] = round(time.perf_counter() - t0, 3)
        return a, debug
    _assign_roles(labels, w, h)
    # a "title" in the same small print as the part labels that is itself a structure name
    # ("Zygomatic arch" printed above a skull) is a part label: it points at something
    hs = [lb.bbox[3] - lb.bbox[1] for lb in labels if lb.role == "part_label"]
    if len(hs) >= 4:
        med_h = float(np.median(hs))
        for lb in labels:
            if lb.role == "title" and is_term(lb.text) and (lb.bbox[3] - lb.bbox[1]) <= 1.2 * med_h:
                lb.role = "part_label"
    title = _title_text(labels)
    lab0 = rgb_to_lab(img)

    # 4. text removal ----------------------------------------------------------------------
    tmask = _text_mask(img, lab0, labels)
    work = cv2.inpaint(img, tmask, 3, cv2.INPAINT_TELEA) if tmask.any() else img.copy()

    # background colour
    bg_rgb = _border_mode_colour(work)
    bg_lab = rgb_to_lab(bg_rgb.reshape(1, 1, 3).astype(np.uint8))[0, 0]
    lab = rgb_to_lab(work)
    de_bg = delta_e(lab, bg_lab)
    nonbg = de_bg > 15

    # 6. mode (text removed; leaders are dark/neutral so they do not change it) -------------
    chroma0 = np.hypot(lab[:, :, 1], lab[:, :, 2])
    colour_frac = float(((chroma0 > 20) & (de_bg > 15)).mean())
    # grey-filled shapes (a grayscale print of a colour diagram) segment like colour regions
    fill = ((de_bg > 15) & (lab[:, :, 0] > 35)).astype(np.uint8)
    fill_frac = float(cv2.morphologyEx(fill, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).mean())
    mode = "lineart" if (colour_frac < LINEART_COLOUR_FRACTION and fill_frac < GREY_FILL_FRACTION) else "colour"
    debug["mode"] = mode
    debug["colour_frac"] = colour_frac
    debug["fill_frac"] = fill_frac

    # 5. leader lines ----------------------------------------------------------------------
    leaders, lmask = _find_leaders(work, labels, nonbg, protect_crossings=(mode == "lineart"))
    if lmask.any():
        work = cv2.inpaint(work, lmask, 3, cv2.INPAINT_TELEA)
        lab = rgb_to_lab(work)
        de_bg = delta_e(lab, bg_lab)
    debug["leaders"] = [{"label": labels[L.label_idx].text, "points": L.points, "end": L.end} for L in leaders]
    leaders_by_label: dict[int, list[_Leader]] = {}
    for L in leaders:
        leaders_by_label.setdefault(L.label_idx, []).append(L)
    # a top label with a pointer line is a part label ("Zygomatic arch" above a skull), not the title
    demoted = [i for i in leaders_by_label if labels[i].role == "title"]
    for i in demoted:
        labels[i].role = "part_label"
    if demoted:
        title = _title_text(labels)
    _split_colon_headings(labels)
    _join_label_lines(labels, leaders_by_label)
    L_ = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])

    thin_m = _thin_dark(work) > 0
    bg_L = float(bg_lab[0])
    ink_dark = (L_ < min(45.0, bg_L - 30)) & (chroma < 18)
    dots: list[tuple[float, float]] = []
    thin_dropped = 0

    # 7/8. segmentation --------------------------------------------------------------------
    _progress(progress, "find")
    if mode == "colour":
        thick_dark = cv2.morphologyEx(ink_dark.astype(np.uint8), cv2.MORPH_OPEN, disk(4)) > 0
        ink = (ink_dark | thin_m) & ~thick_dark
        light_neutral = (chroma < 6) & (L_ > 86)
        cand = ((de_bg > 12) & ~light_neutral & ~ink) | thick_dark
        cands, thin_dropped = _segment_colour(work, lab, cand, N, dots, absorb=(de_bg > 12) & ~light_neutral)
    else:
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        bs = max(15, (min(h, w) // 25) | 1)
        ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, bs, 18)
        ink_u8 = ad | ((gray < min(90, bg_L * 2.55 - 70)).astype(np.uint8) * 255)
        n, cc, st, _ = cv2.connectedComponentsWithStats(ink_u8, connectivity=8)
        speck = np.zeros(n, bool)
        speck[1:] = st[1:, cv2.CC_STAT_AREA] < 5
        ink_u8[speck[cc]] = 0
        ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        ink = ink_u8 > 0
        cands = _segment_lineart(ink_u8, N)

    # leaders that point at something no region covers: an outlined blank shape (white cloud on
    # a coloured page) or, in line-art, a symbol drawn with strokes (bulb, switch)
    if leaders:
        cands += _pointed_extras(cands, ink, leaders, mode, N)

    # remove page-sized candidates (frames / tinted page)
    cands = [c for c in cands if (c.filled > 0).sum() <= MAX_REGION_FRAC * N]

    # small dark dots (e.g. ribosomes) are clutter
    n, cc, st, cen = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    small_min = (COLOUR_MIN_AREA_FRAC if mode == "colour" else LINEART_MIN_AREA_FRAC) * N
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < small_min and _compact_small(st[i]) and st[i, cv2.CC_STAT_AREA] >= 6:
            dots.append((float(cen[i][0]), float(cen[i][1]), max(st[i, 2], st[i, 3]) / 2.0))
    # a real dot sits inside one region (or on the page); fragments where two regions meet are
    # anti-aliasing leftovers, not dots
    if dots:
        ctx = np.zeros((h, w), np.int16)
        ctx[de_bg <= 12] = -1
        for ci, c in enumerate(cands):
            ctx[c.raw] = ci + 1
        real = []
        for (x, y, r) in dots:
            ix, iy, rr = int(round(x)), int(round(y)), int(r) + 3
            win = ctx[max(0, iy - rr):iy + rr + 1, max(0, ix - rr):ix + rr + 1]
            vals = np.unique(win)
            if len(vals[vals != 0]) <= 1:
                real.append((x, y, r))
        dots = real
    dots_arr = np.array([(d[0], d[1]) for d in dots], np.float32).reshape(-1, 2)

    debug["dots"] = dots

    def nearest_dot(px, py):
        if len(dots_arr) == 0:
            return float("inf")
        return float(np.hypot(dots_arr[:, 0] - px, dots_arr[:, 1] - py).min())

    def clutter_near(px, py, r):
        return nearest_dot(px, py) <= r

    def plural(text):
        t = clean_name(text)
        return bool(re.search(r"[a-z]{3,}s$", t)) and not re.search(r"(ss|us|is|[^aeiou]ys)$", t)

    part_label_idx = [i for i, lb in enumerate(labels) if lb.role == "part_label"]

    # highlights / shading that k-means put in another region's colour cluster: a component that
    # sits entirely inside a different region than the rest of its cluster belongs to that region
    if mode == "colour" and len(cands) > 1:
        cands = _absorb_islands(cands)

    # shading merge (needs to know which clusters labels point at)
    if mode == "colour" and len(cands) > 1:
        referenced = set()
        pre = _components(cands)
        for i in part_label_idx:
            for Ld in leaders_by_label.get(i, []):
                t = _point_target(Ld.end[0], Ld.end[1], pre, Ld.heading, Ld.dot_end, Ld.points[-30:])
                if t is not None:
                    referenced.add(pre[t[0]].cand)
            if i not in leaders_by_label:
                x0, y0, x1, y1 = expand_box(labels[i].bbox, 3, w, h)
                for ci, c in enumerate(cands):
                    if c.raw[y0:y1, x0:x1].mean() > 0.3:
                        referenced.add(ci)
        cands = _merge_shading(cands, referenced, len(part_label_idx) >= 2)

    # labels that sit next to an arrow (not next to / on a part) name the arrow, not a part
    pre_lw = _linework(ink, tmask, lmask, [c.filled for c in cands], thin_m, mode, N)
    lw_dist = cv2.distanceTransform((pre_lw == 0).astype(np.uint8), cv2.DIST_L2, 3) if pre_lw.any() else None

    def line_dist(box):
        if lw_dist is None:
            return float("inf")
        x0, y0, x1, y1 = expand_box(box, 2, w, h)
        sub = lw_dist[y0:y1, x0:x1]
        return float(sub.min()) if sub.size else float("inf")

    # 10. label -> component targets -------------------------------------------------------
    comps = _components(cands) if cands else []
    targets: dict[int, tuple[int, float, str]] = {}   # label idx -> (comp idx, score, method)
    clutter_labels: set[int] = set()
    arrow_labels: set[int] = set()                     # captions of arrows / lines, not parts
    for i in part_label_idx:
        lb = labels[i]
        best: Optional[tuple[int, float, str]] = None
        lds = leaders_by_label.get(i, [])
        for Ld in lds:
            if clutter_near(Ld.end[0], Ld.end[1], 7):
                clutter_labels.add(i)
                continue
            if not comps:
                continue
            t = _point_target(Ld.end[0], Ld.end[1], comps, Ld.heading, Ld.dot_end, Ld.points[-30:])
            if t is None:
                continue
            k, d = t
            # a leader ending among many tiny dots probably names the dots (e.g. "ribosomes")
            nd = nearest_dot(Ld.end[0], Ld.end[1])
            pen = 0.6 * max(0.0, 25.0 - nd) + (12.0 if (nd <= 25 and plural(lb.text)) else 0.0)
            score = 300.0 - d - pen
            if best is None or score > best[1]:
                best = (k, score, "leader")
        if i in clutter_labels and best is None:
            continue
        if best is None and not lds and comps:
            bx = lb.bbox
            on = []
            for k, c in enumerate(comps):
                sub = c.mask[bx[1]:bx[3], bx[0]:bx[2]]
                if sub.size and sub.mean() >= 0.5:
                    on.append(k)
            ld = line_dist(bx)
            if on:
                k = min(on, key=lambda k: comps[k].area)
                if ld <= 20:
                    # the outline of the box the label sits in is not an arrow: measure only to
                    # linework that is not that region's own border (flowchart / table cells)
                    ld = _line_dist_off_border(pre_lw, comps[k].mask, bx)
                if ld > 20:
                    best = (k, 200.0, "on")
                else:
                    arrow_labels.add(i)
            else:
                dd = [(box_mask_distance(bx, c.pts), k) for k, c in enumerate(comps)]
                d, k = min(dd)
                pb = mask_bbox(comps[k].mask)
                ovx = min(bx[2], pb[2]) - max(bx[0], pb[0])
                ovy = min(bx[3], pb[3]) - max(bx[1], pb[1])
                aligned = d <= 60 and (ovx >= 0.5 * (bx[2] - bx[0]) or ovy >= 0.5 * (bx[3] - bx[1]))
                if d <= NEAREST_LABEL_PX and (aligned or not (ld <= 45 and ld < d)):
                    best = (k, 100.0 - d / 2.0, "near")
                elif ld <= 45:
                    arrow_labels.add(i)
        elif best is None and not lds and line_dist(lb.bbox) <= 45:
            arrow_labels.add(i)
        if best is not None:
            targets[i] = best

    debug["label_targets"] = {labels[i].text: (targets[i][0], round(targets[i][1], 1), targets[i][2], comps[targets[i][0]].cand)
                              for i in targets}
    debug["arrow_labels"] = [labels[i].text for i in arrow_labels]
    debug["roles"] = [(lb.text, lb.role) for lb in labels]
    # split clusters whose components are named differently (e.g. 'left lung' / 'right lung')
    groups: list[tuple[int, np.ndarray, list[int]]] = []   # (cand idx, filled mask bool, comp idxs)
    comp_to_group: dict[int, int] = {}
    for ci in range(len(cands)):
        my = [k for k, c in enumerate(comps) if c.cand == ci]
        named: dict[str, set] = {}
        for i, (k, sc, meth) in targets.items():
            if k in my and meth in ("leader", "on", "near"):
                if meth == "near" and sc < 80:
                    continue
                named.setdefault(clean_name(labels[i].text), set()).add(k)
        strong_names = {nm: ks for nm, ks in named.items() if nm}
        all_ks = [k for ks in strong_names.values() for k in ks]
        do_split = len(strong_names) >= 2 and len(set(all_ks)) == len(all_ks)
        near_only = all(targets[i][2] == "near" for i in targets if targets[i][0] in my)
        if do_split and near_only and len(strong_names) < 3:
            do_split = False
        if do_split:
            rest = [k for k in my if k not in all_ks]
            for nm, ks in sorted(strong_names.items()):
                ks = sorted(ks)
                m = np.zeros((h, w), bool)
                for k in ks:
                    m |= comps[k].mask
                groups.append((ci, m, ks))
            if rest:
                m = np.zeros((h, w), bool)
                for k in rest:
                    m |= comps[k].mask
                groups.append((ci, m, rest))
        else:
            if my:
                groups.append((ci, cands[ci].filled > 0, my))
        for gi in range(len(groups)):
            for k in groups[gi][2]:
                comp_to_group[k] = gi

    # cap the number of parts
    order = sorted(range(len(groups)), key=lambda g: -int(groups[g][1].sum()))
    if len(order) > MAX_PARTS:
        extra = len(order) - MAX_PARTS
        dropped.append(f"{extra} small regions (too many to feel)")
        order = order[:MAX_PARTS]
    gid_to_part = {g: n_ for n_, g in enumerate(order)}

    # 11. build parts ----------------------------------------------------------------------
    parts: list[Part] = []
    for n_, g in enumerate(order):
        ci, m, ks = groups[g]
        mask = m.astype(np.uint8) * 255
        rawpix = cands[ci].raw & m
        if rawpix.sum() < 10:
            rawpix = m
        col = np.median(img[rawpix], axis=0) if rawpix.any() else np.array([255, 255, 255])
        parts.append(Part(
            id=f"p{n_ + 1}", name="", description="", key="",
            color=(int(col[0]), int(col[1]), int(col[2])),
            mask=mask, polygons=mask_polygons(mask, 1.5), bbox=mask_bbox(mask),
            area=int(m.sum()), count=len(ks), include=True, kind="region"))

    # greedy one-label-per-part assignment
    pairs = []
    for i, (k, sc, meth) in targets.items():
        g = comp_to_group.get(k)
        if g is None or g not in gid_to_part:
            continue
        pairs.append((sc, i, gid_to_part[g], meth))
    # ties (several pointers ending in one region): the WHOLE structure ("temporal bone") owns the
    # region before a part of it ("zygomatic process"), then the pointer that ends deepest inside;
    # the others are carved out of it later. Label order must not decide anatomy.
    from engine.grounding import is_subpart
    _dts: dict[int, np.ndarray] = {}

    def tip_depth(i, k) -> float:
        lds = leaders_by_label.get(i) or []
        if not lds or k >= len(comps):
            return 0.0
        if k not in _dts:
            _dts[k] = cv2.distanceTransform(comps[k].mask.astype(np.uint8), cv2.DIST_L2, 3)
        x = int(min(w - 1, max(0, round(lds[0].end[0]))))
        y = int(min(h - 1, max(0, round(lds[0].end[1]))))
        return float(_dts[k][y, x])
    pairs = [(sc, i, pi, meth, is_subpart(clean_name(labels[i].text)),
              tip_depth(i, targets[i][0]) if meth == "leader" else 0.0) for sc, i, pi, meth in pairs]
    pairs.sort(key=lambda t: (-round(t[0]), t[4], -t[5], t[1]))
    pairs = [t[:4] for t in pairs]
    taken_l, taken_p = set(), set()
    method_names = {"leader": "leader", "on": "inside", "near": "nearest"}
    for sc, i, pi, meth in pairs:
        if i in taken_l or pi in taken_p:
            continue
        nm = clean_name(labels[i].text)
        if not nm:
            continue
        taken_l.add(i)
        taken_p.add(pi)
        parts[pi].name = nm
        parts[pi].label_text = clean_text(labels[i].text) or labels[i].text
        parts[pi].match_method = method_names.get(meth, "nearest")
        labels[i].part_id = parts[pi].id
    # losers whose leader ends among clutter dots name the clutter (e.g. ribosomes)
    pending_named: list = []
    for i in part_label_idx:
        if i in taken_l:
            continue
        lds = leaders_by_label.get(i, [])
        if i in clutter_labels or any(clutter_near(Ld.end[0], Ld.end[1], 25) for Ld in lds):
            nm = clean_name(labels[i].text)
            if not nm:
                continue
            pending_named.append((i, nm, lds))   # TS-01: decided after the leftover-label step

    n_dots = len(dots)
    if n_dots >= 3:
        dropped.insert(0, f"{n_dots} small dots (too small to feel)" if n_dots <= 400
                       else "lots of tiny specks (too small to feel)")

    # leftover labels (e.g. "Heart wall" whose leader ends on a big region another rule skipped):
    # give each to the closest still-unnamed part its leader ends on
    for i in part_label_idx:
        if i in taken_l:
            continue
        nm = clean_name(labels[i].text)
        if not nm or any(d.startswith(nm + " (") for d in dropped) or any(i == q[0] for q in pending_named):
            continue
        best, bd = None, 1e9
        for pi, p in enumerate(parts):
            if p.name or pi in taken_p or p.mask is None or not p.mask.any():
                continue
            hh, ww = p.mask.shape[:2]
            dt = cv2.distanceTransform(255 - (p.mask > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3)
            ends = [Ld.end for Ld in leaders_by_label.get(i, [])]
            if not ends:       # only labels that point somewhere (arrow captions have no leader)
                break
            d = min(float(dt[min(hh - 1, max(0, int(e[1]))), min(ww - 1, max(0, int(e[0])))]) for e in ends)
            lim = 30.0
            if d <= lim and d < bd:
                best, bd = pi, d
        if best is not None:
            taken_l.add(i)
            taken_p.add(best)
            parts[best].name = nm
            parts[best].label_text = labels[i].text
            parts[best].match_method = "leader"
            labels[i].part_id = parts[best].id

    # TS-01: a *named* structure pointed at among small clutter is kept when possible: name the
    # unclaimed region its pointer lands on, or grow the drawn blob there (the tactile step
    # enlarges small parts to a feelable size). Only what's left is reported as dropped.
    for i, nm, lds in pending_named:
        if i in taken_l:
            continue
        grown = None
        if mode == "colour" and not plural(labels[i].text):
            for Ld in lds:
                grown = _grow_named_blob(img, de_bg, Ld.end, parts, N)
                if grown is not None:
                    break
        if grown is None and plural(labels[i].text) and dots:
            # many small dots named together ("ribosomes"): the target is the set of dots in the
            # region the pointer lands in - every instance, not one spot at the tip
            ex_, ey_ = lds[0].end if lds else (None, None)
            host = None
            if ex_ is not None:
                for p in parts:
                    if p.mask is not None and p.mask[min(h - 1, max(0, int(ey_))), min(w - 1, max(0, int(ex_)))]:
                        if host is None or p.area < host.area:
                            host = p
            sel = [(x, y, r) for (x, y, r) in dots
                   if host is None or host.mask[min(h - 1, int(y)), min(w - 1, int(x))]]
            if len(sel) >= 3:
                gm = np.zeros((h, w), np.uint8)
                for (x, y, r) in sel:
                    cv2.circle(gm, (int(round(x)), int(round(y))), max(2, int(round(r)) + 1), 255, -1)
                grown = gm
                dropped[:] = [d for d in dropped if not re.match(r"^\d+ small dots|^lots of tiny specks", d)]
        if grown is None:
            dropped.append(f"{nm} (too small to feel)")
            continue
        if isinstance(grown, Part):
            target = grown
        else:
            col = np.median(img[grown > 0], axis=0)
            target = Part(id=f"p{len(parts) + 1}", name="", description="", key="",
                          color=(int(col[0]), int(col[1]), int(col[2])),
                          mask=grown, polygons=mask_polygons(grown, 1.5), bbox=mask_bbox(grown),
                          area=int((grown > 0).sum()), count=1, include=True, kind="region")
            target.count = max(1, cv2.connectedComponents((grown > 0).astype(np.uint8))[0] - 1)
            parts.append(target)
        target.name = nm
        target.label_text = clean_text(labels[i].text) or labels[i].text
        target.match_method = "leader"
        taken_l.add(i)
        taken_p.add(parts.index(target))
        labels[i].part_id = target.id

    # zoom-and-carve (engine/grounding.py): labels whose pointer ends on a region another label
    # already named, or on no region at all, get their own piece traced on a magnified crop
    if mode == "colour" and parts:
        _ground(work, parts, labels, part_label_idx, taken_l, leaders_by_label, clutter_labels,
                dropped, plural, use_ai=use_ai, title=title, progress=progress)
    # plural labels still unplaced whose pointer ends inside a region holding many small dots
    # ("Ribosomes" -> the dots in the cytoplasm): the target is that set of dots
    for i in part_label_idx:
        if i in taken_l or labels[i].part_id is not None or not plural(labels[i].text) or len(dots) < 3:
            continue
        lds = leaders_by_label.get(i) or []
        nm = clean_name(labels[i].text)
        if not lds or not nm or any(p.name == nm for p in parts):
            continue
        ex_, ey_ = int(min(w - 1, max(0, lds[0].end[0]))), int(min(h - 1, max(0, lds[0].end[1])))
        hosts = [p for p in parts if p.mask is not None and p.mask[ey_, ex_]]
        host = min(hosts, key=lambda p: p.area) if hosts else None
        if host is None or nearest_dot(ex_, ey_) > 60:
            continue
        inner = [p for p in parts if p is not host and p.mask is not None and p.area < host.area]
        sel = [(x, y, r) for (x, y, r) in dots if host.mask[min(h - 1, int(y)), min(w - 1, int(x))]
               and not any(p.mask[min(h - 1, int(y)), min(w - 1, int(x))] for p in inner)]
        if len(sel) < 3:
            continue
        gm = np.zeros((h, w), np.uint8)
        for (x, y, r) in sel:
            cv2.circle(gm, (int(round(x)), int(round(y))), max(2, int(round(r)) + 1), 255, -1)
        col = np.median(img[gm > 0], axis=0)
        gp = Part(id=f"p{len(parts) + 1}", name=nm, description="", key="",
                  color=(int(col[0]), int(col[1]), int(col[2])), mask=gm, polygons=mask_polygons(gm, 1.5),
                  bbox=mask_bbox(gm), area=int((gm > 0).sum()),
                  count=max(1, cv2.connectedComponents((gm > 0).astype(np.uint8))[0] - 1), include=True,
                  kind="region", label_text=clean_text(labels[i].text) or labels[i].text, match_method="leader")
        parts.append(gp)
        labels[i].part_id = gp.id
        taken_l.add(i)
        dropped[:] = [d for d in dropped if not re.match(r"^\d+ small dots|^lots of tiny specks", d)]
    if parts:
        _refine_grounding(work, parts, labels, part_label_idx, taken_l, leaders_by_label, dropped, plural)

    # unlabelled parts: copies of a named shape, page background, or the whole object -----
    ink_clean = to_mask(ink & ~(tmask > 0) & ~(lmask > 0))
    desc_override: dict[int, str] = {}
    parts = _resolve_unnamed(parts, ink_clean, title, dropped, desc_override, N)
    # renumber after merges (area order, ids p1..), keep label links
    order_p = sorted(range(len(parts)), key=lambda i: -parts[i].area)
    remap = {}
    new_parts = []
    new_over = {}
    for n_, i in enumerate(order_p):
        p = parts[i]
        remap[p.id] = f"p{n_ + 1}"
        if id(p) in desc_override:
            new_over[n_] = desc_override[id(p)]
        new_parts.append(p)
    for n_, p in enumerate(new_parts):
        p.id = f"p{n_ + 1}"
    parts = new_parts
    for lb in labels:
        if lb.part_id is not None:
            lb.part_id = remap.get(lb.part_id)

    unnamed = [p for p in parts if not p.name]
    for p in unnamed:
        p.name = f"part {p.id[1:]}"
    if unnamed:
        warnings.append("1 part has no label — rename it in review" if len(unnamed) == 1
                        else f"{len(unnamed)} parts have no label — rename them in review")

    # 12. linework ------------------------------------------------------------------------
    linework = _linework(ink, tmask, lmask, [p.mask for p in parts], thin_m, mode, N)
    # process arrows ("Evaporation" beside the arrow from the sea to the cloud) are learning targets
    try:
        linework = _process_paths(img, ink & ~(tmask > 0), labels, part_label_idx, leaders_by_label,
                                  arrow_labels, parts, linework, dropped)
    except Exception:  # pragma: no cover - never lose the analysis over an optional target type
        pass

    # title & type ------------------------------------------------------------------------
    if not title:
        title = "diagram"
    diagram_type = "diagram"
    if _is_chart(parts, ink_clean):
        diagram_type = "chart"
    elif mode == "lineart":
        diagram_type = "lineart"

    # 13. offline descriptions -------------------------------------------------------------
    _describe(parts)
    for n_, d in new_over.items():
        parts[n_].description = d
    if diagram_type == "chart":
        _describe_chart(parts, labels, img, ink_clean)

    if not parts and ts["good"] < 2:
        # nothing segmented as a diagram (e.g. one dark silhouette, a logo): try the main-object path
        alt = _analyze_photo(img_raw, [replace(lb) for lb in labels], use_ai=use_ai, progress=progress,
                             t0=t0, rotation=rotation, feats=feats, text_boxes_ok=not page_flattened)
        if alt.parts and alt.input_kind == "photo_object":
            alt.notes.append("No labelled parts were found, so the outline of the main shape was kept.")
            alt.timings["analyze"] = round(time.perf_counter() - t0, 3)
            debug["fallback"] = "photo_object"
            return alt, debug
    if not parts:
        warnings.insert(0, "No parts found — try a clearer, flatter photo")

    # labels that look like part names but name nothing (F18): show them to the teacher
    named_texts = {clean_name(p.label_text or "") for p in parts} | {p.name for p in parts}
    unmatched: list[str] = []
    notes: list[str] = []
    arrow_caps = []
    for i in part_label_idx:
        lb = labels[i]
        nm = clean_name(lb.text)
        if lb.part_id is not None or not nm or nm in named_texts:
            continue
        if any(d.startswith(nm + " (") for d in dropped):
            continue                                   # "ribosomes (too small to feel)"
        if i in arrow_labels:
            arrow_caps.append(nm)
            continue
        if any(ch.isdigit() for ch in nm) or (main_noun(title) and main_noun(title) in nm):
            continue                                   # figure numbers, captions repeating the title
        if nm not in unmatched:
            unmatched.append(nm)
    if arrow_caps:
        notes.append("Labels on arrows/lines kept as print only: " + ", ".join(arrow_caps[:8]))
    if rotation:
        notes.append(f"The picture was turned {rotation}° clockwise so its text reads upright.")

    analysis = Analysis(
        width=w, height=h, image=img, title=title, diagram_type=diagram_type, parts=parts, labels=labels,
        linework=linework, dropped=dropped, warnings=warnings, timings={}, ai_used=False,
        page_flattened=page_flattened, input_kind="chart" if diagram_type == "chart" else "diagram",
        unmatched_labels=unmatched, rotation=rotation, notes=notes)
    try:
        analysis.trace = _trace(debug, labels, parts)
    except Exception:  # pragma: no cover - the inspector view is a nicety; never lose the analysis
        analysis.trace = {}
    if debug.get("ocr_failed"):
        analysis.trace = {**(analysis.trace or {}), "ocr_failed": debug["ocr_failed"]}
    _sanitize_names(analysis)

    # 14. AI enrichment -------------------------------------------------------------------
    if use_ai and parts and _ai_ready():
        _progress(progress, "ai")
        try:
            enr = ai.enrich(img, analysis)
        except Exception:
            enr = None
        if enr is None:
            analysis.warnings.append(_ai_error_warning())
        else:
            _apply_ai(analysis, enr)

    analysis.timings["analyze"] = round(time.perf_counter() - t0, 3)
    return analysis, debug


_PROCESS_RE = re.compile(r"^(?:[a-z]+(?:ation|ition|ption|ction|ssion|sion|ing)|run ?off|flow|current|"
                         r"transpiration|evaporation|condensation|precipitation|infiltration|percolation)$")


def _process_paths(img, stroke_ink, labels, part_label_idx, leaders_by_label, arrow_labels, parts, linework,
                   dropped) -> np.ndarray:
    """Labels that caption a drawn ARROW (a process: evaporation, condensation, collection) name the
    arrow as a path target with its direction, instead of being dropped as print-only captions or
    pinned to the region the word happens to sit on. Returns linework without the claimed arrows."""
    h, w = img.shape[:2]
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick = cv2.morphologyEx(stroke_ink.astype(np.uint8), cv2.MORPH_OPEN, k3)   # drop 1-2 px outlines
    if not thick.any():
        return linework
    n, cc, st, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    dt = cv2.distanceTransform(thick, cv2.DIST_L2, 3)
    used: set[int] = set()
    by_id = {p.id: p for p in parts}
    named = [p for p in parts if p.name and not p.name.startswith("part ")]
    for i in part_label_idx:
        lb = labels[i]
        nm = clean_name(lb.text)
        if not nm:
            continue
        if not _PROCESS_RE.match(nm):
            continue              # "Lungs" beside its pointer arrow names the lungs, not the arrow
        lh = max(8, lb.bbox[3] - lb.bbox[1])
        reach = max(60, 3 * lh)
        x0, y0, x1, y1 = lb.bbox
        X0, Y0, X1, Y1 = max(0, x0 - reach), max(0, y0 - reach), min(w, x1 + reach), min(h, y1 + reach)
        sub = cc[Y0:Y1, X0:X1]
        best = None
        for c in np.unique(sub):
            if c == 0 or c in used:
                continue
            bw, bh, area = st[c, cv2.CC_STAT_WIDTH], st[c, cv2.CC_STAT_HEIGHT], st[c, cv2.CC_STAT_AREA]
            length = float(np.hypot(bw, bh))
            if length < 3 * lh or area > 0.02 * h * w:
                continue
            mean_w = area / max(1.0, length)
            if mean_w > 14:                       # a filled shape, not a stroke
                continue
            ys, xs = np.nonzero(sub == c)
            dx = np.maximum(0, np.maximum(x0 - (xs + X0), (xs + X0) - x1))
            dy = np.maximum(0, np.maximum(y0 - (ys + Y0), (ys + Y0) - y1))
            d = float(np.hypot(dx, dy).min())
            if d <= reach and (best is None or d < best[0]):
                best = (d, int(c))
        if best is None:
            continue
        c = best[1]
        comp = (cc == c)
        # the arrow must not just be some part's outline
        if any(p.mask is not None and ((p.mask > 0) & comp).sum() >= 0.9 * comp.sum()
               and getattr(p, "target_type", "region") == "region" and p.area < 4 * comp.sum() for p in parts):
            continue
        used.add(c)
        mask = cv2.dilate(comp.astype(np.uint8) * 255, k3, iterations=1)
        # direction: of the two far ends, the arrowhead is the wider one
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        hull = cv2.convexHull(pts).reshape(-1, 2)
        if len(hull) >= 2:
            dd = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)
            a_i, b_i = np.unravel_index(int(np.argmax(dd)), dd.shape)
            ends = [hull[a_i], hull[b_i]]
        else:
            ends = [pts[0], pts[-1]]

        def width_at(e):
            ex, ey = int(e[0]), int(e[1])
            win = dt[max(0, ey - 14):ey + 15, max(0, ex - 14):ex + 15]
            return float(win.max()) if win.size else 0.0
        wa, wb = width_at(ends[0]), width_at(ends[1])
        flow = ""
        if abs(wa - wb) >= 0.8:
            tail, head = (ends[1], ends[0]) if wa > wb else (ends[0], ends[1])

            def near_name(pt):
                bestq, bd = None, 90.0
                for q in named:
                    if q.mask is None or getattr(q, "target_type", "region") != "region":
                        continue
                    qd = cv2.distanceTransform(255 - (q.mask > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3)
                    v = float(qd[min(h - 1, int(pt[1])), min(w - 1, int(pt[0]))])
                    if v < bd:
                        bestq, bd = q, v
                return bestq.name if bestq is not None else None
            fa, fb = near_name(tail), near_name(head)
            if fa and fb and fa != fb:
                flow = f"from the {fa} to the {fb}"
            elif fb:
                flow = f"towards the {fb}"
            elif fa:
                flow = f"away from the {fa}"
        # the word may have been pinned on the region it sits on / a cloud its arrow points at
        if lb.part_id is not None and lb.part_id in by_id:
            old = by_id[lb.part_id]
            if old.name == nm and old.match_method in ("inside", "leader", "nearest"):
                old.name = f"part {old.id[1:]}"
                old.label_text = None
                old.match_method = "none"
                old.include = False
                dropped.append(f"{nm}: the shape under this arrow caption was switched off (switch on in Fix parts if needed)")
        col = np.median(img[comp], axis=0)
        p = Part(id=f"a{len(parts) + 1}", name=nm, description="", key="",
                 color=(int(col[0]), int(col[1]), int(col[2])), mask=mask,
                 polygons=mask_polygons(mask, 1.5), bbox=mask_bbox(mask), area=int((mask > 0).sum()), count=1,
                 include=True, kind="line", label_text=clean_text(lb.text) or lb.text,
                 match_method="arrow", target_type="path", extent="full", flow=flow)
        parts.append(p)
        lb.part_id = p.id
        if linework is not None and linework.shape == mask.shape:
            linework = np.where(cv2.dilate(mask, k3, iterations=2) > 0, 0, linework).astype(linework.dtype)
    return linework


_ARROW_CHARS = "<>→←↑↓»«"


def _trace(debug: dict, labels: list[Label], parts: list[Part]) -> dict:
    """What the pipeline saw, for the inspector view: pointer lines (simplified) and step counts."""
    leaders = []
    for L in debug.get("leaders", [])[:60]:
        pts = np.asarray(L.get("points") or [], np.float32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            pts = cv2.approxPolyDP(pts, 1.5, False)
        pts = pts.reshape(-1, 2)
        leaders.append({"label": L.get("label", ""), "points": [[int(x), int(y)] for x, y in pts[:24]]})
    good = [lb for lb in labels if lb.role in ("part_label", "title", "other")]
    return {"leaders": leaders, "steps": {
        "read": len(good),
        "repaired": sum(1 for lb in labels if lb.raw),
        "pointers": len(leaders),
        "carved": sum(1 for p in parts if p.match_method == "carved"),
        "ai_located": sum(1 for p in parts if p.match_method == "ai"),
        "shapes": len(parts),
    }}


def _ground(work: np.ndarray, parts: list[Part], labels: list[Label], part_label_idx, taken_l: set,
            leaders_by_label: dict, clutter_labels: set, dropped: list, plural, *, use_ai: bool = False,
            title: str = "", progress=None) -> None:
    from engine import grounding

    h, w = work.shape[:2]

    def tip(i):
        lds = leaders_by_label.get(i) or []
        if not lds:
            return None
        return _nudge_tip(lds[0], parts, w, h)

    seeds: list[grounding.Seed] = []
    seen: set[str] = set()
    named_now = {p.name for p in parts if p.name}
    for i in part_label_idx:
        if i in taken_l or (i in clutter_labels and plural(labels[i].text)):
            continue                     # "ribosomes" pointing at dots: the dots are too small to feel
        nm = clean_name(labels[i].text)
        if plural(labels[i].text) and f"{nm} (too small to feel)" in dropped:
            continue
        t = tip(i)
        if not nm or t is None or nm in seen or nm in named_now or name_problem(nm):
            continue
        seen.add(nm)
        seeds.append(grounding.Seed(i, nm, clean_text(labels[i].text) or labels[i].text, t[0], t[1]))
    if not seeds:
        return
    owners: dict[int, grounding.Seed] = {}
    for pi, p in enumerate(parts):
        if not p.name or p.mask is None:
            continue
        li = next((i for i, lb in enumerate(labels) if lb.part_id == p.id), None)
        t = tip(li) if li is not None else None
        if t is None or not p.mask[t[1], t[0]]:
            dt = cv2.distanceTransform((p.mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape)
            t = (int(xx), int(yy))
        owners[pi] = grounding.Seed(li if li is not None else -1, p.name, p.label_text or p.name, t[0], t[1])

    def on_named(s, part):
        labels[s.label_idx].part_id = part.id
        taken_l.add(s.label_idx)
        dropped[:] = [d for d in dropped if d != f"{s.name} (too small to feel)"]

    grounding.ground_labels(work, parts, seeds, owners, on_named,
                            minor=lambda nm: bool(_LANDMARK_RE.search(nm)))

    # still unplaced (pointer lines lost in a painting): ask the image model where each one is
    rest = [sd for sd in seeds if sd.label_idx not in taken_l]
    if not (rest and use_ai and _ai_ready()) or len(parts) >= grounding.MAX_TOTAL_PARTS:
        return
    _progress(progress, "ai")
    found = ai.locate_structures(work, [sd.name for sd in rest], title) or {}
    N = h * w
    blocked = np.zeros((h, w), bool)
    for p in parts:
        if p.mask is not None and p.area <= grounding.CONTAINER_FRAC * N and p.name and not p.name.startswith("part "):
            blocked |= p.mask > 0
    for sd in rest:
        loc = found.get(sd.name)
        if loc is None or len(parts) >= grounding.MAX_TOTAL_PARTS:
            continue
        pt = grounding.Seed(sd.label_idx, sd.name, sd.text, *loc["point"])
        m = grounding.carve_in_box(work, pt, loc["box"], blocked)
        if m is None:
            continue
        mb = (m > 0) & ~blocked
        if mb.sum() < 150:
            continue
        m = (mb * 255).astype(np.uint8)
        for q in parts:
            if q.mask is not None and ((q.mask > 0) & mb).any():
                rest_m = (q.mask > 0) & ~mb
                if rest_m.sum() >= 40:
                    grounding._set_mask(q, (rest_m * 255).astype(np.uint8))
        np_ = grounding._new_part(work, m, pt, len(parts) + 1)
        np_.match_method = "ai"
        parts.append(np_)
        blocked |= mb
        on_named(pt, np_)


def _nudge_tip(Ld, parts: list[Part], w: int, h: int, reach: int = 15) -> tuple[int, int]:
    """F01: pointer lines often stop on a structure's drawn outline, a few pixels short of its
    coloured region. Step forward along the pointer's own direction until the tip is inside a
    region (at most `reach` px); otherwise keep the drawn end."""
    ex, ey = float(Ld.end[0]), float(Ld.end[1])
    clamp = lambda x, y: (int(min(w - 1, max(0, round(x)))), int(min(h - 1, max(0, round(y)))))
    hd = getattr(Ld, "heading", None)
    if hd is None or getattr(Ld, "dot_end", False):
        return clamp(ex, ey)
    for step in range(0, reach + 1):
        x, y = clamp(ex + hd[0] * step, ey + hd[1] * step)
        if any(p.mask is not None and p.mask[y, x] for p in parts):
            return x, y
    return clamp(ex, ey)


_LANDMARK_RE = re.compile(r"\b(suture|fossa|foramen|meatus|notch|tubercle|tuberosity|protuberance|pterion|"
                          r"apex|sulcus|groove|crest|condyle|fissure|junction|node|point|hilum|hilus)\b")
# what a landmark-type name actually is (handover P0): a boundary path, a point, or a small region
_BOUNDARY_RE = re.compile(r"\b(suture|fissure|sulcus|groove|junction)\b")
_POINT_RE = re.compile(r"\b(pterion|apex|point|bregma|lambda|nasion|inion|asterion)\b")
MAX_LANDMARKS = 8
LANDMARK_MAX_FRAC = 0.015    # a region named like a landmark but bigger than this is a mismatch
LANDMARK_EXTRA_PARTS = 8     # landmark spots may exceed the region cap by this many


def _refine_grounding(work: np.ndarray, parts: list[Part], labels: list[Label], part_label_idx,
                      taken_l: set, leaders_by_label: dict, dropped: list, plural) -> None:
    """F01: after naming and carving, (1) take unexplained distant pieces out of named parts, and
    (2) mark landmark labels (sutures, fossae, tubercles ...) whose pointer lands on the drawing
    but on no traceable region with a small spot at the pointer tip, instead of leaving the target
    silently missing. Markers are tagged match_method="landmark" so review can ask for a check."""
    from engine import grounding
    grounding.ORIGIN.clear()
    h, w = work.shape[:2]
    N = h * w
    tips: dict[str, list[tuple[int, int]]] = {}
    for i, lb in enumerate(labels):
        if lb.part_id is None:
            continue
        for Ld in leaders_by_label.get(i, []):
            tips.setdefault(lb.part_id, []).append(
                (int(min(w - 1, max(0, round(Ld.end[0])))), int(min(h - 1, max(0, round(Ld.end[1]))))))
    # a "part" lying along a traced pointer line is the line itself, not a structure: drop it and
    # release its label (the digestive "ascending colon" and heart "septum" were pointer lines)
    lines = np.zeros((h, w), np.uint8)
    for lds in leaders_by_label.values():
        for Ld in lds:
            pts = np.asarray(Ld.points, np.int32).reshape(-1, 1, 2)
            if len(pts) >= 2:
                cv2.polylines(lines, [pts], False, 255, 7)
    if lines.any():
        on_line = lines > 0
        for p in list(parts):
            if p.mask is None or p.match_method in ("teacher", "ai"):
                continue
            pm = p.mask > 0
            a_ = int(pm.sum())
            if a_ == 0:
                continue
            if (pm & on_line).sum() >= 0.7 * a_:
                parts.remove(p)
                for i, lb in enumerate(labels):
                    if lb.part_id == p.id:
                        lb.part_id = None
                        taken_l.discard(i)
                tips.pop(p.id, None)

    def release(ids):
        for i, lb in enumerate(labels):      # names released because their pointer lands elsewhere
            if lb.part_id in ids:
                lb.part_id = None
                taken_l.discard(i)
        for pid in ids:
            tips.pop(pid, None)

    def name_by_tips() -> bool:
        """A leftover pointer that lands on an unnamed piece names it (the jaw freed from
        "nasal bone"). Tips often stop on the drawn outline, so a small tolerance is used."""
        taken_names = {p.name for p in parts if p.name and not p.name.startswith("part ")}
        cands = []                                   # (fraction of the tip window on the piece, ...)
        for i in part_label_idx:
            if i in taken_l or labels[i].part_id is not None:
                continue
            nm = clean_name(labels[i].text)
            lds = leaders_by_label.get(i) or []
            if not nm or nm in taken_names or not lds or name_problem(nm) or _LANDMARK_RE.search(nm):
                continue
            x = int(min(w - 1, max(0, round(lds[0].end[0]))))
            y = int(min(h - 1, max(0, round(lds[0].end[1]))))
            for qi, q in enumerate(parts):
                if q.mask is None or (q.name and not q.name.startswith("part ")):
                    continue
                win = q.mask[max(0, y - 5):y + 6, max(0, x - 5):x + 6]
                f = float((win > 0).mean()) if win.size else 0.0
                if f >= 0.2:
                    cands.append((f, i, qi, nm, x, y))
        renamed = False
        used_q: set[int] = set()
        for f, i, qi, nm, x, y in sorted(cands, key=lambda c: -c[0]):   # best-placed tips first
            if qi in used_q or i in taken_l or nm in taken_names:
                continue
            q = parts[qi]
            q.name, q.label_text, q.match_method = nm, clean_text(labels[i].text) or labels[i].text, "leader"
            labels[i].part_id = q.id
            taken_l.add(i)
            taken_names.add(nm)
            used_q.add(qi)
            tips[q.id] = [(x, y)]
            renamed = True
        return renamed

    for round_ in range(3):
        try:
            added, ungrounded = grounding.prune_far_fragments(parts, tips, plural, N, img=work)
        except Exception:  # pragma: no cover - never lose the analysis over a refinement
            added, ungrounded = [], []
        if round_ == 0:
            # a suture / fossa / notch is a line or a spot: a big filled region under that name is
            # a leftover that inherited the label, not the structure (the "coronal suture" region)
            for p in parts:
                if not (p.id in tips and p.name and _LANDMARK_RE.search(p.name)
                        and p.match_method not in ("teacher", "ai", "landmark")):
                    continue
                at_tip = any(p.mask[max(0, y - 4):y + 5, max(0, x - 4):x + 5].any() for x, y in tips[p.id])
                if p.area > LANDMARK_MAX_FRAC * N or not at_tip:
                    # a landmark must be where its pointer ends; otherwise mark that spot instead
                    p.name, p.label_text, p.match_method = "", None, "none"
                    ungrounded.append(p.id)
        release(ungrounded)
        try:
            grounding.split_unnamed_pieces(parts, N)
        except Exception:  # pragma: no cover
            pass
        if not name_by_tips() and not added and not ungrounded:
            break
    try:
        left_out = grounding.regroup_unclaimed(parts)
    except Exception:  # pragma: no cover
        grounding.ORIGIN.clear()
        left_out = 0
    try:
        grounding.recover_siblings(work, parts, plural)
    except Exception:  # pragma: no cover - never lose the analysis over a refinement
        pass
    if left_out and any(not p.include for p in parts):
        dropped.append(f"{left_out} unlabelled {'piece' if left_out == 1 else 'pieces'} separated from named parts "
                       "(switched off; switch on in Fix parts if needed)")
    # landmarks
    names = {p.name for p in parts if p.name}
    occupied = np.zeros((h, w), bool)
    for p in parts:
        if p.mask is not None:
            occupied |= p.mask > 0
    r = max(6, int(round(0.011 * float(np.hypot(h, w)))))
    made = 0
    for i in part_label_idx:
        if made >= MAX_LANDMARKS or len(parts) >= grounding.MAX_TOTAL_PARTS + LANDMARK_EXTRA_PARTS:
            break
        if i in taken_l or labels[i].part_id is not None:
            continue
        nm = clean_name(labels[i].text)
        lds = leaders_by_label.get(i) or []
        if not nm or nm in names or not lds or name_problem(nm) or not _LANDMARK_RE.search(nm):
            continue
        x = int(min(w - 1, max(0, round(lds[0].end[0]))))
        y = int(min(h - 1, max(0, round(lds[0].end[1]))))
        disk = np.zeros((h, w), np.uint8)
        cv2.circle(disk, (x, y), r, 255, -1)
        db = disk > 0
        if occupied[db].mean() < 0.3:
            continue                                  # the pointer ends on paper, not on the drawing
        # the target keeps its true type; parent structures are never cut to make room for it
        ttype, extent, mask = "landmark", "unresolved", disk
        if _BOUNDARY_RE.search(nm):
            ttype = "boundary"
            try:
                traced = grounding.trace_boundary(parts, x, y, reach=2 * r, min_len=6 * r)
            except Exception:  # pragma: no cover
                traced = None
            if traced is not None:
                mask, extent = traced, "full"
        elif _POINT_RE.search(nm):
            extent = "full"                           # a point landmark is fully located by its tip
        mb = mask > 0
        col = np.median(work[mb], axis=0)
        mp = Part(id=f"m{len(parts) + 1}", name=nm, description="", key="",
                  color=(int(col[0]), int(col[1]), int(col[2])), mask=mask,
                  polygons=mask_polygons(mask, 1.5), bbox=mask_bbox(mask), area=int(mb.sum()), count=1,
                  include=True, kind="line" if ttype == "boundary" and extent == "full" else "region",
                  label_text=clean_text(labels[i].text) or labels[i].text,
                  match_method="landmark", target_type=ttype, extent=extent)
        parts.append(mp)
        labels[i].part_id = mp.id
        taken_l.add(i)
        names.add(nm)
        dropped[:] = [d for d in dropped if d != f"{nm} (too small to feel)"]
        made += 1
    # a suture that was matched to a small region at its pointer tip is still a boundary: trace it
    for p in parts:
        if (getattr(p, "target_type", "region") != "region" or not p.name or not _BOUNDARY_RE.search(p.name)
                or p.match_method in ("teacher", "ai") or p.id not in tips):
            continue
        x, y = tips[p.id][0]
        others = [q for q in parts if q is not p]
        try:
            traced = grounding.trace_boundary(others, x, y, reach=2 * r, min_len=6 * r)
        except Exception:  # pragma: no cover
            traced = None
        p.target_type = "boundary"
        if traced is not None:
            grounding._set_mask(p, traced)
            p.kind, p.extent = "line", "full"
        else:
            p.extent = "unresolved"


def _outline_blob(img: np.ndarray, x: int, y: int, N: int, R: int = 130):
    """A shape drawn as an outline in pale pencil (brain in a body diagram): close the edges in a
    window around the pointer tip, fill the holes and take the filled blob the tip touches.
    Blobs that run into the window border are background/clutter, not a closed shape."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x - R), max(0, y - R), min(W, x + R), min(H, y + R)
    sub = cv2.cvtColor(np.ascontiguousarray(img[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
    ed = cv2.Canny(cv2.GaussianBlur(sub, (0, 0), 1.5), 20, 60)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ed = cv2.morphologyEx(ed, cv2.MORPH_CLOSE, k9)
    f = fill_holes(ed)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, k9)          # cut thin lines (rulings, arrows)
    n, cc, st, _ = cv2.connectedComponentsWithStats(f)
    best = None
    for i in range(1, n):
        bx, by, bw, bh, a = st[i]
        if a < max(800, int(0.0008 * N)) or a > 0.03 * N:
            continue
        if bx <= 1 or by <= 1 or bx + bw >= (x1 - x0) - 1 or by + bh >= (y1 - y0) - 1:
            continue                                     # touches the window: not a closed shape
        m = (cc == i)
        d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)[y - y0, x - x0]
        if d <= 15 and (best is None or d < best[0]):
            best = (d, m)
    if best is None:
        return None
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1][best[1]] = 255
    return (int((full > 0).sum()), full)


def _grow_named_blob(img: np.ndarray, de_bg: np.ndarray, end, parts: list, N: int):
    """Flood-fill the coloured blob a pointer ends on (TS-01). Returns a uint8 mask or None.

    Accepts only a compact, non-background blob that isn't already (mostly) another part and
    is big enough to be a drawn structure rather than a speck."""
    h, w = de_bg.shape
    x, y = int(round(end[0])), int(round(end[1]))
    x, y = min(w - 1, max(0, x)), min(h - 1, max(0, y))
    # 1. the pointer lands on a region we already found but no label claimed: name it
    for p in parts:
        if p.mask is None or p.name or getattr(p, "match_method", "none") not in ("none", None):
            continue
        win = p.mask[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        if win.size and (win > 0).mean() >= 0.5 and p.area <= 0.06 * N:
            return p
    # 2. otherwise grow the drawn blob there. Pencil/crayon texture defeats a strict flood fill,
    #    so smooth first and use a floating range.
    smooth = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), 2.0)
    best = None
    for dy in (0, -3, 3):
        for dx in (0, -3, 3):
            px, py = min(w - 1, max(0, x + dx)), min(h - 1, max(0, y + dy))
            if de_bg[py, px] <= 6:
                continue                                   # paper, not a drawn shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
            cv2.floodFill(smooth, mask, (px, py), (0, 0, 0), (6, 6, 6), (6, 6, 6), flags)
            m = (mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255   # border of the mask is preset to 1
            area = int((m > 0).sum())
            if area < max(40, int(0.00015 * N)) or area > 0.03 * N:
                continue
            if best is None or area > best[0]:
                best = (area, m)
    if best is None:
        best = _outline_blob(img, x, y, N)                 # pale pencil: use the drawn outline
    if best is None:
        return None
    m = cv2.morphologyEx(best[1], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mb = m > 0
    area = int(mb.sum())
    for p in parts:
        if p.mask is None:
            continue
        pm = p.mask > 0
        ov = int((pm & mb).sum())
        if ov <= 0.3 * area:
            continue
        unnamed = p.name.startswith("part ") or not p.name
        if unnamed and ov >= 0.6 * int(pm.sum()):
            return p                                       # it IS that unnamed part: caller names it
        if not unnamed and int(pm.sum()) < 2.5 * area:
            return None                                    # mostly another named part: leave it
        # carve the named blob out of the bigger region (a heart inside the body outline)
        rest = (pm & ~mb).astype(np.uint8) * 255
        if int((rest > 0).sum()) < 40:
            return None
        p.mask = rest
        p.polygons = mask_polygons(rest, 1.5)
        p.bbox = mask_bbox(rest)
        p.area = int((rest > 0).sum())
    return m


def _split_colon_headings(labels: list[Label]) -> None:
    """UX06: "Chloroplast: site of photosynthesis" names the chloroplast; the definition after the
    colon is explanation, not a name. "Plasmodesmata:" keeps its heading."""
    for lb in labels:
        t = lb.text
        if ":" in t:
            head = t.split(":", 1)[0].strip()
            if len(re.findall(r"[A-Za-z]", head)) >= 3:
                lb.text = head
                lb.colon_heading = True          # the lines under it are its definition, not more name


def _ends_with_connector(text: str) -> bool:
    """F01: "Greater wing of" / "sphenoid bone" - a line ending in a connector continues below."""
    words = text.strip().split()
    return bool(words) and words[-1].lower() in {"of", "and", "the", "to", "for", "with"}


def _join_label_lines(labels: list[Label], leaders_by_label: dict) -> None:
    """TS-02: a label printed on two lines ("Small" / "Intestine") is one name.

    A second line joins the line above it when it has no pointer line of its own, no arrow
    character, sits directly below (small gap, overlapping horizontally) and the joined name stays
    short. A separately labelled part stacked underneath (e.g. "Pancreas" under "Stomach") has its
    own pointer line, so it is not swallowed."""
    idx = [i for i, lb in enumerate(labels) if lb.role == "part_label"]
    for j in sorted(idx, key=lambda k: labels[k].bbox[1]):
        b = labels[j]
        if b.role != "part_label" or any(c in b.text for c in _ARROW_CHARS):
            continue
        bh_ = max(1, b.bbox[3] - b.bbox[1])
        # a stub "leader" (a notebook ruling touching the word) is not a pointer of its own
        b_has_leader = any(np.hypot(L.points[0][0] - L.end[0], L.points[0][1] - L.end[1]) > 2.5 * bh_
                           for L in leaders_by_label.get(j, []))
        if not re.fullmatch(r"[A-Za-z][A-Za-z' -]*", b.text.strip()) or len(b.text.split()) > 2:
            continue
        bx0, by0, bx1, by1 = b.bbox
        best = None
        for i in idx:
            a = labels[i]
            if i == j or a.role != "part_label":
                continue
            ax0, ay0, ax1, ay1 = a.bbox
            h = max(1, min(ay1 - ay0, by1 - by0))
            gap = by0 - ay1
            ov = min(ax1, bx1) - max(ax0, bx0)
            if not (-0.5 * h <= gap <= 0.6 * h) or ov < 0.3 * min(ax1 - ax0, bx1 - bx0):
                continue
            if a.text.rstrip().endswith(":") or getattr(a, "colon_heading", False) \
                    or getattr(b, "colon_heading", False) \
                    or len((a.text + " " + b.text).split()) > (6 if _ends_with_connector(a.text) else 4):
                continue
            if is_term(a.text) and is_term(b.text) and not is_term(f"{a.text} {b.text}") \
                    and not (b.text.strip()[:1].islower() and a.text.strip()[:1].isupper()):
                continue                                  # "Pancreas" + "Large intestine": two names
            a_arrow = any(c in a.text for c in _ARROW_CHARS)
            if not (leaders_by_label.get(i) or a_arrow or (b.text.strip()[:1].islower() and a.text.strip()[:1].isupper())):
                continue                                  # only continue a label that points somewhere
            continuation = b.text.strip()[:1].islower() and a.text.strip()[:1].isupper() and len(a.text.split()) == 1
            if b_has_leader and not (a_arrow and gap <= 0.3 * h) and not continuation:
                continue                                  # b points somewhere itself: a separate label
            if best is None or gap < best[0]:
                best = (gap, i)
        if best is None:
            continue
        a = labels[best[1]]
        a.text = f"{a.text.rstrip()} {b.text.strip()}"
        a.bbox = (min(a.bbox[0], bx0), min(a.bbox[1], by0), max(a.bbox[2], bx1), max(a.bbox[3], by1))
        a.confidence = min(a.confidence, b.confidence)
        b.role = "other"                                  # consumed: not a separate part / unmatched label


def _sanitize_names(analysis: Analysis) -> None:
    """Never show a student OCR junk: sentence fragments, brand text or profanity-like tokens
    become "part N" (the teacher is asked to name them); a junk title becomes "diagram"."""
    from engine.vision_utils import looks_like_structure, name_problem, title_problem
    changed = 0
    for p in analysis.parts:
        bad = name_problem(p.name) or (
            not p.name.startswith("part ") and getattr(p, "match_method", "") not in ("ai", "teacher")
            and not looks_like_structure(p.name) and "caption")
        if not bad:
            continue
        old = p.name
        p.name = f"part {p.id[1:]}" if p.id[1:].isdigit() else "part"
        if getattr(p, "match_method", None) not in (None, "ai", "teacher"):
            p.match_method = "none"
        if hasattr(p, "label_text"):
            p.label_text = None
        if p.description:
            p.description = re.sub(re.escape(old), p.name, p.description, flags=re.I)
        changed += 1
    # UX06: two shapes with the same name are one structure drawn in pieces ("liver" twice) —
    # one part, so students never hear about two livers.
    by_name: dict[str, list[str]] = {}
    for p in analysis.parts:
        if p.include and p.name and not p.name.startswith("part "):
            by_name.setdefault(p.name.lower(), []).append(p.id)
    for ids in by_name.values():
        if len(ids) > 1:
            try:
                merge_parts(analysis, ids)
            except Exception:  # pragma: no cover - never fail analysis over this
                pass
    if title_problem(analysis.title):
        analysis.title = "diagram"
    analysis.unmatched_labels = [u for u in (analysis.unmatched_labels or []) if not name_problem(u)]
    if changed:
        unnamed = sum(1 for p in analysis.parts if p.include and p.name.startswith("part"))
        analysis.warnings = [w_ for w_ in analysis.warnings if "have no label" not in w_ and "has no label" not in w_]
        if unnamed:
            analysis.warnings.append("1 part has no label — rename it in review" if unnamed == 1
                                     else f"{unnamed} parts have no label — rename them in review")


def _children(q: Part, parts: list[Part]) -> list[Part]:
    qm = q.mask > 0
    return [p for p in parts if p is not q and p.area < q.area
            and int((qm & (p.mask > 0)).sum()) >= 0.9 * p.area]


def _reads_as_container(q: Part, parts: list[Part]) -> bool:
    """A container you can feel: its own visible area is a good share of its outline."""
    vis = q.mask > 0
    for c in _children(q, parts):
        vis &= ~(c.mask > 0)
    return int(vis.sum()) >= 0.35 * max(1, q.area)


def _join_names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _merge_into(dst: Part, src: Part, add_count: bool) -> None:
    m = cv2.bitwise_or(dst.mask, src.mask)
    dst.mask = m
    dst.polygons = mask_polygons(m, 1.5)
    dst.bbox = mask_bbox(m)
    dst.area = int((m > 0).sum())
    if add_count:
        dst.count += src.count
    else:
        dst.count = max(1, len(dst.polygons)) if not add_count and dst.count == 1 else dst.count


def _main_contour(mask: np.ndarray):
    cs = outer_contours(mask, approx=False)
    return max(cs, key=cv2.contourArea) if cs else None


def _resolve_unnamed(parts: list[Part], ink_clean: np.ndarray, title: str, dropped: list,
                     desc_override: dict, n_total: int) -> list[Part]:
    h, w = ink_clean.shape
    named = [p for p in parts if p.name]
    alive = list(parts)

    # (a1) small unnamed components hugging a named part are its shading (cloud shadow).
    # Pieces already cut away from a named part (switched off by _refine_grounding) stay apart.
    for u in [p for p in alive if not p.name and p.include]:
        n, cc, st, _ = cv2.connectedComponentsWithStats((u.mask > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        moved = False
        for i in range(1, n):
            comp = (cc == i).astype(np.uint8) * 255
            ring = cv2.dilate(comp, disk(3)) > 0
            host = None
            for q in named:
                if (ring & (q.mask > 0)).any() and st[i, cv2.CC_STAT_AREA] < 0.5 * q.area:
                    host = q
                    break
            if host is not None and n > 2:
                _merge_into(host, Part(id="", name="", description="", key="", color=(0, 0, 0), mask=comp,
                                       polygons=[], bbox=(0, 0, 0, 0), area=0, count=0), add_count=False)
                moved = True
            else:
                keep[i] = True
        if moved:
            m = keep[cc].astype(np.uint8) * 255
            if not m.any():
                alive.remove(u)
                continue
            u.mask, u.polygons, u.bbox = m, mask_polygons(m, 1.5), mask_bbox(m)
            u.area, u.count = int((m > 0).sum()), int(keep.sum())

    # (a2) an unnamed shape that looks like a named one is another one of those
    for u in [p for p in alive if not p.name and p.include]:
        cu = _main_contour(u.mask)
        if cu is None:
            continue
        ua = u.area / max(1, u.count)
        best, best_d = None, 0.15
        for q in named:
            cq = _main_contour(q.mask)
            if cq is None:
                continue
            ratio = ua / max(1.0, q.area / max(1, q.count))
            if not (0.4 <= ratio <= 2.5):
                continue
            d = cv2.matchShapes(cu, cq, cv2.CONTOURS_MATCH_I1, 0)
            if d < best_d:
                best, best_d = q, d
        if best is not None:
            _merge_into(best, u, add_count=True)
            alive.remove(u)

    # frame lines (a drawn panel border) and the image border; solid ink shapes are not frame lines
    solid = cv2.dilate(cv2.morphologyEx(ink_clean, cv2.MORPH_OPEN, disk(SOLID_INK_RADIUS)), disk(3))
    thin_ink = cv2.bitwise_and(ink_clean, cv2.bitwise_not(solid))
    hor = cv2.morphologyEx(thin_ink, cv2.MORPH_OPEN, np.ones((1, max(10, int(0.4 * w))), np.uint8))
    ver = cv2.morphologyEx(thin_ink, cv2.MORPH_OPEN, np.ones((max(10, int(0.3 * h)), 1), np.uint8))
    frame = cv2.bitwise_or(hor, ver) > 0

    def holds_named(u):
        # smaller named parts set in / against u, around its extent (also a globe with a wedge cut away)
        x0, y0, x1, y1 = u.bbox
        mx, my = 0.3 * (x1 - x0), 0.3 * (y1 - y0)
        near_u = cv2.dilate(u.mask, disk(6)) > 0
        out = []
        for q in alive:
            if not q.name or q is u or q.area >= 0.5 * u.area:
                continue
            cx, cy = (q.bbox[0] + q.bbox[2]) / 2, (q.bbox[1] + q.bbox[3]) / 2
            if x0 - mx <= cx <= x1 + mx and y0 - my <= cy <= y1 + my and (near_u & (q.mask > 0)).any():
                out.append(q)
        return out

    for u in [p for p in alive if not p.name]:
        kids = holds_named(u)
        if len(kids) >= 2:
            # (c) the whole object: name it after the title - only when it really surrounds the
            # named parts (inside its outline) and the title names ONE object ("cell", not "organs");
            # never to give a leftover piece next to two organs a name it does not have
            noun = main_noun(title)
            cu = _main_contour(u.mask)
            hull = np.zeros_like(u.mask)
            if cu is not None:
                cv2.drawContours(hull, [cv2.convexHull(cu)], -1, 255, -1)
            inside = [q for q in kids if ((q.mask > 0) & (hull > 0)).sum() >= 0.7 * max(1, q.area)]
            if noun and len(inside) >= 2 and not re.search(r"[a-z]{3,}s$", noun):
                u.name = noun
                kids = sorted(kids, key=lambda q: -max(q.bbox[2] - q.bbox[0], q.bbox[3] - q.bbox[1]))  # outside in
                desc_override[id(u)] = (f"{capitalise(noun)}. Contains the "
                                        f"{_join_names([q.name for q in kids][:6])}.")
            continue
        # (b) background: touches the image border or a drawn frame
        x0, y0, x1, y1 = u.bbox
        # the page/paper hugs the picture edge along a real stretch; a bar or shape that merely
        # touches the edge (or a chart's axes frame) is content, not background
        um = u.mask > 0
        edge_px = int(um[:3, :].sum() + um[-3:, :].sum() + um[:, :3].sum() + um[:, -3:].sum()) / 3.0
        perim = 2.0 * ((x1 - x0) + (y1 - y0))
        on_border = (x0 <= 2 or y0 <= 2 or x1 >= w - 2 or y1 >= h - 2) and edge_px >= 0.15 * max(perim, 1.0)
        on_frame = bool(frame.any() and (cv2.dilate(u.mask, disk(8)) > 0)[frame].any()) and u.area >= 0.10 * n_total
        dark_shape = max(u.color) < 70 and u.area < 0.4 * n_total     # a black silhouette, not the page
        if (on_border or on_frame) and u.area >= 0.02 * n_total and not dark_shape:
            u.name = "background"
            u.include = False
            u.description = "Background."
            dropped.append("large unlabelled background (switch it back on in review if needed)")
    return alive


def _describe(parts: list[Part]) -> None:
    for p in parts:
        s = capitalise(p.name) + "."
        # No "There are N." — a count of separate shapes is not a verified count of objects
        # (a liver drawn in two pieces is still one liver). The teacher UI shows shapes separately.
        cont = None
        m = p.mask > 0
        for q in parts:
            if q is p or q.area <= p.area:
                continue
            if int((m & (q.mask > 0)).sum()) >= 0.9 * p.area and (cont is None or q.area < cont.area):
                cont = q
        if getattr(p, "target_type", "region") in ("path", "boundary"):
            cont = None                          # a line crossing a region is not "inside" it
            if getattr(p, "flow", "") and getattr(p, "match_method", "") == "arrow":
                s += f" An arrow {p.flow}."
        if cont is not None and not cont.name.startswith("part ") and _reads_as_container(cont, parts):
            s += f" Inside the {cont.name}."
        p.description = s


def _apply_ai(analysis: Analysis, enr) -> None:
    by_id = {p.id: p for p in analysis.parts}
    for item in enr.parts:
        p = by_id.get(item.get("id"))
        if p is None:
            continue
        if item.get("name") and item["name"] != p.name:
            p.name = item["name"]
            if p.match_method == "none":
                p.match_method = "ai"            # the name came from the AI, not a label
        if item.get("description") and item["description"] != p.description:
            p.description = item["description"]
            p.description_source = "ai"
        if item.get("drop"):
            p.include = False
            analysis.dropped.append(f"{p.name} (decoration)")
    if enr.title:
        analysis.title = enr.title
    if enr.diagram_type:
        analysis.diagram_type = enr.diagram_type
    analysis.ai_used = True
    analysis.warnings = [w_ for w_ in analysis.warnings if "have no label" not in w_ and "has no label" not in w_]
    still = [p for p in analysis.parts if p.name.startswith("part ")]
    if still:
        analysis.warnings.append("1 part has no label — rename it in review" if len(still) == 1
                                 else f"{len(still)} parts have no label — rename them in review")


# ============================================================================ teacher corrections


def _part_by_id(analysis: Analysis, pid: str) -> Part:
    for p in analysis.parts:
        if p.id == pid:
            return p
    raise ValueError(f"Unknown part id: {str(pid)[:20]}")


def _next_part_id(analysis: Analysis) -> int:
    nums = [int(p.id[1:]) for p in analysis.parts if p.id[:1] == "p" and p.id[1:].isdigit()]
    return max(nums, default=0) + 1


def split_part(analysis: Analysis, part_id: str) -> list[str]:
    """Split a part made of several separate pieces into one part per piece (F01).

    The piece nearest the part's own label keeps the id, name and description; every other piece
    becomes a new part named from the nearest unmatched label (analysis.unmatched_labels) close to
    it, else "part N". Returns the ids of all resulting parts (original id first).
    Raises ValueError if the part is a single piece."""
    p = _part_by_id(analysis, part_id)
    m = (p.mask > 0).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_px = max(20, int(0.01 * max(1, int(m.sum()))))
    comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]
    if len(comps) < 2:
        raise ValueError("This part is one piece — there is nothing to split")
    masks = {i: (cc == i).astype(np.uint8) * 255 for i in comps}
    dts = {i: cv2.distanceTransform(255 - masks[i], cv2.DIST_L2, 3) for i in comps}

    def dist(i, box):
        x0, y0, x1, y1 = box
        sub = dts[i][max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
        return float(sub.min()) if sub.size else float("inf")

    own = [lb for lb in analysis.labels if lb.part_id == part_id]
    keep = min(comps, key=lambda i: min((dist(i, lb.bbox) for lb in own), default=0.0)) if own else \
        max(comps, key=lambda i: st[i, cv2.CC_STAT_AREA])
    # unmatched labels that can name the new pieces
    pool = []
    for lb in analysis.labels:
        nm = clean_name(lb.text)
        if lb.part_id is None and nm and nm in (analysis.unmatched_labels or []):
            pool.append(lb)
    ids = [part_id]
    img = analysis.image
    nxt = _next_part_id(analysis)
    others = sorted([i for i in comps if i != keep], key=lambda i: -st[i, cv2.CC_STAT_AREA])
    _set_part_mask(p, masks[keep])
    p.count = 1
    new_parts = []
    for i in others:
        best, bd = None, SPLIT_LABEL_PX
        for lb in pool:
            d = dist(i, lb.bbox)
            if d < bd:
                best, bd = lb, d
        pid = f"p{nxt}"
        nxt += 1
        if best is not None:
            pool.remove(best)
            nm = clean_name(best.text)
            best.part_id = pid
            analysis.unmatched_labels = [u for u in analysis.unmatched_labels if u != nm]
            q = _new_part(pid, nm, capitalise(nm) + ".", masks[i], img if img is not None else
                          np.full(masks[i].shape + (3,), 255, np.uint8),
                          match_method="nearest", label_text=best.text)
        else:
            nm = f"part {pid[1:]}"
            q = _new_part(pid, nm, capitalise(nm) + ".", masks[i], img if img is not None else
                          np.full(masks[i].shape + (3,), 255, np.uint8), match_method="none")
        q.include = p.include
        q.kind = p.kind
        new_parts.append(q)
        ids.append(pid)
    pos = analysis.parts.index(p)
    analysis.parts[pos + 1:pos + 1] = new_parts
    return ids


def merge_parts(analysis: Analysis, ids: list[str]) -> str:
    """Merge parts into the first id (its name, description and key stay). Returns that id.
    Labels of the merged parts now point at it. Raises ValueError for < 2 ids or unknown ids."""
    ids = list(dict.fromkeys(ids or []))
    if len(ids) < 2:
        raise ValueError("Pick at least two parts to merge")
    parts = [_part_by_id(analysis, pid) for pid in ids]
    dst = parts[0]
    m = (dst.mask > 0).astype(np.uint8) * 255
    count = dst.count
    for q in parts[1:]:
        m = cv2.bitwise_or(m, (q.mask > 0).astype(np.uint8) * 255)
        count += q.count
    _set_part_mask(dst, m)
    dst.count = max(1, count)
    gone = set(ids[1:])
    analysis.parts = [p for p in analysis.parts if p.id not in gone]
    for lb in analysis.labels:
        if lb.part_id in gone:
            lb.part_id = dst.id
    return dst.id


def _set_part_mask(p: Part, mask: np.ndarray) -> None:
    p.mask = mask
    p.polygons = mask_polygons(mask, 1.5)
    p.bbox = mask_bbox(mask)
    p.area = int((mask > 0).sum())


def _describe_chart(parts: list[Part], labels: list[Label], img: np.ndarray, ink_clean: np.ndarray) -> None:
    """Bar values from numbers printed on / above the bars, position and extremes."""
    bars = [p for p in parts if p.count == 1 and p.include and p.polygons
            and cv2.contourArea(np.array(p.polygons[0], np.int32)) >= 0.85 * max(1, (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))]
    if len(bars) < 2:
        return
    values: dict[str, str] = {}
    for lb in labels:
        t = lb.text.strip().replace(",", "")
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", t):
            continue
        cx = (lb.bbox[0] + lb.bbox[2]) / 2
        for b in bars:
            x0, y0, x1, y1 = b.bbox
            if not (x0 <= cx <= x1):
                continue
            above = 0 <= y0 - lb.bbox[3] <= 40
            inside = y0 <= lb.bbox[1] and lb.bbox[3] <= y1
            if (above or inside) and b.id not in values:
                values[b.id] = t
                lb.part_id = b.id
                break
    # small numbers (single digits) are often missed by text detection: read the strip above each bar
    h, w = ink_clean.shape
    for b in bars:
        if b.id in values:
            continue
        x0, y0, x1, y1 = b.bbox
        ry0 = max(0, y0 - 42)
        reg = ink_clean[ry0:max(ry0, y0 - 3), x0:x1] > 0
        g = cv2.cvtColor(img[ry0:max(ry0, y0 - 3), x0:x1], cv2.COLOR_RGB2GRAY)
        reg = reg | (g < 200)
        if reg.sum() < 8:
            continue
        ys, xs = np.nonzero(reg)
        cx0, cx1 = x0 + xs.min() - 6, x0 + xs.max() + 7
        cy0, cy1 = ry0 + ys.min() - 6, ry0 + ys.max() + 7
        if not (x0 <= (cx0 + cx1) / 2 <= x1):
            continue
        best_t, best_s = "", 0.0
        for pad in (0, 4, 10):   # tiny crops are fragile: try a few margins, keep the surest number
            crop = img[max(0, cy0 - pad // 2):min(h, cy1 + pad // 2), max(0, cx0 - pad):min(w, cx1 + pad)]
            txt, sc = read_crop(crop)
            txt = txt.replace(",", "").strip()
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", txt) and sc > best_s:
                best_t, best_s = txt, sc
        if best_s >= 0.3:
            values[b.id] = best_t
    axis_title = _y_axis_title(img, ink_clean, bars)
    order = sorted(bars, key=lambda b: b.bbox[0])
    if all(b.id in values for b in bars):
        heights = {b.id: float(values[b.id].rstrip("%")) for b in bars}
    else:
        heights = {b.id: float(b.bbox[3] - b.bbox[1]) for b in bars}
    tallest = max(bars, key=lambda b: heights[b.id]).id
    shortest = min(bars, key=lambda b: heights[b.id]).id
    for i, b in enumerate(order):
        s_ = capitalise(b.name) + "."
        if b.id in values:
            s_ += f" {values[b.id]}" + (f" — {axis_title}." if axis_title else ".")
        s_ += f" Bar {i + 1} of {len(order)}"
        s_ += ", the tallest." if b.id == tallest else (", the shortest." if b.id == shortest else ".")
        b.description = s_


def _y_axis_title(img: np.ndarray, ink_clean: np.ndarray, bars: list[Part]) -> str:
    """Read a vertical y-axis title left of the bars (rotated text the main OCR pass skips)."""
    h, w = ink_clean.shape
    left = min(b.bbox[0] for b in bars)
    ver = cv2.morphologyEx(ink_clean, cv2.MORPH_OPEN, np.ones((max(10, int(0.25 * h)), 1), np.uint8))
    xs = np.nonzero(ver[:, :left].any(0))[0]
    if len(xs) == 0:
        return ""
    x_axis = int(xs.max())
    ys = np.nonzero(ver[:, x_axis])[0]
    strip = img[int(ys.min()):int(ys.max()) + 1, : max(0, x_axis - 4)]
    if strip.shape[1] < 12:
        return ""
    rot = np.ascontiguousarray(np.rot90(strip, k=-1))   # bottom-to-top text becomes left-to-right
    best = ""
    for lb in run_ocr(rot):
        t = clean_name(lb.text)
        if sum(ch.isalpha() for ch in t) >= 4 and len(t) > len(best) and len(t.split()) <= 5:
            best = t
    return best


def _is_chart(parts: list[Part], linework: np.ndarray) -> bool:
    rects = []
    for p in parts:
        for poly in p.polygons:
            pa = np.array(poly, np.int32)
            if len(pa) < 4:
                continue
            x, y, bw, bh = cv2.boundingRect(pa)
            a = cv2.contourArea(pa)
            if bw * bh > 0 and a >= 0.88 * bw * bh and min(bw, bh) >= 6:
                rects.append((x, y, x + bw, y + bh))
    if len(rects) < 3:
        return False

    def shared(vals, tol=8):
        vals = sorted(vals)
        best = 1
        for i in range(len(vals)):
            best = max(best, sum(1 for v in vals if abs(v - vals[i]) <= tol))
        return best

    baseline = max(shared([r[3] for r in rects]), shared([r[0] for r in rects]))
    if baseline < 3:
        return False
    # TS-10: bars carry data, so their lengths differ. Equal boxes in a grid are a diagram,
    # not a chart (otherwise we'd invent "Bar 5 of 6, the tallest").
    hs = [r[3] - r[1] for r in rects]
    ws = [r[2] - r[0] for r in rects]
    if max(hs) < 1.15 * min(hs) and max(ws) < 1.15 * min(ws):
        return False
    h, w = linework.shape
    if not linework.any():
        return False
    # bridge small breaks (ticks / leader removal) before looking for long axis lines
    hl = cv2.morphologyEx(linework, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8))
    vl = cv2.morphologyEx(linework, cv2.MORPH_CLOSE, np.ones((25, 1), np.uint8))
    hor = cv2.morphologyEx(hl, cv2.MORPH_OPEN, np.ones((1, max(10, int(0.3 * w))), np.uint8))
    ver = cv2.morphologyEx(vl, cv2.MORPH_OPEN, np.ones((max(10, int(0.25 * h)), 1), np.uint8))
    return bool(hor.any() and ver.any())
