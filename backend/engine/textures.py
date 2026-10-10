"""Tactile fill textures at physical scale.

OWNER: Tactile team.

All patterns are anchored to the canvas origin (deterministic, and two regions with the
same texture line up).  Sizes follow common tactile-graphics practice:
  dots  : 1.5 mm dots, 4 mm pitch, staggered rows
  diag  : 1 mm lines at 45 deg, 3.5 mm pitch
  horiz / vert : 1 mm lines, 3.5 mm pitch
  cross : both diagonals, 5 mm pitch
  grid  : horizontal + vertical, 5 mm pitch
  solid : fully raised
  none  : nothing
"""
from __future__ import annotations

import cv2
import numpy as np

TEXTURE_ORDER = ["dots", "diag", "horiz", "vert", "cross", "grid"]
ALL_TEXTURES = ["none", "solid"] + TEXTURE_ORDER

LINE_MM = 1.0
LINE_PITCH_MM = 3.5
CROSS_PITCH_MM = 5.0
DOT_MM = 1.5
DOT_PITCH_MM = 4.0


def _stripes(shape, px_per_mm, pitch_mm, direction, x0=0, y0=0):
    """Binary stripes (0/255) of LINE_MM width. direction: 'h','v','d1' (/), 'd2' (\\).
    (x0, y0) = canvas position of the top-left pixel (keeps the global phase for ROIs)."""
    h, w = shape
    pitch = pitch_mm * px_per_mm
    width = LINE_MM * px_per_mm
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
    if direction == "h":
        t = yy + 0.5
    elif direction == "v":
        t = xx + 0.5
    elif direction == "d1":
        t = (xx + yy + 1.0) / np.sqrt(2.0)
    else:
        t = (xx - yy) / np.sqrt(2.0)
    m = np.mod(t, pitch) < width
    return m.astype(np.uint8) * 255


def dot_centres(shape, px_per_mm) -> np.ndarray:
    """(N, 2) float array of staggered dot centres (x, y) covering `shape`."""
    h, w = shape
    p = DOT_PITCH_MM * px_per_mm
    ys = np.arange(p / 2, h + p, p * np.sqrt(3) / 2)   # hexagonal-ish stagger
    pts = []
    for r, y in enumerate(ys):
        off = p / 2 if r % 2 == 0 else p
        xs = np.arange(off, w + p, p)
        pts.append(np.stack([xs, np.full_like(xs, y)], 1))
    return np.concatenate(pts, 0) if pts else np.zeros((0, 2))


def _draw_dots(shape, px_per_mm, centres) -> np.ndarray:
    img = np.zeros(shape, np.uint8)
    sh = 4
    rad = int(round((DOT_MM / 2 * px_per_mm - 0.35) * (1 << sh)))   # cv2 adds ~0.7 px
    for x, y in centres:
        cv2.circle(img, (int(round((x - 0.5) * 16)), int(round((y - 0.5) * 16))), rad, 255, -1, cv2.LINE_8, sh)
    return img


def fill(name: str, shape: tuple[int, int], px_per_mm: float, x0: int = 0, y0: int = 0) -> np.ndarray:
    """uint8 (0/255) pattern for texture `name` of size `shape`; (x0, y0) = canvas offset."""
    shape = (int(shape[0]), int(shape[1]))
    if name == "none":
        return np.zeros(shape, np.uint8)
    if name == "solid":
        return np.full(shape, 255, np.uint8)
    if name == "dots":
        c = dot_centres((shape[0] + y0, shape[1] + x0), px_per_mm) - np.array([x0, y0])
        return _draw_dots(shape, px_per_mm, c)
    if name == "diag":
        return _stripes(shape, px_per_mm, LINE_PITCH_MM, "d1", x0, y0)
    if name == "horiz":
        return _stripes(shape, px_per_mm, LINE_PITCH_MM, "h", x0, y0)
    if name == "vert":
        return _stripes(shape, px_per_mm, LINE_PITCH_MM, "v", x0, y0)
    if name == "cross":
        return _stripes(shape, px_per_mm, CROSS_PITCH_MM, "d1", x0, y0) | _stripes(shape, px_per_mm, CROSS_PITCH_MM, "d2", x0, y0)
    if name == "grid":
        return _stripes(shape, px_per_mm, CROSS_PITCH_MM, "h", x0, y0) | _stripes(shape, px_per_mm, CROSS_PITCH_MM, "v", x0, y0)
    raise ValueError(f"unknown texture {name!r}")


def fill_region(name: str, region: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Texture clipped to `region` (0/255) so that no fragment is too small to feel.

    * dots: only whole dots whose disc lies fully inside the region
    * line textures: clipped, then slivers shorter than ~3 mm removed
    * solid: the region itself
    """
    if name == "none" or not region.any():
        return np.zeros_like(region)
    if name == "solid":
        return region.copy()
    ys, xs = np.nonzero(region)
    pad = int(np.ceil(DOT_PITCH_MM * px_per_mm)) + 2
    y0, y1 = max(0, ys.min() - pad), min(region.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(region.shape[1], xs.max() + pad + 1)
    roi = region[y0:y1, x0:x1]
    out = np.zeros_like(region)
    if name == "dots":
        r = int(np.ceil(DOT_MM / 2 * px_per_mm))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        inner = cv2.erode(roi, k)
        c = dot_centres(region.shape, px_per_mm)
        sel = (c[:, 0] >= x0) & (c[:, 0] < x1 - 1) & (c[:, 1] >= y0) & (c[:, 1] < y1 - 1)
        c = c[sel]
        if len(c):
            ix = np.clip(np.round(c[:, 0]).astype(int) - x0, 0, x1 - x0 - 1)
            iy = np.clip(np.round(c[:, 1]).astype(int) - y0, 0, y1 - y0 - 1)
            c = c[inner[iy, ix] > 0]
        out = _draw_dots(region.shape, px_per_mm, c)
        return out & region
    # line textures: generate on ROI with the global phase
    pat = fill(name, (y1 - y0, x1 - x0), px_per_mm, x0, y0)
    tex = pat & roi
    # a line clipped along its length can become thinner than LINE_MM: open it away
    r = max(1, int(round(LINE_MM * px_per_mm)))
    tex = cv2.morphologyEx(tex, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    # drop slivers: components shorter than 3 mm
    n, cc, st, _ = cv2.connectedComponentsWithStats(tex, connectivity=8)
    if n > 1:
        min_len = 3.0 * px_per_mm
        min_area = LINE_MM * px_per_mm * 3.0 * px_per_mm * 0.8
        ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
        keep = (ext >= min_len) & (st[:, cv2.CC_STAT_AREA] >= min_area)
        keep[0] = False
        tex = np.where(keep[cc], 255, 0).astype(np.uint8)
    out[y0:y1, x0:x1] = tex
    return out
