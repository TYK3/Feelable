"""Zoom-and-carve: give every label that points at something its own region.

Colour segmentation works on the whole picture at once, so on dense, shaded illustrations
(a sagittal section of the head, a realistic organ painting) neighbouring structures of similar
colour come out as one region and most labels are left without a shape to name. Here each such
label is handled locally, the way a person would do it: go to where its pointer line ends, look
closer, and trace the structure there.

1. Pointer tips that land inside the same region are seeds for a marker-controlled watershed on
   a magnified crop of that region (text and pointer lines already painted out). The region is
   split along its real colour edges, one piece per label.
2. A tip that lands on no region (or on a big container such as the body outline) gets a local
   watershed in a window around it: the tip is one seed, the window border and the other tips
   nearby are the competitors.

Every new piece must be a real drawn structure: big enough, and with its boundary on an actual
edge (mean colour gradient along the cut well above the region's own texture). Pieces that fail
are not added, so a label is never given an invented shape; it stays a teacher issue instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from engine.types import Part
from engine.vision_utils import mask_bbox, mask_polygons

ZOOM_TARGET = 520           # px: a crop is magnified until its long side is about this
MAX_ZOOM = 3.0
CONTAINER_FRAC = 0.22       # a region bigger than this share of the picture is a container
MIN_PIECE_FRAC = 0.0012     # a carved piece smaller than this is a speck, not a structure
EDGE_SUPPORT = 1.35         # boundary gradient / region gradient needed to accept a cut
EDGE_MIN = 6.0              # ... and an absolute floor (Lab units per px, Sobel)
MAX_TOTAL_PARTS = 24


@dataclass
class Seed:
    label_idx: int
    name: str
    text: str
    x: int
    y: int


def _lab_grad(crop_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    g = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        g += cv2.magnitude(gx, gy)
    return g / 3.0


def _zoom(img: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    z = float(np.clip(ZOOM_TARGET / max(1, max(crop.shape[:2])), 1.0, MAX_ZOOM))
    if z > 1.01:
        crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    crop = cv2.bilateralFilter(np.ascontiguousarray(crop), 7, 30, 7)
    return crop, z


def _support(grad: np.ndarray, piece: np.ndarray, region: np.ndarray) -> float:
    """Mean gradient on the piece's boundary inside `region`, relative to the region's texture."""
    k = np.ones((3, 3), np.uint8)
    edge = piece & ~cv2.erode(piece.astype(np.uint8), k).astype(bool)
    edge &= cv2.erode(region.astype(np.uint8), k, iterations=2).astype(bool)
    if edge.sum() < 8:
        return 99.0                                   # the piece is bounded by the region's own outline
    inner = cv2.erode(piece.astype(np.uint8), k, iterations=3).astype(bool)
    base = float(np.median(grad[inner])) if inner.sum() > 20 else float(np.median(grad[region]))
    b = float(grad[edge].mean())
    if b < EDGE_MIN:
        return 0.0
    return b / max(base, 1.0)


def _watershed(crop: np.ndarray, markers: np.ndarray) -> np.ndarray:
    m = markers.astype(np.int32).copy()
    cv2.watershed(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), m)
    return m


def _disk(markers: np.ndarray, x: float, y: float, r: int, val: int) -> None:
    cv2.circle(markers, (int(round(x)), int(round(y))), max(1, r), int(val), -1)


def _to_full(piece_z: np.ndarray, box, z: float, shape) -> np.ndarray:
    x0, y0, x1, y1 = box
    small = cv2.resize(piece_z.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    full = np.zeros(shape, np.uint8)
    full[y0:y1, x0:x1] = small * 255
    full = cv2.morphologyEx(full, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return full


def _largest(mask_u8: np.ndarray, x: int, y: int) -> np.ndarray:
    """Keep the connected piece that contains (or is closest to) the seed."""
    n, cc = cv2.connectedComponents((mask_u8 > 0).astype(np.uint8))
    if n <= 2:
        return mask_u8
    lab_ = cc[min(cc.shape[0] - 1, max(0, y)), min(cc.shape[1] - 1, max(0, x))]
    if lab_ == 0:
        areas = np.bincount(cc.ravel())
        areas[0] = 0
        lab_ = int(np.argmax(areas))
    return ((cc == lab_) * 255).astype(np.uint8)


def _new_part(img: np.ndarray, mask: np.ndarray, seed: Seed, n: int) -> Part:
    col = np.median(img[mask > 0], axis=0)
    return Part(id=f"g{n}", name=seed.name, description="", key="",
                color=(int(col[0]), int(col[1]), int(col[2])),
                mask=mask, polygons=mask_polygons(mask, 1.5), bbox=mask_bbox(mask),
                area=int((mask > 0).sum()), count=1, include=True, kind="region",
                label_text=seed.text, match_method="carved")


def _set_mask(p: Part, m: np.ndarray) -> None:
    p.mask = m
    p.polygons = mask_polygons(m, 1.5)
    p.bbox = mask_bbox(m)
    p.area = int((m > 0).sum())


def split_host(img: np.ndarray, host: Part, seeds: list[Seed], owner: Optional[Seed]) -> list[tuple[Seed, np.ndarray]]:
    """Watershed `host` between `seeds` (+ its current owner). Returns accepted (seed, mask) pieces."""
    h, w = img.shape[:2]
    N = h * w
    hm = host.mask > 0
    x0, y0, x1, y1 = host.bbox
    pad = 8
    box = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad + 1), min(h, y1 + pad + 1))
    crop, z = _zoom(img, box)
    reg = cv2.resize(hm[box[1]:box[3], box[0]:box[2]].astype(np.uint8), (crop.shape[1], crop.shape[0]),
                     interpolation=cv2.INTER_NEAREST).astype(bool)
    markers = np.zeros(crop.shape[:2], np.int32)
    markers[~cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = 1
    all_seeds = ([owner] if owner else []) + seeds
    r = max(2, int(round(2.5 * z)))
    for k, s in enumerate(all_seeds):
        _disk(markers, (s.x - box[0]) * z, (s.y - box[1]) * z, r, k + 2)
    markers[~reg] = 1
    ws = _watershed(crop, markers)
    grad = _lab_grad(crop)
    out = []
    min_area = max(150, int(MIN_PIECE_FRAC * N))
    for k, s in enumerate(all_seeds):
        if owner is not None and k == 0:
            continue
        piece = (ws == k + 2) & reg
        if piece.sum() == 0:
            continue
        sup = _support(grad, piece, reg)
        full = _largest(_to_full(piece, box, z, (h, w)), s.x, s.y)
        area = int((full > 0).sum())
        if area < min_area or area > 0.9 * host.area or sup < EDGE_SUPPORT:
            continue
        out.append((s, full))
    return out


def carve_local(img: np.ndarray, seed: Seed, others: list[Seed], blocked: np.ndarray) -> Optional[np.ndarray]:
    """Trace the structure under one pointer tip: look in a small magnified window first and widen
    it until the structure is closed inside it (a liver needs a bigger window than a gland)."""
    h, w = img.shape[:2]
    for frac in (0.07, 0.14, 0.24):
        R = int(np.clip(frac * max(h, w), 40, 0.5 * max(h, w)))
        m = _carve_window(img, seed, others, blocked, R)
        if m is not None:
            return m
    return None


def _carve_window(img, seed: Seed, others: list[Seed], blocked: np.ndarray, R: int) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    N = h * w
    box = (max(0, seed.x - R), max(0, seed.y - R), min(w, seed.x + R + 1), min(h, seed.y + R + 1))
    crop, z = _zoom(img, box)
    zh, zw = crop.shape[:2]
    markers = np.zeros((zh, zw), np.int32)
    markers[:2, :], markers[-2:, :], markers[:, :2], markers[:, -2:] = 1, 1, 1, 1
    blk = cv2.resize(blocked[box[1]:box[3], box[0]:box[2]].astype(np.uint8), (zw, zh),
                     interpolation=cv2.INTER_NEAREST)
    blk = cv2.erode(blk, np.ones((5, 5), np.uint8)).astype(bool)
    markers[blk] = 1
    r = max(2, int(round(2.5 * z)))
    for k, o in enumerate(others):
        if box[0] <= o.x < box[2] and box[1] <= o.y < box[3]:
            _disk(markers, (o.x - box[0]) * z, (o.y - box[1]) * z, r, 3 + k)
    _disk(markers, (seed.x - box[0]) * z, (seed.y - box[1]) * z, r, 2)
    ws = _watershed(crop, markers)
    piece = ws == 2
    if piece.sum() == 0:
        return None
    # a closed structure: most of it must stay off the window border
    border = np.zeros_like(piece)
    border[:3, :], border[-3:, :], border[:, :3], border[:, -3:] = True, True, True, True
    touch = float((piece & border).sum()) / max(1.0, float(border.sum()))
    if touch > 0.12 or piece.mean() > 0.45:
        return None
    sup = _support(_lab_grad(crop), piece, np.ones_like(piece))
    if sup < EDGE_SUPPORT:
        return None
    full = _largest(_to_full(piece, box, z, (h, w)), seed.x, seed.y)
    area = int((full > 0).sum())
    if area < max(150, int(MIN_PIECE_FRAC * N)):
        return None
    return full


def ground_labels(img: np.ndarray, parts: list[Part], seeds: list[Seed], owners: dict[int, Seed],
                  on_named: Callable[[Seed, Part], None]) -> list[Part]:
    """Carve regions for `seeds` (labels with a pointer but no part). `owners`: part index ->
    the seed of the label that already names it (for splits). Calls `on_named(seed, part)` for each
    new part and returns the new parts (already appended to `parts`, carved out of their hosts)."""
    h, w = img.shape[:2]
    N = h * w
    new: list[Part] = []
    by_host: dict[int, list[Seed]] = {}
    loose: list[Seed] = []
    for s in seeds:
        hi = None
        for pi, p in enumerate(parts):
            if p.mask is None:
                continue
            win = p.mask[max(0, s.y - 2):s.y + 3, max(0, s.x - 2):s.x + 3]
            if win.size and (win > 0).mean() >= 0.5:
                hi = pi
                break
        if hi is None or parts[hi].area > CONTAINER_FRAC * N:
            loose.append(s)
        else:
            by_host.setdefault(hi, []).append(s)
    budget = MAX_TOTAL_PARTS - len(parts)
    # 1. split shared regions
    for hi, ss in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
        if budget <= 0:
            break
        host = parts[hi]
        owner = owners.get(hi)
        if owner is None and len(ss) == 1 and (not host.name or host.name.startswith("part ")):
            host.name, host.label_text, host.match_method = ss[0].name, ss[0].text, "leader"  # plain pointer match
            on_named(ss[0], host)
            continue
        pieces = split_host(img, host, ss, owner)
        hm = host.mask > 0
        for s, m in pieces[:budget]:
            mb = (m > 0) & hm
            if mb.sum() == 0:
                continue
            m = (mb * 255).astype(np.uint8)
            rest = hm & ~mb
            if rest.sum() < 0.1 * host.area:
                break                                    # never swallow the host's own structure
            hm = rest
            p = _new_part(img, m, s, len(parts) + len(new) + 1)
            new.append(p)
            on_named(s, p)
            budget -= 1
        if hm.sum() != (host.mask > 0).sum():
            _set_mask(host, (hm * 255).astype(np.uint8))
    parts.extend(new)
    # 2. pointers into nothing / into a container: trace locally
    blocked = np.zeros((h, w), bool)
    for p in parts:
        if p.mask is not None and p.area <= CONTAINER_FRAC * N and p.name and not p.name.startswith("part "):
            blocked |= p.mask > 0
    for s in loose:
        if budget <= 0:
            break
        others = [o for o in seeds if o is not s] + list(owners.values())
        m = carve_local(img, s, others, blocked)
        if m is None:
            continue
        mb = m > 0
        if (mb & blocked).sum() > 0.3 * mb.sum():
            continue
        mb &= ~blocked
        m = (mb * 255).astype(np.uint8)
        for q in parts:                                   # carve it out of a container / unnamed part
            if q.mask is None:
                continue
            qm = q.mask > 0
            ov = qm & mb
            if ov.any():
                rest = qm & ~mb
                if rest.sum() >= 40:
                    _set_mask(q, (rest * 255).astype(np.uint8))
        p = _new_part(img, m, s, len(parts) + 1)
        parts.append(p)
        new.append(p)
        blocked |= mb
        on_named(s, p)
        budget -= 1
    return new


def carve_in_box(img: np.ndarray, seed: Seed, box: tuple[int, int, int, int],
                 blocked: np.ndarray) -> Optional[np.ndarray]:
    """Segment the structure inside an AI-given box around an AI-given inside point (GrabCut on a
    magnified crop: the box is 'probably the structure', the point is 'surely the structure',
    pixels owned by other named parts and the margin outside the box are 'surely not')."""
    h, w = img.shape[:2]
    N = h * w
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    m = int(0.15 * max(bw, bh)) + 6
    cb = (max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m))
    crop, z = _zoom(img, cb)
    zh, zw = crop.shape[:2]
    gm = np.full((zh, zw), cv2.GC_BGD, np.uint8)
    bx0, by0 = int((x0 - cb[0]) * z), int((y0 - cb[1]) * z)
    bx1, by1 = int((x1 - cb[0]) * z), int((y1 - cb[1]) * z)
    gm[by0:by1, bx0:bx1] = cv2.GC_PR_FGD
    blk = cv2.resize(blocked[cb[1]:cb[3], cb[0]:cb[2]].astype(np.uint8), (zw, zh), interpolation=cv2.INTER_NEAREST)
    gm[cv2.erode(blk, np.ones((3, 3), np.uint8)) > 0] = cv2.GC_BGD
    r = max(3, int(0.08 * min(bx1 - bx0, by1 - by0)))
    cv2.circle(gm, (int((seed.x - cb[0]) * z), int((seed.y - cb[1]) * z)), r, cv2.GC_FGD, -1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), gm, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    piece = (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)
    if piece.sum() == 0:
        return None
    full = _largest(_to_full(piece, cb, z, (h, w)), seed.x, seed.y)
    full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    area = int((full > 0).sum())
    boxa = max(1, bw * bh)
    if area < max(150, int(MIN_PIECE_FRAC * N)) or area < 0.12 * boxa or area > 1.05 * boxa:
        return None
    return full
