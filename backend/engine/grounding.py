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

from dataclasses import dataclass, replace
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
        if area < min_area or sup < EDGE_SUPPORT:
            continue
        # F01: the piece may be the bigger share of the host (a jaw under a "nasal bone" pointer),
        # but it must never take the spot the host's own label points at
        if owner is not None:
            if _hits(full, owner.x, owner.y, 3):
                continue
        elif area > 0.9 * host.area:
            continue
        out.append((s, full))
    return out


def _hits(mask: np.ndarray, x: int, y: int, r: int) -> bool:
    h, w = mask.shape[:2]
    win = mask[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
    return bool(win.size and (win > 0).mean() >= 0.5)


def _comp_at(cc: np.ndarray, x: int, y: int, r: int = 3) -> int:
    """Connected-component label under (x, y), using the most common label in a small window."""
    h, w = cc.shape[:2]
    win = cc[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)].ravel()
    win = win[win > 0]
    if win.size == 0:
        return 0
    return int(np.bincount(win).argmax())


def split_components(host: Part, seeds: list[Seed], owner: Optional[Seed], min_area: int,
                     minor: Optional[Callable[[str], bool]] = None
                     ) -> tuple[list[tuple[Seed, np.ndarray]], list[Seed]]:
    """F01: a region made of physically separate pieces (the nose bone and the jaw share a colour)
    is split by piece first: a pointer that lands on a piece the host's own pointer does not touch
    gets that whole piece. Returns (pieces, seeds still sharing a piece with another label)."""
    hm = (host.mask > 0).astype(np.uint8)
    n, cc = cv2.connectedComponents(hm, connectivity=8)
    if n <= 2:
        return [], list(seeds)
    oc = _comp_at(cc, owner.x, owner.y) if owner is not None else -1
    by_comp: dict[int, list[Seed]] = {}
    rest: list[Seed] = []
    for s in seeds:
        c = _comp_at(cc, s.x, s.y)
        if c <= 0 or c == oc:
            rest.append(s)
        else:
            by_comp.setdefault(c, []).append(s)
    out = []
    for c, ss in by_comp.items():
        m = cc == c
        if len(ss) > 1 and minor is not None:
            major = [s_ for s_ in ss if not minor(s_.name)]
            if len(major) == 1:            # "mandible" takes the jaw; "mental protuberance" is a spot on it
                rest.extend(s_ for s_ in ss if s_ is not major[0])
                ss = major
        if len(ss) > 1 or int(m.sum()) < min_area:
            rest.extend(ss)                      # two labels on one piece: leave it to the watershed
            continue
        if owner is None and int(m.sum()) > 0.9 * host.area:
            rest.extend(ss)
            continue
        out.append((ss[0], (m * 255).astype(np.uint8)))
    return out, rest


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
                  on_named: Callable[[Seed, Part], None],
                  minor: Optional[Callable[[str], bool]] = None) -> list[Part]:
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
        if hi is None:     # F01: tips often stop on the drawn outline, a pixel or two off the region
            best = 0.0
            for pi, p in enumerate(parts):
                if p.mask is None:
                    continue
                win = p.mask[max(0, s.y - 5):s.y + 6, max(0, s.x - 5):s.x + 6]
                f = float((win > 0).mean()) if win.size else 0.0
                if f >= 0.25 and f > best:
                    hi, best = pi, f
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
        min_area = max(150, int(MIN_PIECE_FRAC * N))
        comp_pieces, ss = split_components(host, ss, owner, min_area, minor)
        hm = host.mask > 0
        ws_pieces = []
        if ss:
            ws_host = host
            if comp_pieces:
                left = hm & ~np.any([m > 0 for _, m in comp_pieces], axis=0)
                lm = (left * 255).astype(np.uint8)
                ws_host = replace(host, mask=lm, bbox=mask_bbox(lm), area=int(left.sum()))
            if ws_host.area > 0:
                ws_pieces = split_host(img, ws_host, ss, owner)
        pieces = comp_pieces + ws_pieces
        for s, m in pieces[:budget]:
            mb = (m > 0) & hm
            if mb.sum() == 0:
                continue
            m = (mb * 255).astype(np.uint8)
            rest = hm & ~mb
            if owner is not None:
                if rest.sum() < 40 or not _hits(rest.astype(np.uint8), owner.x, owner.y, 3):
                    continue                             # never take the host's own pointed-at spot
            elif rest.sum() < 0.1 * host.area:
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


# ---------------------------------------------------------------- F01: far fragments
FAR_KEEP_FRAC = 0.6          # keep a piece whose gap to the anchored piece is < this x anchor size
FAR_KEEP_MAX = 0.10          # ... but never a gap wider than this x image diagonal
FAR_PLURAL_DIST = 0.35       # a repeated structure (lungs, kidneys) may sit this far (x image diagonal)


def prune_far_fragments(parts: list[Part], tips: dict[str, list[tuple[int, int]]], is_plural,
                        N: int) -> tuple[list[Part], list[str]]:
    """F01: a name must not cover unexplained distant pieces. For each part with pointer tips,
    the pieces its pointers land on (or the piece nearest a tip) are anchors; other pieces are kept
    only when close to an anchor, or - for a plural / repeated name - when similar in size.
    Pieces that fail are taken out: big ones become new unnamed parts for the teacher to review,
    specks are dropped. Returns the new parts (already appended)."""
    new: list[Part] = []
    ungrounded: list[str] = []
    min_piece = max(150, int(MIN_PIECE_FRAC * N))
    if not parts:
        return new, ungrounded
    h, w = parts[0].mask.shape[:2]
    diag = float(np.hypot(h, w))
    for p in list(parts):
        t = tips.get(p.id)
        if not t or p.mask is None or not p.include:
            continue
        n, cc, st, cent = cv2.connectedComponentsWithStats((p.mask > 0).astype(np.uint8), connectivity=8)
        if n <= 1:
            continue
        anchors = {c for c in (_comp_at(cc, x, y, 4) for x, y in t) if c > 0}
        if not anchors:
            # nearest piece to any tip
            best, bd = 0, 1e18
            for c in range(1, n):
                ys, xs = np.nonzero(cc == c)
                d = min(float(np.min((xs - x) ** 2 + (ys - y) ** 2)) for x, y in t)
                if d < bd:
                    best, bd = c, d
            if bd > (0.05 * diag) ** 2 and p.match_method in ("ai", "teacher"):
                continue                          # placed by the image model / teacher, not by a pointer
            if bd > (0.05 * diag) ** 2:
                # F01: none of this part is where its pointer lands - the name was carried over to
                # a leftover region. Release it so the label can be placed properly (or reviewed).
                p.name, p.label_text, p.match_method = "", None, "none"
                ungrounded.append(p.id)
                continue
            anchors = {best}
        a_area = max(int(st[c, cv2.CC_STAT_AREA]) for c in anchors)
        a_boxes = [st[c, :4] for c in anchors]
        a_size = max(float(np.hypot(b[2], b[3])) for b in a_boxes)
        plural = bool(is_plural(p.label_text or p.name))
        anchor_shapes = [_shape_sig(cc == c, int(st[c, cv2.CC_STAT_AREA])) for c in anchors]
        drop = np.zeros((h, w), bool)
        for c in range(1, n):
            if c in anchors:
                continue
            bx, by, bw, bh, area = (int(v) for v in st[c])
            gap = min(_box_gap((bx, by, bw, bh), tuple(int(v) for v in b)) for b in a_boxes)
            if gap <= min(FAR_KEEP_FRAC * a_size, FAR_KEEP_MAX * diag):
                continue
            if plural and area >= 0.3 * a_area and gap <= FAR_PLURAL_DIST * diag:
                continue
            if _looks_like_copy(cc == c, area, anchor_shapes):
                continue                      # a repeated structure (3 mitochondria, one labelled)
            drop |= cc == c
        if not drop.any():
            continue
        keep = (p.mask > 0) & ~drop
        _set_mask(p, (keep * 255).astype(np.uint8))
        p.count = max(1, cv2.connectedComponents(keep.astype(np.uint8), connectivity=8)[0] - 1)
        n2, cc2, st2, _ = cv2.connectedComponentsWithStats(drop.astype(np.uint8), connectivity=8)
        for c in range(1, n2):
            if int(st2[c, cv2.CC_STAT_AREA]) < min_piece:
                continue
            m = ((cc2 == c) * 255).astype(np.uint8)
            q = Part(id=f"f{len(parts) + len(new) + 1}", name="", description="", key="", color=p.color,
                     mask=m, polygons=mask_polygons(m, 1.5), bbox=mask_bbox(m),
                     area=int((m > 0).sum()), count=1, include=True, kind="region", match_method="none")
            ORIGIN[id(q)] = ORIGIN.get(id(p), id(p))
            new.append(q)
    parts.extend(new)
    return new, ungrounded


def _box_gap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0, max(ay, by) - min(ay + ah, by + bh))
    return float(np.hypot(dx, dy))


def split_unnamed_pieces(parts: list[Part], N: int) -> int:
    """F01: an unnamed leftover region made of physically separate pieces is split into one part
    per piece, so each leftover pointer can claim only the piece it lands on. Specks are dropped."""
    min_piece = max(150, int(MIN_PIECE_FRAC * N))
    added = 0
    for p in list(parts):
        if p.mask is None or (p.name and not p.name.startswith("part ")) or not p.include:
            continue
        n, cc, st, _ = cv2.connectedComponentsWithStats((p.mask > 0).astype(np.uint8), connectivity=8)
        if n <= 2:
            continue
        order = sorted(range(1, n), key=lambda c: -int(st[c, cv2.CC_STAT_AREA]))
        keep = order[0]
        _set_mask(p, ((cc == keep) * 255).astype(np.uint8))
        p.count = 1
        for c in order[1:]:
            if int(st[c, cv2.CC_STAT_AREA]) < min_piece:
                continue
            m = ((cc == c) * 255).astype(np.uint8)
            q = Part(id=f"u{len(parts) + 1}", name="", description="", key="", color=p.color,
                     mask=m, polygons=mask_polygons(m, 1.5), bbox=mask_bbox(m),
                     area=int((m > 0).sum()), count=1, include=True, kind="region", match_method="none")
            ORIGIN[id(q)] = ORIGIN.get(id(p), id(p))
            parts.append(q)
            added += 1
    return added


def _shape_sig(m: np.ndarray, area: int):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return area, 0.0, 1.0, None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = float(cv2.contourArea(c) / hull) if hull > 0 else 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    elong = float(max(rw, rh) / max(1.0, min(rw, rh)))
    return area, solidity, elong, c


def _looks_like_copy(m: np.ndarray, area: int, anchors: list) -> bool:
    """Same size (0.4-2.5x) and same kind of shape (solidity, elongation) as an anchored piece."""
    _, sol, el, c = _shape_sig(m, area)
    for a_area, a_sol, a_el, a_c in anchors:
        if 0.4 * a_area <= area <= 2.5 * a_area and abs(sol - a_sol) <= 0.15 \
                and max(el, a_el) / max(1e-6, min(el, a_el)) <= 1.8 \
                and c is not None and a_c is not None \
                and cv2.matchShapes(c, a_c, cv2.CONTOURS_MATCH_I1, 0) <= 0.2:
            return True
    return False


# piece -> id() of the part it was taken from (only for one analysis run; see regroup_unclaimed)
ORIGIN: dict[int, int] = {}


def regroup_unclaimed(parts: list[Part]) -> int:
    """After leftover pointers have claimed their pieces, put the unclaimed pieces back together:
    all unnamed pieces taken from the same region become one part again (into that region itself
    when it is still unnamed), so splitting for precision never floods review with fragments."""
    by_id = {id(p): p for p in parts}
    unnamed = lambda p: not p.name or p.name.startswith("part ")
    left_out = 0
    groups: dict[int, list[Part]] = {}
    for p in parts:
        o = ORIGIN.get(id(p))
        if o is not None and unnamed(p):
            groups.setdefault(o, []).append(p)
    for o, pieces in groups.items():
        host = by_id.get(o)
        target = host if (host is not None and unnamed(host)) else pieces[0]
        m = (target.mask > 0).copy()
        for q in pieces:
            if q is target:
                continue
            m |= q.mask > 0
            parts.remove(q)
        _set_mask(target, (m * 255).astype(np.uint8))
        target.count = max(1, cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1)
        if host is not None and not unnamed(host):
            # cut away from a named part and claimed by no pointer: keep it, but switched off,
            # so review is not flooded with fragments; the teacher can switch it back on
            target.include = False
            left_out += 1
    ORIGIN.clear()
    return left_out


# ---------------------------------------------------------------- assisted outline (teacher tool)
def propose_outline(img: np.ndarray, inside: list[tuple[float, float]], outside: list[tuple[float, float]],
                    box: Optional[tuple[float, float, float, float]] = None,
                    lasso: Optional[list[tuple[float, float]]] = None) -> Optional[np.ndarray]:
    """Trace the structure a teacher roughly marked (taps inside, optional taps outside, an optional
    rough box or loose lasso), using only the picture around the marks - never the label name.
    GrabCut on a magnified crop: inside taps are sure-foreground, outside taps and everything
    beyond the box are sure-background, a lasso's interior is probable foreground.
    Returns a uint8 mask in source coordinates, or None when no clear outline is found."""
    h, w = img.shape[:2]
    N = h * w
    pts = list(inside) + list(lasso or [])
    if not pts and box is None:
        return None
    if inside and box is None and not lasso:
        # taps only: trace the closed region around the first tap (widening window watershed);
        # outside taps compete as other seeds. Accept it only when the piece is closed inside the
        # window (not cut off by it) and holds every inside tap.
        x, y = (int(round(v)) for v in inside[0])
        others = [Seed(-1, "", "", int(round(ox)), int(round(oy))) for ox, oy in outside]
        for frac in (0.1, 0.2, 0.35, 0.5):
            R = int(np.clip(frac * max(h, w), 40, max(h, w)))
            m = _carve_window(img, Seed(-1, "", "", x, y), others, np.zeros((h, w), bool), R)
            if m is None:
                continue
            bx0, by0, bx1, by1 = mask_bbox(m)
            wx0, wy0, wx1, wy1 = max(0, x - R), max(0, y - R), min(w, x + R + 1), min(h, y + R + 1)
            cut = (bx0 <= wx0 + 2 and wx0 > 0) or (by0 <= wy0 + 2 and wy0 > 0) or \
                  (bx1 >= wx1 - 3 and wx1 < w) or (by1 >= wy1 - 3 and wy1 < h)
            if cut:
                continue
            if all(m[min(h - 1, int(yy)), min(w - 1, int(xx))] for xx, yy in inside):
                return m
    if box is None:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if lasso and len(lasso) >= 3:
            box = (min(xs), min(ys), max(xs), max(ys))
        else:
            r = 0.08 * max(h, w)
            box = (min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    bw, bh = x1 - x0, y1 - y0
    m = int(0.15 * max(bw, bh)) + 6
    cb = (max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m))
    crop, z = _zoom(img, cb)
    zh, zw = crop.shape[:2]
    gm = np.full((zh, zw), cv2.GC_BGD, np.uint8)
    bx0, by0 = int((x0 - cb[0]) * z), int((y0 - cb[1]) * z)
    bx1, by1 = int((x1 - cb[0]) * z), int((y1 - cb[1]) * z)
    gm[by0:by1, bx0:bx1] = cv2.GC_PR_BGD if (lasso and len(lasso) >= 3) else cv2.GC_PR_FGD
    if lasso and len(lasso) >= 3:
        poly = np.array([[(x - cb[0]) * z, (y - cb[1]) * z] for x, y in lasso], np.int32)
        cv2.fillPoly(gm, [poly], int(cv2.GC_PR_FGD))
    r = max(3, int(0.012 * max(zh, zw)))
    for x, y in outside:
        cv2.circle(gm, (int((x - cb[0]) * z), int((y - cb[1]) * z)), r, int(cv2.GC_BGD), -1)
    for x, y in inside:
        cv2.circle(gm, (int((x - cb[0]) * z), int((y - cb[1]) * z)), r, int(cv2.GC_FGD), -1)
    if not ((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)).any():
        return None
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), gm, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    piece = (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)
    if piece.sum() == 0:
        return None
    full = _to_full(piece, cb, z, (h, w))
    full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # keep the pieces the inside taps touch (or the biggest one)
    n, cc = cv2.connectedComponents((full > 0).astype(np.uint8), connectivity=8)
    keep = {cc[min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))] for x, y in inside} - {0}
    if not keep and n > 1:
        areas = np.bincount(cc.ravel())
        areas[0] = 0
        keep = {int(np.argmax(areas))}
    out = (np.isin(cc, list(keep)) * 255).astype(np.uint8)
    area = int((out > 0).sum())
    if area < max(60, int(0.0002 * N)) or area > 0.6 * N:
        return None
    return out
