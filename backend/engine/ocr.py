"""Offline OCR (RapidOCR / ONNX runtime) -> list[Label] in SOURCE coords.

OWNER: Vision team.
"""
from __future__ import annotations

import difflib
import os
import re
import threading

import cv2
import numpy as np

from engine.types import Label

MIN_CONFIDENCE = 0.5

_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()


def _get_engine():
    """Lazy singleton: model loading takes ~0.5 s, so do it once per process."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _limit_ort_threads()
                from rapidocr_onnxruntime import RapidOCR  # heavy import, keep lazy

                eng = RapidOCR()
                # diagrams are upright: skipping the 0/180 classifier saves ~40 % of OCR time
                eng.use_angle_cls = False
                _ENGINE = eng
    return _ENGINE


def _limit_ort_threads() -> None:
    """On small/throttled hosts set FEELABLE_THREADS=1: ONNX Runtime otherwise starts one
    busy-spinning thread per host core, which burns a tiny CPU quota (e.g. 0.1 CPU) for nothing."""
    n = os.environ.get("FEELABLE_THREADS")
    if not n:
        return
    import onnxruntime as ort
    from rapidocr_onnxruntime import utils as rutils
    base = ort.SessionOptions

    def make_opts():
        o = base()
        o.intra_op_num_threads = int(n)
        o.inter_op_num_threads = 1
        try:
            o.add_session_config_entry("session.intra_op.allow_spinning", "0")
            o.add_session_config_entry("session.inter_op.allow_spinning", "0")
        except Exception:  # pragma: no cover
            pass
        return o

    rutils.SessionOptions = make_opts


_FAILED = threading.local()


def take_failure() -> str | None:
    """The text-reading engine error since the last call (this thread), or None."""
    e = getattr(_FAILED, "error", None)
    _FAILED.error = None
    return e


def run_ocr(image_rgb: np.ndarray, min_confidence: float = MIN_CONFIDENCE,
            restore_spaces: bool = True) -> list[Label]:
    """Detect + recognise text. Returns labels with axis-aligned int boxes (x0, y0, x1, y1).

    `min_confidence=0` also returns unsure reads (used to spot sideways / upside-down text).
    Never raises for an ordinary image: OCR failures return [].
    """
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[0] < 8 or image_rgb.shape[1] < 8:
        return []
    h, w = image_rgb.shape[:2]
    try:
        engine = _get_engine()
        with _RUN_LOCK:  # onnx sessions are thread-safe, but rapidocr keeps some state; be safe
            result, _elapse = engine(np.ascontiguousarray(image_rgb))
    except Exception as e:  # pragma: no cover - defensive
        # an engine failure is not "no text": remember it so the analysis can say so (audit P1)
        _FAILED.error = f"{type(e).__name__}"
        return []
    if not result:
        return []
    labels: list[Label] = []
    for item in result:
        try:
            pts, text, score = item[0], item[1], item[2]
            conf = float(score)
        except (TypeError, ValueError, IndexError):
            continue
        text = str(text).strip()
        if conf < min_confidence or not text:
            continue
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        x0 = int(max(0, np.floor(p[:, 0].min())))
        y0 = int(max(0, np.floor(p[:, 1].min())))
        x1 = int(min(w, np.ceil(p[:, 0].max())))
        y1 = int(min(h, np.ceil(p[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            continue
        labels.append(Label(text=text, bbox=(x0, y0, x1, y1), confidence=round(conf, 4)))
    try:
        if restore_spaces:
            _restore_spaces(engine, image_rgb, labels)
    except Exception:  # pragma: no cover - spacing is a nicety, never fail OCR for it
        pass
    try:
        _zoom_reread(engine, image_rgb, labels)
    except Exception:  # pragma: no cover - a second look is a nicety too
        pass
    labels.sort(key=lambda lb: (lb.bbox[1], lb.bbox[0]))
    return labels


def _word_segments(crop: np.ndarray) -> list[tuple[int, int]]:
    """Split a single text line crop into words using vertical gaps in the ink."""
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.int16)
    h, w = g.shape
    if h < 6 or w < 6:
        return [(0, w)]
    rim = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    ref = float(np.median(rim))
    spread = float(np.percentile(np.abs(g - ref), 99))
    ink = np.abs(g - ref) > max(25.0, 0.35 * spread)
    cols = ink.sum(0) > 0
    segs, start = [], None
    for x in range(w):
        if cols[x] and start is None:
            start = x
        elif not cols[x] and start is not None:
            segs.append([start, x])
            start = None
    if start is not None:
        segs.append([start, w])
    # drop leader-line stubs caught inside the box (flat, wide, no glyph height)
    rows_any = np.nonzero(ink.any(1))[0]
    if len(rows_any) == 0:
        return [(0, w)]
    ink_h = float(rows_any.max() - rows_any.min() + 1)
    kept = []
    for a, b in segs:
        r = np.nonzero(ink[:, a:b].any(1))[0]
        span = r.max() - r.min() + 1
        if span <= max(3, 0.25 * ink_h) and (b - a) >= 2 * span:
            continue
        kept.append([a, b])
    segs = kept
    if len(segs) < 2:
        return [(segs[0][0], segs[0][1])] if segs else [(0, w)]
    gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
    thresh = max(3.0, 2.2 * float(np.median(gaps)), 0.22 * ink_h)
    words = [segs[0][:]]
    for gap, sg in zip(gaps, segs[1:]):
        if gap >= thresh:
            words.append(sg[:])
        else:
            words[-1][1] = sg[1]
    return [(a, b) for a, b in words]


def _restore_spaces(engine, image_rgb: np.ndarray, labels: list[Label]) -> None:
    """RapidOCR's recogniser often drops spaces ("Cellwall"): re-read each word separately."""
    jobs = []  # (label, [crops])
    for lb in labels:
        if len(lb.text) > 48:
            continue
        x0, y0, x1, y1 = lb.bbox
        crop = image_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        words = _word_segments(crop)
        if len(words) <= len(lb.text.split()):
            continue
        crops = [np.ascontiguousarray(crop[:, max(0, a - 2):min(crop.shape[1], b + 2)]) for a, b in words]
        jobs.append((lb, crops))
    if not jobs:
        return
    flat = [c for _, crops in jobs for c in crops]
    rec, _ = engine.text_recognizer(flat)
    k = 0
    for lb, crops in jobs:
        res = rec[k:k + len(crops)]
        k += len(crops)
        parts = [str(r[0]).strip() for r in res]
        squash = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
        joined = " ".join(p for p in parts if p)
        if parts and all(parts) and difflib.SequenceMatcher(None, squash(joined), squash(lb.text)).ratio() >= 0.8:
            lb.text = joined


REREAD_BELOW = 0.9      # second, zoomed look at readings less sure than this


def _sharpen(c: np.ndarray) -> np.ndarray:
    b = cv2.GaussianBlur(c, (0, 0), 1.2)
    return cv2.addWeighted(c, 1.6, b, -0.6, 0)


def _zoom_reread(engine, image_rgb: np.ndarray, labels: list[Label]) -> None:
    """Small or blurry labels: look again, closer, and repair the spelling.

    Each unsure reading is re-recognised from a padded crop and from a 2x sharpened zoom of it;
    every candidate is snapped to the science vocabulary (engine/lexicon.py) and the reading
    that needed the fewest repairs wins (ties: the recogniser's confidence). Sure readings are
    only spell-checked."""
    from engine.lexicon import correct, is_known

    h, w = image_rgb.shape[:2]
    jobs: list[tuple[Label, int]] = []
    crops: list[np.ndarray] = []
    for lb in labels:
        if lb.confidence >= REREAD_BELOW or len(lb.text) > 40:
            continue
        x0, y0, x1, y1 = lb.bbox
        ph = max(2, int(0.25 * (y1 - y0)))
        c = image_rgb[max(0, y0 - ph):min(h, y1 + ph), max(0, x0 - ph):min(w, x1 + ph)]
        if c.size == 0 or min(c.shape[:2]) < 4:
            continue
        z = np.ascontiguousarray(_sharpen(cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)))
        jobs.append((lb, len(crops)))
        crops.extend([np.ascontiguousarray(c), z])
    rec = engine.text_recognizer(crops)[0] if crops else []
    squash = lambda t: re.sub(r"[^a-z]", "", t.lower())
    picked: dict[int, str] = {}
    for lb, k in jobs:
        cands = [(lb.text, lb.confidence)] + [(str(r[0]).strip(), float(r[1])) for r in rec[k:k + 2]]
        best = None
        for raw, conf in cands:
            raw = raw.strip(" -_.,:;")
            if not raw:
                continue
            fixed = correct(raw)
            edits = difflib.SequenceMatcher(None, squash(raw), squash(fixed)).ratio()
            key = (is_known(fixed), edits, conf)
            if best is None or key > best[0]:
                best = (key, fixed)
        if best is not None:
            picked[id(lb)] = best[1]
    for lb in labels:
        new = picked[id(lb)] if id(lb) in picked else (correct(lb.text) if lb.confidence < 0.97 else lb.text)
        if squash(new) != squash(lb.text):
            lb.raw = lb.text                    # keep what was first read, for the inspector view
        lb.text = new


def read_crop(crop_rgb: np.ndarray) -> tuple[str, float]:
    """Recognise one short line of text in a tight crop (no detection step). ('', 0) on failure."""
    try:
        if crop_rgb is None or crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 4:
            return "", 0.0
        engine = _get_engine()
        with _RUN_LOCK:
            rec, _ = engine.text_recognizer([np.ascontiguousarray(crop_rgb)])
        if not rec:
            return "", 0.0
        return str(rec[0][0]).strip(), float(rec[0][1])
    except Exception:  # pragma: no cover
        return "", 0.0


def warm() -> None:
    """Load the OCR model now (used by the persistent worker so the first job is fast)."""
    _get_engine()
