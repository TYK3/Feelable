"""Bulk regression harness: run the local Feelable engine on a folder of diagrams.

Usage (from backend/):
    python ../qa/bulk_eval.py OUT_DIR IMAGE [IMAGE ...] [--workers 2] [--timeout 300]

For every image it runs pipeline.run_job (fresh processing, no sample cache, AI off) in its own
process with a deadline, then measures:

* status / error / seconds
* labels read (part labels), parts included, named parts, unnamed parts
* pointer grounding: of the traced pointer lines whose label became a part, how many end inside
  that part's outline ("endpoint hits"). A miss means the name is attached to the wrong shape.
* far fragments: parts with a sizeable component far from the part's main body (the "nasal bone
  also selects the jaw" failure)
* readiness status and issue codes, warnings, missing-part suggestions

It writes OUT_DIR/<name>.json per image, OUT_DIR/<name>-overlay.jpg (each part tinted and named,
pointer endpoints marked green = hit / red = miss) and OUT_DIR/summary.csv.
These are internal-consistency measures, not semantic accuracy against a human annotation.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "backend")
sys.path.insert(0, BACKEND)


def _worker(path: str, job_dir: str, q):
    try:
        os.environ.setdefault("GEMINI_API_KEY", "")
        from engine import pipeline
        with open(path, "rb") as f:
            data = f.read()
        t0 = time.perf_counter()
        res = pipeline.run_job(data, job_dir, use_ai=False)
        q.put({"ok": True, "seconds": time.perf_counter() - t0, "readiness": res.get("readiness"),
               "warnings": res.get("warnings")})
    except Exception as e:  # noqa: BLE001
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]})


def _run_one(path: str, out_dir: str, timeout: float) -> dict:
    name = os.path.splitext(os.path.basename(path))[0]
    job_dir = tempfile.mkdtemp(prefix="fb-")
    q = mp.get_context("spawn").Queue()
    p = mp.get_context("spawn").Process(target=_worker, args=(path, job_dir, q))
    t0 = time.perf_counter()
    p.start()
    p.join(timeout)
    rec: dict = {"name": name, "path": path}
    if p.is_alive():
        p.kill()
        p.join()
        rec.update(status="timeout", seconds=round(time.perf_counter() - t0, 1))
        return rec
    try:
        r = q.get(timeout=5)
    except Exception:  # noqa: BLE001
        r = {"ok": False, "error": f"worker exited {p.exitcode}"}
    if not r.get("ok"):
        rec.update(status="error", error=r.get("error"), seconds=round(time.perf_counter() - t0, 1))
        return rec
    rec.update(status="done", seconds=round(r["seconds"], 1))
    with open(os.path.join(job_dir, "analysis.pkl"), "rb") as f:
        state = pickle.load(f)
    a = state["analysis"]
    rec.update(measure(a, r.get("readiness") or {}))
    rec["warnings"] = r.get("warnings") or []
    try:
        overlay(a, os.path.join(out_dir, f"{name}-overlay.jpg"), rec)
    except Exception as e:  # noqa: BLE001
        rec["overlay_error"] = str(e)
    return rec


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _target_end(points, label_bbox):
    """The pointer end farther from its label is the end that touches the structure."""
    if not points or len(points) < 2:
        return None
    x0, y0, x1, y1 = label_bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def d(pt):
        dx = max(x0 - pt[0], 0, pt[0] - x1)
        dy = max(y0 - pt[1], 0, pt[1] - y1)
        return (dx * dx + dy * dy) ** 0.5 + 1e-3 * ((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2) ** 0.5
    a, b = points[0], points[-1]
    return tuple(a) if d(a) > d(b) else tuple(b)


def measure(a, readiness: dict) -> dict:
    import cv2
    H, W = a.height, a.width
    diag = (H * H + W * W) ** 0.5
    inc = [p for p in a.parts if p.include and p.mask is not None and p.mask.any()]
    part_labels = [lb for lb in a.labels if lb.role == "part_label"]
    lb_by_text = {}
    for lb in a.labels:
        lb_by_text.setdefault(_norm(lb.text), lb)
    by_label = {}
    for p in inc:
        for key in {_norm(p.label_text or ""), _norm(p.name)}:
            if key:
                by_label.setdefault(key, p)
    hits, misses, unmatched_leaders = [], [], []
    for L in (a.trace or {}).get("leaders", []):
        lab = _norm(L.get("label", ""))
        lb = lb_by_text.get(lab)
        p = by_label.get(lab)
        if lb is None:
            continue
        end = _target_end(L.get("points") or [], lb.bbox)
        if end is None:
            continue
        if p is None:
            unmatched_leaders.append(L.get("label"))
            continue
        x, y = int(round(end[0])), int(round(end[1]))
        r = max(4, int(0.012 * diag))
        y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
        inside = bool(p.mask[y0:y1, x0:x1].any())
        (hits if inside else misses).append({"label": L.get("label"), "part": p.name, "end": [x, y]})
    far = []
    for p in inc:
        n, cc, stats, cents = cv2.connectedComponentsWithStats((p.mask > 0).astype(np.uint8), 8)
        if n <= 2:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        main = 1 + int(np.argmax(areas))
        total = float(areas.sum())
        for i in range(1, n):
            if i == main:
                continue
            frac = stats[i, cv2.CC_STAT_AREA] / total
            dist = float(np.hypot(*(cents[i] - cents[main]))) / diag
            if frac >= 0.08 and dist >= 0.18:
                far.append({"part": p.name, "frac": round(frac, 2), "dist": round(dist, 2)})
                break
    import re
    unnamed = [p.name for p in inc if re.match(r"^(part \d+|main object|.* detail)$", p.name.strip().lower())]
    issues = readiness.get("issues") or []
    return {
        "input_kind": a.input_kind,
        "diagram_type": a.diagram_type,
        "labels_read": len(part_labels),
        "parts": len(inc),
        "named_parts": len(inc) - len(unnamed),
        "unnamed": unnamed,
        "names": [p.name for p in inc],
        "pointers": len((a.trace or {}).get("leaders", [])),
        "endpoint_hits": len(hits),
        "endpoint_misses": len(misses),
        "miss_detail": misses[:12],
        "pointer_label_no_part": unmatched_leaders[:20],
        "far_fragments": far,
        "readiness": readiness.get("status"),
        "issue_codes": sorted({i.get("code") for i in issues}),
        "missing_suggestions": [i.get("target") for i in issues if i.get("action") == "draw"],
        "unmatched_labels": list(a.unmatched_labels or [])[:30],
    }


def overlay(a, out_path: str, rec: dict) -> None:
    import cv2
    img = a.image.copy()
    rng = np.random.default_rng(7)
    tint = img.copy()
    inc = [p for p in a.parts if p.include and p.mask is not None and p.mask.any()]
    for p in inc:
        c = tuple(int(v) for v in rng.integers(40, 230, 3))
        tint[p.mask > 0] = c
    img = cv2.addWeighted(img, 0.45, tint, 0.55, 0)
    for p in inc:
        cnts, _ = cv2.findContours((p.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 0, 0), 1)
        ys, xs = np.nonzero(p.mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(img, p.name[:22], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(img, p.name[:22], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for h in rec.get("miss_detail", []):
        cv2.circle(img, tuple(h["end"]), 7, (255, 0, 0), 2)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    s = 1400 / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])


COLS = ["name", "status", "seconds", "input_kind", "labels_read", "parts", "named_parts", "pointers",
        "endpoint_hits", "endpoint_misses", "far_fragments", "readiness", "issue_codes",
        "missing_suggestions", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    recs = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_run_one, p, args.out, args.timeout): p for p in args.images}
        for f in futs:
            pass
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            rec = f.result()
            recs.append(rec)
            with open(os.path.join(args.out, rec["name"] + ".json"), "w") as fh:
                json.dump(rec, fh, indent=1, default=str)
            print(f"{rec['status']:7} {rec.get('seconds', 0):6.1f}s {rec['name'][:45]:45} parts={rec.get('parts')} "
                  f"hits={rec.get('endpoint_hits')}/{(rec.get('endpoint_hits') or 0) + (rec.get('endpoint_misses') or 0)} "
                  f"far={len(rec.get('far_fragments') or [])} ready={rec.get('readiness')}", flush=True)
    recs.sort(key=lambda r: r["name"])
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in recs:
            w.writerow([json.dumps(r.get(c)) if isinstance(r.get(c), (list, dict)) else r.get(c, "") for c in COLS])


if __name__ == "__main__":
    main()
