"""Score a Feelable analysis against an independent target annotation (qa/precision/truth/*.json).

Truth is written by hand from the SOURCE image (never from Feelable output). Each target has a
name, a type and either interior points or a ground-truth mask:

    region    every listed point must lie inside the part named for it; no point of another region
              target may (unless one is declared to contain the other, e.g. temporal bone contains
              squamous temporal). Two points for "kidneys" means both kidneys must be included.
    path      a boundary / arrow / route: every listed point along its course must be covered.
              A tiny disk at one end does not cover the course and fails.
    landmark  a point: the part must cover the point (a spot is fine here).
    mask      intersection-over-union with the ground-truth mask >= MASK_IOU.

A target is valid only when its name exists AND its geometry passes. A *critical* error is a part
named for one target that covers another target's point (wrong anatomy / wrong node), or a
forbidden point. Label count alone is never reported as accuracy.
"""
from __future__ import annotations

import json
import os
import re

import cv2
import numpy as np

MASK_IOU = 0.85
TOL = {"region": 4, "path": 7, "landmark": 12}


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in s.split())


def _names(t: dict) -> list[str]:
    return [norm(t["name"])] + [norm(x) for x in t.get("synonyms", [])]


def matches(part_name: str, t: dict) -> bool:
    pn = norm(part_name)
    for n in _names(t):
        rest = pn[len(n) + 1:] if pn.startswith(n + " ") else None
        if pn == n or rest is not None and (re.match(r"^[a-z]\d+$", n) or re.match(r"^\d", rest)):
            # "r1 1 00 ohm" ~ "r1", "battery 12 0 volt" ~ "battery"
            return True
    return False


def load_truth(truth_dir: str, name: str) -> dict | None:
    p = os.path.join(truth_dir, name + ".json")
    if not os.path.exists(p):
        alias = os.path.join(truth_dir, "aliases.json")
        if os.path.exists(alias):
            m = json.load(open(alias))
            if name in m:
                p = os.path.join(truth_dir, m[name] + ".json")
    if not os.path.exists(p):
        return None
    t = json.load(open(p))
    t["_dir"] = truth_dir
    return t


def _covers(mask: np.ndarray, x: float, y: float, r: int) -> bool:
    h, w = mask.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1, y0, y1 = max(0, xi - r), min(w, xi + r + 1), max(0, yi - r), min(h, yi + r + 1)
    return x0 < x1 and y0 < y1 and bool(mask[y0:y1, x0:x1].any())


def score(a, truth: dict) -> dict:
    W, H = truth["size"]
    if (a.width > a.height) != (W > H) and abs(a.width / a.height - W / H) > 0.05:
        return {"error": f"working image {a.width}x{a.height} is not in the annotated orientation {W}x{H}",
                "valid": 0, "total": len(truth["targets"]), "critical": 0, "targets": []}
    sx, sy = a.width / W, a.height / H
    parts = [p for p in a.parts if p.include and p.mask is not None and p.mask.any()]
    targets = truth["targets"]
    regions = [t for t in targets if t.get("type", "region") == "region" and t.get("points")]

    def related(t, u):
        return norm(u["name"]) in [norm(x) for x in t.get("contains", [])] or \
            norm(t["name"]) in [norm(x) for x in u.get("contains", [])]

    out, criticals = [], []
    for t in targets:
        typ = "mask" if "mask" in t else t.get("type", "region")
        mine = [p for p in parts if matches(p.name, t)]
        rec = {"name": t["name"], "type": typ, "parts": [p.name for p in mine]}
        if not mine:
            rec["result"] = "missing"
            out.append(rec)
            continue
        m = np.zeros((a.height, a.width), np.uint8)
        for p in mine:
            m |= (p.mask > 0).astype(np.uint8)
        if typ == "mask":
            g = cv2.imread(os.path.join(truth["_dir"], t["mask"]), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, (a.width, a.height), interpolation=cv2.INTER_NEAREST) > 127
            inter, uni = np.logical_and(g, m > 0).sum(), np.logical_or(g, m > 0).sum()
            iou = inter / uni if uni else 0.0
            rec.update(iou=round(float(iou), 3), result="valid" if iou >= MASK_IOU else "wrong_extent")
            out.append(rec)
            continue
        r = max(1, int(round(TOL.get(typ, 4) * (sx + sy) / 2)))
        pts = [(x * sx, y * sy) for x, y in t.get("points", [])]
        hit = [_covers(m, x, y, r) for x, y in pts]
        bad = []
        if typ == "region":
            for u in regions:
                if u is t or related(t, u):
                    continue
                for x, y in u["points"]:
                    if _covers(m, x * sx, y * sy, 0):
                        bad.append(u["name"])
                        break
        for x, y in t.get("forbid_extra", []):
            if _covers(m, x * sx, y * sy, 0):
                bad.append(f"forbidden point {x},{y}")
        rec.update(points_covered=f"{sum(hit)}/{len(hit)}")
        if bad:
            rec["result"] = "critical"
            rec["covers"] = bad
            criticals.append(f"{t['name']} covers {', '.join(bad)}")
        elif all(hit):
            rec["result"] = "valid"
        else:
            rec["result"] = "location_only" if any(hit) else "wrong_place"
        out.append(rec)
    graph = None
    if truth.get("routes"):
        # route membership from the truth polylines vs what each route part says it stops at
        stations = {norm(t["name"]): t["points"][0] for t in targets if t.get("type") == "landmark"}
        ok = total = 0
        for rname, _col, pts in truth["routes"]:
            want = {n for n, p in stations.items() if list(p) in [list(q) for q in pts]}
            rp = next((p for p in parts if matches(p.name, {"name": rname})), None)
            got = set()
            if rp is not None and getattr(rp, "flow", ""):
                got = {norm(x) for x in re.split(r",| and ", rp.flow.replace("stops at", "")) if x.strip()}
            total += len(want | got)
            ok += len(want & got)
        graph = {"route_membership": f"{ok}/{total}"}
    if truth.get("edges"):
        # flow edges: each truth arrow (tail point, head point) joins the nodes nearest its ends
        nodes = [(norm(t["name"]), t["points"][0]) for t in targets]

        def node_at(pt):
            return min(nodes, key=lambda n: (n[1][0] - pt[0]) ** 2 + (n[1][1] - pt[1]) ** 2)[0]
        want = {(node_at(a_), node_at(b_)) for a_, b_ in truth["edges"]}
        got = {(norm(e["from"]), norm(e["to"])) for e in ((getattr(a, "trace", None) or {}).get("edges") or [])}
        graph = {**(graph or {}), "edges_correct": f"{len(want & got)}/{len(want)}", "edges_wrong": len(got - want)}
    scored = [o for o in out]
    valid = sum(o["result"] == "valid" for o in scored)
    return {"valid": valid, "total": len(scored), "critical": len(criticals),
            "valid_pct": round(100 * valid / max(1, len(scored)), 1),
            "criticals": criticals, "targets": out, "graph": graph}
