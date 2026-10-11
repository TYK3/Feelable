"""Pre-compute every sample (A4, offline mode) so clicking a sample on a slow server is instant.

Usage:  FEELABLE_SAMPLE_CACHE=/path/to/cache python precompute_samples.py
The API serves /api/jobs/from-sample/<id>?paper=a4 (and byte-identical uploads) from this cache.
Cache entries never hold an edit token: every clone gets a fresh one from the API
(pipeline.clone_job skips per-job files), and anything left from an older build is removed here.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STRICT = os.environ.get("FEELABLE_STRICT_REVIEW") == "1"


def _thin(poly, n=380):
    k = max(1, -(-len(poly) // n))
    return [list(map(float, q)) for q in poly[::k]]


def apply_review(out: Path, review: list) -> dict:
    """Replay a prepared example's reviewed corrections (samples/manifest.json "review") with the
    same edit operations a teacher uses in Feelable: rename, describe, remove, merge, and
    tap-to-outline (find_outline + add_part). Coordinates are in the analysed image's pixels.
    Each op names parts by their current name, so a correction that no longer applies is reported
    (and fails the build check when FEELABLE_STRICT_REVIEW=1) instead of silently doing nothing."""
    r = json.loads((out / pipeline.RESULT_FILE).read_text())
    for op in review:
        byname = {p["name"].lower(): p for p in r["parts"]}

        def pid(name):
            p = byname.get(name.lower())
            if p is None:
                raise KeyError(f"review: no part named {name!r} (have: {sorted(byname)})")
            return p["id"]
        try:
            if "rename" in op:
                pe = {"id": pid(op["rename"]), "name": op["to"]}
                if op.get("description"):
                    pe["description"] = op["description"]
                r = pipeline.rerender_job(str(out), {"parts": [pe]})
            elif "describe" in op:
                r = pipeline.rerender_job(str(out), {"parts": [{"id": pid(op["describe"]), "description": op["text"]}]})
            elif "remove" in op:
                r = pipeline.rerender_job(str(out), {"parts": [{"id": pid(op["remove"]), "include": False}]})
            elif "merge" in op:
                ids = [pid(n) for n in op["merge"]]
                r = pipeline.rerender_job(str(out), {"merge": ids})
                if op.get("name"):
                    keep = next(p for p in r["parts"] if p["id"] == ids[0])
                    r = pipeline.rerender_job(str(out), {"parts": [{"id": keep["id"], "name": op["name"]}]})
            elif "outline" in op:
                req = {k: op[k] for k in ("inside", "outside", "box", "lasso") if k in op}
                found = pipeline.find_outline(str(out), {"rev": 1, **req})
                if not found.get("found"):
                    raise ValueError(f"review: no outline found for {op['outline']!r}")
                ap = {"name": op["outline"], "polygon": _thin(found["polygons"][0]),
                      "overlap": op.get("overlap", "cut")}
                if op.get("into"):
                    ap["into"] = pid(op["into"])
                    ap["mode"] = op.get("mode", "add")
                r = pipeline.rerender_job(str(out), {"add_part": ap})
            elif "draw" in op:
                ap = {"name": op["draw"], "polygon": [list(map(float, q)) for q in op["polygon"]],
                      "overlap": op.get("overlap", "cut")}
                if op.get("within"):
                    ap["within"] = pid(op["within"])
                if op.get("into"):
                    ap["into"] = pid(op["into"])
                    ap["mode"] = op.get("mode", "add")
                r = pipeline.rerender_job(str(out), {"add_part": ap})
            elif "title" in op:
                r = pipeline.rerender_job(str(out), {"title": op["title"]})
            else:
                raise ValueError(f"review: unknown op {op}")
        except Exception as e:
            if STRICT:
                raise
            print(f"   review step skipped ({e})", flush=True)
            r = json.loads((out / pipeline.RESULT_FILE).read_text())
    return r
cache = Path(os.environ.get("FEELABLE_SAMPLE_CACHE") or ROOT / "backend" / "sample_cache")
only = set(sys.argv[1:])                   # optional: precompute just these sample ids
for s in json.loads((ROOT / "samples" / "manifest.json").read_text()):
    if only and s["id"] not in only:
        continue
    out = cache / "a4" / s["id"]
    if out.exists():
        shutil.rmtree(out)                 # start clean: no stale tokens/status/old artifacts
    out.mkdir(parents=True)
    t = time.time()
    r = pipeline.run_job((ROOT / "samples" / s["file"]).read_bytes(), str(out), use_ai=False, paper="a4")
    if r.get("title") in ("", "diagram") and s.get("title"):
        r = pipeline.rerender_job(str(out), {"title": s["title"]})   # textbook figure without a printed title
    if s.get("review"):
        r = apply_review(out, s["review"])
    for name in pipeline._NO_CLONE - {pipeline.STATE_FILE, pipeline.RESULT_FILE}:
        (out / name).unlink(missing_ok=True)
    print(f"{s['id']:18s} {len(r['parts'])} parts  {r['readiness']['status']:12s} {time.time() - t:.1f}s", flush=True)
print("cache:", cache)
