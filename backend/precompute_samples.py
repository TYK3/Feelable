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
cache = Path(os.environ.get("FEELABLE_SAMPLE_CACHE") or ROOT / "backend" / "sample_cache")
for s in json.loads((ROOT / "samples" / "manifest.json").read_text()):
    out = cache / "a4" / s["id"]
    if out.exists():
        shutil.rmtree(out)                 # start clean: no stale tokens/status/old artifacts
    out.mkdir(parents=True)
    t = time.time()
    r = pipeline.run_job((ROOT / "samples" / s["file"]).read_bytes(), str(out), use_ai=False, paper="a4")
    if r.get("title") in ("", "diagram") and s.get("title"):
        r = pipeline.rerender_job(str(out), {"title": s["title"]})   # textbook figure without a printed title
    for name in pipeline._NO_CLONE - {pipeline.STATE_FILE, pipeline.RESULT_FILE}:
        (out / name).unlink(missing_ok=True)
    print(f"{s['id']:18s} {len(r['parts'])} parts  {r['readiness']['status']:12s} {time.time() - t:.1f}s", flush=True)
print("cache:", cache)
