"""Curation helper for prepared gallery examples.

    python ../qa/curate.py SAMPLE_ID OUT_DIR

Runs the sample exactly as the gallery cache does (fresh analysis + its reviewed corrections,
strict), then prints every part, every readiness issue, and writes OUT_DIR/<id>-overlay.jpg
(each part tinted and named) plus OUT_DIR/<id>-tactile.png and -legend.png for inspection.
The release gate for a prepared example: no dropped_label / unmatched_label / unnamed issues.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "backend")
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)

GATE = ("dropped_label", "unmatched_label", "unnamed", "braille_unsupported", "braille_overflow", "no_parts")


def run(sid: str, out: str) -> dict:
    cache = os.path.join(out, "cache")
    env = {**os.environ, "FEELABLE_SAMPLE_CACHE": cache, "FEELABLE_STRICT_REVIEW": "1",
           "FEELABLE_DATA": os.path.join(out, "jobs")}
    subprocess.run([sys.executable, os.path.join(BACKEND, "precompute_samples.py"), sid], env=env, check=True, cwd=BACKEND)
    jd = os.path.join(cache, "a4", sid)
    r = json.load(open(os.path.join(jd, "result.json")))
    a = pickle.load(open(os.path.join(jd, "analysis.pkl"), "rb"))["analysis"]
    from bulk_eval import overlay
    overlay(a, os.path.join(out, f"{sid}-overlay.jpg"), {})
    for f, dst in (("tactile.png", "tactile"), ("legend.png", "legend")):
        if os.path.exists(os.path.join(jd, f)):
            shutil.copy(os.path.join(jd, f), os.path.join(out, f"{sid}-{dst}.png"))
    return r


if __name__ == "__main__":
    sid, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    r = run(sid, out)
    print(f"\n{sid}: title={r['title']!r} parts={len(r['parts'])} readiness={r['readiness']['status']}")
    for p in r["parts"]:
        print(f"  {p['id']:4} {'on ' if p['include'] else 'OFF'} {p['key']:3} {p['name'][:40]:40} {p['match_method']}")
    bad = [i for i in r["readiness"]["issues"] if i["code"] in GATE]
    for i in r["readiness"]["issues"]:
        print(f"  [{i['severity']}] {i['code']}: {i['message'][:110]}")
    print("GATE", "PASS" if not bad else f"FAIL ({len(bad)})")
