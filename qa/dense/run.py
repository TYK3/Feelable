"""Dense-diagram benchmark: how many of the known labels does Feelable read, and turn into named parts?

usage (repo root):  python qa/dense/run.py [--only id1,id2] [--out /tmp/dense] [--jobs 4] [--ai]
Prints a table and writes <out>/results.json. AI is off unless --ai (needs GEMINI_API_KEY).

Metrics per image (labels = the expected list in manifest.json):
  read   = expected labels found in the OCR text (what the engine *read*)
  named  = expected labels that became an included, named part (what a student can *feel*)
  junk   = included parts whose name matches no expected label (only for images with a label list)
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import pickle
import re
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def matches(expected: str, text: str) -> bool:
    e, t = norm(expected), norm(text)
    if not e or not t:
        return False
    if e in t or (len(t) >= 4 and t in e):
        return True
    if difflib.SequenceMatcher(None, e, t).ratio() >= 0.82:
        return True
    # plural / singular
    return e.rstrip("s") == t.rstrip("s")


def run_case(case: dict, out: str, use_ai: bool) -> dict:
    os.environ.setdefault("FEELABLE_THREADS", "2")
    from engine import pipeline

    job_dir = os.path.join(out, case["id"])
    data = (ROOT / case["file"]).read_bytes()
    t0 = time.perf_counter()
    try:
        r = pipeline.run_job(data, job_dir, use_ai=use_ai)
    except Exception as e:  # noqa: BLE001 - the benchmark records failures
        return {"id": case["id"], "error": f"{type(e).__name__}: {e}"}
    dt = time.perf_counter() - t0
    with open(os.path.join(job_dir, "analysis.pkl"), "rb") as f:
        a = pickle.load(f)["analysis"]
    texts = [lb.text for lb in a.labels]
    joined = " | ".join(texts)
    parts = [p["name"] for p in r["parts"] if p["include"]]
    exp = case["labels"]
    read = [e for e in exp if any(matches(e, t) for t in texts) or matches(e, joined)]
    named = [e for e in exp if any(matches(e, n) for n in parts)]
    junk = [n for n in parts if exp and not any(matches(e, n) for e in exp)]
    return {
        "id": case["id"], "kind": case["kind"], "n": len(exp), "read": len(read), "named": len(named),
        "missing_read": [e for e in exp if e not in read], "missing_named": [e for e in exp if e not in named],
        "parts": parts, "junk": junk, "issues": len(r["readiness"]["issues"]), "status": r["readiness"]["status"],
        "secs": round(dt, 1), "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "input_kind": r.get("input_kind"), "labels_read": len(texts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="/tmp/feelable-dense")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--ai", action="store_true")
    args = ap.parse_args()
    cases = json.loads((ROOT / "qa/dense/manifest.json").read_text())["cases"]
    by_id = {c["id"]: c for c in cases}
    for c in cases:
        if isinstance(c["labels"], str):
            c["labels"] = by_id[c["labels"][1:]]["labels"]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    cases = [c for c in cases if (ROOT / c["file"]).exists()]
    os.makedirs(args.out, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs, max_tasks_per_child=1) as ex:
        results = list(ex.map(run_case, cases, [args.out] * len(cases), [args.ai] * len(cases)))
    tot_n = sum(r.get("n", 0) for r in results)
    tot_read = sum(r.get("read", 0) for r in results)
    tot_named = sum(r.get("named", 0) for r in results)
    print(f"{'case':26} {'read':>7} {'named':>7} {'junk':>4} {'secs':>5} {'MB':>5}")
    for r in results:
        if "error" in r:
            print(f"{r['id']:26} ERROR {r['error']}")
            continue
        n = r["n"] or 1
        print(f"{r['id']:26} {r['read']:>3}/{r['n']:<3} {r['named']:>3}/{r['n']:<3} {len(r['junk']):>4} {r['secs']:>5} {r['peak_mb']:>5}")
    print(f"TOTAL read {tot_read}/{tot_n} = {tot_read / max(tot_n, 1):.0%}   named {tot_named}/{tot_n} = {tot_named / max(tot_n, 1):.0%}")
    (pathlib.Path(args.out) / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
