#!/usr/bin/env python3
"""Feelable robustness harness: run every image in manifest.csv through the engine.

Each image runs in its own subprocess (`run_job(bytes, jobdir, use_ai=False, paper="a4")`) with a
timeout and peak-RSS tracking, so one crash/OOM can't take the whole run down.

    python qa/robustness/run.py --repo /path/to/repo --out qa/robustness/runs/<name>
    python qa/robustness/run.py --repo . --out qa/robustness/runs/smoke --only smoke   # 25-image smoke set (also --smoke)
    python qa/robustness/run.py --out qa/robustness/runs/x --only ob-mouse,fx-user-mouse-photo
    python qa/robustness/run.py --out qa/robustness/runs/x --category objects,maps
    python qa/robustness/run.py --out qa/robustness/runs/baseline --report-only        # rebuild report

Outputs in --out: results.json, results.csv, report.html, thumbs/ (small JPEGs), jobs/ (full job
folders, git-ignored). Human verdicts are merged from qa/robustness/verdicts.csv (id,verdict,reason).
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SERVER_BUDGET_MB = 512

DRAWABLE = {"textbook", "unlabelled", "charts", "maps", "flow_circuits", "objects", "animals_people",
            "scenes", "logos_icons", "fixtures", "phone_variants"}
NON_DRAWABLE_IDS = {"fx-blank-page"}
TEXTY = {"text_screens", "handwriting"}

CATEGORY_ORDER = ["textbook", "unlabelled", "charts", "maps", "flow_circuits", "objects", "animals_people",
                  "scenes", "text_screens", "handwriting", "logos_icons", "weird", "fixtures", "phone_variants"]


# ============================================================================ child
def child_main(image_path: str, job_dir: str, repo: str, paper: str) -> int:
    """Runs inside the per-image subprocess. Writes job_dir/qa_child.json."""
    sys.path.insert(0, os.path.join(repo, "backend"))
    os.makedirs(job_dir, exist_ok=True)
    out = {"status": "crash", "error": "", "seconds": None}
    t0 = time.perf_counter()
    try:
        from engine import pipeline
        with open(image_path, "rb") as f:
            data = f.read()
        bad = getattr(pipeline, "BadImage", ValueError)
        try:
            pipeline.run_job(data, job_dir, use_ai=False, paper=paper)
            out["status"] = "ok"
        except (bad, ValueError) as e:  # the API maps these to 4xx with a message
            out["status"] = "clean_error"
            out["error"] = f"{type(e).__name__}: {e}"[:300]
        except MemoryError as e:
            out["status"] = "oom"
            out["error"] = f"MemoryError: {e}"[:300]
    except BaseException as e:  # noqa: BLE001 -- we want *everything* here
        out["status"] = "crash"
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        out["traceback"] = traceback.format_exc()[-3000:]
    out["seconds"] = round(time.perf_counter() - t0, 2)
    # VmHWM is per address space (reset at exec); ru_maxrss would inherit the parent's high-water mark
    out["child_maxrss_mb"] = _hwm_mb()
    with open(os.path.join(job_dir, "qa_child.json"), "w") as f:
        json.dump(out, f)
    return 0


def _hwm_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return 0.0


# ============================================================================ parent
def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def run_one(row: dict, args) -> dict:
    img = os.path.join(HERE, row["file"])
    job_dir = os.path.join(args.out, "jobs", row["id"])
    if os.path.isdir(job_dir):
        import shutil
        shutil.rmtree(job_dir)
    os.makedirs(job_dir, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("FEELABLE_THREADS", str(args.threads))
    env.setdefault("OMP_NUM_THREADS", str(args.threads))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        env.pop(k, None)  # use_ai=False anyway; make sure nothing leaks out
    cmd = [sys.executable, os.path.abspath(__file__), "--child", img, job_dir, "--repo", args.repo, "--paper", args.paper]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    peak = 0.0
    killed = None
    out_chunks: list[bytes] = []

    def pump():
        for chunk in iter(lambda: p.stdout.read(4096), b""):
            out_chunks.append(chunk)
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    while p.poll() is None:
        peak = max(peak, _rss_mb(p.pid))
        el = time.perf_counter() - t0
        if el > args.timeout:
            killed = "timeout"
        elif args.rss_limit_mb and peak > args.rss_limit_mb:
            killed = "oom"
        if killed:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
            break
        time.sleep(0.05)
    th.join(timeout=2)
    wall = round(time.perf_counter() - t0, 2)
    rec = {"id": row["id"], "category": row["category"], "file": row["file"], "variant_of": row.get("variant_of", ""),
           "expected": row.get("expected", ""), "source_url": row.get("source_url", ""),
           "wall_s": wall, "peak_mb": round(peak, 1)}
    child_path = os.path.join(job_dir, "qa_child.json")
    if killed:
        rec["status"] = killed
        rec["error"] = f"killed after {wall}s ({killed}); peak {peak:.0f} MB"
    elif os.path.exists(child_path):
        with open(child_path) as f:
            ch = json.load(f)
        rec["status"] = ch["status"]
        rec["error"] = ch.get("error", "")
        rec["engine_s"] = ch.get("seconds")
        rec["peak_mb"] = round(max(peak, ch.get("child_maxrss_mb") or 0), 1)
        if ch.get("traceback"):
            rec["traceback"] = ch["traceback"]
    else:
        rc = p.returncode
        rec["status"] = "oom" if rc == -9 else "crash"
        tail = b"".join(out_chunks).decode("utf-8", "replace")[-600:]
        rec["error"] = f"child exited rc={rc} without result; {tail}"
    rec["log_tail"] = b"".join(out_chunks).decode("utf-8", "replace")[-400:]
    _add_result_fields(rec, job_dir)
    _ink(rec, job_dir)
    _junk_signals(rec, row)
    _thumbs(rec, img, job_dir, args.out)
    print(f"[{rec['status']:>11}] {rec['id']:<40} {wall:6.1f}s {rec['peak_mb']:6.0f}MB parts={rec.get('n_parts', '-')}"
          f" {('JUNK:' + ','.join(rec['junk'])) if rec.get('junk') else ''}", flush=True)
    return rec


def _ink(rec: dict, job_dir: str) -> None:
    """Fraction of the tactile page that is raised (dark), and the largest solid-black blob as a page fraction."""
    tp = os.path.join(job_dir, "tactile.png")
    if not os.path.exists(tp):
        return
    try:
        import numpy as np
        import cv2
        from PIL import Image
        with Image.open(tp) as im:
            a = np.asarray(im.convert("L"))
        dark = (a < 128).astype(np.uint8)
        rec["ink_frac"] = round(float(dark.mean()), 4)
        # solid regions: erode away lines/hatching (<= ~8 px thick at this scale), keep true fills
        k = max(3, int(round(min(a.shape) / 120)))
        solid = cv2.erode(dark, np.ones((k, k), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        rec["solid_frac"] = round(float(stats[1:, cv2.CC_STAT_AREA].max() / dark.size) if n > 1 else 0.0, 4)
    except Exception:
        pass


def _add_result_fields(rec: dict, job_dir: str) -> None:
    rp = os.path.join(job_dir, "result.json")
    if not os.path.exists(rp):
        return
    with open(rp, encoding="utf-8") as f:
        r = json.load(f)
    parts = r.get("parts", []) or []
    rec["title"] = r.get("title", "")
    rec["diagram_type"] = r.get("diagram_type", "")
    rec["input_kind"] = r.get("input_kind", r.get("kind", ""))
    rd = r.get("readiness")
    rec["readiness"] = json.dumps(rd, ensure_ascii=False)[:300] if isinstance(rd, (dict, list)) else (rd or "")
    rec["n_parts"] = len(parts)
    rec["n_included"] = sum(1 for p in parts if p.get("include"))
    rec["part_names"] = [p.get("name", "") for p in parts]
    rec["warnings"] = r.get("warnings", []) or []
    rec["dropped"] = r.get("dropped", []) or []
    checks = r.get("checks", []) or []
    rec["checks"] = f"{sum(1 for c in checks if c.get('passed'))}/{len(checks)}"
    rec["source_wh"] = [r.get("source", {}).get("width"), r.get("source", {}).get("height")]
    rec["timings"] = r.get("timings", {})


# ----------------------------------------------------------------- junk detection
_WORDS: set[str] | None = None
GENERIC_OK = {"part", "parts", "shape", "region", "area", "label", "x", "y", "axis", "bar", "line", "key"}
PROFANE = re.compile(r"(fuck|shit|cunt|bitch|dick|cock|piss|slut|whore|nigg|fag)", re.I)
FUNCTION_WORDS = {"the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with", "by", "from", "is", "are", "at", "as"}
VERBISH = {"makes", "make", "allow", "allows", "form", "forms", "is", "are", "contains", "produces", "carries", "helps", "the"}
BRANDISH = re.compile(r"(basics|inc\b|ltd|corp|\.com|www|logitech|amazon|amagon|microsoft|apple inc|samsung|tm\b|®|©)", re.I)


def _words() -> set[str]:
    global _WORDS
    if _WORDS is None:
        _WORDS = set()
        for p in ("/usr/share/dict/words", "/usr/share/dict/american-english"):
            if os.path.exists(p):
                with open(p, encoding="utf-8", errors="ignore") as f:
                    _WORDS |= {w.strip().lower() for w in f if w.strip()}
        try:  # optional, much better coverage
            from wordfreq import zipf_frequency  # noqa: F401
            _WORDS.add("__wordfreq__")
        except Exception:
            pass
    return _WORDS


def _is_word(tok: str) -> bool:
    t = tok.lower().strip(".,;:'\"()[]-")
    if not t:
        return True
    if t.isdigit():
        return True
    w = _words()
    if "__wordfreq__" in w:
        from wordfreq import zipf_frequency
        if zipf_frequency(t, "en") >= 1.0 or zipf_frequency(t.rstrip("s"), "en") >= 1.0:
            return True
        return False
    if not w:  # no dictionary available: don't guess
        return True
    if t in w or t.rstrip("s") in w:
        return True
    return False


def name_is_garbage(name: str) -> str | None:
    """Return a reason if a part name looks like OCR garbage / brand text, else None."""
    n = name.strip()
    if not n or re.fullmatch(r"part \d+", n, re.I):
        return None
    if BRANDISH.search(n):
        return "brand"
    toks = re.findall(r"[A-Za-z0-9]+", n)
    if not toks:
        return "no-letters"
    letters = "".join(t for t in toks if not t.isdigit())
    if len(letters) <= 2 and not any(t.lower() in GENERIC_OK for t in toks):
        return "too-short"
    if any(re.search(r"[A-Za-z]\d|\d[A-Za-z]", t) for t in toks) and not re.fullmatch(r"[A-Za-z]{1,3}\d{1,2}", toks[0]):
        return "mixed-alnum"
    non_words = [t for t in toks if not _is_word(t) and t.lower() not in GENERIC_OK]
    if non_words and len(non_words) / len(toks) > 0.5:
        return "non-word:" + non_words[0][:20]
    low = [t.lower() for t in toks]
    if len(toks) >= 5 or (len(toks) >= 2 and (low[-1] in FUNCTION_WORDS or low[0] in {"and", "or", "of", "to", "the"})) \
            or (len(toks) >= 3 and any(t in VERBISH for t in low)):
        return "fragment"
    return None


_MANIFEST_CAT: dict | None = None


def _base_category(row: dict) -> str:
    """Category of the original image (phone variants inherit their base's category)."""
    global _MANIFEST_CAT
    if _MANIFEST_CAT is None:
        try:
            _MANIFEST_CAT = {r["id"]: r["category"] for r in load_manifest(os.path.join(HERE, "manifest.csv"))}
        except OSError:
            _MANIFEST_CAT = {}
    base = row.get("variant_of") or ""
    if row.get("category") == "phone_variants" and base in _MANIFEST_CAT:
        return _MANIFEST_CAT[base]
    return row.get("category", "")


def _junk_signals(rec: dict, row: dict) -> None:
    junk: list[str] = []
    names = rec.get("part_names") or []
    if rec.get("status") == "ok":
        garbage = [(n, name_is_garbage(n)) for n in names]
        garbage = [(n, why) for n, why in garbage if why and "background" not in n.lower()]
        rec["garbage_names"] = [f"{n} ({why})" for n, why in garbage]
        if any(why != "fragment" for _, why in garbage):
            junk.append("ocr_garbage_names")
        t = (rec.get("title") or "").strip()
        if t and t.lower() not in ("diagram", "tactile diagram") and name_is_garbage(t):
            junk.append("garbage_title")
        if any(PROFANE.search(n) for n in names + [t]):
            junk.append("profanity")
        if any("background" in n.lower() for n in names):
            junk.append("background_part")
        if len(names) > 12:
            junk.append("too_many_parts")
        unnamed = [n for n in names if re.fullmatch(r"part \d+", n.strip(), re.I)]
        if names and len(unnamed) == len(names):
            junk.append("all_unnamed")
        elif names and len(unnamed) * 2 > len(names):
            junk.append("mostly_unnamed")
        if any(why == "fragment" for _, why in garbage):
            junk.append("label_fragments")
        base_cat = _base_category(row)
        drawable = base_cat in DRAWABLE and row["id"] not in NON_DRAWABLE_IDS and "blank" not in row["id"]
        if not names and drawable:
            junk.append("zero_parts_drawable")
        if drawable and rec.get("ink_frac") is not None and rec["ink_frac"] < 0.004 and not names:
            pass  # covered by zero_parts_drawable
        elif drawable and rec.get("ink_frac") is not None and rec["ink_frac"] < 0.004:
            junk.append("blank_tactile_with_parts")
        if (rec.get("solid_frac") or 0) > 0.03:
            junk.append("solid_black_fill")
        if names and base_cat in TEXTY:
            junk.append("text_page_has_parts")
        if row["id"] in NON_DRAWABLE_IDS and names:
            junk.append("blank_has_parts")
    if rec.get("peak_mb", 0) > SERVER_BUDGET_MB:
        junk.append("over_512mb")
    if rec.get("wall_s", 0) > 60:
        junk.append("slow_over_60s")
    rec["junk"] = junk


# ----------------------------------------------------------------- thumbnails
def _thumb(src_path: str, dst_path: str, max_side: int = 260) -> bool:
    try:
        from PIL import Image, ImageOps
        with Image.open(src_path) as im:  # default MAX_IMAGE_PIXELS: refuse bombs, keep the harness small
            im.draft("RGB", (max_side * 2, max_side * 2))
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P", "PA"):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                bg.alpha_composite(im)
                im = bg
            elif im.mode.startswith("I"):
                import numpy as np
                a = np.asarray(im, dtype="float64")
                im = Image.fromarray((a / max(a.max(), 1) * 255).astype("uint8"))
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            im.save(dst_path, "JPEG", quality=70)
        return True
    except Exception:
        return False


def _thumbs(rec: dict, img: str, job_dir: str, out: str) -> None:
    td = os.path.join(out, "thumbs")
    os.makedirs(td, exist_ok=True)
    s = os.path.join(td, rec["id"] + ".src.jpg")
    t = os.path.join(td, rec["id"] + ".tac.jpg")
    rec["thumb_src"] = os.path.relpath(s, out) if _thumb(img, s) else ""
    tp = os.path.join(job_dir, "tactile.png")
    rec["thumb_tac"] = os.path.relpath(t, out) if os.path.exists(tp) and _thumb(tp, t, 320) else ""


# ============================================================================ outputs
CSV_FIELDS = ["id", "category", "variant_of", "status", "wall_s", "engine_s", "peak_mb", "input_kind", "diagram_type",
              "title", "n_parts", "n_included", "ink_frac", "solid_frac", "part_names", "garbage_names", "warnings", "readiness", "checks",
              "junk", "verdict", "verdict_reason", "error", "expected", "source_url"]


def load_verdicts(path: str) -> dict:
    v = {}
    if path and os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v[r["id"]] = (r.get("verdict", ""), r.get("reason", ""))
    return v


def write_outputs(results: list[dict], out: str, verdicts: dict, meta: dict) -> None:
    for r in results:
        if r["id"] in verdicts:
            r["verdict"], r["verdict_reason"] = verdicts[r["id"]]
    results.sort(key=lambda r: (CATEGORY_ORDER.index(r["category"]) if r["category"] in CATEGORY_ORDER else 99, r["id"]))
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            for k in ("part_names", "garbage_names", "warnings", "junk"):
                if isinstance(row.get(k), list):
                    row[k] = " | ".join(row[k])
            w.writerow(row)
    write_report(results, out, meta)


def _pct(vals, q):
    if not vals:
        return 0
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def write_report(results: list[dict], out: str, meta: dict, path: str | None = None, prefix: str = "") -> None:
    path = path or os.path.join(out, "report.html")
    from collections import Counter, defaultdict
    st = Counter(r["status"] for r in results)
    junk = Counter(j for r in results for j in r.get("junk", []))
    verd = Counter(r.get("verdict", "") for r in results if r.get("verdict"))
    times = [r["wall_s"] for r in results if r["status"] == "ok"]
    mems = [r["peak_mb"] for r in results]
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    e = html.escape
    css = """
:root{--bg:#fbfaf6;--fg:#1d1d1b;--mut:#6b6a63;--line:#e3e0d6;--ok:#2f7d32;--bad:#b3261e;--warn:#9a6b00;--chip:#f0ede4}
@media (prefers-color-scheme: dark){:root{--bg:#151513;--fg:#ecebe6;--mut:#a3a29a;--line:#33322d;--ok:#7cc47f;--bad:#f2948d;--warn:#e7c063;--chip:#26251f}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.mut{color:var(--mut)}.sum{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.sum span{background:var(--chip);padding:4px 10px;border-radius:99px}
.row{display:grid;grid-template-columns:150px 190px 1fr;gap:12px;padding:10px 0;border-bottom:1px solid var(--line);align-items:start}
.row img{max-width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}
.st{font-weight:600}.ok{color:var(--ok)}.crash,.oom,.timeout{color:var(--bad)}.clean_error{color:var(--warn)}
.j{display:inline-block;background:var(--chip);color:var(--bad);border-radius:4px;padding:0 5px;margin:1px;font-size:12px}
.v-useful{color:var(--ok);font-weight:600}.v-meh{color:var(--warn);font-weight:600}.v-useless{color:var(--bad);font-weight:600}
.names{font-family:ui-monospace,monospace;font-size:12px;word-break:break-word}
@media (max-width:640px){.row{grid-template-columns:1fr 1fr}.row .info{grid-column:1/-1}}
nav a{margin-right:10px;color:inherit}
"""
    h = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Feelable Robustness</title><style>{css}</style></head><body>",
         "<h1>Feelable robustness contact sheet</h1>",
         f"<div class='mut'>{e(meta.get('when', ''))} · repo {e(meta.get('repo', ''))} @ {e(meta.get('commit', ''))} · "
         f"{len(results)} images · use_ai=False · paper=a4 · threads={e(str(meta.get('threads', '')))}</div>",
         "<div class='sum'>" + "".join(f"<span>{e(k)}: <b>{v}</b></span>" for k, v in st.most_common()) + "</div>",
         "<div class='sum'>" + "".join(f"<span>verdict {e(k)}: <b>{v}</b></span>" for k, v in verd.most_common()) + "</div>",
         f"<div class='sum'><span>time ok p50 {_pct(times, .5):.1f}s · p90 {_pct(times, .9):.1f}s · max {max(times or [0]):.1f}s</span>"
         f"<span>peak RSS p50 {_pct(mems, .5):.0f} MB · p90 {_pct(mems, .9):.0f} MB · max {max(mems or [0]):.0f} MB</span></div>",
         "<div class='sum'>" + "".join(f"<span class='j'>{e(k)} ×{v}</span>" for k, v in junk.most_common()) + "</div>",
         "<nav>" + "".join(f"<a href='#{c}'>{c} ({len(by_cat[c])})</a>" for c in CATEGORY_ORDER if c in by_cat) + "</nav>"]
    for c in CATEGORY_ORDER:
        if c not in by_cat:
            continue
        h.append(f"<h2 id='{c}'>{e(c)} <span class='mut'>({len(by_cat[c])})</span></h2>")
        for r in by_cat[c]:
            src = f"<img loading='lazy' src='{e(prefix + r['thumb_src'])}' alt='source'>" if r.get("thumb_src") else "<div class='mut'>no preview</div>"
            tac = f"<img loading='lazy' src='{e(prefix + r['thumb_tac'])}' alt='tactile'>" if r.get("thumb_tac") else "<div class='mut'>no tactile</div>"
            names = ", ".join(r.get("part_names") or []) or "—"
            v = r.get("verdict", "")
            vh = f"<div><span class='v-{e(v)}'>{e(v)}</span> — {e(r.get('verdict_reason', ''))}</div>" if v else ""
            jj = "".join(f"<span class='j'>{e(j)}</span>" for j in r.get("junk", []))
            warn = "; ".join(r.get("warnings") or [])
            h.append(
                f"<div class='row'>{src}{tac}<div class='info'><div><b>{e(r['id'])}</b> "
                f"<span class='st {e(r['status'])}'>{e(r['status'])}</span> · {r['wall_s']}s · {r['peak_mb']:.0f} MB"
                f"{' · kind ' + e(str(r.get('input_kind'))) if r.get('input_kind') else ''}"
                f"{' · type ' + e(str(r.get('diagram_type'))) if r.get('diagram_type') else ''}</div>"
                f"<div>title: <i>{e(r.get('title', '') or '—')}</i> · parts {r.get('n_parts', '—')} (incl {r.get('n_included', '—')})"
                f" · checks {e(r.get('checks', '—'))}</div>"
                f"<div class='names'>{e(names)}</div>{jj}{vh}"
                f"{'<div class=mut>⚠ ' + e(warn) + '</div>' if warn else ''}"
                f"{'<div class=mut>err: ' + e(r.get('error', '')) + '</div>' if r.get('error') else ''}"
                f"<div class='mut'>expected: {e(r.get('expected', ''))}</div></div></div>")
    h.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(h))


# ============================================================================ main
def load_manifest(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", nargs=2, metavar=("IMAGE", "JOBDIR"), help=argparse.SUPPRESS)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="repo root whose backend/engine is tested")
    ap.add_argument("--out", default=None, help="output folder (default qa/robustness/runs/<timestamp>)")
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    ap.add_argument("--verdicts", default=os.path.join(HERE, "verdicts.csv"))
    ap.add_argument("--jobs", type=int, default=2, help="images in parallel (default 2)")
    ap.add_argument("--threads", type=int, default=1, help="FEELABLE_THREADS for each child (default 1)")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--rss-limit-mb", type=float, default=0, help="kill + mark 'oom' above this RSS (0 = off; 512 emulates the server)")
    ap.add_argument("--paper", default="a4")
    ap.add_argument("--only", default="", help="comma-separated ids")
    ap.add_argument("--category", default="", help="comma-separated categories")
    ap.add_argument("--smoke", action="store_true", help="only ids listed in qa/robustness/smoke.txt")
    ap.add_argument("--merge", action="store_true", help="replace these ids inside an existing results.json in --out instead of starting fresh")
    ap.add_argument("--report-only", action="store_true", help="re-merge verdicts + rebuild report from existing results.json")
    args = ap.parse_args()

    if args.child:
        return child_main(args.child[0], args.child[1], os.path.abspath(args.repo), args.paper)

    args.repo = os.path.abspath(args.repo)
    args.out = os.path.abspath(args.out or os.path.join(HERE, "runs", time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(args.out, exist_ok=True)
    verdicts = load_verdicts(args.verdicts)

    if args.report_only:
        with open(os.path.join(args.out, "results.json"), encoding="utf-8") as f:
            d = json.load(f)
        man = {r["id"]: r for r in load_manifest(args.manifest)}
        for r in d["results"]:  # recompute junk signals so heuristic tweaks don't need a re-run
            jd = os.path.join(args.out, "jobs", r["id"])
            if os.path.isdir(jd):
                _ink(r, jd)
            _junk_signals(r, man.get(r["id"], {"id": r["id"], "category": r["category"]}))
        write_outputs(d["results"], args.out, verdicts, d["meta"])
        print("report:", os.path.join(args.out, "report.html"))
        return 0

    if not os.path.exists(os.path.join(args.repo, "backend", "engine", "pipeline.py")):
        print("no backend/engine/pipeline.py under", args.repo, file=sys.stderr)
        return 2
    rows = load_manifest(args.manifest)
    if args.only == "smoke":  # `--only smoke` == --smoke
        args.smoke, args.only = True, ""
    if args.smoke:
        with open(os.path.join(HERE, "smoke.txt")) as f:
            keep = {l.split("#")[0].strip() for l in f if l.split("#")[0].strip()}
        rows = [r for r in rows if r["id"] in keep]
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    if args.category:
        keep = set(args.category.split(","))
        rows = [r for r in rows if r["category"] in keep]
    try:
        commit = subprocess.run(["git", "-C", args.repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "?"
    meta = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "repo": args.repo, "commit": commit, "n": len(rows),
            "threads": args.threads, "jobs": args.jobs, "timeout": args.timeout, "rss_limit_mb": args.rss_limit_mb}
    print(f"running {len(rows)} images from {args.repo} @ {commit} -> {args.out}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max(1, args.jobs)) as ex:
        results = list(ex.map(lambda r: run_one(r, args), rows))
    meta["total_s"] = round(time.time() - t0, 1)
    prev = os.path.join(args.out, "results.json")
    if args.merge and os.path.exists(prev):
        with open(prev, encoding="utf-8") as f:
            old = json.load(f)
        new_ids = {r["id"] for r in results}
        results = [r for r in old["results"] if r["id"] not in new_ids] + results
        meta = dict(old["meta"], merged=old["meta"].get("merged", []) + [{"when": meta["when"], "commit": commit, "ids": sorted(new_ids)}])
    write_outputs(results, args.out, verdicts, meta)
    from collections import Counter
    print("status:", dict(Counter(r["status"] for r in results)), f"in {meta['total_s']}s")
    print("report:", os.path.join(args.out, "report.html"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
