"""Pipeline orchestration: analyze -> render -> export, plus re-render after teacher edits.

OWNER: API team.  Engine modules are imported lazily (inside functions) so the server
starts even if one of them is temporarily broken.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import pickle
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

STATE_FILE = "analysis.pkl"
RESULT_FILE = "result.json"

# Text limits (CONTRACT-v2 §2.3). Over-limit input is rejected with a clear message, never cut.
NAME_MAX = 60
DESCRIPTION_MAX = 400
TITLE_MAX = 80
RETENTION_HOURS = 24
MAX_COMFORTABLE_PARTS = 8

# files the async machinery keeps in a job dir (mirrors app/storage.py)
PROGRESS_FILE = "progress.json"
CANCEL_FILE = "cancel"
# never copied when a cached sample is cloned: each clone gets its own token and state
_NO_CLONE = {STATE_FILE, RESULT_FILE, ".edit_token", "status.json", PROGRESS_FILE, CANCEL_FILE, "upload.bin", "published.json", "pub", "tts"}

# export_all keys -> default basenames (used if export omits a key)
FILE_KEYS = {
    "pdf": "tactile.pdf",
    "stl": "relief.stl",
    "png": "tactile.png",
    "print_png": "tactile_print.png",
    "original": "original.png",
    "legend_png": "legend.png",
}


class BadImage(ValueError):
    """The uploaded bytes could not be decoded as an image."""


class JobCancelled(Exception):
    """The teacher cancelled the job (cancel marker seen at a stage boundary)."""


class Conflict(ValueError):
    """PATCH base_version doesn't match the job's current version (HTTP 409)."""


class NotSupported(Exception):
    """An edit needs an engine helper that isn't available yet (HTTP 501)."""


# ---------------------------------------------------------------- locks
_locks_guard = threading.Lock()
_job_locks: dict[str, threading.Lock] = {}


def job_lock(job_dir: str) -> threading.Lock:
    key = os.path.abspath(job_dir)
    with _locks_guard:
        lock = _job_locks.get(key)
        if lock is None:
            lock = _job_locks[key] = threading.Lock()
        return lock


def _mod(name: str):
    return importlib.import_module(f"engine.{name}")


# ---------------------------------------------------------------- JSON helpers
def _to_int(v) -> int:
    return int(round(float(v)))


def _point(p) -> list[int]:
    a = np.asarray(p).reshape(-1)
    return [_to_int(a[0]), _to_int(a[1])]


def _polygons(polys) -> list[list[list[int]]]:
    out: list[list[list[int]]] = []
    for poly in polys or []:
        a = np.asarray(poly)
        if a.size < 2:
            continue
        a = a.reshape(-1, 2)
        out.append([[_to_int(x), _to_int(y)] for x, y in a])
    return out


def _clean(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and tuples to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _check_dict(c) -> dict:
    if isinstance(c, dict):
        return {"rule": str(c.get("rule", "")), "passed": bool(c.get("passed", False)),
                "detail": str(c.get("detail", ""))}
    return {"rule": str(c.rule), "passed": bool(c.passed), "detail": str(c.detail)}


def _file_urls(job_id: str, files: dict, version: int) -> dict[str, str]:
    q = f"?v={version}" if version else ""
    out = {}
    for key, default in FILE_KEYS.items():
        path = files.get(key) if files else None
        base = os.path.basename(str(path)) if path else default
        out[key] = f"/files/{job_id}/{base}{q}"
    return out


def expires_at(created_at: str) -> str:
    try:
        t = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        t = datetime.now(timezone.utc)
    return (t + timedelta(hours=RETENTION_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")


_UNNAMED_RE = re.compile(r"^part\s*\d+$", re.I)
_PHOTO_KINDS = ("photo_object", "photo_scene")
_NO_DIAGRAM_KINDS = ("text_page", "blank")


def _part_fields(p) -> dict:
    lt = getattr(p, "label_text", None)
    src = getattr(p, "description_source", "auto") or "auto"
    return {
        "label_text": str(lt) if lt else None,
        "match_method": str(getattr(p, "match_method", "none") or "none"),
        "description_source": src if src in ("auto", "ai", "teacher") else "auto",
        "target_type": str(getattr(p, "target_type", "region") or "region"),
        "extent": str(getattr(p, "extent", "full") or "full"),
    }


def build_readiness(analysis, checks: list[dict], approved: bool, ai_error: str | None) -> dict:
    """CONTRACT-v2 §2.7 readiness. `checks` are JSON check dicts (so quick edits can recompute it)."""
    issues: list[dict] = []
    parts = list(analysis.parts or [])
    inc = [p for p in parts if p.include]
    kind = str(getattr(analysis, "input_kind", "diagram") or "diagram")
    if kind in _NO_DIAGRAM_KINDS:
        what = "a blank page" if kind == "blank" else "a page of text"
        issues.append({"code": "no_diagram", "severity": "block", "action": "retake",
                       "message": f"This looks like {what}, not a diagram. Take a photo of the diagram itself "
                                  "and try again."})
    for it in ((getattr(analysis, "trace", None) or {}).get("issues") or []):   # F06 / F13 reading checks
        if isinstance(it, dict) and it.get("code") and it.get("message"):
            issues.append({"code": str(it["code"]), "severity": str(it.get("severity", "warn")),
                           "action": str(it.get("action", "retake")), "message": str(it["message"])})
    if not inc and not (kind in _NO_DIAGRAM_KINDS and not parts):
        issues.append({"code": "no_parts", "severity": "block",
                       "action": "include" if parts else "retake",
                       "message": "No parts are switched on, so there is nothing to feel. "
                                  + ("Switch at least one part on." if parts else "Try a clearer photo of the diagram.")})
    from engine.vision_utils import looks_like_structure
    inc_names = {str(p.name).strip().lower() for p in inc}

    def label_box(name: str):
        for lb in (getattr(analysis, "labels", None) or []):
            try:
                txt = re.sub(r"[^a-z0-9 ]+", " ", str(lb.text).lower())
                if " ".join(txt.split()) == name.lower():
                    return [int(v) for v in lb.bbox]
            except Exception:  # pragma: no cover
                continue
        return None

    def target_box(name: str):
        """F08: where the label's pointer ends (the structure), not where the text is."""
        lb_box = label_box(name)
        W_, H_ = int(getattr(analysis, "width", 0) or 0), int(getattr(analysis, "height", 0) or 0)
        if not W_ or not H_:
            return None
        norm = lambda t: " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(t).lower()).split())
        nm = norm(name)
        for L in ((getattr(analysis, "trace", None) or {}).get("leaders") or []):
            ll = norm(L.get("label", ""))
            pts = L.get("points") or []
            if not ll or len(pts) < 2 or not (nm == ll or nm.startswith(ll + " ")):
                continue
            a, b = pts[0], pts[-1]
            if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 < 0.03 * (W_ ** 2 + H_ ** 2) ** 0.5:
                return None          # a stub (e.g. stopped at a bracket): its end is not the structure
            if lb_box:
                cx, cy = (lb_box[0] + lb_box[2]) / 2, (lb_box[1] + lb_box[3]) / 2
                end = a if (a[0] - cx) ** 2 + (a[1] - cy) ** 2 > (b[0] - cx) ** 2 + (b[1] - cy) ** 2 else b
            else:
                end = b
            r = max(30, int(0.06 * max(W_, H_)))
            return [max(0, int(end[0]) - r), max(0, int(end[1]) - r), min(W_, int(end[0]) + r), min(H_, int(end[1]) + r)]
        return None

    from engine.vision_utils import group_members_present
    inc_name_list = [str(p.name) for p in inc]
    groups_seen: set[str] = set()

    def group_info(nm: str) -> bool:
        """A group heading whose members are parts: say so (info), don't ask to draw it."""
        members = group_members_present(nm, inc_name_list)
        if len(members) < 2 or nm.lower() in groups_seen:
            return False
        groups_seen.add(nm.lower())
        issues.append({"code": "group_label", "severity": "info",
                       "message": f"“{nm[:1].upper()}{nm[1:]}” is a group heading for "
                                  f"{', '.join(members[:6])}{' …' if len(members) > 6 else ''}."})
        return True

    # 1. required targets the engine left out (UX07: most important first, one specific repair)
    missing: list[dict] = []
    for d in (getattr(analysis, "dropped", None) or []):
        m = re.match(r"^(.+?) \(too small to feel\)$", str(d))
        if not m or re.match(r"^(\d+ |lots of )", m.group(1)):
            continue                          # "285 small dots", "lots of tiny specks": clutter, fine
        nm = m.group(1)
        if nm.lower() in inc_names or not looks_like_structure(nm) or group_info(nm):
            continue                          # already added by the teacher / a caption fragment / a group
        missing.append({"code": "dropped_label", "severity": "warn", "action": "draw", "target": nm,
                        "bbox": label_box(nm), "target_bbox": target_box(nm),
                        "message": f"{nm[:1].upper()}{nm[1:]} has no matching outline. It's labelled in the "
                                   "picture but drawn too small to feel."})
    # 2. uncertain names
    names: list[dict] = []
    for p in inc:
        if _UNNAMED_RE.match(str(p.name).strip()):
            names.append({"code": "unnamed", "severity": "warn", "action": "rename", "part_id": str(p.id),
                          "message": f"“{p.name}” has no name yet. Give it a name a student will understand."})
    for p in inc:
        if str(getattr(p, "extent", "full")) == "unresolved":
            what = {"boundary": "its course is not traced yet", "path": "its course is not traced yet"}.get(
                str(getattr(p, "target_type", "region")), "its outline is not traced yet")
            nm = str(p.name)
            boundary = str(getattr(p, "target_type", "region")) == "boundary"
            names.append({"code": "extent_unresolved", "severity": "warn", "action": "trace" if boundary else "outline",
                          "part_id": str(p.id), "target": nm, "target_bbox": [int(v) for v in p.bbox],
                          "message": f"{nm[:1].upper()}{nm[1:]}: location found, {what}. "
                                     + (" Trace the border, or draw it." if boundary else " Suggest an outline or draw it.")})
    unmatched = []
    for lb in (getattr(analysis, "unmatched_labels", None) or []):
        lb = str(lb)
        if lb.lower() in inc_names or not looks_like_structure(lb) or lb in unmatched or group_info(lb):
            continue                          # explanatory text, duplicates, already a part, or a group
        unmatched.append(lb)
        names.append({"code": "unmatched_label", "severity": "warn", "action": "draw", "target": lb,
                      "bbox": label_box(lb), "target_bbox": target_box(lb),
                      "message": f"{lb[:1].upper()}{lb[1:]} has no matching outline."})
    # 3. layout suggestions (crowding reported once, not twice)
    layout: list[dict] = []
    if len(inc) > MAX_COMFORTABLE_PARTS:
        layout.append({"code": "too_many", "severity": "warn", "action": "reduce",
                       "message": f"The full image has {len(inc)} targets. Create detail pages (close-ups) to "
                                  "make them easier to explore, or choose the parts for this lesson."})
    for c in checks:
        if not c.get("passed"):
            if len(inc) > MAX_COMFORTABLE_PARTS and "parts or fewer" in str(c.get("rule", "")):
                continue
            detail = f": {c.get('detail')}" if c.get("detail") else ""
            layout.append({"code": "geometry", "severity": "warn",
                           "message": f"Print layout check: {c.get('rule', '')}{detail}"})
    issues += missing + names + layout
    if ai_error:
        issues.append({"code": "ai_failed", "severity": "warn",
                       "message": "Descriptions could not be added. Your diagram is saved. Retry descriptions "
                                  "or continue.", "detail": str(ai_error)})
    if kind in _PHOTO_KINDS:
        issues.append({"code": "photo", "severity": "info",
                       "message": "Simplified from a photo — check the outline matches what you want to teach."})
    # F02: characters the Braille code cannot represent are a publish blocker, never a silent omission.
    from engine import braille as _braille
    for p in inc:
        bad = _braille.unsupported_chars(str(p.name))
        if bad:
            shown = " ".join(sorted(bad)[:8])
            issues.append({"code": "braille_unsupported", "severity": "block", "action": "rename",
                           "part_id": str(p.id),
                           "message": f"Feelable cannot currently translate {shown} in “{p.name}”. "
                                      "Rename it using plain letters and numbers."})
    # P1 (precision audit): the raised title is Braille too - unsupported symbols there must block as
    # well, never be dropped silently (H₂O would become "HO")
    tbad = _braille.unsupported_chars(str(getattr(analysis, "title", "") or ""))
    if tbad:
        shown = " ".join(sorted(tbad)[:8])
        issues.append({"code": "braille_unsupported_title", "severity": "block", "action": "retitle",
                       "part_id": None,
                       "message": f"Feelable cannot currently translate {shown} in the title "
                                  f"“{analysis.title}”. Edit the title using plain letters and numbers "
                                  "(for example H2O for H₂O)."})
    # F03: braille text that could not be printed in full blocks the lesson (never publish a cut key)
    _OVF = _braille.OVERFLOW_PREFIX
    for w in (getattr(analysis, "warnings", None) or []):
        if str(w).startswith(_OVF):
            issues.append({"code": "braille_overflow", "severity": "block", "action": "shorten",
                           "part_id": None,
                           "message": str(w)[len(_OVF):]})
    blocked = any(i["severity"] == "block" for i in issues)
    open_warnings = sum(1 for i in issues if i["severity"] == "warn")
    # F04: approval records the teacher's decision; "Ready to teach" needs no open checks as well.
    status = "blocked" if blocked else ("ready" if approved and open_warnings == 0 else "needs_review")
    return {
        "status": status,
        "approved": bool(approved),
        # TS-12: layout checks are meaningless with nothing to check
        "geometry": ({"passed": sum(1 for c in checks if c.get("passed")), "total": len(checks)}
                     if inc else {"passed": 0, "total": 0}),
        "issues": issues,
        "unmatched_labels": unmatched,
    }


def _stl_size(files: dict | None, tactile) -> list[float] | None:
    path = (files or {}).get("png")
    if not path:
        return None
    try:
        fn = getattr(_mod("export"), "stl_size_mm", None)
        if not callable(fn):
            return None
        v = fn(str(path), float(tactile.px_per_mm))
        if v is None:
            return None
        v = [round(float(x), 1) for x in list(v)[:3]]
        return v if len(v) == 3 else None
    except Exception:
        return None


def build_result(job_id: str, created_at: str, analysis, tactile, files: dict, *,
                 version: int = 0, timings: dict | None = None, approved: bool = False,
                 ai_error: str | None = None) -> dict:
    """Build the JobResult dict (CONTRACT §4 + v2 §2.7). `files` = export_all() output (abs paths)."""
    urls = _file_urls(job_id, files, version)
    textures = getattr(tactile, "textures", None) or {}
    part_polys = getattr(tactile, "part_polygons", None) or {}
    key_pos = getattr(tactile, "key_positions", None) or {}

    parts_sorted = sorted(enumerate(analysis.parts), key=lambda ip: (-int(ip[1].area), ip[0]))
    parts = []
    for _, p in parts_sorted:
        inc = bool(p.include)
        kp = key_pos.get(p.id) if inc else None
        parts.append({
            "id": str(p.id),
            "key": str(p.key or ""),
            "name": str(p.name),
            "description": str(p.description),
            "include": inc,
            "color": [_to_int(c) for c in list(p.color)[:3]],
            "texture": str(textures.get(p.id, "none") or "none"),
            "count": int(p.count),
            "area": int(p.area),
            "source_bbox": [_to_int(v) for v in list(p.bbox)[:4]],
            "source_polygons": _polygons(p.polygons),
            "tactile_polygons": _polygons(part_polys.get(p.id, [])) if inc else [],
            "key_position": _point(kp) if kp is not None else None,
            **_part_fields(p),
        })

    checks = [_check_dict(c) for c in (getattr(tactile, "checks", None) or [])]
    t = {k: round(float(v), 3) for k, v in (timings or {}).items()}
    kind = str(getattr(analysis, "input_kind", "diagram") or "diagram")
    return {
        "id": job_id,
        "created_at": created_at,
        "version": int(version),
        "title": str(analysis.title),
        "diagram_type": str(analysis.diagram_type),
        "input_kind": kind,
        "ai_used": bool(analysis.ai_used),
        "page_flattened": bool(analysis.page_flattened),
        "source": {"width": int(analysis.width), "height": int(analysis.height), "url": urls["original"]},
        "tactile": {
            "width": int(tactile.width), "height": int(tactile.height),
            "px_per_mm": float(tactile.px_per_mm),
            "url": urls["png"], "print_url": urls["print_png"], "legend_url": urls["legend_png"],
            "stl_size_mm": _stl_size(files, tactile),
        },
        "parts": parts,
        "dropped": [str(s) for s in (analysis.dropped or [])],
        "warnings": [str(s) for s in (analysis.warnings or [])],
        "notes": [str(s) for s in (getattr(analysis, "notes", None) or [])],
        "checks": checks,
        "readiness": build_readiness(analysis, checks, approved, ai_error),
        "retention_hours": RETENTION_HOURS,
        "expires_at": expires_at(created_at),
        "timings": _clean(t),
        "files": urls,
    }


# ---------------------------------------------------------------- persistence
def _write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


PUBLISHED_FILE = "published.json"


def _save(job_dir: str, state: dict, result: dict) -> None:
    _write_atomic(os.path.join(job_dir, STATE_FILE), pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    body = json.dumps(result, ensure_ascii=False).encode("utf-8")
    _write_atomic(os.path.join(job_dir, RESULT_FILE), body)
    # TS-06 / UX01: students only ever see the last teacher-approved snapshot. Its JSON *and all
    # its media* are frozen together in an immutable bundle (pub/v<N>/), so later drafts (which
    # rewrite tactile.png etc. in the job folder) can't change what students touch.
    # F04: a teacher's approval publishes the reviewed snapshot even while checks remain open
    # (status "needs_review" then says so to students); blocked lessons are never published.
    rd = result.get("readiness") or {}
    if rd.get("approved") and rd.get("status") != "blocked":
        _publish_bundle(job_dir, result)


PUB_DIR = "pub"
PUB_FILES = ("original.png", "tactile.png", "tactile_print.png", "legend.png", "tactile.pdf", "relief.stl")


def _publish_bundle(job_dir: str, result: dict) -> None:
    """Copy the current media into pub/v<version>/ (written to a temp dir, then renamed, so a
    half-written bundle is never visible), point a copy of the result at it, then swap
    published.json. Failures leave the previous bundle and published.json untouched."""
    import shutil
    ver = int(result.get("version") or 0)
    jid = str(result.get("id") or os.path.basename(job_dir.rstrip("/")))
    pub_root = os.path.join(job_dir, PUB_DIR)
    os.makedirs(pub_root, exist_ok=True)
    final = os.path.join(pub_root, f"v{ver}")
    if not os.path.isdir(final):
        tmp = final + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for name in PUB_FILES:
            src = os.path.join(job_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp, name))
        os.replace(tmp, final)
    pub = json.loads(json.dumps(result))
    old_prefix = f"/files/{jid}/"
    new_prefix = f"/files/{jid}/{PUB_DIR}/v{ver}/"

    def rew(u):
        if isinstance(u, str) and u.startswith(old_prefix):
            return new_prefix + u[len(old_prefix):].split("?", 1)[0]
        return u
    pub["files"] = {k: rew(v) for k, v in (pub.get("files") or {}).items()}
    if isinstance(pub.get("source"), dict):
        pub["source"]["url"] = rew(pub["source"].get("url"))
    if isinstance(pub.get("tactile"), dict):
        for k in ("url", "print_url", "legend_url"):
            pub["tactile"][k] = rew(pub["tactile"].get(k))
    pub["published_version"] = ver
    _write_atomic(os.path.join(job_dir, PUBLISHED_FILE), json.dumps(pub, ensure_ascii=False).encode("utf-8"))
    # keep the two newest bundles (an open student page may still be loading the previous one)
    vers = sorted((int(d[1:]) for d in os.listdir(pub_root) if d.startswith("v") and d[1:].isdigit()), reverse=True)
    for old in vers[2:]:
        shutil.rmtree(os.path.join(pub_root, f"v{old}"), ignore_errors=True)


def ensure_pub_stl(job_dir: str, ver: int) -> str:
    """Lazy STL inside a published bundle, built from that bundle's own tactile.png."""
    pub = load_published(job_dir) or {}
    ppm = float(((pub.get("tactile") or {}).get("px_per_mm")) or 5.0)
    bdir = os.path.join(job_dir, PUB_DIR, f"v{int(ver)}")
    return _mod("export").ensure_stl(bdir, ppm)


def load_published(job_dir: str) -> dict | None:
    try:
        with open(os.path.join(job_dir, PUBLISHED_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _load_state(job_dir: str) -> dict:
    with open(os.path.join(job_dir, STATE_FILE), "rb") as f:
        return pickle.load(f)


def _load_result(job_dir: str) -> dict:
    with open(os.path.join(job_dir, RESULT_FILE), encoding="utf-8") as f:
        return json.load(f)


def _emit(progress, stage: str) -> None:
    if progress is not None:
        progress(stage)


def file_progress(job_dir: str):
    """Progress callback for async jobs: writes progress.json (read by GET /status) and stops the job
    at the next stage boundary if the teacher cancelled it."""
    path = os.path.join(job_dir, PROGRESS_FILE)
    cancel = os.path.join(job_dir, CANCEL_FILE)

    def cb(stage: str) -> None:
        if os.path.exists(cancel):
            raise JobCancelled()
        try:
            _write_atomic(path, json.dumps({"stage": str(stage), "at": time.time()}).encode("utf-8"))
        except OSError:  # pragma: no cover - progress is best effort
            pass
    return cb


def _accepts_kw(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _render_and_export(analysis, options, job_dir: str, progress=None) -> tuple[Any, dict, float, float]:
    tactile_mod = _mod("tactile")
    export_mod = _mod("export")
    _emit(progress, "layout")
    tactile_mod.assign_keys(analysis.parts)
    t0 = time.perf_counter()
    tactile = tactile_mod.render_tactile(analysis, options)
    t1 = time.perf_counter()
    _emit(progress, "files")
    files = export_mod.export_all(analysis, tactile, job_dir)
    t2 = time.perf_counter()
    return tactile, files, t1 - t0, t2 - t1


def _ai_will_run(use_ai: bool) -> bool:
    if not use_ai:
        return False
    try:
        return bool(_mod("ai").get_provider_info().get("provider"))
    except Exception:
        return False


def _ai_last_error() -> str | None:
    try:
        fn = getattr(_mod("ai"), "last_error", None)
        msg = fn() if callable(fn) else None
        return str(msg)[:200] if msg else None
    except Exception:
        return None


# ---------------------------------------------------------------- public API
def run_job(image_bytes: bytes, job_dir: str, *, use_ai: bool = True, paper: str = "a4",
            progress=None) -> dict:
    """Full pipeline for a new upload. Returns JobResult dict; saves analysis.pkl + result.json.

    `progress`: optional Callable[[str], None] called with "read" (and, from the engine, "find"/"ai"),
    then "layout" and "files". It may raise JobCancelled to stop the job."""
    from engine.types import TactileOptions

    if paper not in ("a4", "letter"):
        raise ValueError("paper must be 'a4' or 'letter'")
    os.makedirs(job_dir, exist_ok=True)
    job_id = os.path.basename(os.path.normpath(job_dir))
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    t_start = time.perf_counter()
    _emit(progress, "read")
    analyze_mod = _mod("analyze")
    image_notes: list[str] = []
    try:
        if _accepts_kw(analyze_mod.load_image, "notes"):
            image = analyze_mod.load_image(image_bytes, notes=image_notes)
        else:
            image = analyze_mod.load_image(image_bytes)
    except ValueError as e:
        raise BadImage(str(e) or "Not an image") from e

    with job_lock(job_dir):
        t0 = time.perf_counter()
        if progress is not None and _accepts_kw(analyze_mod.analyze, "progress"):
            analysis = analyze_mod.analyze(image, use_ai=use_ai, progress=progress)
        else:
            analysis = analyze_mod.analyze(image, use_ai=use_ai)
        del image
        if image_notes:   # F10: decode notes (incomplete file, animation) are shown in review
            analysis.warnings = list(analysis.warnings or []) + image_notes
        measured = time.perf_counter() - t0
        t_analyze = float((analysis.timings or {}).get("analyze", measured))
        ai_error = None
        if _ai_will_run(use_ai) and not analysis.ai_used:
            ai_error = _ai_last_error() or "the AI service didn't answer"

        options = TactileOptions(paper=paper)
        tactile, files, t_render, t_export = _render_and_export(analysis, options, job_dir, progress)
        timings = {"analyze": t_analyze, "render": t_render, "export": t_export,
                   "total": time.perf_counter() - t_start}
        result = build_result(job_id, created_at, analysis, tactile, files, version=0, timings=timings,
                              approved=False, ai_error=ai_error)
        state = {"job_id": job_id, "created_at": created_at, "analysis": analysis,
                 "options": options, "version": 0, "use_ai": use_ai, "approved": False,
                 "ai_error": ai_error}
        _save(job_dir, state, result)
    return result


def run_job_file(upload_path: str, job_dir: str, *, use_ai: bool = True, paper: str = "a4") -> dict:
    """Async entry point (runs in the warm worker): read the queued upload from disk and run the job,
    reporting stages through progress.json."""
    with open(upload_path, "rb") as f:
        data = f.read()
    return run_job(data, job_dir, use_ai=use_ai, paper=paper, progress=file_progress(job_dir))


def _capitalise(s: str) -> str:
    return s[:1].upper() + s[1:]


def _too_long(what: str, n: int, limit: int) -> ValueError:
    return ValueError(f"{what} is too long ({n} characters; the limit is {limit}). Please shorten it.")


def _validate_edit(edit: dict, analysis) -> tuple[str | None, list[dict], dict]:
    """Validate a JobEdit; returns (title or None, normalised part edits, ops). Raises ValueError.

    ops = {"split": id|None, "merge": [ids]|None, "approved": bool|None, "base_version": int|None}.
    Whitespace is tidied; text over a limit is rejected, never shortened."""
    if not isinstance(edit, dict):
        raise ValueError("Edit must be a JSON object")
    # unknown top-level / per-part fields are ignored (forward compatible)
    title = None
    if "title" in edit and edit["title"] is not None:
        if not isinstance(edit["title"], str):
            raise ValueError("title must be text")
        title = " ".join(edit["title"].split())
        if not title:
            raise ValueError("title can't be empty")
        if len(title) > TITLE_MAX:
            raise _too_long("The title", len(title), TITLE_MAX)

    part_ids = {p.id for p in analysis.parts}
    norm: list[dict] = []
    raw_parts = edit.get("parts") or []
    if not isinstance(raw_parts, list):
        raise ValueError("parts must be a list")
    for pe in raw_parts:
        if not isinstance(pe, dict) or not isinstance(pe.get("id"), str):
            raise ValueError("Each part edit needs an id")
        if pe["id"] not in part_ids:
            raise ValueError(f"Unknown part id: {pe['id'][:20]}")
        n: dict = {"id": pe["id"]}
        if pe.get("name") is not None:
            if not isinstance(pe["name"], str):
                raise ValueError("name must be text")
            name = " ".join(pe["name"].split())
            if not name:
                raise ValueError("A part name can't be empty")
            if len(name) > NAME_MAX:
                raise _too_long("A part name", len(name), NAME_MAX)
            n["name"] = name
        if pe.get("description") is not None:
            if not isinstance(pe["description"], str):
                raise ValueError("description must be text")
            desc = pe["description"].strip()
            if len(desc) > DESCRIPTION_MAX:
                raise _too_long("A description", len(desc), DESCRIPTION_MAX)
            n["description"] = desc
        if pe.get("include") is not None:
            if not isinstance(pe["include"], bool):
                raise ValueError("include must be true or false")
            n["include"] = pe["include"]
        norm.append(n)

    ops: dict = {"split": None, "merge": None, "approved": None, "base_version": None, "add_part": None}
    if edit.get("split") is not None:
        if not isinstance(edit["split"], str) or edit["split"] not in part_ids:
            raise ValueError("split must be the id of a part")
        ops["split"] = edit["split"]
    if edit.get("merge") is not None:
        m = edit["merge"]
        if not isinstance(m, list) or not all(isinstance(x, str) for x in m):
            raise ValueError("merge must be a list of part ids")
        m = list(dict.fromkeys(m))
        if len(m) < 2:
            raise ValueError("Choose at least two parts to merge")
        bad = [x for x in m if x not in part_ids]
        if bad:
            raise ValueError(f"Unknown part id: {bad[0][:20]}")
        ops["merge"] = m
    if edit.get("trace_boundary") is not None:
        tb = edit["trace_boundary"]
        if not isinstance(tb, str) or tb not in part_ids:
            raise ValueError("trace_boundary must be a part id")
        ops["trace_boundary"] = tb
    if edit.get("add_part") is not None:
        ap = edit["add_part"]
        if not isinstance(ap, dict):
            raise ValueError("add_part must be an object with a name and a polygon")
        name = " ".join(str(ap.get("name") or "").split())
        into = ap.get("into")
        if into is not None and not isinstance(into, str):
            raise ValueError("into must be a part id")
        if not name and not into:
            raise ValueError("Give the new part a name")
        if len(name) > NAME_MAX:
            raise ValueError(f"A name is too long ({len(name)} characters; the limit is {NAME_MAX}). Please shorten it.")
        poly = ap.get("polygon")
        if (not isinstance(poly, list) or not 3 <= len(poly) <= 400
                or not all(isinstance(q, (list, tuple)) and len(q) == 2
                           and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in q) for q in poly)):
            raise ValueError("Draw at least 3 points around the missing part")
        overlap = ap.get("overlap", "cut")
        mode = ap.get("mode", "add")
        within = ap.get("within")
        if within is not None and (not isinstance(within, str) or into):
            raise ValueError("within must be a part id (and can't be combined with into)")
        if overlap not in ("cut", "keep", "replace"):
            raise ValueError("overlap must be cut, keep or replace")
        if mode not in ("add", "remove"):
            raise ValueError("mode must be add or remove")
        more = ap.get("more_polygons") or []
        # several pieces of one part (both kidneys, a cell's many mitochondria): every piece kept
        if (not isinstance(more, list) or len(more) > 12
                or not all(isinstance(pg, list) and 3 <= len(pg) <= 400
                           and all(isinstance(q, (list, tuple)) and len(q) == 2
                                   and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in q)
                                   for q in pg) for pg in more)):
            raise ValueError("more_polygons must be up to 12 outlines of 3-400 points")
        ops["add_part"] = {"name": name, "polygon": [[float(q[0]), float(q[1])] for q in poly],
                           "more_polygons": [[[float(q[0]), float(q[1])] for q in pg] for pg in more],
                           "overlap": overlap, "into": into, "mode": mode, "within": within}
    if edit.get("approved") is not None:
        if not isinstance(edit["approved"], bool):
            raise ValueError("approved must be true or false")
        ops["approved"] = edit["approved"]
    if edit.get("base_version") is not None:
        bv = edit["base_version"]
        if isinstance(bv, bool) or not isinstance(bv, int):
            raise ValueError("base_version must be a whole number")
        ops["base_version"] = bv
    return title, norm, ops


def _set(obj, attr: str, value) -> None:
    try:
        setattr(obj, attr, value)
    except Exception:  # pragma: no cover - e.g. a slotted/frozen object
        pass


def _check_version(state: dict, edit) -> None:
    if not isinstance(edit, dict):
        return
    bv = edit.get("base_version")
    if bv is None or isinstance(bv, bool) or not isinstance(bv, int):
        return
    cur = int(state.get("version", 0))
    if bv != cur:
        raise Conflict(f"This diagram was changed somewhere else (now version {cur}, you edited version {bv}). "
                       "Reload to see the latest, then make your change again.")


def apply_edit(analysis, edit: dict) -> bool:
    """Validate then apply a JobEdit to an Analysis in place. Returns True if the content changed.

    Order: part edits, then split, then merge. Raises ValueError (nothing is saved by callers) or
    NotSupported when split/merge helpers are missing."""
    title, part_edits, ops = _validate_edit(edit, analysis)
    changed = False
    if title is not None and title != analysis.title:
        analysis.title = title
        changed = True
    by_id = {p.id: p for p in analysis.parts}
    for pe in part_edits:
        p = by_id[pe["id"]]
        if "name" in pe and pe["name"] != p.name:
            old_name, new_name = p.name, pe["name"]
            p.name = new_name
            _set(p, "match_method", "teacher")
            changed = True
            old_cap = _capitalise(old_name)
            if "description" not in pe and old_cap and p.description.startswith(old_cap):
                renamed = _capitalise(new_name) + p.description[len(old_cap):]
                if len(renamed) <= DESCRIPTION_MAX:
                    p.description = renamed
        if "description" in pe and pe["description"] != p.description:
            p.description = pe["description"]
            _set(p, "description_source", "teacher")
            changed = True
        if "include" in pe and pe["include"] != bool(p.include):
            p.include = pe["include"]
            changed = True
    if ops["split"] or ops["merge"]:
        analyze_mod = _mod("analyze")
        if ops["split"]:
            fn = getattr(analyze_mod, "split_part", None)
            if not callable(fn):
                raise NotSupported("Splitting a part isn't available on this server yet")
            fn(analysis, ops["split"])
            changed = True
        if ops["merge"]:
            fn = getattr(analyze_mod, "merge_parts", None)
            if not callable(fn):
                raise NotSupported("Merging parts isn't available on this server yet")
            ids = [i for i in ops["merge"] if i in {p.id for p in analysis.parts}]
            if len(ids) < 2:
                raise ValueError("Choose at least two parts to merge")
            fn(analysis, ids)
            changed = True
    if ops.get("add_part"):
        _add_drawn_part(analysis, ops["add_part"])
        changed = True
    if ops.get("trace_boundary"):
        _trace_boundary_part(analysis, ops["trace_boundary"])
        changed = True
    if analysis.parts and not any(p.include for p in analysis.parts):
        raise ValueError("Keep at least one part switched on")
    return changed


def _trace_boundary_part(analysis, part_id: str) -> None:
    """Boundary tool (precision audit): a suture / fissure marked only at its pointer tip is traced
    along the border shared by the two outlined parts that meet there."""
    import cv2
    import numpy as np
    from engine import grounding
    p = next((q for q in analysis.parts if q.id == part_id), None)
    if p is None or p.mask is None or not p.mask.any():
        raise ValueError("That part no longer exists")
    ys, xs = np.nonzero(p.mask > 0)
    cx, cy = int(xs.mean()), int(ys.mean())
    h, w = p.mask.shape[:2]
    r = max(6, int(round(0.011 * float(np.hypot(h, w)))))
    others = [q for q in analysis.parts if q is not p and q.include]
    m = grounding.trace_boundary(others, cx, cy, reach=4 * r, min_len=6 * r)
    if m is None:
        raise ValueError(f"No two outlined parts meet near “{p.name[:40]}”. Outline the parts on either side "
                         "first, or draw the boundary yourself.")
    grounding._set_mask(p, m)
    p.kind, p.target_type, p.extent = "line", "boundary", "full"
    p.count = max(1, cv2.connectedComponents((m > 0).astype(np.uint8))[0] - 1)
    if getattr(p, "match_method", "") == "landmark":
        p.match_method = "teacher"


def _add_drawn_part(analysis, spec: dict) -> None:
    """TS-09: the teacher outlines a missing structure on the original photo.

    spec: {"name", "polygon"[, "overlap": "cut"|"keep"|"replace"][, "into": part id, "mode": "add"|"remove"]}
    * new part (no "into"): "cut" (default) cuts it out of the regions it was drawn over so it can
      be felt on its own, but refuses to make an existing part vanish; "replace" does the same and
      switches fully covered parts off (an explicit teacher choice); "keep" leaves other parts as
      they are (nested detail).
    * "into": add the area to an existing part (incomplete lungs) or remove it ("mode": "remove")."""
    import cv2
    import numpy as np
    from engine.types import Part
    from engine.vision_utils import mask_bbox, mask_polygons
    h, w = int(analysis.height), int(analysis.width)
    mask = np.zeros((h, w), np.uint8)
    for poly in [spec["polygon"]] + list(spec.get("more_polygons") or []):
        pts = np.array([[min(w - 1, max(0, round(x))), min(h - 1, max(0, round(y)))] for x, y in poly], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    area = int((mask > 0).sum())
    if area < 30:
        raise ValueError("That shape is too small; draw around the whole part")
    mb = mask > 0
    overlap = spec.get("overlap") or "cut"
    into = spec.get("into")
    within = spec.get("within")
    if within:
        # cut a piece out of an existing part along the drawn outline: the piece keeps that
        # part's own edges; the drawing only says where to cut (the bends of one long colon)
        host = next((p for p in analysis.parts if p.id == within), None)
        if host is None or host.mask is None:
            raise ValueError("That part no longer exists")
        hm = host.mask > 0
        piece = hm & mb
        if int(piece.sum()) < 30:
            raise ValueError(f"The outline doesn't overlap “{host.name[:40]}”.")
        rest = hm & ~piece
        if int(rest.sum()) < 30:
            raise ValueError(f"That takes all of “{host.name[:40]}”. Rename it instead.")
        hmm = rest.astype(np.uint8) * 255
        host.mask, host.polygons, host.bbox, host.area = hmm, mask_polygons(hmm, 1.5), mask_bbox(hmm), int(rest.sum())
        host.count = max(1, cv2.connectedComponents(rest.astype(np.uint8), connectivity=8)[0] - 1)
        mask = piece.astype(np.uint8) * 255
        mb, area, overlap = piece, int(piece.sum()), "keep"

    def set_mask(p, m):
        mm = m.astype(np.uint8) * 255
        p.mask, p.polygons, p.bbox, p.area = mm, mask_polygons(mm, 1.5), mask_bbox(mm), int(m.sum())
        p.count = max(1, cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1)

    def cut_others(exclude_id):
        if overlap == "keep":
            return
        for p in analysis.parts:
            if p.mask is None or p.id == exclude_id or not p.include:
                continue
            pm = p.mask > 0
            rest = pm & ~mb
            if int(rest.sum()) == int(pm.sum()):
                continue
            if int(rest.sum()) < 30:
                if overlap != "replace":
                    raise ValueError(f"It covers all of “{p.name[:40]}”. Choose Replace it or Keep both.")
                p.include = False                  # an explicit teacher choice
                continue
            set_mask(p, rest)

    if into:
        tgt = next((p for p in analysis.parts if p.id == into), None)
        if tgt is None or tgt.mask is None:
            raise ValueError("That part no longer exists")
        tm = tgt.mask > 0
        if spec.get("mode") == "remove":
            rest = tm & ~mb
            if int(rest.sum()) < 30:
                raise ValueError(f"That removes all of “{tgt.name[:40]}”. Use Remove part instead.")
            set_mask(tgt, rest)
        else:
            cut_others(tgt.id)
            set_mask(tgt, tm | mb)
        try:
            tgt.match_method = "teacher"
        except Exception:  # pragma: no cover
            pass
        return

    cut_others(None)
    img = getattr(analysis, "image", None)
    col = (np.median(img[mb], axis=0) if img is not None and img.ndim == 3 else np.array([128, 128, 128]))
    ids = {p.id for p in analysis.parts}
    n = len(analysis.parts) + 1
    while f"p{n}" in ids:
        n += 1
    name = spec["name"]
    part = Part(id=f"p{n}", name=name, description=f"{name[:1].upper()}{name[1:]}.", key="",
                color=(int(col[0]), int(col[1]), int(col[2])), mask=mask,
                polygons=mask_polygons(mask, 1.5), bbox=mask_bbox(mask), area=area,
                count=1, include=True, kind="region")
    for attr, val in (("label_text", name), ("match_method", "teacher"), ("description_source", "teacher")):
        try:
            setattr(part, attr, val)
        except Exception:  # pragma: no cover
            pass
    analysis.parts.append(part)


def outline_overlaps(analysis, mask) -> list[dict]:
    """Which included parts a proposed outline would cover, and how much of each."""
    out = []
    mb = mask > 0
    for p in analysis.parts:
        if p.mask is None or not p.include:
            continue
        pm = p.mask > 0
        ov = int((pm & mb).sum())
        if ov:
            out.append({"part_id": str(p.id), "name": str(p.name), "covered": round(ov / max(1, int(pm.sum())), 3)})
    return sorted(out, key=lambda d: -d["covered"])


def find_outline(job_dir: str, req: dict) -> dict:
    """Assisted outline for the teacher's draw tool. req: {"rev": int, "inside": [[x,y]],
    "outside": [[x,y]], "box": [x0,y0,x1,y1] | None, "lasso": [[x,y]] | None, "polygon": [[x,y]] | None}.
    With "polygon" only the overlaps are reported. Coordinates are source (original image) px.
    Never changes the lesson."""
    import cv2
    import numpy as np
    from engine import grounding
    from engine.vision_utils import mask_polygons
    with open(os.path.join(job_dir, STATE_FILE), "rb") as f:
        state = pickle.load(f)
    a = state["analysis"]
    h, w = int(a.height), int(a.width)

    def pts(key, lim=400):
        v = req.get(key) or []
        if not isinstance(v, list) or len(v) > lim:
            raise ValueError(f"{key} must be a list of at most {lim} points")
        out = []
        for q in v:
            if not (isinstance(q, (list, tuple)) and len(q) == 2):
                raise ValueError(f"{key} must contain [x, y] points")
            out.append((min(w - 1, max(0.0, float(q[0]))), min(h - 1, max(0.0, float(q[1])))))
        return out

    rev = int(req.get("rev") or 0)
    poly = pts("polygon")
    if poly:
        if len(poly) < 3:
            raise ValueError("Draw at least 3 points")
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.array([[round(x), round(y)] for x, y in poly], np.int32)], 255)
        return {"rev": rev, "found": True, "polygons": [], "overlaps": outline_overlaps(a, m)}
    inside, outside, lasso = pts("inside", 50), pts("outside", 50), pts("lasso")
    box = req.get("box")
    if box is not None:
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            raise ValueError("box must be [x0, y0, x1, y1]")
        box = tuple(float(v) for v in box)
    if not inside and not lasso and box is None:
        raise ValueError("Tap inside the part, or draw roughly around it")
    m = grounding.propose_outline(a.image, inside, outside, box, lasso or None)
    if m is None:
        return {"rev": rev, "found": False, "polygons": [], "overlaps": [],
                "message": "We couldn't find a clear outline. Keep drawing or try another selection."}
    polys = [[[int(x), int(y)] for x, y in poly] for poly in mask_polygons(m, max(1.5, 0.002 * max(h, w)))]
    polys = sorted(polys, key=len, reverse=True)
    return {"rev": rev, "found": True, "polygons": polys, "area": int((m > 0).sum()),
            "overlaps": outline_overlaps(a, m)}


def _finish_edit(state: dict, edit: dict, changed: bool) -> None:
    """Every successful edit bumps the version; a content change clears approval unless the same
    request explicitly sets `approved` (an explicit value in the request wins)."""
    state["version"] = int(state.get("version", 0)) + 1
    if changed:
        state["approved"] = False
    if isinstance(edit, dict) and isinstance(edit.get("approved"), bool):
        state["approved"] = edit["approved"]


def rerender_job(job_dir: str, edit: dict) -> dict:
    """Apply a JobEdit (no re-analysis), re-render + re-export, update result.json."""
    with job_lock(job_dir):
        state = _load_state(job_dir)
        _check_version(state, edit)
        analysis = state["analysis"]
        changed = apply_edit(analysis, edit)  # ValueError -> 400 in the API
        t_start = time.perf_counter()
        tactile, files, t_render, t_export = _render_and_export(analysis, state["options"], job_dir)
        _finish_edit(state, edit, changed)
        timings = {"analyze": 0.0, "render": t_render, "export": t_export,
                   "total": time.perf_counter() - t_start}
        result = build_result(state["job_id"], state["created_at"], analysis, tactile, files,
                              version=state["version"], timings=timings,
                              approved=bool(state.get("approved")), ai_error=state.get("ai_error"))
        state["analysis"] = analysis
        _save(job_dir, state, result)
    return result


def _refresh_quick(result: dict, analysis, state: dict) -> None:
    """Update the parts of result.json that change without a re-render."""
    by_id = {p.id: p for p in analysis.parts}
    for rp in result.get("parts", []):
        p = by_id.get(rp.get("id"))
        if p is not None:
            rp["description"] = str(p.description)
            rp.update(_part_fields(p))
    result["version"] = int(state.get("version", 0))
    result["ai_used"] = bool(analysis.ai_used)
    result["warnings"] = [str(s) for s in (analysis.warnings or [])]
    result["readiness"] = build_readiness(analysis, result.get("checks", []), bool(state.get("approved")),
                                          state.get("ai_error"))


# ---------------------------------------------------------------- pre-made sample jobs
def clone_job(src_dir: str, dst_dir: str) -> dict:
    """Copy a finished job (e.g. a pre-computed sample) to a new job id. Returns its JobResult.
    The edit token, status and progress files are never copied: each clone gets its own."""
    import shutil

    new_id = os.path.basename(os.path.normpath(dst_dir))
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if name not in _NO_CLONE and not name.endswith(".tmp"):
            shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))
    with open(os.path.join(src_dir, RESULT_FILE), encoding="utf-8") as f:
        result = json.load(f)
    old_id = result["id"]
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = json.loads(json.dumps(result).replace(f"/files/{old_id}/", f"/files/{new_id}/"))
    result["id"], result["created_at"] = new_id, created_at
    if "expires_at" in result or "retention_hours" in result:
        result["expires_at"] = expires_at(created_at)
        result["retention_hours"] = RETENTION_HOURS
    with open(os.path.join(src_dir, STATE_FILE), "rb") as f:
        state = pickle.load(f)
    state["job_id"], state["created_at"] = new_id, created_at
    _save(dst_dir, state, result)
    return result


def quick_edit(job_dir: str, edit: dict) -> dict | None:
    """Apply an edit that doesn't change the drawing (descriptions / approval) without re-rendering.

    Returns the updated JobResult, or None when the edit needs a full re-render (title, a name,
    include, split or merge changed). Raises ValueError / Conflict like rerender_job."""
    with job_lock(job_dir):
        state = _load_state(job_dir)
        _check_version(state, edit)
        analysis = state["analysis"]
        title, norm, ops = _validate_edit(edit, analysis)
        if ops["split"] or ops["merge"] or ops.get("add_part") or ops.get("trace_boundary"):
            return None
        by_id = {p.id: p for p in analysis.parts}
        if title is not None and title != analysis.title:
            return None
        for n in norm:
            p = by_id[n["id"]]
            if ("name" in n and n["name"] != p.name) or ("include" in n and n["include"] != bool(p.include)):
                return None
        result = _load_result(job_dir)
        changed = False
        for n in norm:
            p = by_id[n["id"]]
            if "description" in n and n["description"] != p.description:
                p.description = n["description"]
                _set(p, "description_source", "teacher")
                changed = True
        _finish_edit(state, edit, changed)
        state["analysis"] = analysis
        _refresh_quick(result, analysis, state)
        _save(job_dir, state, result)
    return result


_AI_FAIL_WARNING = "AI description unavailable — used labels from the image"


def _fit_description(desc: str) -> str:
    """AI text is ours to shorten (teacher text never is): keep whole sentences/words under the limit."""
    desc = " ".join(str(desc).split())
    if len(desc) <= DESCRIPTION_MAX:
        return desc
    cut = desc[:DESCRIPTION_MAX]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end > 40:
        return cut[:end + 1]
    return cut[:cut.rfind(" ")].rstrip(",;:") + "." if " " in cut else cut


def ai_describe_job(job_dir: str, overwrite_teacher: bool = False) -> dict:
    """Ask the AI for spoken descriptions of every included part, keeping the drawing, names and keys,
    so no re-render is needed. Teacher-written descriptions are kept unless overwrite_teacher.
    Prefers ai.describe_parts(image, analysis, ids) -> {id: description}; falls back to ai.enrich.
    Returns the JobResult (on AI failure: a warning + an `ai_failed` readiness issue)."""
    ai = _mod("ai")
    with job_lock(job_dir):
        state = _load_state(job_dir)
        analysis = state["analysis"]
        result = _load_result(job_dir)
        targets = [p.id for p in analysis.parts if p.include and
                   (overwrite_teacher or getattr(p, "description_source", "auto") != "teacher")]
        descs: dict | None = {}
        if targets:
            try:
                fn = getattr(ai, "describe_parts", None)
                if callable(fn):
                    descs = fn(analysis.image, analysis, targets)
                else:
                    enr = ai.enrich(analysis.image, analysis)
                    descs = None if enr is None else {
                        it.get("id"): it.get("description") for it in (enr.parts or []) if isinstance(it, dict)}
            except Exception:
                descs = None
        changed = False
        if descs is None:
            state["ai_error"] = _ai_last_error() or "the AI service didn't answer"
            if _AI_FAIL_WARNING not in analysis.warnings:
                analysis.warnings.append(_AI_FAIL_WARNING)
        else:
            state["ai_error"] = None
            if _AI_FAIL_WARNING in analysis.warnings:
                analysis.warnings.remove(_AI_FAIL_WARNING)
            by_id = {p.id: p for p in analysis.parts}
            for pid in targets:
                desc = descs.get(pid) if isinstance(descs, dict) else None
                if isinstance(desc, str) and desc.strip():
                    p = by_id[pid]
                    new = _fit_description(desc)
                    if new != p.description:
                        p.description = new
                        changed = True
                    _set(p, "description_source", "ai")
            analysis.ai_used = True
        state["version"] = int(state.get("version", 0)) + 1
        if changed:
            state["approved"] = False
        state["use_ai"] = True
        state["analysis"] = analysis
        _refresh_quick(result, analysis, state)
        _save(job_dir, state, result)
    return result


# ---------------------------------------------------------------- worker-process helpers
def ensure_stl(job_dir: str) -> str:
    with open(os.path.join(job_dir, RESULT_FILE), encoding="utf-8") as f:
        ppm = float(json.load(f)["tactile"]["px_per_mm"])
    with job_lock(job_dir):
        return _mod("export").ensure_stl(job_dir, ppm)


def warm_worker() -> None:
    """Run once in a fresh worker process: import the engine and load the OCR model up front."""
    n = os.environ.get("FEELABLE_THREADS")
    if n:
        import cv2
        cv2.setNumThreads(int(n))
    for name in ("analyze", "tactile", "export"):
        _mod(name)
    try:
        from engine import ocr
        ocr.warm()
    except Exception:
        pass


def worker_call(fn_name: str, *args, **kwargs):
    """Run a pipeline function in the worker, then hand freed memory back to the OS."""
    import gc
    try:
        return globals()[fn_name](*args, **kwargs)
    finally:
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def vision_trace(job_dir: str) -> dict:
    """What the engine saw in the current draft, for the teacher's "What Feelable saw" view:
    every label it read (with the first reading when it was repaired), the pointer lines it
    followed, which part each label named and how, plus step counts. SOURCE (working) coords."""
    st = _load_state(job_dir)
    a = st["analysis"]
    trace = getattr(a, "trace", None) or {}
    parts = {p.id: p for p in a.parts}
    labels = []
    for lb in a.labels:
        if lb.role == "title":
            role = "title"
        elif lb.part_id in parts:
            role = "named"
        else:
            role = lb.role
        labels.append({"text": lb.text, "raw": getattr(lb, "raw", None), "bbox": list(map(int, lb.bbox)),
                       "part_id": lb.part_id if lb.part_id in parts else None, "role": role,
                       "confidence": round(float(lb.confidence), 2)})
    methods: dict[str, int] = {}
    for p in a.parts:
        if p.include:
            methods[p.match_method or "none"] = methods.get(p.match_method or "none", 0) + 1
    return {"width": int(a.width), "height": int(a.height), "input_kind": a.input_kind,
            "labels": labels, "leaders": trace.get("leaders", []),
            "parts": [{"id": p.id, "name": p.name, "match_method": p.match_method, "include": p.include}
                      for p in a.parts],
            "steps": {**(trace.get("steps") or {}), "methods": methods}}
