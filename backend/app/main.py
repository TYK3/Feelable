"""Feelable HTTP server (FastAPI). OWNER: API team.  Implements CONTRACT.md §4 + CONTRACT-v2.md §2.

Run from backend/:  python3 -m uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import storage
from engine import pipeline

VERSION = "0.1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD_BYTES = int(float(os.environ.get("FEELABLE_MAX_UPLOAD_MB", "15")) * 1024 * 1024)

log = logging.getLogger("feelable")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

FILE_TYPES = {
    ".png": "image/png",
    ".pdf": "application/pdf",
    ".stl": "model/stl",
    ".json": "application/json",
}
SAMPLE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

MSG_NOT_IMAGE = "That file isn't an image we can read (PNG, JPEG or WebP)"
MSG_EMPTY = "That file is empty. Please choose a picture of a diagram."
MSG_NO_FILE = "No file uploaded. Please choose a picture of a diagram."
MSG_DECODE = ("That image looks damaged or uses a format we can't open. "
              "Please choose a different file (PNG, JPEG, WebP or HEIC).")
MSG_TOO_BIG_PIXELS = "That image has too many pixels to process. Try a smaller photo or a screenshot."
MSG_OOM = "The server ran out of memory on this image; try a smaller photo"
MSG_BUSY = "The server is busy; please try again in a minute"
MSG_VIEW_ONLY = "This is a view-only link — open the teacher link to edit."
MSG_EXPIRED = ("This link has expired or the diagram was removed. Links last 24 h on the free demo server "
               "— upload the diagram again to make a new one.")
MSG_AI_OFF = "AI descriptions aren't switched on for this server."
AI_SENDS = ("A downscaled copy of the image and the detected label text is sent to Google Gemini to write "
            "names and descriptions. Nothing is stored by Feelable beyond 24 h.")
MAX_QUEUED = int(os.environ.get("FEELABLE_MAX_QUEUED", "12"))
ALL_STAGES = ("queued", "read", "find", "ai", "layout", "files", "done")


def samples_dir() -> Path:
    return Path(os.environ.get("FEELABLE_SAMPLES") or (REPO_ROOT / "samples"))


def web_dist_dir() -> Path:
    return Path(os.environ.get("FEELABLE_WEB_DIST") or (REPO_ROOT / "web" / "dist"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        n = storage.cleanup_old_jobs()
        if n:
            log.info("removed %d job(s) older than 24 h", n)
    except Exception:  # pragma: no cover
        log.exception("startup cleanup failed")
    try:
        quota = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        quota = "?"
    log.info("host cpus=%s affinity=%s cgroup cpu.max=%s threads=%s", os.cpu_count(),
             len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "?", quota,
             os.environ.get("FEELABLE_THREADS", "default"))
    if os.environ.get("FEELABLE_ISOLATE") == "1":  # start the warm worker in the background
        import threading
        threading.Thread(target=lambda: _safe_warm(), daemon=True).start()
    yield


def _safe_warm():
    try:
        _engine_call("warm_worker")
    except Exception:  # pragma: no cover
        log.exception("worker warm-up failed")


app = FastAPI(title="Feelable", version=VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- errors (CONTRACT-v2 §2.6)
# Every error body is {"detail": "<human message>", "code": "<machine code>"}.
class ApiError(HTTPException):
    def __init__(self, status: int, detail: str, code: str):
        super().__init__(status, detail)
        self.code = code


_DEFAULT_CODES = {400: "invalid", 403: "view_only", 404: "not_found", 405: "invalid", 409: "conflict",
                  413: "too_big", 422: "invalid", 500: "server", 501: "server", 503: "busy"}


@app.exception_handler(StarletteHTTPException)
async def _http_error(_req: Request, exc: StarletteHTTPException):
    code = getattr(exc, "code", None) or _DEFAULT_CODES.get(exc.status_code, "server")
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail, "code": code},
                        headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def _validation_error(_req: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(x) for x in first.get("loc", []) if x not in ("body", "query"))
    msg = first.get("msg", "Invalid request")
    return JSONResponse(status_code=400, content={"detail": f"{msg} ({where})" if where else msg, "code": "invalid"})


@app.exception_handler(Exception)
async def _unhandled(_req: Request, exc: Exception):  # pragma: no cover - last resort
    log.error("unhandled error:\n%s", "".join(traceback.format_exception(exc)))
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server. Please try again.",
                                                  "code": "server"})


# ---------------------------------------------------------------- helpers
def _parse_bool(value: str | bool | None, name: str) -> bool:
    if isinstance(value, bool):
        return value
    v = (value or "true").strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise ApiError(400, f"{name} must be true or false", "invalid")


def _parse_paper(value: str | None) -> str:
    v = (value or "a4").strip().lower()
    if v not in ("a4", "letter"):
        raise ApiError(400, "paper must be a4 or letter", "invalid")
    return v


def _short_reason(e: BaseException) -> str:
    msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
    if not msg or "/" in msg or "\\" in msg or len(msg) > 120:
        return type(e).__name__
    return msg


# ---------------------------------------------------------------- job isolation
# On small hosts (e.g. 512 MB free tiers) set FEELABLE_ISOLATE=1: heavy jobs run in ONE warm
# worker process (OCR model + imports loaded once at startup, memory trimmed after every job,
# recycled every FEELABLE_WORKER_JOBS jobs), so the web process itself stays small.
_POOL = None
_POOL_LOCK = __import__("threading").Lock()


_CANCELLED: set[str] = set()
_FUTURES: dict[str, object] = {}      # cancel token -> Future of the job it started


class JobCancelled(Exception):
    pass


class WorkerOOM(RuntimeError):
    pass


class WorkerBusy(RuntimeError):
    pass


class EngineTimeout(TimeoutError):
    """F05: one engine stage ran past its deadline; the worker was stopped, not left running."""


# F05: a single engine call may not run longer than this (seconds). Fresh uploads normally take
# tens of seconds; a stalled worker must become a recoverable failure instead of a hung job.
ENGINE_DEADLINE_S = float(os.environ.get("FEELABLE_ENGINE_DEADLINE_S", "300"))
MSG_TIMEOUT = ("This diagram took too long to process. Try again, or use a smaller or clearer photo. "
               "Your other lessons are not affected.")


def _new_pool():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(
        max_workers=1, mp_context=mp.get_context("spawn"),
        initializer=pipeline.warm_worker,
        max_tasks_per_child=int(os.environ.get("FEELABLE_WORKER_JOBS", "20")))


def _engine_call(fn_name: str, *args, _token: str | None = None, **kwargs):
    if os.environ.get("FEELABLE_ISOLATE") != "1":
        return getattr(pipeline, fn_name)(*args, **kwargs)
    global _POOL
    from concurrent.futures import CancelledError
    from concurrent.futures import TimeoutError as FuturesTimeout
    from concurrent.futures.process import BrokenProcessPool
    for attempt in range(2):
        with _POOL_LOCK:
            if _token and _token in _CANCELLED:   # cancelled while the upload was still arriving
                _CANCELLED.discard(_token)
                raise JobCancelled()
            if _POOL is None:
                _POOL = _new_pool()
            pool = _POOL
            fut = pool.submit(pipeline.worker_call, fn_name, *args, **kwargs)
            if _token:
                _FUTURES[_token] = fut
        try:
            return fut.result(timeout=ENGINE_DEADLINE_S)
        except CancelledError:
            raise JobCancelled()
        except FuturesTimeout:
            # F05: stop the stuck worker so the queue is released; the next job starts a fresh one
            _kill_pool_for(pool)
            if _token:
                _CANCELLED.discard(_token)
            raise EngineTimeout(MSG_TIMEOUT)
        except BrokenProcessPool:
            with _POOL_LOCK:
                if _POOL is pool:
                    _POOL = None      # worker died: start a fresh pool next time
            if _token and _token in _CANCELLED:
                raise JobCancelled()
            if attempt == 0 and _was_killed_for_cancel(pool):
                continue              # someone else's cancel killed the shared worker: retry
            raise WorkerOOM(MSG_OOM)
        finally:
            if _token:
                _FUTURES.pop(_token, None)
                _CANCELLED.discard(_token)
    raise WorkerBusy(MSG_BUSY)


_KILLED_POOLS: "weakref.WeakSet" = __import__("weakref").WeakSet()


def _kill_pool_for(pool) -> None:
    """Kill the worker process behind `pool` and retire the pool (used for deadlines)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is pool:
            _POOL = None
    _KILLED_POOLS.add(pool)
    for proc in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            proc.kill()
        except Exception:  # pragma: no cover
            pass
    pool.shutdown(wait=False, cancel_futures=True)


def _was_killed_for_cancel(pool) -> bool:
    return pool in _KILLED_POOLS


def _cancel(token: str) -> bool:
    """Stop the job started with this token: drop it from the queue, or kill the worker running it."""
    global _POOL
    with _POOL_LOCK:
        if len(_CANCELLED) > 500:
            _CANCELLED.clear()
        _CANCELLED.add(token)
        fut = _FUTURES.get(token)
        if fut is None:
            return True                       # not started yet: it will be dropped when it arrives
        if fut.cancel():                      # still queued
            return True
        if fut.running() and _POOL is not None:
            pool, _POOL = _POOL, None
            _KILLED_POOLS.add(pool)
            for proc in list((getattr(pool, "_processes", None) or {}).values()):
                try:
                    proc.kill()
                except Exception:  # pragma: no cover
                    pass
            pool.shutdown(wait=False, cancel_futures=False)
            return True
    return False


_SAMPLE_HASHES: dict[str, str] | None = None


def _sample_for_bytes(data: bytes) -> str | None:
    """If an upload is byte-identical to a bundled sample, return that sample's id."""
    global _SAMPLE_HASHES
    if _SAMPLE_HASHES is None:
        import hashlib
        table = {}
        for s in _load_manifest():
            f = samples_dir() / str(s.get("file", ""))
            if f.is_file():
                table[hashlib.sha256(f.read_bytes()).hexdigest()] = str(s["id"])
        _SAMPLE_HASHES = table
    import hashlib
    return _SAMPLE_HASHES.get(hashlib.sha256(data).hexdigest())


def _clone_into(sample_id: str, paper: str, use_ai: bool, jdir: Path, progress=None) -> dict | None:
    """Copy the pre-made result for a sample into an existing job dir (instant). With AI on, the copy
    then gets AI-written descriptions (a few seconds of network time; the drawing doesn't change).
    The job dir already holds its own fresh edit token; clone_job never copies one."""
    cached = _sample_cache_dir(sample_id, paper)
    if cached is None:
        return None
    try:
        if progress:
            progress("files")
        result = pipeline.clone_job(str(cached), str(jdir))
    except (JobCancelled, pipeline.JobCancelled):
        raise
    except Exception:
        log.warning("sample cache for %s unusable; running live", sample_id)
        return None
    if use_ai and _ai_configured():
        try:
            if progress:
                progress("ai")
            result = pipeline.ai_describe_job(str(jdir))
        except (JobCancelled, pipeline.JobCancelled):
            raise
        except Exception:
            log.warning("AI descriptions for cached sample %s failed:\n%s", sample_id, traceback.format_exc())
    _log_job(jdir.name, f"sample {sample_id} (cached{', ai' if result.get('ai_used') else ''})", result)
    return result


def _new_job() -> tuple[str, Path, str]:
    """Create an empty job dir with a fresh teacher edit token. Returns (id, dir, token)."""
    job_id = storage.new_job_id()
    while storage.job_dir_exists(job_id):  # pragma: no cover - astronomically unlikely
        job_id = storage.new_job_id()
    jdir = storage.create_job_dir(job_id)
    tok = storage.save_edit_token(jdir)
    return job_id, jdir, tok


_IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"BM", b"II*\x00", b"MM\x00*")


def _looks_like_image(head: bytes) -> bool:
    """Does the file *claim* to be an image (so a failed decode means it's damaged, not the wrong type)?"""
    if head.startswith(_IMAGE_MAGIC):
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    return head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1",
                                                   b"msf1", b"avif")


def _classify_error(e: BaseException, head: bytes = b"") -> tuple[int, str, str]:
    """Map a job failure to (HTTP status, code, human message). CONTRACT-v2 §2.6."""
    if isinstance(e, (JobCancelled, pipeline.JobCancelled)):
        return 409, "cancelled", "Cancelled"
    if isinstance(e, pipeline.BadImage):
        if "too big" in str(e).lower():
            return 400, "too_big", MSG_TOO_BIG_PIXELS
        if _looks_like_image(head):
            return 400, "decode", MSG_DECODE
        return 400, "not_image", MSG_NOT_IMAGE
    if isinstance(e, (WorkerOOM, MemoryError)):
        return 413, "too_big", MSG_OOM
    if isinstance(e, EngineTimeout):
        return 504, "timeout", MSG_TIMEOUT
    if isinstance(e, (WorkerBusy, TimeoutError, ConnectionError)):
        return 503, "busy", MSG_BUSY
    return 500, "server", f"Processing failed: {_short_reason(e)}"


def _run_new_job(data: bytes, use_ai: bool, paper: str, origin: str, token: str | None = None) -> dict:
    """Synchronous create (also used by from-sample). Returns the JobResult plus `edit_token`."""
    job_id, jdir, edit_token = _new_job()
    try:
        result = None
        if os.environ.get("FEELABLE_SAMPLE_CACHE"):
            sid = _sample_for_bytes(data)
            if sid:
                result = _clone_into(sid, paper, use_ai, jdir)
        if result is None:
            result = _engine_call("run_job", data, str(jdir), use_ai=use_ai, paper=paper, _token=token)
            _log_job(job_id, origin, result)
    except Exception as e:
        storage.delete_job(job_id)
        status, code, msg = _classify_error(e, data[:16])
        if code == "server":
            log.error("job %s (%s) failed:\n%s", job_id, origin, traceback.format_exc())
        raise ApiError(status, msg, code)
    return {**result, "edit_token": edit_token}


def _log_job(job_id: str, origin: str, result: dict) -> None:
    t = result.get("timings", {})
    n_inc = sum(1 for p in result.get("parts", []) if p.get("include"))
    log.info("job %s %s: %d parts (%d included) ai=%s analyze=%.2fs render=%.2fs export=%.2fs total=%.2fs",
             job_id, origin, len(result.get("parts", [])), n_inc, result.get("ai_used"),
             t.get("analyze", 0), t.get("render", 0), t.get("export", 0), t.get("total", 0))


def _load_manifest() -> list[dict]:
    path = samples_dir() / "manifest.json"
    if not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        log.warning("samples/manifest.json is not valid JSON")
        return []
    return [s for s in data if isinstance(s, dict) and s.get("id") and s.get("file")] if isinstance(data, list) else []


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:40].strip("-")
    return s or "diagram"


# ---------------------------------------------------------------- async jobs (CONTRACT-v2 §2.1)
# One background thread runs queued uploads in order (each in the warm worker when ISOLATE=1), so
# jobs stay serialised and memory stays flat. Upload bytes wait on disk, not in RAM.
_ASYNC_EXEC = None
_ASYNC_LOCK = threading.Lock()
_STATUS_LOCK = threading.Lock()
_ASYNC_PENDING = 0


def _async_executor():
    global _ASYNC_EXEC
    with _ASYNC_LOCK:
        if _ASYNC_EXEC is None:
            from concurrent.futures import ThreadPoolExecutor
            _ASYNC_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ts-async")
        return _ASYNC_EXEC


def _read_status(jdir: Path) -> dict | None:
    return storage.read_json(jdir / storage.STATUS_FILE)


def _write_status(jdir: Path, *, force: bool = False, **fields) -> bool:
    """Merge fields into status.json. A cancelled job is never moved to another state (unless force)."""
    with _STATUS_LOCK:
        st = _read_status(jdir) or {}
        if st.get("status") == "cancelled" and not force:
            return False
        st.update(fields)
        storage.write_json_atomic(jdir / storage.STATUS_FILE, st)
        return True


_KEEP_AFTER_FAIL = {storage.STATUS_FILE, storage.TOKEN_FILE, storage.CANCEL_FILE, storage.PROGRESS_FILE}


def _discard_outputs(jdir: Path) -> None:
    """Remove a failed/cancelled job's outputs but keep its status (so /status can explain)."""
    try:
        for f in jdir.iterdir():
            if f.name not in _KEEP_AFTER_FAIL and f.is_file():
                f.unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        pass


def _stages_for(use_ai: bool, cached: bool) -> list[str]:
    ai_on = use_ai and _ai_configured()
    if cached:
        return ["ai"] if ai_on else ["files"]
    return ["read", "find"] + (["ai"] if ai_on else []) + ["layout", "files"]


def _run_async(job_id: str, use_ai: bool, paper: str, sid: str | None) -> None:
    global _ASYNC_PENDING
    jdir = storage.job_dir(job_id)
    upload = jdir / storage.UPLOAD_FILE
    head = b""
    try:
        head = upload.open("rb").read(16) if upload.is_file() else b""
        if (jdir / storage.CANCEL_FILE).exists() or not _write_status(jdir, status="running", started=time.time()):
            raise JobCancelled()
        result = None
        if sid:
            result = _clone_into(sid, paper, use_ai, jdir, progress=pipeline.file_progress(str(jdir)))
        if result is None:
            result = _engine_call("run_job_file", str(upload), str(jdir), use_ai=use_ai, paper=paper,
                                  _token=f"job:{job_id}")
            _log_job(job_id, f"async upload {upload.stat().st_size // 1024 if upload.exists() else 0} KB", result)
        upload.unlink(missing_ok=True)
        if (jdir / storage.CANCEL_FILE).exists() or not _write_status(
                jdir, status="done", stage="done", finished=time.time()):
            raise JobCancelled()
    except Exception as e:
        status, code, msg = _classify_error(e, head)
        _discard_outputs(jdir)
        if code == "cancelled":
            _write_status(jdir, force=True, status="cancelled", finished=time.time())
            log.info("job %s cancelled", job_id)
        else:
            if code == "server":
                log.error("async job %s failed:\n%s", job_id, traceback.format_exc())
            _write_status(jdir, status="error", error={"code": code, "message": msg}, finished=time.time())
    finally:
        upload.unlink(missing_ok=True)
        with _ASYNC_LOCK:
            _ASYNC_PENDING -= 1


def _start_async(data: bytes, use_ai: bool, paper: str) -> JSONResponse:
    global _ASYNC_PENDING
    with _ASYNC_LOCK:
        if _ASYNC_PENDING >= MAX_QUEUED:
            raise ApiError(503, "Lots of diagrams are being converted right now. Please try again in a minute.",
                           "busy")
        _ASYNC_PENDING += 1
    try:
        sid = None
        if os.environ.get("FEELABLE_SAMPLE_CACHE"):
            sid = _sample_for_bytes(data)
            if sid and _sample_cache_dir(sid, paper) is None:
                sid = None
        job_id, jdir, tok = _new_job()
        (jdir / storage.UPLOAD_FILE).write_bytes(data)
        _write_status(jdir, id=job_id, status="queued", stage="queued", stages=_stages_for(use_ai, bool(sid)),
                      created=time.time(), started=None, finished=None, error=None)
        _async_executor().submit(_run_async, job_id, use_ai, paper, sid)
    except Exception:
        with _ASYNC_LOCK:
            _ASYNC_PENDING -= 1
        raise
    return JSONResponse(status_code=202, content={"id": job_id, "edit_token": tok, "status": "queued"})


# ---------------------------------------------------------------- per-job web lock + access checks
_WEB_LOCKS: dict[str, threading.Lock] = {}
_WEB_LOCKS_GUARD = threading.Lock()


def _web_lock(job_id: str) -> threading.Lock:
    """Serialises edits of one job across the web process and the worker (whose locks are per-process)."""
    with _WEB_LOCKS_GUARD:
        if len(_WEB_LOCKS) > 2000:
            for k in [k for k, v in _WEB_LOCKS.items() if not v.locked()]:
                _WEB_LOCKS.pop(k, None)
        return _WEB_LOCKS.setdefault(job_id, threading.Lock())


def _require_job(job_id: str) -> Path:
    """Path of a finished job, else a 404/409 that says why (expired, still processing, failed...)."""
    if not storage.is_valid_job_id(job_id):
        raise ApiError(404, "Job not found", "not_found")
    if storage.job_exists(job_id):
        return storage.job_dir(job_id)
    jdir = storage.job_dir(job_id)
    if jdir.is_dir():
        st = _read_status(jdir) or {}
        s = st.get("status")
        if s == "cancelled":
            raise ApiError(404, "This job was cancelled.", "cancelled")
        if s == "error":
            err = st.get("error") or {}
            raise ApiError(404, str(err.get("message") or "Processing failed"), str(err.get("code") or "server"))
        raise ApiError(409, "This diagram is still being processed. Try again in a moment.", "busy")
    raise ApiError(404, MSG_EXPIRED, "expired")


def _require_token(job_id: str, token: str | None) -> None:
    if not storage.check_edit_token(job_id, token):
        raise ApiError(403, MSG_VIEW_ONLY, "view_only")


# ---------------------------------------------------------------- API
@app.get("/api/health")
def health():
    ai = {"provider": None, "model": None}
    try:
        from engine import ai as ai_mod
        info = ai_mod.get_provider_info() or {}
        ai = {"provider": info.get("provider"), "model": info.get("model")}
    except Exception:
        pass
    try:
        from engine import ai as _ai_mod
        ai["sends"] = getattr(_ai_mod, "SENDS", None) or AI_SENDS
    except Exception:  # pragma: no cover
        ai["sends"] = AI_SENDS
    # which build is answering (audit: "health exposes a version, not a deployed commit")
    commit = (os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("FEELABLE_COMMIT") or "").strip()[:40] or None
    return {"status": "ok", "version": VERSION, "commit": commit, "ai": ai,
            "limits": {"name": pipeline.NAME_MAX, "description": pipeline.DESCRIPTION_MAX,
                       "title": pipeline.TITLE_MAX},
            "retention_hours": storage.RETENTION_HOURS}


@app.get("/api/samples")
def list_samples():
    return [
        {"id": str(s["id"]), "title": str(s.get("title", s["id"])),
         "thumb_url": f"/samples/{s['file']}", "kind": str(s.get("kind", "colour")),
         "group": str(s.get("group", "simple")), "objective": str(s.get("objective", "")),
         "corrections": len(s.get("review") or []), "featured": bool(s.get("featured", False)),
         **({"credit": str(s["credit"])} if s.get("credit") else {})}
        for s in _load_manifest()
    ]


@app.post("/api/jobs", status_code=201)
def create_job(file: UploadFile | None = File(None), use_ai: str = Form("true"), paper: str = Form("a4"),
               cancel_token: str = Form(""), async_: str = Form("", alias="async")):
    use_ai_b = _parse_bool(use_ai, "use_ai")
    paper_s = _parse_paper(paper)
    run_async = _parse_bool(async_ or "false", "async")
    if file is None:
        raise ApiError(400, MSG_NO_FILE, "not_image")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise ApiError(400, MSG_EMPTY, "not_image")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ApiError(400, f"That image is too big (max {MAX_UPLOAD_BYTES // (1024 * 1024) or 1} MB). "
                            "Try a smaller photo or a screenshot.", "too_big")
    if run_async:
        return _start_async(data, use_ai_b, paper_s)
    return _run_new_job(data, use_ai_b, paper_s, origin=f"upload {len(data) // 1024} KB",
                        token=_valid_token(cancel_token))


def _valid_token(t: str | None) -> str | None:
    return t if t and re.match(r"^[A-Za-z0-9_-]{8,64}$", t) else None


@app.post("/api/jobs/cancel/{token}")
def cancel_job(token: str):
    """Stop a synchronous upload that is still being processed (legacy Cancel button)."""
    tok = _valid_token(token)
    if tok is None:
        raise ApiError(404, "Not found", "not_found")
    return {"cancelled": _cancel(tok)}


@app.get("/api/jobs/{job_id}/status")
def job_status(job_id: str):
    if not storage.is_valid_job_id(job_id):
        raise ApiError(404, "Job not found", "not_found")
    jdir = storage.job_dir(job_id)
    if not jdir.is_dir():
        raise ApiError(404, MSG_EXPIRED, "expired")
    st = _read_status(jdir)
    now = time.time()
    if st is None:   # created synchronously
        done = storage.job_exists(job_id)
        return {"id": job_id, "status": "done" if done else "running", "stage": "done" if done else "read",
                "stages": _stages_for(False, False), "elapsed": 0.0, "error": None}
    status = st.get("status", "queued")
    prog = storage.read_json(jdir / storage.PROGRESS_FILE) or {}
    stage = prog.get("stage") if prog.get("stage") in ALL_STAGES else None
    if status in ("queued", "running"):
        # "running" only once the worker has actually started (it reports "read" first)
        status, stage = ("running", stage) if stage else ("queued", "queued")
    elif status == "done":
        stage = "done"
    else:
        stage = stage or "queued"
    created = float(st.get("created") or now)
    end = st.get("finished") or now
    err = st.get("error") if status == "error" else None
    return {"id": job_id, "status": status, "stage": stage, "stages": list(st.get("stages") or []),
            "elapsed": round(max(0.0, float(end) - created), 1),
            "error": {"code": str(err.get("code", "server")), "message": str(err.get("message", ""))} if err else None}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_by_id(job_id: str, x_edit_token: str | None = Header(None, alias="X-Edit-Token")):
    """Cancel a queued or running async job (teacher token required)."""
    if not storage.job_dir_exists(job_id):
        _require_job(job_id)      # raises not_found / expired
    _require_token(job_id, x_edit_token)
    jdir = storage.job_dir(job_id)
    with _STATUS_LOCK:
        st = _read_status(jdir)
        current = (st or {}).get("status") or ("done" if storage.job_exists(job_id) else "running")
        if current in ("done", "error", "cancelled"):
            return {"cancelled": False, "status": current}
        (jdir / storage.CANCEL_FILE).write_text("1")
        st = {**(st or {}), "status": "cancelled", "finished": time.time()}
        storage.write_json_atomic(jdir / storage.STATUS_FILE, st)
    _cancel(f"job:{job_id}")      # drop it from the worker queue, or stop the worker running it
    return {"cancelled": True, "status": "cancelled"}


@app.post("/api/jobs/from-sample/{sample_id}", status_code=201)
def create_job_from_sample(sample_id: str, use_ai: str = "true", paper: str = "a4"):
    use_ai_b = _parse_bool(use_ai, "use_ai")
    paper_s = _parse_paper(paper)
    sample = next((s for s in _load_manifest() if str(s["id"]) == sample_id), None)
    if sample is None:
        raise ApiError(404, "Sample not found", "not_found")
    fname = str(sample["file"])
    path = samples_dir() / fname
    if not SAFE_NAME_RE.match(fname) or not path.is_file():
        raise ApiError(404, "Sample not found", "not_found")
    if _sample_cache_dir(sample_id, paper_s) is not None:
        job_id, jdir, tok = _new_job()
        hit = _clone_into(sample_id, paper_s, use_ai_b, jdir)
        if hit is not None:
            return {**hit, "edit_token": tok}
        storage.delete_job(job_id)
    return _run_new_job(path.read_bytes(), use_ai_b, paper_s, origin=f"sample {sample_id}")


def _sample_cache_dir(sample_id: str, paper: str):
    root = os.environ.get("FEELABLE_SAMPLE_CACHE")
    if not root or not re.match(r"^[a-z0-9-]{1,40}$", sample_id):
        return None
    d = Path(root) / paper / sample_id
    if not ((d / "result.json").is_file() and (d / "analysis.pkl").is_file()):
        return None
    # a prepared lesson is served only for the source + annotation version it was built from
    key = d / "cache_key.json"
    if key.is_file():
        try:
            k = json.loads(key.read_text())
            entry = next((s for s in _load_manifest() if s.get("id") == sample_id), None)
            if entry is not None and str(entry.get("annotation_version", "")) != str(k.get("annotation_version", "")):
                return None
        except (OSError, ValueError):
            return None
    return d


def _ai_configured() -> bool:
    try:
        from engine import ai
        return bool(ai.get_provider_info().get("provider"))
    except Exception:
        return False


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    _require_job(job_id)
    result = storage.load_result(job_id)
    if result is None:  # pragma: no cover - deleted between the two reads
        raise ApiError(404, MSG_EXPIRED, "expired")
    result.pop("edit_token", None)
    return result


def _norm_say(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


@app.get("/api/jobs/{job_id}/vision")
def job_vision(job_id: str):
    """The image-AI pipeline made visible: labels read (and repaired), pointer lines followed,
    how each part was found. Read-only, for the review page's "What Feelable saw" view."""
    jdir = _require_job(job_id)
    try:
        return pipeline.vision_trace(str(jdir))
    except (OSError, KeyError, EOFError) as e:
        raise ApiError(404, "Nothing to show for this lesson", "not_found") from e


@app.post("/api/jobs/{job_id}/tts")
def job_tts(job_id: str, body: dict = Body(...)):
    """Natural voice clip (WAV) for text that belongs to this lesson: a part's name or description,
    or the title. Anything else is refused, so this can't be used as a free TTS service.
    503 when no Gemini key is configured; the browser voice is used then."""
    from engine import tts
    jdir = _require_job(job_id)
    text = " ".join(str((body or {}).get("text") or "").split())
    if not text or len(text) > tts.MAX_CHARS:
        raise ApiError(400, "Text missing or too long", "invalid")
    if not tts.available():
        raise ApiError(503, "Natural voice is off on this server", "ai_off")
    allowed = set()
    for res in (storage.load_result(job_id), pipeline.load_published(str(jdir))):
        if not res:
            continue
        allowed.add(_norm_say(res.get("title", "")))
        for p in res.get("parts", []):
            allowed.add(_norm_say(p.get("name", "")))
            allowed.add(_norm_say(p.get("description", "")))
    if _norm_say(text) not in allowed:
        raise ApiError(403, "Only this lesson's own text can be spoken", "invalid")
    import hashlib
    cache = jdir / "tts"
    cache.mkdir(exist_ok=True)
    f = cache / (hashlib.sha1(text.encode("utf-8")).hexdigest()[:20] + ".wav")
    if not f.is_file():
        audio = tts.synth(text)
        if audio is None:
            raise ApiError(503, "Natural voice unavailable right now", "busy")
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(audio)
        os.replace(tmp, f)
    return FileResponse(f, media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs/{job_id}/published")
def get_published(job_id: str):
    """The teacher-approved snapshot that students see (404 not_published until approved)."""
    jdir = _require_job(job_id)
    pub = pipeline.load_published(str(jdir))
    if pub is None:
        raise ApiError(404, "This lesson is not ready yet. Ask your teacher to approve it.", "not_published")
    pub.pop("edit_token", None)
    return pub


@app.patch("/api/jobs/{job_id}")
def update_job(job_id: str, edit: dict = Body(...),
               x_edit_token: str | None = Header(None, alias="X-Edit-Token")):
    jdir = _require_job(job_id)
    _require_token(job_id, x_edit_token)
    with _web_lock(job_id):
        try:
            result = pipeline.quick_edit(str(jdir), edit)   # descriptions/approval only: no redraw
            if result is None:
                result = _engine_call("rerender_job", str(jdir), edit)
        except pipeline.Conflict as e:
            raise ApiError(409, str(e), "conflict")
        except pipeline.NotSupported as e:
            raise ApiError(501, str(e), "server")
        except ValueError as e:
            raise ApiError(400, _short_reason(e), "invalid")
        except Exception as e:
            status, code, msg = _classify_error(e)
            if code == "server":
                log.error("rerender %s failed:\n%s", job_id, traceback.format_exc())
            raise ApiError(status, msg, code)
    _log_job(job_id, "edit", result)
    return result


@app.post("/api/jobs/{job_id}/ai-describe")
def ai_describe(job_id: str, body: dict | None = Body(None),
                x_edit_token: str | None = Header(None, alias="X-Edit-Token")):
    """Re-describe every included part with AI (teacher-written descriptions kept unless
    {"overwrite_teacher": true}). Runs in the web process: it waits on the network, not the CPU."""
    jdir = _require_job(job_id)
    _require_token(job_id, x_edit_token)
    if not _ai_configured():
        raise ApiError(503, MSG_AI_OFF, "ai_off")
    overwrite = (body or {}).get("overwrite_teacher", False)
    if not isinstance(overwrite, bool):
        raise ApiError(400, "overwrite_teacher must be true or false", "invalid")
    with _web_lock(job_id):
        try:
            result = pipeline.ai_describe_job(str(jdir), overwrite_teacher=overwrite)
        except Exception as e:
            log.error("ai-describe %s failed:\n%s", job_id, traceback.format_exc())
            status, code, msg = _classify_error(e)
            raise ApiError(status, msg, code)
    _log_job(job_id, "ai-describe", result)
    return result


@app.post("/api/jobs/{job_id}/outline")
def job_outline(job_id: str, body: dict = Body(...),
                x_edit_token: str | None = Header(None, alias="X-Edit-Token")):
    """Assisted outline for the draw tool: propose the outline of a structure the teacher roughly
    marked, or report which parts a drawn outline would overlap. Read-only: nothing is saved
    until the teacher saves the part. The request's "rev" comes back so the app can ignore
    answers to older drafts."""
    jdir = _require_job(job_id)
    _require_token(job_id, x_edit_token)
    if not isinstance(body, dict):
        raise ApiError(400, "Expected a JSON object", "invalid")
    try:
        return pipeline.find_outline(str(jdir), body)
    except ValueError as e:
        raise ApiError(400, str(e), "invalid")
    except (OSError, KeyError, EOFError) as e:
        raise ApiError(404, "Nothing to outline for this lesson", "not_found") from e


# ---------------------------------------------------------------- files
@app.get("/files/{job_id}/{filename}")
def job_file(job_id: str, filename: str):
    if not storage.is_valid_job_id(job_id) or not SAFE_NAME_RE.match(filename):
        raise ApiError(404, "File not found", "not_found")
    if not storage.job_dir_exists(job_id):
        raise ApiError(404, MSG_EXPIRED, "expired")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in FILE_TYPES:
        raise HTTPException(404, "File not found")
    jdir = storage.job_dir(job_id).resolve()
    path = (jdir / filename).resolve()
    if filename == "relief.stl" and path.parent == jdir and not path.is_file() \
            and (jdir / "tactile.png").is_file():
        try:  # STL is built on first download (saves ~1.5 s on every job)
            _engine_call("ensure_stl", str(jdir))
        except Exception:
            raise ApiError(500, "Could not build the 3D model", "server")
    if path.parent != jdir or not path.is_file():
        raise HTTPException(404, "File not found")
    headers = {"Cache-Control": "no-cache"}
    if ext in (".pdf", ".stl"):
        title = (storage.load_result(job_id) or {}).get("title", "")
        headers["Content-Disposition"] = f'attachment; filename="feelable-{_slug(title)}{ext}"'
    return FileResponse(path, media_type=FILE_TYPES[ext], headers=headers)


@app.get("/files/{job_id}/pub/{ver}/{filename}")
def published_file(job_id: str, ver: str, filename: str):
    """Immutable media of an approved version (UX01). Never rewritten, so it can be cached."""
    if (not storage.is_valid_job_id(job_id) or not re.fullmatch(r"v\d{1,6}", ver)
            or not SAFE_NAME_RE.match(filename)):
        raise ApiError(404, "File not found", "not_found")
    if not storage.job_dir_exists(job_id):
        raise ApiError(404, MSG_EXPIRED, "expired")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in FILE_TYPES or filename not in pipeline.PUB_FILES:
        raise HTTPException(404, "File not found")
    bdir = (storage.job_dir(job_id) / pipeline.PUB_DIR / ver).resolve()
    path = (bdir / filename).resolve()
    if path.parent != bdir:
        raise HTTPException(404, "File not found")
    if filename == "relief.stl" and not path.is_file() and (bdir / "tactile.png").is_file():
        try:
            pipeline.ensure_pub_stl(str(storage.job_dir(job_id)), int(ver[1:]))
        except Exception:
            raise ApiError(500, "Could not build the 3D model", "server")
    if not path.is_file():
        raise HTTPException(404, "File not found")
    headers = {"Cache-Control": "public, max-age=86400, immutable"}
    if ext in (".pdf", ".stl"):
        title = (pipeline.load_published(str(storage.job_dir(job_id))) or {}).get("title", "")
        headers["Content-Disposition"] = f'attachment; filename="feelable-{_slug(title)}{ext}"'
    return FileResponse(path, media_type=FILE_TYPES[ext], headers=headers)


@app.get("/samples/{filename}")
def sample_file(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if not SAFE_NAME_RE.match(filename) or ext not in SAMPLE_TYPES:
        raise HTTPException(404, "File not found")
    sdir = samples_dir().resolve()
    path = (sdir / filename).resolve()
    if path.parent != sdir or not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type=SAMPLE_TYPES[ext])


# ---------------------------------------------------------------- catch-alls (must be registered last)
@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
def api_not_found(rest: str):
    if rest.startswith("jobs/"):
        raise ApiError(404, "Job not found", "not_found")
    raise ApiError(404, "Not found", "not_found")


@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    first = full_path.split("/", 1)[0]
    if first in ("api", "files", "samples"):
        raise HTTPException(404, "Not found")
    dist = web_dist_dir()
    index = dist / "index.html"
    if not index.is_file():
        if full_path in ("", "index.html"):
            return {"name": "Feelable API", "version": VERSION,
                    "hint": "Web app not built. Run `cd web && npm run build` (or ./run.sh), "
                            "or use `npm run dev` on :5173. API lives under /api."}
        raise HTTPException(404, "Not found")
    if full_path:
        root = dist.resolve()
        candidate = (root / full_path).resolve()
        if candidate.is_file() and root in candidate.parents:
            return FileResponse(candidate)
    return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})
