"""Job storage on disk. OWNER: API team.

Layout: <data root>/<job id>/{result.json, analysis.pkl, original.png, tactile.png, ...}
Data root = backend/data/jobs (override with env FEELABLE_DATA).
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import time
from pathlib import Path

log = logging.getLogger("feelable.storage")

BACKEND_DIR = Path(__file__).resolve().parents[1]
JOB_ID_RE = re.compile(r"^[0-9a-f]{10}$")
RESULT_FILE = "result.json"


class InvalidJobId(ValueError):
    pass


def data_root() -> Path:
    root = Path(os.environ.get("FEELABLE_DATA") or (BACKEND_DIR / "data" / "jobs"))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def new_job_id() -> str:
    return secrets.token_hex(5)  # 10 lowercase hex chars


def is_valid_job_id(job_id: str) -> bool:
    return isinstance(job_id, str) and bool(JOB_ID_RE.match(job_id))


def job_dir(job_id: str) -> Path:
    """Path of a job directory (not necessarily existing). Raises InvalidJobId for bad ids."""
    if not is_valid_job_id(job_id):
        raise InvalidJobId("Invalid job id")
    return data_root() / job_id


def create_job_dir(job_id: str) -> Path:
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=False)
    return d


def job_exists(job_id: str) -> bool:
    try:
        return (job_dir(job_id) / RESULT_FILE).is_file()
    except InvalidJobId:
        return False


def load_result(job_id: str) -> dict | None:
    try:
        p = job_dir(job_id) / RESULT_FILE
    except InvalidJobId:
        return None
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_result(job_dir_path: str | os.PathLike, result: dict) -> None:
    write_json_atomic(Path(job_dir_path) / RESULT_FILE, result)


def write_json_atomic(path: str | os.PathLike, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def job_dir_exists(job_id: str) -> bool:
    """True if the job directory exists (an async job may not have a result yet)."""
    try:
        return job_dir(job_id).is_dir()
    except InvalidJobId:
        return False


# ---------------------------------------------------------------- teacher edit token (CONTRACT-v2 §2.2)
# Stored in the job dir under a dot-name, which /files/ never serves (SAFE_NAME_RE needs an alnum start
# and the extension allow-list has no match). Never written to result.json.
TOKEN_FILE = ".edit_token"
STATUS_FILE = "status.json"       # async job state, written by the web process
PROGRESS_FILE = "progress.json"   # current stage, written by the worker
UPLOAD_FILE = "upload.bin"        # queued upload bytes (async), deleted when the job finishes
CANCEL_FILE = "cancel"            # marker: the teacher cancelled this job
RETENTION_HOURS = 24


def new_edit_token() -> str:
    return secrets.token_hex(16)  # 32 hex chars


def save_edit_token(job_dir_path: str | os.PathLike) -> str:
    tok = new_edit_token()
    p = Path(job_dir_path) / TOKEN_FILE
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(tok, encoding="ascii")
    os.replace(tmp, p)
    return tok


def check_edit_token(job_id: str, token: str | None) -> bool:
    import hmac
    if not token or not isinstance(token, str):
        return False
    try:
        stored = (job_dir(job_id) / TOKEN_FILE).read_text(encoding="ascii").strip()
    except (InvalidJobId, OSError, UnicodeDecodeError):
        return False
    if not stored:
        return False
    return hmac.compare_digest(stored.encode("ascii"), token.strip().encode("utf-8", "replace"))


def read_json(path: str | os.PathLike) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def delete_job(job_id: str) -> None:
    try:
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
    except InvalidJobId:
        pass


def cleanup_old_jobs(max_age_s: float = 24 * 3600) -> int:
    """Delete job dirs older than max_age_s. Returns number removed."""
    removed = 0
    now = time.time()
    try:
        root = data_root()
        for d in root.iterdir():
            if not d.is_dir() or not is_valid_job_id(d.name):
                continue
            try:
                if now - d.stat().st_mtime > max_age_s:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    except OSError as e:  # pragma: no cover
        log.warning("cleanup failed: %s", e)
    return removed
