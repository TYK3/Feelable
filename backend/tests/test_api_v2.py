"""CONTRACT-v2 §2 API tests (fake engine): async jobs + progress, edit tokens, limits, readiness,
conflicts, split/merge, AI on demand. OWNER: API team."""
from __future__ import annotations

import threading
import time

import pytest

from conftest import assert_job_result_schema, fake_analysis, png_bytes


def _create(client, **form):
    data = {"use_ai": "false", "paper": "a4", **form}
    r = client.post("/api/jobs", files={"file": ("d.png", png_bytes(), "image/png")}, data=data)
    assert r.status_code == 201, r.text
    job = r.json()
    return job, job.pop("edit_token")


def _create_async(client, payload=None, **form):
    data = {"use_ai": "false", "paper": "a4", "async": "1", **form}
    r = client.post("/api/jobs", files={"file": ("d.png", payload or png_bytes(), "image/png")}, data=data)
    assert r.status_code == 202, r.text
    body = r.json()
    assert set(body) == {"id", "edit_token", "status"} and body["status"] == "queued"
    assert len(body["edit_token"]) == 32
    return body["id"], body["edit_token"]


def _wait(client, jid, until=("done", "error", "cancelled"), timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/api/jobs/{jid}/status").json()
        if st["status"] in until:
            return st
        time.sleep(0.02)
    raise AssertionError(f"job {jid} stuck: {st}")


def _patch(client, jid, tok, body):
    return client.patch(f"/api/jobs/{jid}", json=body, headers={"X-Edit-Token": tok})


@pytest.fixture
def gate(fake_engine):
    """Make the fake analyze accept `progress` and block until released (to observe queued/running)."""
    started = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    def analyze(image_rgb, *, use_ai=True, progress=None):
        if progress:
            progress("read")
            progress("find")
        started.set()
        assert release.wait(10)
        return fake_analysis(use_ai)

    fake_engine["engine.analyze"].analyze = analyze
    yield started, release, seen
    release.set()


# ---------------------------------------------------------------- async jobs (F13)
def test_async_job_runs_with_real_stages(client, fake_engine, monkeypatch):
    from engine import pipeline
    seen = []
    real = pipeline.file_progress

    def spy(job_dir):
        cb = real(job_dir)
        return lambda s: (seen.append(s), cb(s))
    monkeypatch.setattr(pipeline, "file_progress", spy)

    def analyze(image_rgb, *, use_ai=True, progress=None):   # engine reports its own stages
        progress("read")
        progress("find")
        return fake_analysis(use_ai)
    fake_engine["engine.analyze"].analyze = analyze

    jid, tok = _create_async(client)
    st = _wait(client, jid)
    assert st["status"] == "done" and st["stage"] == "done" and st["error"] is None
    assert st["stages"] == ["read", "find", "layout", "files"]           # no "ai": AI isn't configured
    assert isinstance(st["elapsed"], float) and st["elapsed"] >= 0
    assert seen == ["read", "read", "find", "layout", "files"]
    job = client.get(f"/api/jobs/{jid}").json()
    assert_job_result_schema(job)
    assert tok not in str(job)
    d = __import__("app.storage", fromlist=["x"]).job_dir(jid)
    assert not (d / "upload.bin").exists()
    assert _patch(client, jid, tok, {"title": "async cell"}).status_code == 200


def test_async_stages_include_ai_only_when_ai_runs(client, fake_engine, monkeypatch):
    fake_engine["engine.ai"].get_provider_info = lambda: {"provider": "gemini", "model": "m"}
    jid, _ = _create_async(client, use_ai="true")
    assert client.get(f"/api/jobs/{jid}/status").json()["stages"] == ["read", "find", "ai", "layout", "files"]
    _wait(client, jid)
    jid2, _ = _create_async(client, use_ai="false")
    assert "ai" not in client.get(f"/api/jobs/{jid2}/status").json()["stages"]
    _wait(client, jid2)


def test_async_engine_without_progress_param_still_works(client):
    jid, _ = _create_async(client)     # conftest's fake analyze has no `progress` argument
    assert _wait(client, jid)["status"] == "done"


def test_async_error_is_stored_with_code(client):
    jid, _ = _create_async(client, payload=b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    st = _wait(client, jid)
    assert st["status"] == "error" and st["error"]["code"] == "decode" and "damaged" in st["error"]["message"]
    r = client.get(f"/api/jobs/{jid}")
    assert r.status_code == 404 and r.json()["code"] == "decode"
    jid, _ = _create_async(client, payload=b"just text, not a picture")
    assert _wait(client, jid)["error"]["code"] == "not_image"


def test_async_queued_then_running_and_cancel_both(client, gate):
    started, release, _ = gate
    a, tok_a = _create_async(client)
    assert started.wait(5)
    st = client.get(f"/api/jobs/{a}/status").json()
    assert st["status"] == "running" and st["stage"] == "find"
    r = client.get(f"/api/jobs/{a}")
    assert r.status_code == 409 and r.json()["code"] == "busy"

    b, tok_b = _create_async(client)                       # waits behind a: jobs are serialised
    st = client.get(f"/api/jobs/{b}/status").json()
    assert st["status"] == "queued" and st["stage"] == "queued"

    # cancel needs the teacher token
    r = client.post(f"/api/jobs/{b}/cancel")
    assert r.status_code == 403 and r.json()["code"] == "view_only"
    assert client.post(f"/api/jobs/{b}/cancel", headers={"X-Edit-Token": tok_a}).status_code == 403
    # cancel queued b, then running a
    assert client.post(f"/api/jobs/{b}/cancel", headers={"X-Edit-Token": tok_b}).json() == \
        {"cancelled": True, "status": "cancelled"}
    assert client.post(f"/api/jobs/{a}/cancel", headers={"X-Edit-Token": tok_a}).json()["cancelled"] is True
    assert client.get(f"/api/jobs/{a}/status").json()["status"] == "cancelled"
    release.set()
    for jid in (a, b):
        st = _wait(client, jid)
        assert st["status"] == "cancelled"
    time.sleep(0.3)   # let the worker thread finish a and skip b
    for jid in (a, b):
        assert client.get(f"/api/jobs/{jid}/status").json()["status"] == "cancelled"
        r = client.get(f"/api/jobs/{jid}")
        assert r.status_code == 404 and r.json()["code"] == "cancelled"
    # a later job still runs fine
    c, tok_c = _create_async(client)
    assert _wait(client, c)["status"] == "done"
    assert client.post(f"/api/jobs/{c}/cancel", headers={"X-Edit-Token": tok_c}).json() == \
        {"cancelled": False, "status": "done"}


def test_async_queue_limit(client, gate, monkeypatch):
    from app import main
    started, release, _ = gate
    monkeypatch.setattr(main, "MAX_QUEUED", 2)
    a, _ = _create_async(client)
    assert started.wait(5)
    b, _ = _create_async(client)
    r = client.post("/api/jobs", files={"file": ("d.png", png_bytes(), "image/png")},
                    data={"async": "1", "use_ai": "false"})
    assert r.status_code == 503 and r.json()["code"] == "busy"
    release.set()
    assert _wait(client, a)["status"] == _wait(client, b)["status"] == "done"


def test_status_of_sync_job(client):
    job, _ = _create(client)
    st = client.get(f"/api/jobs/{job['id']}/status").json()
    assert st["status"] == "done" and st["stage"] == "done" and st["error"] is None
    assert client.get("/api/jobs/nothex!!!!/status").json()["code"] == "not_found"


# ---------------------------------------------------------------- edit token (F04)
def test_edits_need_the_teacher_token(client):
    job, tok = _create(client)
    jid = job["id"]
    for headers in ({}, {"X-Edit-Token": "0" * 32}, {"X-Edit-Token": tok[:-1]}):
        r = client.patch(f"/api/jobs/{jid}", json={"title": "hacked"}, headers=headers)
        assert r.status_code == 403
        assert r.json() == {"detail": "This is a view-only link — open the teacher link to edit.", "code": "view_only"}
        assert client.post(f"/api/jobs/{jid}/ai-describe", json={}, headers=headers).status_code == 403
    assert client.get(f"/api/jobs/{jid}").json() == job           # read-only access stays public
    assert client.get(job["files"]["pdf"]).status_code == 200
    assert _patch(client, jid, tok, {"title": "teacher"}).status_code == 200


# ---------------------------------------------------------------- limits (F06)
@pytest.mark.parametrize("edit,needle", [
    ({"title": "t" * 81}, "title is too long (81 characters; the limit is 80)"),
    ({"parts": [{"id": "p1", "name": "n" * 61}]}, "part name is too long (61 characters; the limit is 60)"),
    ({"parts": [{"id": "p1", "description": "d" * 401}]}, "description is too long (401 characters; the limit is 400)"),
])
def test_over_limit_is_rejected_not_truncated(client, edit, needle):
    job, tok = _create(client)
    r = _patch(client, job["id"], tok, edit)
    assert r.status_code == 400 and needle in r.json()["detail"] and r.json()["code"] == "invalid"
    assert client.get(f"/api/jobs/{job['id']}").json() == job


def test_at_limit_round_trips_unchanged(client):
    job, tok = _create(client)
    desc = ("Word " * 80).strip()[:399] + "."
    r = _patch(client, job["id"], tok, {"title": "T" * 80, "parts": [{"id": "p2", "name": "N" * 60, "description": desc}]})
    assert r.status_code == 200
    p2 = next(p for p in r.json()["parts"] if p["id"] == "p2")
    assert r.json()["title"] == "T" * 80 and p2["name"] == "N" * 60 and p2["description"] == desc


# ---------------------------------------------------------------- readiness / approval (F02, F11)
def test_excluding_every_part_is_400(client):
    job, tok = _create(client)
    r = _patch(client, job["id"], tok, {"parts": [{"id": p["id"], "include": False} for p in job["parts"]]})
    assert r.status_code == 400 and r.json()["detail"] == "Keep at least one part switched on"
    assert client.get(f"/api/jobs/{job['id']}").json() == job


def test_readiness_rules(fake_engine):
    from engine import pipeline
    a = fake_analysis()
    for p in a.parts:
        p.include = False
    rd = pipeline.build_readiness(a, [{"rule": "r", "passed": True, "detail": ""}], True, None)
    assert rd["status"] == "blocked" and rd["approved"] is True
    assert [i["code"] for i in rd["issues"]] == ["no_parts"] and rd["issues"][0]["action"] == "include"

    a = fake_analysis()
    a.input_kind = "text_page"
    a.parts = []
    rd = pipeline.build_readiness(a, [], False, None)
    assert rd["status"] == "blocked" and [(i["code"], i.get("action")) for i in rd["issues"]] == [("no_diagram", "retake")]

    a = fake_analysis()
    a.parts[0].name = "part 3"
    a.unmatched_labels = ["vacuole"]
    a.input_kind = "photo_object"
    checks = [{"rule": "Lines at least 1 mm", "passed": False, "detail": "0.5 mm"},
              {"rule": "ok", "passed": True, "detail": ""}]
    rd = pipeline.build_readiness(a, checks, False, "Gemini rate limit")
    codes = [i["code"] for i in rd["issues"]]
    assert codes == ["unnamed", "unmatched_label", "geometry", "ai_failed", "photo"]
    assert rd["issues"][0]["part_id"] == a.parts[0].id and rd["issues"][0]["action"] == "rename"
    assert "Gemini rate limit" in rd["issues"][3]["message"] and rd["issues"][4]["severity"] == "info"
    assert rd["status"] == "needs_review" and rd["geometry"] == {"passed": 1, "total": 2}
    assert rd["unmatched_labels"] == ["vacuole"]
    assert pipeline.build_readiness(a, checks, True, None)["status"] == "ready"

    a = fake_analysis()
    a.parts = a.parts * 3
    assert "too_many" in [i["code"] for i in pipeline.build_readiness(a, [], False, None)["issues"]]


def test_approval_and_reset_on_content_change(client):
    job, tok = _create(client)
    jid = job["id"]
    assert job["readiness"]["status"] == "needs_review" and job["version"] == 0
    r = _patch(client, jid, tok, {"approved": True, "base_version": 0}).json()
    assert r["readiness"]["status"] == "ready" and r["readiness"]["approved"] and r["version"] == 1
    r = _patch(client, jid, tok, {"parts": [{"id": "p1", "description": "New words."}]}).json()
    assert r["readiness"]["approved"] is False and r["readiness"]["status"] == "needs_review"
    r = _patch(client, jid, tok, {"title": "approved in the same save", "approved": True}).json()
    assert r["readiness"]["approved"] is True and r["version"] == 3
    r = _patch(client, jid, tok, {"title": "changed again"}).json()
    assert r["readiness"]["approved"] is False


def test_conflict_on_stale_base_version(client):
    job, tok = _create(client)
    jid = job["id"]
    assert _patch(client, jid, tok, {"title": "first", "base_version": 0}).json()["version"] == 1
    for body in ({"title": "second", "base_version": 0},
                 {"parts": [{"id": "p1", "description": "x"}], "base_version": 0}):
        r = _patch(client, jid, tok, body)
        assert r.status_code == 409 and r.json()["code"] == "conflict" and "version 1" in r.json()["detail"]
    assert client.get(f"/api/jobs/{jid}").json()["title"] == "first"
    assert _patch(client, jid, tok, {"title": "second", "base_version": 1}).status_code == 200


# ---------------------------------------------------------------- split / merge
def test_split_merge_501_when_engine_lacks_them(client):
    job, tok = _create(client)
    for body in ({"split": "p3"}, {"merge": ["p2", "p3"]}):
        r = _patch(client, job["id"], tok, body)
        assert r.status_code == 501 and r.json()["code"] == "server"


def test_split_merge_call_the_engine(client, fake_engine):
    import numpy as np
    from conftest import _part
    calls = []

    def split_part(analysis, pid):
        p = next(p for p in analysis.parts if p.id == pid)
        if p.count < 2:
            raise ValueError("That part is a single piece")
        calls.append(("split", pid))
        m = np.zeros_like(p.mask); m[100:180, 200:240] = 255
        analysis.parts.append(_part("p4", "part 4", (60, 160, 60), m))
        return [pid, "p4"]

    def merge_parts(analysis, ids):
        calls.append(("merge", ids))
        analysis.parts = [p for p in analysis.parts if p.id not in ids[1:]]
        return ids[0]

    fake_engine["engine.analyze"].split_part = split_part
    fake_engine["engine.analyze"].merge_parts = merge_parts
    job, tok = _create(client)
    r = _patch(client, job["id"], tok, {"split": "p3"})
    assert r.status_code == 200 and calls == [("split", "p3")]
    assert_job_result_schema(r.json())
    assert {p["id"] for p in r.json()["parts"]} == {"p1", "p2", "p3", "p4"}
    assert any(i["code"] == "unnamed" for i in r.json()["readiness"]["issues"])
    r = _patch(client, job["id"], tok, {"split": "p2"})
    assert r.status_code == 400 and "single piece" in r.json()["detail"]
    r = _patch(client, job["id"], tok, {"merge": ["p3", "p4"]})
    assert r.status_code == 200 and {p["id"] for p in r.json()["parts"]} == {"p1", "p2", "p3"}
    for bad in ({"merge": ["p3"]}, {"merge": ["p3", "zz"]}, {"merge": "p3"}, {"split": "zz"}):
        assert _patch(client, job["id"], tok, bad).status_code == 400


# ---------------------------------------------------------------- AI on demand
def test_ai_describe_503_when_ai_off(client):
    job, tok = _create(client)
    r = client.post(f"/api/jobs/{job['id']}/ai-describe", json={}, headers={"X-Edit-Token": tok})
    assert r.status_code == 503 and r.json()["code"] == "ai_off"


def test_ai_describe_keeps_teacher_text_unless_overwrite(client, fake_engine):
    ai = fake_engine["engine.ai"]
    ai.get_provider_info = lambda: {"provider": "gemini", "model": "m"}
    asked = []

    def describe_parts(image, analysis, ids):
        asked.append(list(ids))
        return {i: f"AI words for {i}." for i in ids}
    ai.describe_parts = describe_parts
    job, tok = _create(client)
    jid = job["id"]
    _patch(client, jid, tok, {"parts": [{"id": "p2", "description": "Teacher words."}], "approved": True})
    h = {"X-Edit-Token": tok}
    r = client.post(f"/api/jobs/{jid}/ai-describe", json={}, headers=h)
    assert r.status_code == 200, r.text
    j = r.json()
    assert_job_result_schema(j)
    by = {p["id"]: p for p in j["parts"]}
    assert sorted(asked[0]) == ["p1", "p3"]
    assert by["p1"]["description"] == "AI words for p1." and by["p1"]["description_source"] == "ai"
    assert by["p2"]["description"] == "Teacher words." and by["p2"]["description_source"] == "teacher"
    assert j["ai_used"] is True and j["readiness"]["approved"] is False and j["version"] == 2
    j = client.post(f"/api/jobs/{jid}/ai-describe", json={"overwrite_teacher": True}, headers=h).json()
    assert next(p for p in j["parts"] if p["id"] == "p2")["description"] == "AI words for p2."
    assert client.get(f"/api/jobs/{jid}").json() == j

    # failure: warning + ai_failed issue with the AI's own reason
    ai.describe_parts = lambda *a: None
    ai.last_error = lambda: "Gemini rate limit — try again in a minute"
    j = client.post(f"/api/jobs/{jid}/ai-describe", json={"overwrite_teacher": True}, headers=h).json()
    issue = next(i for i in j["readiness"]["issues"] if i["code"] == "ai_failed")
    assert "rate limit" in issue["message"] and issue["severity"] == "warn"
    assert client.post(f"/api/jobs/{jid}/ai-describe", json={"overwrite_teacher": "yes"},
                       headers=h).status_code == 400


def test_ai_failure_during_upload_is_a_warning(client, fake_engine):
    ai = fake_engine["engine.ai"]
    ai.get_provider_info = lambda: {"provider": "gemini", "model": "m"}
    ai.last_error = lambda: "AI timed out"
    job, _ = _create(client, use_ai="true")     # fake analyze leaves ai_used False
    issue = next(i for i in job["readiness"]["issues"] if i["code"] == "ai_failed")
    assert "AI timed out" in issue["message"]


def test_students_see_only_the_approved_snapshot(client):
    job, tok = _create(client)
    jid = job["id"]
    r = client.get(f"/api/jobs/{jid}/published")
    assert r.status_code == 404 and r.json()["code"] == "not_published"
    _patch(client, jid, tok, {"approved": True})
    pub = client.get(f"/api/jobs/{jid}/published").json()
    assert pub["readiness"]["status"] == "ready" and "edit_token" not in pub
    _patch(client, jid, tok, {"title": "draft title not approved yet"})
    pub2 = client.get(f"/api/jobs/{jid}/published").json()
    assert pub2["title"] == pub["title"] and pub2["version"] == pub["version"]


def test_teacher_can_draw_a_missing_part(client):
    job, tok = _create(client)
    jid = job["id"]
    n0 = len(job["parts"])
    w, h = job["source"]["width"], job["source"]["height"]
    poly = [[w * 0.05, h * 0.05], [w * 0.2, h * 0.05], [w * 0.2, h * 0.2], [w * 0.05, h * 0.2]]
    r = _patch(client, jid, tok, {"add_part": {"name": "pancreas", "polygon": poly}})
    assert r.status_code == 200, r.text
    parts = r.json()["parts"]
    assert len(parts) == n0 + 1
    new = next(p for p in parts if p["name"] == "pancreas")
    assert new["match_method"] == "teacher" and new["include"]
    bad = _patch(client, jid, tok, {"add_part": {"name": "x", "polygon": [[1, 1], [2, 2]]}})
    assert bad.status_code == 400


def test_published_files_served_from_immutable_bundle(client):
    job, tok = _create(client)
    jid = job["id"]
    _patch(client, jid, tok, {"approved": True})
    pub = client.get(f"/api/jobs/{jid}/published").json()
    url = pub["tactile"]["url"]
    assert "/pub/v" in url
    r = client.get(url)
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"]
    assert client.get(f"/files/{jid}/pub/v1/../result.json").status_code == 404
    assert client.get(f"/files/{jid}/pub/x1/tactile.png").status_code == 404
