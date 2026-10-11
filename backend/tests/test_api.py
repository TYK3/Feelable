"""HTTP API tests (fake engine). OWNER: API team."""
from __future__ import annotations

import json

import pytest

from conftest import assert_job_result_schema, png_bytes


TOKENS: dict[str, str] = {}   # job id -> teacher edit token (create responses only)


def _create(client, **form):
    data = {"use_ai": "false", "paper": "a4", **form}
    r = client.post("/api/jobs", files={"file": ("diagram.png", png_bytes(), "image/png")}, data=data)
    assert r.status_code == 201, r.text
    job = r.json()
    tok = job.pop("edit_token")
    assert isinstance(tok, str) and len(tok) == 32 and int(tok, 16) >= 0
    TOKENS[job["id"]] = tok
    return job


def _h(jid):
    return {"X-Edit-Token": TOKENS.get(jid, "")}


def _patch(client, jid, body):
    return client.patch(f"/api/jobs/{jid}", json=body, headers=_h(jid))


# ---------------------------------------------------------------- health / samples
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    sends = body["ai"].pop("sends")
    commit = body.pop("commit")
    assert commit is None or isinstance(commit, str)
    assert "Gemini" in sends and "24 h" in sends
    assert body == {"status": "ok", "version": "0.1.0", "ai": {"provider": None, "model": None},
                    "limits": {"name": 60, "description": 400, "title": 80}, "retention_hours": 24}


def test_samples_without_manifest(client):
    r = client.get("/api/samples")
    assert r.status_code == 200 and r.json() == []


def test_samples_with_manifest_and_from_sample(client, samples_dir):
    (samples_dir / "plant-cell.png").write_bytes(png_bytes())
    (samples_dir / "manifest.json").write_text(json.dumps([
        {"id": "plant-cell", "title": "Plant cell", "file": "plant-cell.png", "kind": "colour",
         "expected_parts": ["nucleus"], "min_parts": 1, "photo": False}]))
    r = client.get("/api/samples")
    assert r.json() == [{"id": "plant-cell", "title": "Plant cell", "thumb_url": "/samples/plant-cell.png",
                         "kind": "colour", "group": "simple", "objective": "", "corrections": 0}]
    img = client.get("/samples/plant-cell.png")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert client.get("/samples/manifest.json").status_code == 404
    assert client.get("/samples/..%2Fsecret.png").status_code == 404

    r = client.post("/api/jobs/from-sample/plant-cell?use_ai=false&paper=letter")
    assert r.status_code == 201, r.text
    job = r.json()
    assert len(job.pop("edit_token")) == 32
    assert_job_result_schema(job)
    assert job["tactile"]["width"] == 1397  # fake renderer honours letter paper

    r = client.post("/api/jobs/from-sample/nope")
    assert r.status_code == 404 and r.json() == {"detail": "Sample not found", "code": "not_found"}


# ---------------------------------------------------------------- create job
def test_create_job_schema(client):
    job = _create(client)
    assert_job_result_schema(job)
    assert [p["id"] for p in job["parts"]] == ["p1", "p2", "p3"]  # area desc
    assert job["title"] == "plant cell" and job["warnings"] == ["fake warning"]
    assert job["timings"]["analyze"] == 0.25
    assert job["files"]["pdf"] == f"/files/{job['id']}/tactile.pdf"
    # GET returns the same thing
    r = client.get(f"/api/jobs/{job['id']}")
    assert r.status_code == 200 and r.json() == job


@pytest.mark.parametrize("payload,needle,code", [
    (b"", "empty", "not_image"),
    (b"hello, I am a text file", "isn't an image", "not_image"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "damaged", "decode"),          # corrupt PNG: choose a new file
    (b"\xff\xd8\xff\xe0" + b"garbage" * 10, "damaged", "decode"),
])
def test_bad_uploads(client, payload, needle, code):
    r = client.post("/api/jobs", files={"file": ("x.png", payload, "image/png")})
    assert r.status_code == 400
    assert needle in r.json()["detail"] and r.json()["code"] == code


def test_oversize_upload(client, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/api/jobs", files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 400 and "too big" in r.json()["detail"] and r.json()["code"] == "too_big"


def test_bad_params(client):
    f = {"file": ("x.png", png_bytes(), "image/png")}
    r = client.post("/api/jobs", files=f, data={"paper": "a3"})
    assert r.status_code == 400 and r.json()["code"] == "invalid"
    assert client.post("/api/jobs", files=f, data={"use_ai": "maybe"}).status_code == 400
    assert client.post("/api/jobs", files=f, data={"async": "perhaps"}).status_code == 400
    r = client.post("/api/jobs", data={"paper": "a4"})
    assert r.status_code == 400 and "No file" in r.json()["detail"]


def test_bad_upload_leaves_no_job_dir(client, data_dir):
    client.post("/api/jobs", files={"file": ("x.png", b"nope", "image/png")})
    assert not data_dir.exists() or list(data_dir.iterdir()) == []


def test_engine_failure_is_500_without_paths(client, fake_engine):
    def boom(*a, **k):
        raise RuntimeError("cannot open /home/secret/thing")
    fake_engine["engine.tactile"].render_tactile = boom
    r = client.post("/api/jobs", files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 500
    assert r.json() == {"detail": "Processing failed: RuntimeError", "code": "server"}


# ---------------------------------------------------------------- get / 404
@pytest.mark.parametrize("jid", ["nothex!!!!", "ABCDEF0123", "..%2F..%2Fetc"])
def test_get_job_404(client, jid):
    r = client.get(f"/api/jobs/{jid}")
    assert r.status_code == 404 and r.json() == {"detail": "Job not found", "code": "not_found"}


def test_get_job_expired(client):
    """A well-formed id with no job: the link expired (or the free server restarted)."""
    for r in (client.get("/api/jobs/0123456789"), client.get("/api/jobs/0123456789/status"),
              client.patch("/api/jobs/0123456789", json={"title": "x"}),
              client.post("/api/jobs/0123456789/ai-describe", json={}),
              client.post("/api/jobs/0123456789/cancel"),
              client.get("/files/0123456789/tactile.png")):
        assert r.status_code == 404, r.request.url
        assert r.json()["code"] == "expired" and "24 h" in r.json()["detail"]


def test_path_traversal(client):
    job = _create(client)
    for url in ["/api/jobs/../../etc", "/files/../../etc/passwd", f"/files/{job['id']}/..%2Fresult.json",
                f"/files/{job['id']}/analysis.pkl", "/files/..%2F..%2Fetc/passwd"]:
        assert client.get(url).status_code == 404, url
    assert client.patch("/api/jobs/..%2F..%2Fetc", json={"title": "x"}).status_code == 404
    assert client.get(f"/files/{job['id']}/.edit_token").status_code == 404
    assert client.get(f"/files/{job['id']}/edit_token").status_code == 404


# ---------------------------------------------------------------- PATCH
def test_patch_rename_description_prefix_and_version(client):
    job = _create(client)
    jid = job["id"]
    nuc = next(p for p in job["parts"] if p["id"] == "p2")
    assert nuc["description"].startswith("Nucleus.") and nuc["key"] == "n"

    r = _patch(client, jid, {"parts": [{"id": "p2", "name": "  Brain  of the cell "}]})
    assert r.status_code == 200, r.text
    j1 = r.json()
    assert_job_result_schema(j1)
    p2 = next(p for p in j1["parts"] if p["id"] == "p2")
    assert p2["name"] == "Brain of the cell"
    assert p2["description"] == "Brain of the cell. A part of the cell."
    assert p2["key"] == "b"  # keys re-assigned after rename
    assert j1["id"] == jid and j1["created_at"] == job["created_at"]
    assert j1["files"]["pdf"].endswith("tactile.pdf?v=1") and j1["tactile"]["url"].endswith("?v=1")
    assert j1["timings"]["analyze"] == 0

    # name + description together: description wins
    j2 = _patch(client, jid, {"parts": [
        {"id": "p2", "name": "core", "description": "  The control centre.  "}]}).json()
    p2 = next(p for p in j2["parts"] if p["id"] == "p2")
    assert p2["name"] == "core" and p2["description"] == "The control centre."
    assert j2["files"]["stl"].endswith("?v=2")

    # rename when description doesn't start with old name -> unchanged
    j3 = _patch(client, jid, {"parts": [{"id": "p2", "name": "kernel"}]}).json()
    assert next(p for p in j3["parts"] if p["id"] == "p2")["description"] == "The control centre."

    # long values (at the limits) round-trip unchanged; over the limit is a 400, never truncated
    j4 = _patch(client, jid, {"parts": [{"id": "p3", "name": "x" * 60, "description": "y" * 400}]}).json()
    p3 = next(p for p in j4["parts"] if p["id"] == "p3")
    assert p3["name"] == "x" * 60 and p3["description"] == "y" * 400
    assert j4["version"] == 4

    # persisted
    assert client.get(f"/api/jobs/{jid}").json() == j4


def test_patch_include_and_title(client):
    job = _create(client)
    jid = job["id"]
    r = _patch(client, jid, {"title": "  Leaf cell ", "parts": [{"id": "p3", "include": False}]})
    assert r.status_code == 200
    j = r.json()
    assert_job_result_schema(j)
    assert j["title"] == "Leaf cell"
    p3 = next(p for p in j["parts"] if p["id"] == "p3")
    assert p3["include"] is False and p3["tactile_polygons"] == [] and p3["key_position"] is None
    assert p3["texture"] == "none" and p3["source_polygons"]
    assert [p["id"] for p in j["parts"]] == ["p1", "p2", "p3"]
    # download filename follows the new title
    d = client.get(f"/files/{jid}/tactile.pdf")
    assert d.headers["content-disposition"] == 'attachment; filename="feelable-leaf-cell.pdf"'
    # re-include
    j = _patch(client, jid, {"parts": [{"id": "p3", "include": True}]}).json()
    p3 = next(p for p in j["parts"] if p["id"] == "p3")
    assert p3["include"] and p3["tactile_polygons"] and p3["key_position"] is not None


@pytest.mark.parametrize("edit", [
    {"parts": [{"id": "p99", "name": "ghost"}]},
    {"parts": [{"id": "p1", "name": "   "}]},
    {"parts": [{"id": "p1", "include": "yes"}]},
    {"parts": [{"name": "no id"}]},
    {"title": ""},
    {"title": 5},
    {"parts": "p1"},
])
def test_patch_bad_edits(client, edit):
    job = _create(client)
    r = _patch(client, job["id"], edit)
    assert r.status_code == 400, r.text
    assert isinstance(r.json()["detail"], str) and r.json()["code"] == "invalid"
    # nothing changed (atomic)
    assert client.get(f"/api/jobs/{job['id']}").json() == job


def test_patch_partial_bad_edit_is_atomic(client):
    job = _create(client)
    r = _patch(client, job["id"], {"title": "new", "parts": [
        {"id": "p1", "name": "outer"}, {"id": "zzz", "name": "x"}]})
    assert r.status_code == 400 and "Unknown part id" in r.json()["detail"]
    assert client.get(f"/api/jobs/{job['id']}").json() == job


def test_patch_404_and_non_object(client):
    assert client.patch("/api/jobs/0123456789", json={"title": "x"}).status_code == 404
    job = _create(client)
    assert _patch(client, job["id"], [1, 2]).status_code == 400


# ---------------------------------------------------------------- files
def test_files_content_types_and_disposition(client):
    job = _create(client)
    expect = {"pdf": "application/pdf", "stl": "model/stl", "png": "image/png", "print_png": "image/png",
              "original": "image/png", "legend_png": "image/png"}
    for key, ctype in expect.items():
        r = client.get(job["files"][key])
        assert r.status_code == 200, key
        assert r.headers["content-type"].startswith(ctype), (key, r.headers["content-type"])
        cd = r.headers.get("content-disposition", "")
        if key in ("pdf", "stl"):
            assert cd == f'attachment; filename="feelable-plant-cell.{key}"'
        else:
            assert "attachment" not in cd
    r = client.get(f"/files/{job['id']}/result.json")
    assert r.status_code == 200 and r.json()["id"] == job["id"]
    assert client.get(f"/files/{job['id']}/missing.png").status_code == 404
    assert client.get("/files/0123456789/tactile.png").status_code == 404


# ---------------------------------------------------------------- SPA
def test_root_without_dist(client):
    r = client.get("/")
    assert r.status_code == 200 and "hint" in r.json()
    assert client.get("/whatever").status_code == 404


def test_spa_fallback(client, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Feelable</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setenv("FEELABLE_WEB_DIST", str(dist))
    r = client.get("/")
    assert r.status_code == 200 and "Feelable" in r.text and r.headers["content-type"].startswith("text/html")
    r = client.get("/assets/app.js")
    assert r.status_code == 200 and "console.log" in r.text
    r = client.get("/some/deep/route")
    assert r.status_code == 200 and "Feelable" in r.text
    assert client.get("/api/nope").status_code == 404
    assert client.get("/files/nope").status_code == 404
    assert client.get("/samples/nope.txt").status_code == 404
    assert "Feelable" in client.get("/..%2F..%2Fetc%2Fpasswd").text  # falls back to index, no traversal


def test_cors(client):
    r = client.get("/api/health", headers={"Origin": "http://example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_description_only_edit_skips_rerender(client, monkeypatch):
    from engine import pipeline
    job = _create(client)
    pid = job["parts"][0]["id"]
    calls = []
    real = pipeline.rerender_job
    monkeypatch.setattr(pipeline, "rerender_job", lambda *a, **k: calls.append(1) or real(*a, **k))
    r2 = _patch(client, job["id"], {"parts": [{"id": pid, "description": "Spoken text."}]})
    assert r2.status_code == 200 and not calls
    p = next(p for p in r2.json()["parts"] if p["id"] == pid)
    assert p["description"] == "Spoken text." and p["description_source"] == "teacher"
    assert r2.json()["version"] == 1 and r2.json()["files"] == job["files"]   # drawing untouched
    assert next(p for p in client.get(f"/api/jobs/{job['id']}").json()["parts"] if p["id"] == pid)["description"] == "Spoken text."
    r3 = _patch(client, job["id"], {"parts": [{"id": pid, "name": "renamed"}]})
    assert r3.status_code == 200 and calls
    p = next(p for p in r3.json()["parts"] if p["id"] == pid)
    assert p["match_method"] == "teacher" and p["description"] == "Spoken text."
    assert r3.json()["version"] == 2 and r3.json()["files"]["pdf"].endswith("?v=2")


def test_cancel_endpoint(client):
    assert client.post("/api/jobs/cancel/bad").status_code == 404
    r = client.post("/api/jobs/cancel/abcdef012345")
    assert r.status_code == 200 and r.json() == {"cancelled": True}


def test_upload_identical_to_sample_uses_cache(client, samples_dir, tmp_path, monkeypatch):
    from app import main
    (samples_dir / "plant-cell.png").write_bytes(png_bytes())
    (samples_dir / "manifest.json").write_text(json.dumps([
        {"id": "plant-cell", "title": "Plant cell", "file": "plant-cell.png", "kind": "colour"}]))
    # build a cache entry for the sample by running it once
    first = client.get("/api/samples").json()[0]
    job = client.post(f"/api/jobs/from-sample/{first['id']}?use_ai=false").json()
    cache = tmp_path / "cache" / "a4" / first["id"]
    import shutil
    shutil.copytree(main.storage.job_dir(job["id"]), cache)
    monkeypatch.setenv("FEELABLE_SAMPLE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(main, "_SAMPLE_HASHES", None)
    called = []
    monkeypatch.setattr(main, "_engine_call", lambda *a, **k: called.append(a) or {})
    data = (samples_dir / "plant-cell.png").read_bytes()
    r = client.post("/api/jobs", files={"file": ("x.png", data, "image/png")}, data={"use_ai": "false"})
    assert r.status_code == 201 and r.json()["id"] != job["id"]
    assert called == []
    clone = r.json()
    tok = clone.pop("edit_token")
    assert len(tok) == 32 and tok != job["edit_token"]          # every clone gets a fresh token
    assert_job_result_schema(clone)
    stored = (main.storage.job_dir(clone["id"]) / ".edit_token").read_text()
    assert stored == tok and tok not in json.dumps(client.get(f"/api/jobs/{clone['id']}").json())
    # the old token of the cached job does not open the clone
    assert client.patch(f"/api/jobs/{clone['id']}", json={"title": "x"},
                        headers={"X-Edit-Token": job["edit_token"]}).status_code == 403
    assert client.patch(f"/api/jobs/{clone['id']}", json={"title": "x"},
                        headers={"X-Edit-Token": tok}).status_code == 200
    # from-sample uses the cache too, and also hands out a new token
    r = client.post(f"/api/jobs/from-sample/{first['id']}?use_ai=false")
    assert r.status_code == 201 and not [c for c in called if c[0] == "run_job"]
    assert r.json()["edit_token"] not in (tok, job["edit_token"])
