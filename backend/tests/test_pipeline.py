"""Pipeline tests. OWNER: API team.

Most tests use the fake engine (conftest). `test_real_engine_end_to_end` runs the REAL engine
and is skipped until engine.analyze / engine.tactile / engine.export import cleanly.
"""
from __future__ import annotations

import importlib
import json
import os
import pickle
import threading

import cv2
import numpy as np
import pytest

from conftest import assert_job_result_schema, png_bytes


def _job_dir(tmp_path, jid="abcdef0123"):
    d = tmp_path / "jobs" / jid
    d.mkdir(parents=True)
    return str(d)


def test_run_job_writes_result_and_pickle(fake_engine, tmp_path):
    from engine import pipeline
    jd = _job_dir(tmp_path)
    r = pipeline.run_job(png_bytes(), jd, use_ai=False, paper="a4")
    assert_job_result_schema(r)
    assert r["id"] == "abcdef0123"
    assert json.load(open(os.path.join(jd, "result.json"))) == r
    state = pickle.load(open(os.path.join(jd, "analysis.pkl"), "rb"))
    assert state["analysis"].title == "plant cell" and state["options"].paper == "a4"
    assert state["version"] == 0
    assert all(p.key for p in state["analysis"].parts)
    assert set(r["timings"]) == {"analyze", "render", "export", "total"}


def test_run_job_bad_image(fake_engine, tmp_path):
    from engine import pipeline
    with pytest.raises(pipeline.BadImage):
        pipeline.run_job(b"not an image", _job_dir(tmp_path))


def test_run_job_bad_paper(fake_engine, tmp_path):
    from engine import pipeline
    with pytest.raises(ValueError):
        pipeline.run_job(png_bytes(), _job_dir(tmp_path), paper="a3")


def test_build_result_converts_numpy_and_tuples(fake_engine):
    from engine import pipeline
    from conftest import fake_analysis
    tac_mod = fake_engine["engine.tactile"]
    a = fake_analysis()
    tac_mod.assign_keys(a.parts)
    a.parts[0].polygons = [np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.int32)]  # cv2 contour shape
    t = tac_mod.render_tactile(a)
    r = pipeline.build_result("abcdef0123", "2026-10-10T00:00:00Z", a, t, {}, version=3,
                              timings={"analyze": np.float64(1), "render": 0.1, "export": 0.2, "total": 1.3})
    assert_job_result_schema(r)
    p2 = next(p for p in r["parts"] if p["id"] == "p2")
    assert p2["source_polygons"] == [[[1, 2], [3, 4], [5, 6]]]
    assert r["files"]["original"] == "/files/abcdef0123/original.png?v=3"
    assert r["source"]["url"] == r["files"]["original"]


def test_rerender_concurrent_edits_are_serialised(fake_engine, tmp_path):
    from engine import pipeline
    jd = _job_dir(tmp_path)
    pipeline.run_job(png_bytes(), jd, use_ai=False)
    errors = []

    def edit(i):
        try:
            pipeline.rerender_job(jd, {"title": f"t{i}"})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=edit, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    r = json.load(open(os.path.join(jd, "result.json")))
    assert r["files"]["pdf"].endswith("?v=6")


def test_rerender_unknown_part(fake_engine, tmp_path):
    from engine import pipeline
    jd = _job_dir(tmp_path)
    pipeline.run_job(png_bytes(), jd, use_ai=False)
    with pytest.raises(ValueError):
        pipeline.rerender_job(jd, {"parts": [{"id": "nope", "include": False}]})


# ---------------------------------------------------------------- real engine
def _real_engine_available() -> tuple[bool, str]:
    try:
        a = importlib.import_module("engine.analyze")
        t = importlib.import_module("engine.tactile")
        e = importlib.import_module("engine.export")
        for mod, names in ((a, ("load_image", "analyze")), (t, ("assign_keys", "render_tactile")),
                           (e, ("export_all",))):
            for n in names:
                if not hasattr(mod, n):
                    return False, f"{mod.__name__}.{n} missing"
        return True, ""
    except Exception as ex:  # noqa: BLE001
        return False, f"real engine not importable: {type(ex).__name__}: {ex}"


_REAL_OK, _REAL_WHY = _real_engine_available()


def synthetic_diagram() -> bytes:
    """A 1200x800 coloured diagram: 3 filled shapes with leader lines + text labels."""
    img = np.full((800, 1200, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Simple Cell", (420, 70), font, 1.6, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.rectangle(img, (150, 150), (850, 700), (60, 140, 60), -1)        # big green region (BGR)
    cv2.rectangle(img, (150, 150), (850, 700), (20, 60, 20), 4)
    cv2.circle(img, (400, 420), 120, (200, 120, 170), -1)                 # purple circle
    cv2.circle(img, (400, 420), 120, (90, 40, 80), 3)
    cv2.ellipse(img, (680, 330), (90, 50), 20, 0, 360, (40, 170, 240), -1)  # orange ellipse
    cv2.ellipse(img, (680, 330), (90, 50), 20, 0, 360, (20, 80, 120), 3)
    for (x0, y0), (x1, y1), text in [((520, 420), (930, 420), "nucleus"),
                                      ((770, 330), (930, 250), "vacuole"),
                                      ((800, 650), (930, 600), "cytoplasm")]:
        cv2.line(img, (x0, y0), (x1, y1), (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x1 + 10, y1 + 12), font, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.mark.skipif(not _REAL_OK, reason=_REAL_WHY)
def test_real_engine_end_to_end(tmp_path, monkeypatch):
    from engine import pipeline
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    jd = _job_dir(tmp_path, "0123456789")
    r = pipeline.run_job(synthetic_diagram(), jd, use_ai=False, paper="a4")
    assert_job_result_schema(r)
    assert len(r["parts"]) >= 2, [p["name"] for p in r["parts"]]
    for key in ("pdf", "stl", "png", "print_png", "original", "legend_png"):
        base = r["files"][key].split("/")[-1].split("?")[0]
        assert os.path.getsize(os.path.join(jd, base)) > 0, key
    assert open(os.path.join(jd, "tactile.pdf"), "rb").read(5) == b"%PDF-"

    # an edit round-trip on the real engine
    first = r["parts"][0]
    r2 = pipeline.rerender_job(jd, {"title": "my cell", "parts": [{"id": first["id"], "name": "outer layer"},
                                                                 {"id": r["parts"][-1]["id"], "include": False}]})
    assert_job_result_schema(r2)
    assert r2["title"] == "my cell"
    assert next(p for p in r2["parts"] if p["id"] == first["id"])["name"] == "outer layer"
    assert r2["files"]["pdf"].endswith("?v=1")


def test_clone_job_rewrites_ids_and_urls(tmp_path):
    import json, pickle
    from engine import pipeline
    src = tmp_path / "aaaaaaaaaa"; src.mkdir()
    (src / "tactile.png").write_bytes(b"png")
    result = {"id": "aaaaaaaaaa", "created_at": "x", "files": {"png": "/files/aaaaaaaaaa/tactile.png?v=0"}}
    (src / pipeline.RESULT_FILE).write_text(json.dumps(result))
    (src / pipeline.STATE_FILE).write_bytes(pickle.dumps({"job_id": "aaaaaaaaaa", "created_at": "x", "version": 0}))
    dst = tmp_path / "bbbbbbbbbb"
    r = pipeline.clone_job(str(src), str(dst))
    assert r["id"] == "bbbbbbbbbb" and r["files"]["png"].startswith("/files/bbbbbbbbbb/")
    assert (dst / "tactile.png").read_bytes() == b"png"
    assert pickle.loads((dst / pipeline.STATE_FILE).read_bytes())["job_id"] == "bbbbbbbbbb"
    assert json.loads((src / pipeline.RESULT_FILE).read_text())["id"] == "aaaaaaaaaa"   # source untouched


@pytest.mark.skipif(not _REAL_OK, reason=_REAL_WHY)
def test_lazy_stl_built_on_demand(tmp_path, monkeypatch):
    from engine import pipeline
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("FEELABLE_LAZY_STL", "1")
    jd = _job_dir(tmp_path, "1123456789")
    pipeline.run_job(synthetic_diagram(), jd, use_ai=False, paper="a4")
    stl = os.path.join(jd, "relief.stl")
    assert not os.path.exists(stl)
    assert pipeline.ensure_stl(jd) == stl
    data = open(stl, "rb").read()
    assert len(data) > 1000
    assert pipeline.ensure_stl(jd) == stl and open(stl, "rb").read() == data   # second call reuses it


def test_worker_call_runs_named_function(fake_engine, tmp_path):
    from engine import pipeline
    jd = _job_dir(tmp_path, "2123456789")
    r = pipeline.worker_call("run_job", png_bytes(), jd, use_ai=False, paper="a4")
    assert r["id"] == "2123456789"


def test_ai_describe_job_updates_descriptions_only(fake_engine, tmp_path, monkeypatch):
    from engine import ai, pipeline
    jd = _job_dir(tmp_path, "3123456789")
    r0 = pipeline.run_job(png_bytes(), jd, use_ai=False, paper="a4")
    first = r0["parts"][0]

    class Enr:
        title, diagram_type = "ignored title", None
        parts = [{"id": first["id"], "description": "AI wrote this.", "name": "ignored name"}]

    monkeypatch.setattr(ai, "enrich", lambda img, analysis: Enr(), raising=False)
    r = pipeline.ai_describe_job(jd)
    p = next(x for x in r["parts"] if x["id"] == first["id"])
    assert r["ai_used"] is True and p["description"] == "AI wrote this."
    assert p["name"] == first["name"] and r["title"] == r0["title"]
    monkeypatch.setattr(ai, "enrich", lambda img, analysis: None)
    r = pipeline.ai_describe_job(jd)
    assert any("AI description unavailable" in w for w in r["warnings"])


@pytest.mark.skipif(not _REAL_OK, reason=_REAL_WHY)
def test_published_bundle_is_immutable_until_reapproved(tmp_path, monkeypatch):
    """UX01: an unapproved draft must not change ANY approved media; re-approval switches all of them."""
    import hashlib
    from engine import pipeline
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    jd = _job_dir(tmp_path, "4123456789")
    r = pipeline.run_job(synthetic_diagram(), jd, use_ai=False, paper="a4")
    assert pipeline.quick_edit(jd, {"approved": True}) is not None
    pub1 = pipeline.load_published(jd)
    v1 = pub1["version"]
    assert pub1["tactile"]["url"] == f"/files/4123456789/pub/v{v1}/tactile.png"
    assert "?" not in pub1["files"]["pdf"]
    bundle = os.path.join(jd, "pub", f"v{v1}")
    sha = {n: hashlib.sha256(open(os.path.join(bundle, n), "rb").read()).hexdigest()
           for n in ("tactile.png", "tactile_print.png", "legend.png", "tactile.pdf", "original.png")}
    n_inc = sum(p["include"] for p in pub1["parts"])

    # unapproved draft: switch a part off (re-renders everything in the job folder)
    first = r["parts"][0]["id"]
    pipeline.rerender_job(jd, {"parts": [{"id": first, "include": False}]})
    pub_after = pipeline.load_published(jd)
    assert pub_after["version"] == v1 and sum(p["include"] for p in pub_after["parts"]) == n_inc
    for n, h in sha.items():
        assert hashlib.sha256(open(os.path.join(bundle, n), "rb").read()).hexdigest() == h, n
    # the draft really did change the working image
    assert hashlib.sha256(open(os.path.join(jd, "tactile.png"), "rb").read()).hexdigest() != sha["tactile.png"]

    # re-approval publishes the new version: JSON and every asset switch together
    res = pipeline.quick_edit(jd, {"approved": True}) or pipeline.rerender_job(jd, {"approved": True})
    pub2 = pipeline.load_published(jd)
    assert pub2["version"] > v1 and pub2["tactile"]["url"].startswith(f"/files/4123456789/pub/v{pub2['version']}/")
    assert sum(p["include"] for p in pub2["parts"]) == n_inc - 1
    b2 = os.path.join(jd, "pub", f"v{pub2['version']}")
    assert hashlib.sha256(open(os.path.join(b2, "tactile.png"), "rb").read()).hexdigest() != sha["tactile.png"]
    # lazy STL is built inside the bundle from the bundle's own image
    stl = pipeline.ensure_pub_stl(jd, pub2["version"])
    assert stl.startswith(b2) and os.path.getsize(stl) > 1000
