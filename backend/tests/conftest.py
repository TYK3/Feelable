"""Shared fixtures for API/pipeline tests. OWNER: API team.

The real engine modules are developed in parallel, so these tests inject a small FAKE engine
(engine.analyze / engine.tactile / engine.export / engine.ai) via sys.modules.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from engine.types import (  # noqa: E402
    PX_PER_MM, Analysis, Check, Label, Part, TactileOptions, TactileResult,
)


# ---------------------------------------------------------------- fake engine
def _part(pid, name, color, mask, kind="region"):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [[(int(p[0][0]), int(p[0][1])) for p in c] for c in cnts]
    x, y, w, h = cv2.boundingRect(mask)
    return Part(id=pid, name=name, description=f"{name[:1].upper()}{name[1:]}. A part of the cell.",
                key="", color=np.array(color, dtype=np.uint8), mask=mask, polygons=polys,
                bbox=(np.int64(x), np.int64(y), np.int64(x + w), np.int64(y + h)),
                area=np.int64(int((mask > 0).sum())), count=len(cnts), kind=kind)


def fake_analysis(use_ai: bool = False) -> Analysis:
    W, H = 300, 200
    img = np.full((H, W, 3), 255, np.uint8)
    m1 = np.zeros((H, W), np.uint8); cv2.rectangle(m1, (10, 10), (290, 190), 255, -1)
    m2 = np.zeros((H, W), np.uint8); cv2.circle(m2, (100, 100), 40, 255, -1)
    m3 = np.zeros((H, W), np.uint8)
    cv2.ellipse(m3, (220, 70), (20, 12), 0, 0, 360, 255, -1)
    cv2.ellipse(m3, (220, 140), (20, 12), 0, 0, 360, 255, -1)
    parts = [
        _part("p2", "nucleus", (170, 120, 200), m2),
        _part("p1", "cell wall", (40, 100, 40), m1),
        _part("p3", "chloroplast", (60, 160, 60), m3),
    ]
    return Analysis(width=W, height=H, image=img, title="plant cell", diagram_type="biology",
                    parts=parts, labels=[Label("nucleus", (90, 150, 140, 165), 0.9, "p2", "part_label")],
                    linework=np.zeros((H, W), np.uint8), dropped=["12 small dots (too small to feel)"],
                    warnings=["fake warning"], timings={"analyze": 0.25}, ai_used=False)


def make_fake_engine():
    analyze = types.ModuleType("engine.analyze")

    def load_image(data: bytes) -> np.ndarray:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("Not an image")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def analyze_fn(image_rgb, *, use_ai=True):
        return fake_analysis(use_ai)

    analyze.load_image = load_image
    analyze.analyze = analyze_fn

    tactile = types.ModuleType("engine.tactile")

    def assign_keys(parts):
        used = set()
        for p in parts:
            letters = "".join(c for c in p.name.lower() if "a" <= c <= "z") or "x"
            for k in (letters[0], letters[:2]) + tuple("abcdefghijklmnopqrstuvwxyz"):
                if k not in used:
                    p.key = k
                    used.add(k)
                    break

    def render_tactile(analysis, options=TactileOptions()):
        w, h = (1485, 1050) if options.paper == "a4" else (1397, 1080)
        inc = [p for p in analysis.parts if p.include]
        polys = {p.id: [[(x * 4 + 60, y * 4 + 80) for x, y in np.asarray(poly).reshape(-1, 2)] for poly in p.polygons]
                 for p in inc}
        return TactileResult(
            width=w, height=h, px_per_mm=PX_PER_MM,
            ink=np.zeros((h, w), np.uint8), legend_ink=np.zeros((h, w), np.uint8),
            print_layer=np.full((h, w, 3), 255, np.uint8), part_polygons=polys,
            key_positions={p.id: (np.int32(50 + i * 30), np.float64(40.0)) for i, p in enumerate(inc)},
            textures={p.id: t for p, t in zip(inc, ["none", "horiz", "dots"])},
            legend=[{"part_id": p.id, "key": p.key, "name": p.name} for p in inc],
            checks=[Check("Lines at least 1 mm", np.bool_(True), "thinnest line 1.5 mm")],
            transform=(4.0, 60.0, 80.0))

    tactile.assign_keys = assign_keys
    tactile.render_tactile = render_tactile

    export = types.ModuleType("engine.export")

    def export_all(analysis, tac, outdir):
        out = {}
        tiny = np.full((8, 8, 3), 255, np.uint8)
        for key, name in [("original", "original.png"), ("png", "tactile.png"),
                          ("print_png", "tactile_print.png"), ("legend_png", "legend.png")]:
            p = os.path.join(outdir, name)
            cv2.imwrite(p, tiny)
            out[key] = os.path.abspath(p)
        p = os.path.join(outdir, "tactile.pdf")
        Path(p).write_bytes(b"%PDF-1.4\n%fake\n%%EOF\n")
        out["pdf"] = os.path.abspath(p)
        p = os.path.join(outdir, "relief.stl")
        Path(p).write_text("solid fake\nendsolid fake\n")
        out["stl"] = os.path.abspath(p)
        return out

    export.export_all = export_all

    ai = types.ModuleType("engine.ai")
    ai.get_provider_info = lambda: {"provider": None, "model": None}
    return {"engine.analyze": analyze, "engine.tactile": tactile, "engine.export": export, "engine.ai": ai}


@pytest.fixture
def fake_engine(monkeypatch):
    import engine
    mods = make_fake_engine()
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(engine, name.split(".")[1], mod, raising=False)
    return mods


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "jobs"
    monkeypatch.setenv("FEELABLE_DATA", str(d))
    return d


@pytest.fixture
def samples_dir(tmp_path, monkeypatch):
    d = tmp_path / "samples"
    d.mkdir()
    monkeypatch.setenv("FEELABLE_SAMPLES", str(d))
    return d


@pytest.fixture
def client(fake_engine, data_dir, samples_dir, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FEELABLE_WEB_DIST", str(tmp_path / "no-dist"))
    from app.main import app
    with TestClient(app) as c:
        yield c


def png_bytes(w=120, h=80, color=(200, 50, 50)) -> bytes:
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), color, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------- schema checker (mirrors web/src/types.ts)
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_point(pt):
    assert isinstance(pt, list) and len(pt) == 2 and all(_is_int(v) for v in pt), pt


def _check_polygons(polys):
    assert isinstance(polys, list)
    for poly in polys:
        assert isinstance(poly, list)
        for pt in poly:
            _check_point(pt)


PART_KEYS = {"id", "key", "name", "description", "include", "color", "texture", "count", "area",
             "source_bbox", "source_polygons", "tactile_polygons", "key_position",
             "label_text", "match_method", "description_source"}                       # v2
RESULT_KEYS = {"id", "created_at", "title", "diagram_type", "ai_used", "page_flattened", "source", "tactile",
               "parts", "dropped", "warnings", "checks", "timings", "files",
               "version", "input_kind", "notes", "retention_hours", "expires_at", "readiness"}  # v2
INPUT_KINDS = {"diagram", "chart", "photo_object", "photo_scene", "text_page", "blank"}
ISSUE_ACTIONS = {"rename", "include", "split", "merge", "retake", "reduce", "draw"}
FILE_KEYS = {"pdf", "stl", "png", "print_png", "original", "legend_png"}
TEXTURES = {"none", "dots", "diag", "horiz", "vert", "cross", "solid", "grid"}


def assert_job_result_schema(r: dict) -> None:
    # round-trip through JSON: no tuples / numpy leftovers
    assert json.loads(json.dumps(r)) == r
    assert set(r) == RESULT_KEYS, set(r) ^ RESULT_KEYS
    assert isinstance(r["id"], str) and len(r["id"]) == 10
    assert isinstance(r["created_at"], str) and r["created_at"].endswith("Z") and "T" in r["created_at"]
    for k in ("title", "diagram_type"):
        assert isinstance(r[k], str)
    for k in ("ai_used", "page_flattened"):
        assert isinstance(r[k], bool)
    s = r["source"]
    assert set(s) == {"width", "height", "url"} and _is_int(s["width"]) and _is_int(s["height"])
    assert s["url"].startswith(f"/files/{r['id']}/original.png")
    t = r["tactile"]
    assert set(t) == {"width", "height", "px_per_mm", "url", "print_url", "legend_url", "stl_size_mm"}
    assert t["stl_size_mm"] is None or (len(t["stl_size_mm"]) == 3 and all(_is_num(v) for v in t["stl_size_mm"]))
    assert _is_int(t["width"]) and _is_int(t["height"]) and isinstance(t["px_per_mm"], float)
    for k in ("url", "print_url", "legend_url"):
        assert isinstance(t[k], str) and t[k].startswith(f"/files/{r['id']}/")
    assert isinstance(r["parts"], list)
    areas = [p["area"] for p in r["parts"]]
    assert areas == sorted(areas, reverse=True), "parts must be ordered by area desc"
    for p in r["parts"]:
        assert set(p) == PART_KEYS, set(p) ^ PART_KEYS
        for k in ("id", "key", "name", "description", "texture"):
            assert isinstance(p[k], str), k
        assert p["texture"] in TEXTURES
        assert isinstance(p["include"], bool)
        assert isinstance(p["color"], list) and len(p["color"]) == 3 and all(_is_int(c) for c in p["color"])
        assert _is_int(p["count"]) and _is_int(p["area"])
        assert isinstance(p["source_bbox"], list) and len(p["source_bbox"]) == 4
        assert all(_is_int(v) for v in p["source_bbox"])
        _check_polygons(p["source_polygons"])
        _check_polygons(p["tactile_polygons"])
        if p["key_position"] is not None:
            _check_point(p["key_position"])
        if not p["include"]:
            assert p["tactile_polygons"] == [] and p["key_position"] is None
        assert p["label_text"] is None or isinstance(p["label_text"], str)
        assert isinstance(p["match_method"], str)
        assert p["description_source"] in ("auto", "ai", "teacher")
    for k in ("dropped", "warnings"):
        assert isinstance(r[k], list) and all(isinstance(x, str) for x in r[k])
    assert isinstance(r["checks"], list)
    for c in r["checks"]:
        assert set(c) == {"rule", "passed", "detail"}
        assert isinstance(c["rule"], str) and isinstance(c["passed"], bool) and isinstance(c["detail"], str)
    assert isinstance(r["timings"], dict)
    for k in ("analyze", "render", "export", "total"):
        assert _is_num(r["timings"][k]), k
    assert set(r["files"]) == FILE_KEYS
    # v2 (CONTRACT-v2 §2.7)
    assert "edit_token" not in r
    assert _is_int(r["version"]) and r["version"] >= 0
    assert r["input_kind"] in INPUT_KINDS
    assert isinstance(r["notes"], list) and all(isinstance(x, str) for x in r["notes"])
    assert r["retention_hours"] == 24
    assert isinstance(r["expires_at"], str) and r["expires_at"].endswith("Z") and r["expires_at"] > r["created_at"]
    rd = r["readiness"]
    assert set(rd) == {"status", "approved", "geometry", "issues", "unmatched_labels"}
    assert rd["status"] in ("blocked", "needs_review", "ready") and isinstance(rd["approved"], bool)
    assert set(rd["geometry"]) == {"passed", "total"}
    assert rd["geometry"]["total"] == len(r["checks"])
    assert rd["geometry"]["passed"] == sum(c["passed"] for c in r["checks"])
    assert all(isinstance(x, str) for x in rd["unmatched_labels"])
    for i in rd["issues"]:
        assert {"code", "severity", "message"} <= set(i) <= {"code", "severity", "message", "part_id", "action", "target", "bbox"}
        assert i["severity"] in ("block", "warn", "info") and isinstance(i["message"], str)
        assert i.get("action", "rename") in ISSUE_ACTIONS
    blocked = any(i["severity"] == "block" for i in rd["issues"])
    assert (rd["status"] == "blocked") == blocked
    if not blocked:
        assert rd["status"] == ("ready" if rd["approved"] else "needs_review")
    for v in r["files"].values():
        assert isinstance(v, str) and v.startswith(f"/files/{r['id']}/")
        assert "/home" not in v and "data/jobs" not in v
