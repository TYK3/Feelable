"""Dense / complex diagrams: spelling repair of blurry labels, zoom-and-carve grounding,
AI visual grounding, and slightly truncated downloads. (The full 24-image benchmark is
qa/dense/run.py; these are the fast, deterministic regressions.)"""
from __future__ import annotations

import io
import json

import cv2
import httpx
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from engine import ai, grounding
from engine.analyze import analyze, load_image
from engine.lexicon import correct, is_known

FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"


def _font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:  # pragma: no cover
        return ImageFont.load_default()


# ------------------------------------------------------------------ spelling repair
@pytest.mark.parametrize("raw, fixed", [
    ("Parktatbone", "Parietal bone"),          # real misreads from a 420 px skull screenshot
    ("Witroouschamber", "Vitreous chamber"),
    ("Corocalsure", "Coronal suture"),
    ("Occpitai bone", "Occipital bone"),
    ("Stytoidprocess", "Styloid process"),
    ("Scloral enousSirus", "Scleral venous sinus"),
    ("Lons", "Lens"),
    ("Maxila", "Maxilla"),
    ("Tounge", "Tongue"),
])
def test_lexicon_repairs_ocr_misreads(raw, fixed):
    assert correct(raw) == fixed


@pytest.mark.parametrize("text", [
    "Planet", "animal cell", "Cleveland Clinic", "Right lateral view", "Ovaries (female)",
    "Stapes (attached to oval window)", "HUMAN INTERNAL ORGANS", "Heart Left ventricle", "Rosna",
])
def test_lexicon_never_touches_real_words_or_unknowns(text):
    assert correct(text) == text


def test_is_known():
    assert is_known("Superior vena cava") and not is_known("Parktatbone")


# ------------------------------------------------------------------ zoom-and-carve
def make_shared_region_diagram() -> np.ndarray:
    """Two neighbouring organs of almost the same colour (one colour cluster) separated only by a
    darker crease, each with its own pointer line, plus a third organ that is easy."""
    W, H = 1000, 700
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((260, 180, 520, 520), fill=(196, 92, 92))          # "liver"
    d.ellipse((470, 200, 740, 500), fill=(198, 94, 93))          # "stomach": the same red
    d.arc((470, 200, 740, 500), 140, 220, fill=(150, 60, 60), width=4)   # only a faint crease between
    # (without zoom-and-carve both pointers land in one region and only one of them gets a shape)
    d.ellipse((560, 540, 700, 640), fill=(90, 120, 200))         # "kidney"
    lab = _font(24)
    for text, tx, ty, px, py in [("Liver", 40, 330, 340, 350), ("Stomach", 830, 330, 650, 350),
                                 ("Kidney", 830, 590, 640, 590)]:
        tw = d.textlength(text, font=lab)
        sx = tx - 8 if tx > 500 else tx + tw + 8
        d.line((sx, ty + 14, px, py), fill=(30, 30, 30), width=2)
        d.text((tx, ty), text, font=lab, fill=(20, 20, 20))
    return np.array(im)


def test_zoom_and_carve_gives_each_pointer_its_own_part():
    a = analyze(make_shared_region_diagram(), use_ai=False)
    names = {p.name for p in a.parts if p.include}
    assert {"liver", "stomach", "kidney"} <= names, names
    by = {p.name: p for p in a.parts}
    # the two halves really are separate shapes on their own sides of the crease
    s = a.width / 1000
    assert by["liver"].bbox[0] < 300 * s and by["stomach"].bbox[2] > 700 * s
    assert not ((by["liver"].mask > 0) & (by["stomach"].mask > 0)).any()


def test_carve_rejects_cuts_with_no_real_edge():
    # one flat region, two seeds: there is no structure boundary to split along
    img = np.full((400, 400, 3), 255, np.uint8)
    cv2.circle(img, (200, 200), 150, (200, 90, 90), -1)
    mask = np.zeros((400, 400), np.uint8)
    cv2.circle(mask, (200, 200), 150, 255, -1)
    from engine.types import Part
    from engine.vision_utils import mask_bbox, mask_polygons
    host = Part(id="p1", name="blob", description="", key="", color=(200, 90, 90), mask=mask,
                polygons=mask_polygons(mask), bbox=mask_bbox(mask), area=int((mask > 0).sum()))
    pieces = grounding.split_host(img, host, [grounding.Seed(1, "left", "Left", 120, 200)],
                                  grounding.Seed(0, "blob", "Blob", 280, 200))
    assert pieces == []


# ------------------------------------------------------------------ AI visual grounding
def _gemini_reply(obj) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(obj)}]},
                            "finishReason": "STOP"}]}


def test_locate_structures_parses_points_and_boxes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTESTKEY-0123456789abcdefghijkl")
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_gemini_reply({"items": [
            {"name": "liver", "found": True, "point": [500, 250], "box_2d": [400, 100, 600, 400]},
            {"name": "heart", "found": False},
            {"name": "everything", "found": True, "point": [500, 500], "box_2d": [0, 0, 1000, 1000]},
        ]}))

    ai._TRANSPORT = httpx.MockTransport(handler)
    try:
        out = ai.locate_structures(np.zeros((200, 400, 3), np.uint8), ["liver", "heart", "everything"])
    finally:
        ai._TRANSPORT = None
    assert out == {"liver": {"point": (100, 100), "box": (40, 80, 160, 120)}}
    prompt = seen["body"]["contents"][0]["parts"][0]["text"]
    assert "liver" in prompt and "INSIDE" in prompt


def test_ai_grounding_places_a_label_whose_pointer_is_lost(monkeypatch):
    """A label with a pointer that ends in empty space gets its shape from the image model."""
    W, H = 1000, 700
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((300, 200, 500, 420), fill=(180, 70, 70))          # heart (named by its pointer)
    d.ellipse((600, 240, 760, 380), fill=(90, 150, 90))          # spleen: its pointer stops short
    lab = _font(24)
    d.text((40, 300), "Heart", font=lab, fill=(20, 20, 20))
    d.line((120, 314, 380, 314), fill=(30, 30, 30), width=2)
    d.text((860, 120), "Spleen", font=lab, fill=(20, 20, 20))
    d.line((850, 134, 800, 150), fill=(30, 30, 30), width=2)     # ends in white space
    img = np.array(im)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTESTKEY-0123456789abcdefghijkl")

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        text = body["contents"][0]["parts"][0]["text"]
        if "find the structure itself" in text:
            return httpx.Response(200, json=_gemini_reply({"items": [
                {"name": "spleen", "found": True, "point": [443, 680], "box_2d": [330, 590, 556, 770]}]}))
        return httpx.Response(500, json={"error": {"message": "only locate is mocked"}})

    monkeypatch.setattr(ai, "_SLEEP", lambda s: None)
    ai._TRANSPORT = httpx.MockTransport(handler)
    try:
        a = analyze(img, use_ai=True)
    finally:
        ai._TRANSPORT = None
    by = {p.name: p for p in a.parts}
    assert "spleen" in by, list(by)
    assert by["spleen"].match_method == "ai"
    s = a.width / 1000
    x0, y0, x1, y1 = by["spleen"].bbox
    assert 560 * s <= x0 <= 640 * s and 720 * s <= x1 <= 800 * s


# ------------------------------------------------------------------ downloads
def test_slightly_truncated_jpeg_still_loads():
    im = Image.new("RGB", (400, 300), (240, 240, 240))
    ImageDraw.Draw(im).ellipse((100, 60, 300, 240), fill=(200, 60, 60))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=90)
    data = b.getvalue()
    arr = load_image(data[:-5])                                  # the last few bytes are missing
    assert arr.shape == (300, 400, 3) and arr[150, 200, 0] > 150
    with pytest.raises(ValueError):
        load_image(data[: len(data) // 2])                       # half a file is still refused


# ------------------------------------------------------------------ "What Feelable saw"
def test_vision_trace_reports_the_pipeline(tmp_path):
    from engine import pipeline
    b = io.BytesIO()
    Image.fromarray(make_shared_region_diagram()).save(b, "PNG")
    r = pipeline.run_job(b.getvalue(), str(tmp_path / "job1"), use_ai=False)
    v = pipeline.vision_trace(str(tmp_path / "job1"))
    assert v["width"] > 0 and v["height"] > 0
    texts = {lb["text"].lower() for lb in v["labels"]}
    assert {"liver", "stomach", "kidney"} <= texts
    named = [lb for lb in v["labels"] if lb["role"] == "named"]
    assert len(named) >= 3 and all(lb["part_id"] in {p["id"] for p in r["parts"]} for lb in named)
    assert len(v["leaders"]) >= 3 and all(len(L["points"]) >= 2 for L in v["leaders"])
    st = v["steps"]
    assert st["read"] >= 3 and st["pointers"] >= 3 and st["carved"] >= 1
    assert sum(st["methods"].values()) == sum(1 for p in r["parts"] if p["include"])


def test_trace_survives_single_point_pointer():
    from engine.analyze import _trace
    from engine.types import Label
    t = _trace({"leaders": [{"label": "x", "points": [(5, 5)]}, {"label": "y", "points": [(1, 1), (9, 9)]}]},
               [Label("x", (0, 0, 4, 4), 0.9)], [])
    assert [len(L["points"]) for L in t["leaders"]] == [1, 2]
