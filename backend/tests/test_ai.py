"""AI team tests for engine.ai (Gemini). Network is always mocked with httpx.MockTransport."""
from __future__ import annotations

import base64
import json
import logging

import httpx
import numpy as np
import pytest

from engine import ai
from engine.types import Analysis, Label, Part

KEY = "AIzaTESTKEY-0123456789abcdefghijkl"


def _analysis() -> Analysis:
    h, w = 300, 400
    img = np.full((h, w, 3), 255, np.uint8)
    m1 = np.zeros((h, w), np.uint8)
    m1[50:250, 50:350] = 255
    m2 = np.zeros((h, w), np.uint8)
    m2[120:180, 170:230] = 255
    m3 = np.zeros((h, w), np.uint8)
    m3[0:10, 0:400] = 255
    parts = [
        Part(id="p1", name="cytoplasm", description="Cytoplasm.", key="", color=(250, 200, 160), mask=m1,
             polygons=[[(50, 50), (350, 50), (350, 250), (50, 250)]], bbox=(50, 50, 350, 250), area=60000),
        Part(id="p2", name="part 2", description="Part 2.", key="", color=(90, 110, 210), mask=m2,
             polygons=[[(170, 120), (230, 120), (230, 180), (170, 180)]], bbox=(170, 120, 230, 180), area=3600),
        Part(id="p3", name="part 3", description="Part 3.", key="", color=(0, 0, 0), mask=m3,
             polygons=[[(0, 0), (400, 0), (400, 10), (0, 10)]], bbox=(0, 0, 400, 10), area=4000),
    ]
    labels = [Label(text="Cytoplasm", bbox=(360, 60, 395, 75), confidence=0.9, part_id="p1", role="part_label"),
              Label(text="Nucleus", bbox=(360, 140, 395, 155), confidence=0.9, role="part_label")]
    return Analysis(width=w, height=h, image=img, title="animal cell", diagram_type="diagram", parts=parts,
                    labels=labels, linework=np.zeros((h, w), np.uint8))


def _gemini_reply(obj, finish="STOP") -> dict:
    text = json.dumps(obj) if not isinstance(obj, str) else obj
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}],
            "modelVersion": "gemini-3.5-flash"}


GOOD = {
    "title": "Animal Cell", "diagram_type": "biology",
    "parts": [
        {"id": "p1", "name": "Cytoplasm", "description": "Fills the whole cell, around the nucleus. A jelly where "
                                                         "the cell's work happens.", "drop": False},
        {"id": "p2", "name": "nucleus", "description": "In the centre, inside the cytoplasm. Holds the cell's DNA.",
         "drop": False},
        {"id": "p3", "name": "border", "description": "A strip along the top edge.", "drop": True},
        {"id": "p99", "name": "ghost", "description": "x", "drop": True},
    ]}


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    sleeps = []
    monkeypatch.setattr(ai, "_SLEEP", sleeps.append)
    yield sleeps
    ai._TRANSPORT = None


def _img(h=50, w=50):
    return np.zeros((h, w, 3), np.uint8)


# ------------------------------------------------------------------ provider
def test_no_key_no_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert ai.get_provider_info() == {"provider": None, "model": None}
    assert ai.enrich(_img(10, 10), _analysis()) is None
    assert "no Gemini key" in ai.last_error()
    assert ai.describe_object(_img()) is None
    assert ai.describe_parts(_img(), _analysis(), ["p1"]) is None


def test_provider_info_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert ai.get_provider_info() == {"provider": "gemini", "model": "gemini-3.5-flash"}
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    assert ai.get_provider_info()["model"] == "gemini-test"


# ------------------------------------------------------------------ enrich
def test_enrich_success_and_request_shape(with_key):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_gemini_reply(GOOD))

    ai._TRANSPORT = httpx.MockTransport(handler)
    res = ai.enrich(np.full((1200, 1600, 3), 200, np.uint8), _analysis())
    assert res is not None and ai.last_error() is None
    assert res.title == "animal cell" and res.diagram_type == "biology"
    by = {p["id"]: p for p in res.parts}
    assert list(by) == ["p1", "p2", "p3"]                          # unknown id ignored
    # labelled part keeps its label; unlabelled part gets the AI name
    assert by["p1"]["name"] == "cytoplasm" and by["p2"]["name"] == "nucleus"
    # description = "Name." + 1-2 sentences saying where + what
    assert by["p2"]["description"] == "Nucleus. In the centre, inside the cytoplasm. Holds the cell's DNA."
    assert by["p1"]["description"].startswith("Cytoplasm. Fills the whole cell")
    assert by["p3"]["drop"] is True and by["p1"]["drop"] is False
    # request: key in header only, never in the URL
    assert "models/gemini-3.5-flash:generateContent" in seen["url"]
    assert KEY not in seen["url"] and seen["headers"]["x-goog-api-key"] == KEY
    body = seen["body"]
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    assert cfg["responseSchema"]["type"] == "OBJECT"
    assert cfg["responseSchema"]["properties"]["parts"]["items"]["required"] == ["id", "name", "description", "drop"]
    assert cfg["thinkingConfig"] == {"thinkingLevel": "low"}
    parts = body["contents"][0]["parts"]
    assert body["contents"][0]["role"] == "user"
    assert "p2" in parts[0]["text"] and "Nucleus" in parts[0]["text"] and "WHERE" in parts[0]["text"]
    jpg = base64.b64decode(parts[1]["inline_data"]["data"])
    assert jpg[:2] == b"\xff\xd8" and parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    import cv2
    dec = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert max(dec.shape[:2]) <= 1024


def test_labelled_part_keeps_label_and_is_never_dropped(with_key):
    reply = {"title": "cell", "diagram_type": "biology", "parts": [
        {"id": "p1", "name": "cell interior", "description": "Big.", "drop": True},     # rename + drop refused
        {"id": "p2", "name": "nucleus", "description": "Middle.", "drop": False}]}
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply(reply)))
    res = ai.enrich(_img(), _analysis())
    p1 = next(p for p in res.parts if p["id"] == "p1")
    assert p1["name"] == "cytoplasm" and p1["drop"] is False


def test_labelled_part_accepts_ocr_spelling_fix(with_key):
    an = _analysis()
    an.parts[0].name = "cytoplasn"            # OCR misread
    reply = {"title": "cell", "diagram_type": "biology",
             "parts": [{"id": "p1", "name": "cytoplasm", "description": "Around the middle.", "drop": False}]}
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply(reply)))
    assert ai.enrich(_img(), an).parts[0]["name"] == "cytoplasm"


def test_never_drops_every_part(with_key):
    reply = {"title": "x", "diagram_type": "other", "parts": [
        {"id": i, "name": "thing", "description": "Somewhere.", "drop": True} for i in ("p2", "p3")]}
    an = _analysis()
    an.parts[0].include = False
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply(reply)))
    res = ai.enrich(_img(), an)
    assert not any(p["drop"] for p in res.parts)


def test_long_description_capped_to_two_short_sentences(with_key):
    long = ("Top left, above everything else on the page. " + " ".join(["word"] * 40) + ". Third sentence here.")
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply(
        {"title": "t", "diagram_type": "biology",
         "parts": [{"id": "p2", "name": "nucleus", "description": long, "drop": "true"}]})))
    res = ai.enrich(_img(), _analysis())
    d = res.parts[0]["description"]
    assert d.startswith("Nucleus. Top left") and "Third" not in d
    assert len(d.split()) <= ai.DESC_MAX_WORDS + 3 and len(d) <= ai.DESC_MAX_CHARS
    assert res.parts[0]["drop"] is True


def test_model_repeating_name_is_not_doubled(with_key):
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply(
        {"title": "t", "diagram_type": "biology",
         "parts": [{"id": "p2", "name": "nucleus", "description": "Nucleus: in the middle", "drop": False}]})))
    assert ai.enrich(_img(), _analysis()).parts[0]["description"] == "Nucleus. In the middle."


def test_schema_violating_json_returns_none(with_key):
    for bad in ('{"title": 3, "parts": "nope"}', "[1, 2, 3]", "{not json at all"):
        ai._TRANSPORT = httpx.MockTransport(lambda r, b=bad: httpx.Response(200, json=_gemini_reply(b)))
        assert ai.enrich(_img(), _analysis()) is None
        assert "Gemini" in ai.last_error()


def test_bad_json_reason(with_key):
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply("{not json")))
    assert ai.enrich(_img(), _analysis()) is None
    assert "couldn't read" in ai.last_error()
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply('{"title": "x', "MAX_TOKENS")))
    assert ai.enrich(_img(), _analysis()) is None
    assert "cut off" in ai.last_error()


def test_markdown_fenced_json_and_thought_parts(with_key):
    txt = "```json\n" + json.dumps({"title": "plant cell", "diagram_type": "biology",
                                    "parts": [{"id": "p2", "name": "cell wall", "description": "Edge.", "drop": False}]}) + "\n```"
    reply = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": txt}]},
                             "finishReason": "STOP"}]}
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=reply))
    res = ai.enrich(_img(), _analysis())
    assert res is not None and res.parts[0]["name"] == "cell wall"


def test_429_then_success(with_key):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "2"},
                                  json={"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
        return httpx.Response(200, json=_gemini_reply(GOOD))

    ai._TRANSPORT = httpx.MockTransport(handler)
    assert ai.enrich(_img(), _analysis()) is not None
    assert len(calls) == 2 and all("gemini-3.5-flash:" in u for u in calls)
    assert with_key == [2.0] and ai.last_error() is None


def test_503_twice_gives_reason(with_key):
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE"}}))
    assert ai.enrich(_img(), _analysis()) is None
    assert "503" in ai.last_error() and len(with_key) == 1


def test_404_falls_back_to_lite_model(with_key):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "gemini-3.5-flash:" in str(request.url):
            return httpx.Response(404, json={"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
        return httpx.Response(200, json=_gemini_reply(GOOD))

    ai._TRANSPORT = httpx.MockTransport(handler)
    assert ai.enrich(_img(), _analysis()) is not None
    assert len(calls) == 2 and "gemini-3.5-flash-lite:generateContent" in calls[1]


def test_persistent_429_falls_back_then_reports_rate_limit(with_key):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})

    ai._TRANSPORT = httpx.MockTransport(handler)
    assert ai.enrich(_img(), _analysis()) is None
    assert [("lite" in u) for u in calls] == [False, False, True, True]
    assert ai.last_error() == "Gemini rate limit — try again in a minute"


def test_400_on_structured_config_retries_plain(with_key):
    bodies = []

    def handler(request):
        b = json.loads(request.content)
        bodies.append(b)
        if "responseSchema" in b["generationConfig"]:
            return httpx.Response(400, json={"error": {"code": 400, "message": "Unknown name thinkingLevel",
                                                       "status": "INVALID_ARGUMENT"}})
        return httpx.Response(200, json=_gemini_reply(GOOD))

    ai._TRANSPORT = httpx.MockTransport(handler)
    assert ai.enrich(_img(), _analysis()) is not None
    assert bodies[1]["generationConfig"] == {"responseMimeType": "application/json"}


def test_bad_key_message_never_contains_key(with_key, caplog):
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(400, json={"error": {
        "code": 400, "message": f"API key not valid. Please pass a valid API key. ({KEY})", "status": "INVALID_ARGUMENT"}}))
    with caplog.at_level(logging.DEBUG, logger="feelable.ai"):
        assert ai.enrich(_img(), _analysis()) is None
    assert "API key" in ai.last_error() and KEY not in ai.last_error()
    assert KEY not in caplog.text
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(403, json={"error": {"code": 403, "status": "PERMISSION_DENIED"}}))
    assert ai.enrich(_img(), _analysis()) is None and "key" in ai.last_error()


def test_timeout_and_network_errors(with_key):
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    ai._TRANSPORT = httpx.MockTransport(slow)
    assert ai.enrich(_img(), _analysis()) is None
    assert ai.last_error() == "AI timed out — Gemini took too long to answer"

    def down(request):
        raise httpx.ConnectError(f"cannot connect {KEY}", request=request)

    ai._TRANSPORT = httpx.MockTransport(down)
    assert ai.enrich(_img(), _analysis()) is None
    assert "network" in ai.last_error() and KEY not in ai.last_error()


def test_safety_block_reasons(with_key):
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    assert ai.enrich(_img(), _analysis()) is None and "safety" in ai.last_error()
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json={
        "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}))
    assert ai.enrich(_img(), _analysis()) is None and "safety" in ai.last_error()
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json={"candidates": []}))
    assert ai.enrich(_img(), _analysis()) is None and ai.last_error()


def test_success_clears_last_error(with_key):
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(500))
    assert ai.enrich(_img(), _analysis()) is None and ai.last_error()
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply(GOOD)))
    assert ai.enrich(_img(), _analysis()) is not None and ai.last_error() is None


def test_analyze_applies_mocked_gemini(with_key):
    """End-to-end: analyze() -> ai.enrich() over a mocked HTTP transport."""
    from engine.analyze import analyze
    try:
        from tests.test_analyze import make_colour_diagram
    except ImportError:  # pragma: no cover - when tests/ itself is on sys.path
        from test_analyze import make_colour_diagram

    def handler(request):
        prompt = json.loads(request.content)["contents"][0]["parts"][0]["text"]
        ids = sorted(set(__import__("re").findall(r'"id": "(p\d+)"', prompt)))
        parts = [{"id": i, "name": f"thing {i[1:]}", "description": f"Spot {i[1:]}. Made up.", "drop": i == ids[-1]}
                 for i in ids]
        return httpx.Response(200, json=_gemini_reply({"title": "cell (ai)", "diagram_type": "biology", "parts": parts}))

    ai._TRANSPORT = httpx.MockTransport(handler)
    a = analyze(make_colour_diagram(), use_ai=True)
    assert a.ai_used and a.title == "cell (ai)" and a.diagram_type == "biology"
    assert all(p.description.endswith("Made up.") for p in a.parts)


# ------------------------------------------------------------------ describe_object
def test_describe_object_box_conversion_and_clipping(with_key):
    seen = {}
    reply = {"title": "Computer Mouse", "summary": "a computer mouse, seen from above",
             "parts": [
                 {"name": "computer mouse", "description": "The whole mouse, filling the middle.", "box_2d": [100, 250, 900, 750]},
                 {"name": "left button", "description": "Front left, beside the scroll wheel. Click it to select.",
                  "box_2d": [100, 250, 400, 500]},
                 {"name": "scroll wheel", "description": "Front centre.", "box_2d": [-50, 480, 1200, 520]},  # clipped
                 {"name": "right button", "description": "Front right.", "box_2d": [400, 750, 100, 500]},   # swapped
                 {"name": "speck", "description": "x", "box_2d": [500, 500, 500, 500]},                     # degenerate
                 {"name": "bad", "description": "x", "box_2d": [1, 2, 3]},                                  # wrong length
                 {"name": "left button", "description": "dup", "box_2d": [0, 0, 10, 10]},                   # duplicate
             ]}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_gemini_reply(reply))

    ai._TRANSPORT = httpx.MockTransport(handler)
    res = ai.describe_object(np.zeros((600, 800, 3), np.uint8))     # h=600, w=800
    assert isinstance(res, ai.ObjectDescription)
    assert res.title == "computer mouse" and res.summary.startswith("A computer mouse")
    names = [p.name for p in res.parts]
    assert names == ["computer mouse", "left button", "scroll wheel", "right button"]
    assert res.parts[0].box == (200, 60, 600, 540)
    assert res.parts[1].box == (200, 60, 400, 240)
    assert res.parts[2].box == (384, 0, 416, 600)
    assert res.parts[3].box == (400, 60, 600, 240)
    assert res.parts[1].description == "Left button. Front left, beside the scroll wheel. Click it to select."
    cfg = seen["body"]["generationConfig"]
    box_schema = cfg["responseSchema"]["properties"]["parts"]["items"]["properties"]["box_2d"]
    assert box_schema == {"type": "ARRAY", "minItems": 4, "maxItems": 4, "items": {"type": "INTEGER"}}
    assert "[ymin, xmin, ymax, xmax]" in seen["body"]["contents"][0]["parts"][0]["text"]


def test_describe_object_caps_parts_and_handles_empty(with_key):
    many = {"title": "thing", "summary": "s",
            "parts": [{"name": f"piece {i}", "description": "d", "box_2d": [0, 0, 500, 500]} for i in range(12)]}
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply(many)))
    assert len(ai.describe_object(_img(100, 100)).parts) == 8
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply({"title": "x", "summary": "", "parts": []})))
    assert ai.describe_object(_img()) is None and "object" in ai.last_error()


# ------------------------------------------------------------------ describe_parts
def test_describe_parts(with_key):
    seen = {}

    def handler(request):
        seen["prompt"] = json.loads(request.content)["contents"][0]["parts"][0]["text"]
        return httpx.Response(200, json=_gemini_reply({"parts": [
            {"id": "p2", "description": "In the centre of the cell. Holds the DNA."},
            {"id": "p1", "description": "Not asked for."},
            {"id": "p2", "description": "duplicate"},
        ]}))

    ai._TRANSPORT = httpx.MockTransport(handler)
    an = _analysis()
    an.parts[1].name = "nucleus"
    out = ai.describe_parts(_img(), an, ["p2", "p404"])
    assert out == {"p2": "Nucleus. In the centre of the cell. Holds the DNA."}
    assert '["p2"]' in seen["prompt"]
    ai._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json=_gemini_reply({"parts": []})))
    assert ai.describe_parts(_img(), an, ["p2"]) is None and ai.last_error()
    assert ai.describe_parts(_img(), an, []) is None


def test_timeout_falls_back_to_lite_model(with_key):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "gemini-3.5-flash:" in str(request.url):
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=_gemini_reply(GOOD))

    ai._TRANSPORT = httpx.MockTransport(handler)
    assert ai.enrich(_img(), _analysis()) is not None
    assert len(calls) == 2 and "flash-lite" in calls[1]
