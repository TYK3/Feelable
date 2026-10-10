"""Optional AI enrichment (Gemini) for part names / spoken descriptions.

OWNER: AI team.

Offline by default: only active when GEMINI_API_KEY is set. Every public function here
never raises: on any problem (no key, no network, timeout, rate limit, bad JSON, safety
block) it returns None, and `last_error()` gives a short human reason the caller can show.

Request format (Gemini REST `models/{model}:generateContent`, v1beta):
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  header  x-goog-api-key: <key>          (never in the URL, so it can't leak into logs)
  body    {"contents": [{"role": "user", "parts": [{"text": ...},
                         {"inline_data": {"mime_type": "image/jpeg", "data": <base64>}}]}],
           "generationConfig": {"responseMimeType": "application/json",
                                "responseSchema": <OpenAPI-subset schema>,
                                "thinkingConfig": {"thinkingLevel": "low"}}}
  reply   {"candidates": [{"content": {"parts": [{"text": "<json>"}]}, "finishReason": "STOP"}],
           "promptFeedback": {"blockReason": ...}}            (promptFeedback only when blocked)
  errors  {"error": {"code": 429, "message": "...", "status": "RESOURCE_EXHAUSTED"}}

If the server rejects the optional parts of the config (400), we retry once with a plain
JSON-mode request (no schema, no thinking config); the validators below never trust the
reply's shape anyway.
"""
from __future__ import annotations

import base64
import difflib
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from engine.types import Analysis

log = logging.getLogger("feelable.ai")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.5-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"
TIMEOUT_S = 22.0            # per HTTP attempt (leaves the lite fallback ~25 s of the budget)
TOTAL_BUDGET_S = 50.0       # whole call, including the retry and the fallback model
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_IMAGE_SIDE = 1024
THINKING_LEVEL = "low"      # Gemini 3.x flash defaults to "medium", which is slow for this job
MAX_OBJECT_PARTS = 8        # main subject + 7
DESC_MAX_WORDS = 32         # 1-2 short spoken sentences (not counting the leading name)
DESC_MAX_CHARS = 300        # API limit is 400

# Plain-English data notice for the upload switch and /api/health `ai.sends` (audit F21).
SENDS = ("When AI descriptions are on, a smaller copy of your image and the label text we read from it "
         "are sent to Google Gemini to name the parts and write the spoken descriptions. "
         "Nothing else is sent, and Feelable deletes its copy after 24 hours.")

# tests may inject an httpx transport (e.g. httpx.MockTransport)
_TRANSPORT = None
# tests replace this to avoid real sleeping during backoff
_SLEEP = time.sleep

_state = threading.local()


# ------------------------------------------------------------------ public data types
@dataclass
class AIEnrichment:
    title: str
    diagram_type: str
    parts: list[dict] = field(default_factory=list)  # [{"id","name","description","drop": bool}]


@dataclass
class ObjectPart:
    name: str
    description: str
    box: tuple[int, int, int, int]      # x0, y0, x1, y1 in pixels of the image passed in


@dataclass
class ObjectDescription:
    title: str
    summary: str
    parts: list[ObjectPart]             # main subject first, at most 8


# ------------------------------------------------------------------ provider / errors
def get_provider_info() -> dict:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return {"provider": None, "model": None}
    return {"provider": "gemini", "model": os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL}


def last_error() -> Optional[str]:
    """Human reason for the most recent failed AI call in this thread (None after a success)."""
    return getattr(_state, "error", None)


def _set_error(msg: Optional[str]) -> None:
    _state.error = msg


class _AIError(Exception):
    """Internal: carries a human, key-free reason."""


def _redact(text: str) -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    text = str(text)
    if key:
        text = text.replace(key, "***")
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "***", text)


# ------------------------------------------------------------------ image / helpers
def _encode_jpeg(image_rgb: np.ndarray) -> str:
    img = np.ascontiguousarray(image_rgb)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = np.ascontiguousarray(img[:, :, :3])
    img = img.astype(np.uint8)
    h, w = img.shape[:2]
    s = min(1.0, MAX_IMAGE_SIDE / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise _AIError("Couldn't prepare the image for AI")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _nearby_labels(analysis: Analysis, bbox, radius: int = 150) -> list[str]:
    x0, y0, x1, y1 = bbox
    out = []
    for lb in analysis.labels:
        if lb.role == "title":
            continue
        bx0, by0, bx1, by1 = lb.bbox
        dx = max(bx0 - x1, 0, x0 - bx1)
        dy = max(by0 - y1, 0, y0 - by1)
        if (dx * dx + dy * dy) ** 0.5 <= radius:
            out.append(lb.text)
    return out[:6]


def _position_words(bbox, w: int, h: int) -> str:
    """Coarse page position of a box centre, e.g. 'top left', 'middle', 'bottom right'."""
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / max(1, w)
    cy = (y0 + y1) / 2 / max(1, h)
    v = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
    hz = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "centre")
    if v == "middle" and hz == "centre":
        return "centre"
    if v == "middle":
        return f"middle {hz}"
    return f"{v} {hz}" if hz != "centre" else f"{v} centre"


def _label_for(analysis: Analysis, p) -> Optional[str]:
    """The label printed in the image that names this part, if any."""
    lt = getattr(p, "label_text", None)
    if isinstance(lt, str) and lt.strip():
        return lt.strip()
    return next((lb.text for lb in analysis.labels if lb.part_id == p.id), None)


def _is_labelled(analysis: Analysis, p) -> bool:
    if getattr(p, "description_source", "auto") == "teacher" or getattr(p, "match_method", "none") == "teacher":
        return True
    if getattr(p, "match_method", "none") in ("leader", "inside", "nearest"):
        return True
    return _label_for(analysis, p) is not None and not p.name.startswith("part ")


def _part_facts(analysis: Analysis, part_ids=None) -> list[dict]:
    out = []
    for p in analysis.parts:
        if part_ids is not None and p.id not in part_ids:
            continue
        out.append({
            "id": p.id,
            "current_name": p.name,
            "label_in_image": _label_for(analysis, p) if _is_labelled(analysis, p) else None,
            "position": _position_words(p.bbox, analysis.width, analysis.height),
            "bbox_xyxy": [int(v) for v in p.bbox],
            "colour_rgb": [int(v) for v in p.color],
            "count": int(p.count),
            "area_px": int(p.area),
            "nearby_label_texts": _nearby_labels(analysis, p.bbox),
        })
    return out


def _capitalise(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


_NAME_STRIP = re.compile(r"[^\w' \-]", re.UNICODE)


def _clean_name(name, max_words: int = 4) -> str:
    if not isinstance(name, str):
        return ""
    name = _NAME_STRIP.sub(" ", name.lower().replace("_", " "))
    name = " ".join(name.split()[:max_words]).strip(" -'")
    return name[:40].strip()


def _clean_sentences(desc, name: str = "") -> str:
    """1-2 short sentences for speech. If `name` is given the result starts with 'Name.'."""
    if not isinstance(desc, str):
        return ""
    desc = re.sub(r"\s+", " ", desc.replace("*", "").replace("`", "")).strip().strip('"').strip()
    if not desc:
        return ""
    # drop a leading "Name." / "Name:" / "Name -" that the model may have added; we add our own
    if name:
        m = re.match(r"^\s*" + re.escape(name) + r"\s*[.:\-–—]\s*", desc, re.IGNORECASE)
        if m and m.end() < len(desc):
            desc = desc[m.end():]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", desc) if s.strip()]
    sents = sents[:2]
    if len(" ".join(sents).split()) > DESC_MAX_WORDS and len(sents) > 1:
        sents = sents[:1]
    body = " ".join(sents)
    words = body.split()
    if len(words) > DESC_MAX_WORDS:
        body = " ".join(words[:DESC_MAX_WORDS]).rstrip(",;:-–—") + "."
    if body and body[-1] not in ".!?":
        body += "."
    body = _capitalise(body)
    out = f"{_capitalise(name)}. {body}" if name else body
    if len(out) > DESC_MAX_CHARS:
        out = out[:DESC_MAX_CHARS].rsplit(" ", 1)[0].rstrip(",;:-–—") + "."
    return out


# ------------------------------------------------------------------ HTTP core
def _schema_obj(props: dict, required: list[str]) -> dict:
    return {"type": "OBJECT", "properties": props, "required": required, "propertyOrdering": list(props)}


def _call_gemini(prompt: str, image_rgb: np.ndarray, schema: dict, what: str):
    """Send prompt + image, return the parsed JSON reply. Raises _AIError with a human reason."""
    info = get_provider_info()
    if info["provider"] != "gemini":
        raise _AIError("AI is off — no Gemini key is set on the server")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    import httpx

    image_b64 = _encode_jpeg(image_rgb)
    contents = [{"role": "user", "parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
    ]}]
    full_cfg = {"responseMimeType": "application/json", "responseSchema": schema,
                "thinkingConfig": {"thinkingLevel": THINKING_LEVEL}}
    plain_cfg = {"responseMimeType": "application/json"}

    main = info["model"]
    models = [main] + ([FALLBACK_MODEL] if main != FALLBACK_MODEL else [])
    t_start = time.monotonic()
    deadline = t_start + TOTAL_BUDGET_S
    kwargs = {"timeout": TIMEOUT_S}
    if _TRANSPORT is not None:
        kwargs["transport"] = _TRANSPORT
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}

    last: Optional[_AIError] = None
    with httpx.Client(**kwargs) as client:
        for mi, model in enumerate(models):
            cfg = full_cfg
            retried_status = False
            retried_plain = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining < 2.0:
                    raise last or _AIError("AI timed out — Gemini took too long to answer")
                try:
                    resp = client.post(GEMINI_URL.format(model=model), headers=headers,
                                       json={"contents": contents, "generationConfig": cfg},
                                       timeout=min(TIMEOUT_S, remaining))
                except httpx.TimeoutException:
                    last = _AIError("AI timed out — Gemini took too long to answer")
                    if mi == 0 and len(models) > 1 and deadline - time.monotonic() > 8.0:
                        break                   # slow main model -> try the faster lite model
                    raise last
                except httpx.HTTPError as e:
                    raise _AIError(f"Couldn't reach Gemini (network error: {type(e).__name__})")

                st = resp.status_code
                if st == 200:
                    data = _parse_reply(resp)
                    log.info("AI %s ok: model=%s %.1fs", what, model, time.monotonic() - t_start)
                    return data
                err_msg, err_status = _error_detail(resp)
                if st == 400 and re.search(r"api[ _]?key", err_msg, re.I):
                    raise _AIError("Gemini rejected the API key — check GEMINI_API_KEY on the server")
                if st in (401, 403):
                    raise _AIError("Gemini rejected the API key or this project has no access "
                                   f"({st} {err_status or ''})".rstrip())
                if st == 400 and cfg is full_cfg and not retried_plain:
                    # schema / thinking config not accepted by this model: try plain JSON mode once
                    log.warning("AI %s: model %s rejected structured config (%s); retrying plain JSON",
                                what, model, _redact(err_msg)[:160])
                    cfg, retried_plain = plain_cfg, True
                    continue
                if st == 400:
                    raise _AIError(f"Gemini rejected the request ({_redact(err_msg)[:120] or 'bad request'})")
                if st == 404:
                    last = _AIError(f"Gemini model '{model}' was not found")
                    break                       # -> fallback model
                if st == 429:
                    last = _AIError("Gemini rate limit — try again in a minute")
                elif st in RETRY_STATUSES:
                    last = _AIError(f"Gemini is having problems (error {st}) — try again later")
                else:
                    raise _AIError(f"Gemini error {st}: {_redact(err_msg)[:120]}")
                if not retried_status:
                    retried_status = True
                    _SLEEP(_backoff(resp))
                    continue
                if st == 429:
                    break                       # still rate limited -> fallback model
                raise last
            if mi == 0 and len(models) > 1:
                log.warning("AI %s: %s on %s; trying %s", what, last, model, FALLBACK_MODEL)
    raise last or _AIError("Gemini didn't answer")


def _backoff(resp) -> float:
    ra = resp.headers.get("retry-after")
    try:
        v = float(ra) if ra is not None else 1.5
    except ValueError:
        v = 1.5
    return max(0.5, min(v, 4.0))


def _error_detail(resp) -> tuple[str, str]:
    try:
        e = resp.json().get("error") or {}
        return str(e.get("message", "")), str(e.get("status", ""))
    except Exception:
        return "", ""


_BLOCK_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "RECITATION"}


def _parse_reply(resp):
    try:
        payload = resp.json()
    except Exception:
        raise _AIError("Gemini sent an answer we couldn't read (not JSON)")
    if not isinstance(payload, dict):
        raise _AIError("Gemini sent an answer we couldn't read")
    block = (payload.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise _AIError(f"Gemini blocked this image (safety filter: {str(block).lower()})")
    cands = payload.get("candidates") or []
    if not cands or not isinstance(cands[0], dict):
        raise _AIError("Gemini returned no answer")
    cand = cands[0]
    finish = str(cand.get("finishReason") or "")
    parts = ((cand.get("content") or {}).get("parts")) or []
    text = "".join(p.get("text", "") for p in parts
                   if isinstance(p, dict) and not p.get("thought") and isinstance(p.get("text"), str))
    if finish in _BLOCK_FINISH and not text.strip():
        raise _AIError(f"Gemini blocked this image (safety filter: {finish.lower()})")
    if not text.strip():
        raise _AIError("Gemini returned an empty answer")
    try:
        return _extract_json(text)
    except ValueError:
        if finish == "MAX_TOKENS":
            raise _AIError("Gemini's answer was cut off (too long)")
        raise _AIError("Gemini sent an answer we couldn't read (bad JSON)")


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _run(what: str, fn):
    """Run fn(); record success/failure for last_error(); never raise."""
    _set_error(None)
    try:
        out = fn()
    except _AIError as e:
        msg = _redact(str(e))
        _set_error(msg)
        log.warning("AI %s failed: %s", what, msg)
        return None
    except Exception as e:  # programming / unexpected errors: still never raise
        msg = f"AI failed unexpectedly ({type(e).__name__})"
        _set_error(msg)
        log.warning("AI %s failed: %s: %s", what, msg, _redact(str(e))[:200])
        return None
    if out is None and last_error() is None:
        _set_error("Gemini returned no usable descriptions")
    return out


# ------------------------------------------------------------------ enrich (diagram)
DIAGRAM_TYPES = ["biology", "chemistry", "physics", "chart", "map", "circuit", "geometry", "diagram", "other"]

ENRICH_SCHEMA = _schema_obj({
    "title": {"type": "STRING", "description": "short lowercase subject of the diagram"},
    "diagram_type": {"type": "STRING", "enum": DIAGRAM_TYPES},
    "parts": {"type": "ARRAY", "items": _schema_obj({
        "id": {"type": "STRING"},
        "name": {"type": "STRING"},
        "description": {"type": "STRING"},
        "drop": {"type": "BOOLEAN"},
    }, ["id", "name", "description", "drop"])},
}, ["title", "diagram_type", "parts"])

_SPEECH_RULES = (
    "Descriptions are read aloud to a blind student who is exploring a raised-line (tactile) copy with "
    "their fingers. Write 1-2 short, plain sentences (at most 25 words in total), in this order:\n"
    "  1. WHERE it is on the page, relative to other named parts (e.g. \"Top left, above the left ventricle.\").\n"
    "  2. WHAT it is or does, in words a 12-year-old understands (e.g. \"Receives oxygen-rich blood from the lungs.\").\n"
    "Do NOT start with the part's name (we say the name first ourselves). Do not mention colours, "
    "pixels, bounding boxes, labels, arrows or 'the image'. If there are several copies, say how many.\n"
)


def build_prompt(analysis: Analysis) -> str:
    texts = [lb.text for lb in analysis.labels]
    return (
        "You help make tactile graphics for blind students. The image is a diagram "
        f"({analysis.width}x{analysis.height} px; the bboxes below use these pixel coordinates).\n"
        f"Detected title text: {analysis.title!r}. All text read from the image: {json.dumps(texts)}.\n"
        "Computer vision found these regions (parts):\n"
        f"{json.dumps(_part_facts(analysis), indent=1)}\n\n"
        "Return JSON: {\"title\", \"diagram_type\", \"parts\": [{\"id\", \"name\", \"description\", \"drop\"}]}, "
        "one entry per part id above (keep the ids, invent none).\n"
        "Rules:\n"
        "- title: short lowercase subject of the diagram, no 'Fig.' numbers (e.g. \"human heart\").\n"
        f"- diagram_type: one of {', '.join(DIAGRAM_TYPES)}.\n"
        "- name: if label_in_image is set, use that label (only fix obvious spelling mistakes from text "
        "reading). Otherwise (current_name like \"part 3\") name it from what you see: 1-3 lowercase words. "
        "Never use brand names, logo text or figure numbers as names.\n"
        "- description: " + _SPEECH_RULES +
        "- drop: true only if the part is decoration, shading, background, border or noise, not content. "
        "Never drop a part that has label_in_image.\n"
    )


def _validate(data, analysis: Analysis) -> Optional[AIEnrichment]:
    if not isinstance(data, dict):
        raise _AIError("Gemini's answer didn't have the expected shape")
    by_id = {p.id: p for p in analysis.parts}
    title = data.get("title")
    dtype = data.get("diagram_type")
    title = re.sub(r"\s+", " ", title).strip().lower()[:80] if isinstance(title, str) else ""
    title = re.sub(r"^(fig(ure)?\.?\s*\d+[a-z]?[.:]?\s*)", "", title).strip()
    dtype = re.sub(r"[^a-z ]", "", dtype.lower()).strip()[:30] if isinstance(dtype, str) else ""
    raw_parts = data.get("parts", [])
    if not isinstance(raw_parts, list):
        raise _AIError("Gemini's answer didn't have the expected shape")
    out, seen = [], set()
    for item in raw_parts:
        if not isinstance(item, dict):
            continue
        pid = item.get("id")
        if pid not in by_id or pid in seen:
            continue  # unknown ids ignored
        seen.add(pid)
        p = by_id[pid]
        labelled = _is_labelled(analysis, p)
        ai_name = _clean_name(item.get("name"))
        if labelled:
            # keep the printed label; accept the AI's spelling only if it's a near-identical fix
            name = p.name
            if ai_name and ai_name != p.name and difflib.SequenceMatcher(None, ai_name, p.name).ratio() >= 0.8:
                name = ai_name
        else:
            name = ai_name or p.name
        desc = _clean_sentences(item.get("description"), name)
        drop = item.get("drop", False)
        drop = drop if isinstance(drop, bool) else str(drop).lower() == "true"
        if labelled:
            drop = False
        out.append({"id": pid, "name": name, "description": desc, "drop": drop})
    # never let the AI switch every included part off
    included = {p.id for p in analysis.parts if p.include}
    if included and all(it["drop"] for it in out if it["id"] in included) and \
            included <= {it["id"] for it in out}:
        for it in out:
            it["drop"] = False
    if not out and not title:
        raise _AIError("Gemini returned no usable descriptions")
    return AIEnrichment(title=title, diagram_type=dtype, parts=out)


def enrich(image_rgb: np.ndarray, analysis: Analysis) -> Optional[AIEnrichment]:
    """Ask Gemini for names, spoken descriptions and decoration flags. Never raises; None on any
    problem (see last_error()). Labelled parts keep their label as the name."""
    def go():
        data = _call_gemini(build_prompt(analysis), image_rgb, ENRICH_SCHEMA, "enrich")
        return _validate(data, analysis)
    return _run("enrich", go)


# ------------------------------------------------------------------ describe_parts (on demand)
PARTS_SCHEMA = _schema_obj({
    "parts": {"type": "ARRAY", "items": _schema_obj({
        "id": {"type": "STRING"},
        "description": {"type": "STRING"},
    }, ["id", "description"])},
}, ["parts"])


def describe_parts(image_rgb: np.ndarray, analysis: Analysis, part_ids) -> Optional[dict[str, str]]:
    """Fresh spoken descriptions for the given part ids (names are kept as they are).
    Returns {id: description} for the parts Gemini described, or None (see last_error())."""
    want = [pid for pid in dict.fromkeys(part_ids or []) if any(p.id == pid for p in analysis.parts)]

    def go():
        if not want:
            raise _AIError("No parts to describe")
        all_parts = [{"id": f["id"], "name": f["current_name"], "position": f["position"],
                      "bbox_xyxy": f["bbox_xyxy"]} for f in _part_facts(analysis)]
        prompt = (
            "You help make tactile graphics for blind students. The image is a diagram titled "
            f"{analysis.title!r} ({analysis.width}x{analysis.height} px; bboxes use these pixel coordinates).\n"
            f"All parts, for context: {json.dumps(all_parts)}\n"
            f"Write a new description for each of these part ids: {json.dumps(want)}. "
            "Keep each part's name as given.\n"
            "Return JSON: {\"parts\": [{\"id\", \"description\"}]}.\n"
            "description: " + _SPEECH_RULES
        )
        data = _call_gemini(prompt, image_rgb, PARTS_SCHEMA, "describe_parts")
        items = data.get("parts") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise _AIError("Gemini's answer didn't have the expected shape")
        by_id = {p.id: p for p in analysis.parts}
        out: dict[str, str] = {}
        for it in items:
            if not isinstance(it, dict) or it.get("id") not in want or it["id"] in out:
                continue
            d = _clean_sentences(it.get("description"), by_id[it["id"]].name)
            if d:
                out[it["id"]] = d
        if not out:
            raise _AIError("Gemini returned no usable descriptions")
        return out
    return _run("describe_parts", go)


# ------------------------------------------------------------------ describe_object (photos)
OBJECT_SCHEMA = _schema_obj({
    "title": {"type": "STRING"},
    "summary": {"type": "STRING"},
    "parts": {"type": "ARRAY", "maxItems": MAX_OBJECT_PARTS, "items": _schema_obj({
        "name": {"type": "STRING"},
        "description": {"type": "STRING"},
        "box_2d": {"type": "ARRAY", "minItems": 4, "maxItems": 4, "items": {"type": "INTEGER"}},
    }, ["name", "description", "box_2d"])},
}, ["title", "summary", "parts"])

OBJECT_PROMPT = (
    "This is a photo of a real object. We will turn its outline into a raised-line (tactile) picture "
    "for a blind student, who explores it with their fingers while a phone reads descriptions aloud.\n"
    "Return JSON: {\"title\", \"summary\", \"parts\": [{\"name\", \"description\", \"box_2d\"}]}.\n"
    "- title: what the object is, 1-4 lowercase words (e.g. \"computer mouse\"). No brand names.\n"
    "- summary: one short sentence describing the whole object for the student.\n"
    "- parts: FIRST the whole main subject, with a box around all of it. Then up to 7 parts of it that "
    "can be felt and matter for understanding it (e.g. a computer mouse: body, left button, right button, "
    "scroll wheel, side button). Skip the background, shadows, reflections, logos and printed text.\n"
    "- name: 1-3 lowercase words. Never a brand name or logo text.\n"
    "- description: " + _SPEECH_RULES.replace("relative to other named parts", "on the object, relative to its other parts") +
    "- box_2d: [ymin, xmin, ymax, xmax] of the part, normalized to 0-1000 (integers).\n"
)


def _box_to_px(box, w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, v)) for v in vals)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    x0 = int(math.floor(xmin / 1000.0 * w))
    y0 = int(math.floor(ymin / 1000.0 * h))
    x1 = int(math.ceil(xmax / 1000.0 * w))
    y1 = int(math.ceil(ymax / 1000.0 * h))
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)


def _validate_object(data, w: int, h: int) -> ObjectDescription:
    if not isinstance(data, dict) or not isinstance(data.get("parts"), list):
        raise _AIError("Gemini's answer didn't have the expected shape")
    title = _clean_name(data.get("title"), max_words=5) or "object"
    summary = _clean_sentences(data.get("summary"))
    parts: list[ObjectPart] = []
    names: set[str] = set()
    for it in data["parts"]:
        if not isinstance(it, dict):
            continue
        box = _box_to_px(it.get("box_2d"), w, h)
        name = _clean_name(it.get("name"))
        if box is None or not name or name in names:
            continue
        names.add(name)
        parts.append(ObjectPart(name=name, description=_clean_sentences(it.get("description"), name), box=box))
        if len(parts) >= MAX_OBJECT_PARTS:
            break
    if not parts:
        raise _AIError("Gemini didn't find an object in this photo")
    return ObjectDescription(title=title, summary=summary, parts=parts)


def describe_object(image_rgb: np.ndarray) -> Optional[ObjectDescription]:
    """Name a photographed object and its touchable parts, with boxes in pixels of `image_rgb`.
    Main subject first, at most 8 parts. Never raises; None on any problem (see last_error())."""
    def go():
        h, w = image_rgb.shape[:2]
        data = _call_gemini(OBJECT_PROMPT, image_rgb, OBJECT_SCHEMA, "describe_object")
        return _validate_object(data, w, h)
    return _run("describe_object", go)


# ------------------------------------------------------------------ locate (visual grounding)
LOCATE_MAX = 16
LOCATE_SCHEMA = _schema_obj({
    "items": {"type": "ARRAY", "maxItems": LOCATE_MAX, "items": _schema_obj({
        "name": {"type": "STRING"},
        "found": {"type": "BOOLEAN"},
        "point": {"type": "ARRAY", "minItems": 2, "maxItems": 2, "items": {"type": "INTEGER"}},
        "box_2d": {"type": "ARRAY", "minItems": 4, "maxItems": 4, "items": {"type": "INTEGER"}},
    }, ["name", "found"])},
}, ["items"])


def locate_structures(image_rgb: np.ndarray, names: list[str], title: str = "") -> Optional[dict]:
    """Visual grounding: where in the picture is each named structure?

    Used when a label's pointer line couldn't be followed to its shape (realistic paintings, faint
    lines). Returns {name: {"point": (x, y), "box": (x0, y0, x1, y1)}} in pixels of `image_rgb` for
    the structures Gemini could see, or None (see last_error()). Never raises."""
    want = [n for n in dict.fromkeys(names) if n][:LOCATE_MAX]

    def go():
        if not want:
            raise _AIError("Nothing to locate")
        h, w = image_rgb.shape[:2]
        prompt = (
            "This is a labelled science diagram" + (f" titled {title!r}" if title else "") + ". "
            "For each structure name below, find the structure itself in the picture — the drawn "
            "thing the label points at, not the label text.\n"
            f"Names: {json.dumps(want)}\n"
            "Return JSON: {\"items\": [{\"name\", \"found\", \"point\", \"box_2d\"}]} with one item per name, "
            "in the same order. found=false if you can't see it. point: [y, x] of a spot clearly "
            "INSIDE the structure (not on its edge or on a label line). box_2d: [ymin, xmin, ymax, xmax] "
            "tightly around the structure. All coordinates normalized to 0-1000, integers."
        )
        data = _call_gemini(prompt, image_rgb, LOCATE_SCHEMA, "locate")
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise _AIError("Gemini's answer didn't have the expected shape")
        out: dict = {}
        for it in items:
            if not isinstance(it, dict) or not it.get("found") or it.get("name") not in want:
                continue
            box = _box_to_px(it.get("box_2d"), w, h)
            pt = it.get("point")
            if box is None or not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            try:
                py, px = float(pt[0]), float(pt[1])
            except (TypeError, ValueError):
                continue
            x, y = int(round(px / 1000.0 * w)), int(round(py / 1000.0 * h))
            x0, y0, x1, y1 = box
            if not (x0 <= x < x1 and y0 <= y < y1):
                x, y = (x0 + x1) // 2, (y0 + y1) // 2
            area = (x1 - x0) * (y1 - y0)
            if area > 0.5 * w * h:
                continue                       # "the whole picture" is not a location
            out[it["name"]] = {"point": (x, y), "box": box}
        if not out:
            raise _AIError("Gemini couldn't find the labelled structures")
        return out
    return _run("locate", go)
