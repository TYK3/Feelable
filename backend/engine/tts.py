"""Natural-sounding speech for touch mode via Gemini TTS (optional).

The browser's built-in voices vary a lot (Windows' default "Desktop" voices sound robotic), so when
a Gemini key is set the server can synthesise each part's name and description with a neural
voice. Clips are cached per job. Any failure returns None and the browser voice is used instead.

Two request shapes are tried, because the Gemini API exposes TTS through both:
  1. POST /v1beta/interactions  (current docs; audio is base64 in steps[].content[].data, WAV)
  2. POST /v1beta/models/{m}:generateContent with responseModalities=["AUDIO"]
     (audio is base64 raw PCM L16 24 kHz mono in candidates[0].content.parts[].inlineData.data)
"""
from __future__ import annotations

import base64
import io
import logging
import os
import struct
import threading
from typing import Optional

import httpx

log = logging.getLogger("feelable.tts")

BASE = "https://generativelanguage.googleapis.com/v1beta"
MODELS = ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-2.5-flash-preview-tts")
DEFAULT_VOICE = "Sulafat"          # described by Google as "warm"; falls back to Kore if unknown
FALLBACK_VOICE = "Kore"
TIMEOUT_S = 20.0
MAX_CHARS = 420
STYLE = "Read this aloud warmly and clearly, at a calm, natural pace, for a student: "

_TRANSPORT = None                   # tests inject httpx.MockTransport
_local = threading.local()
_working: dict = {}                 # remembers which (shape, model) worked last, to try it first


def last_error() -> Optional[str]:
    return getattr(_local, "err", None)


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip()) and os.environ.get("FEELABLE_TTS", "1") != "0"


def _wav(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    buf.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE")
    buf.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
    buf.write(b"data" + struct.pack("<I", len(pcm)) + pcm)
    return buf.getvalue()


def _from_interactions(js: dict) -> Optional[bytes]:
    for step in reversed(js.get("steps") or []):
        for c in reversed(step.get("content") or []):
            if isinstance(c, dict) and c.get("type") == "audio" and c.get("data"):
                raw = base64.b64decode(c["data"])
                return raw if raw[:4] == b"RIFF" else _wav(raw)
    # some responses use outputs[] instead of steps[]
    for c in reversed(js.get("outputs") or []):
        if isinstance(c, dict) and c.get("type") == "audio" and c.get("data"):
            raw = base64.b64decode(c["data"])
            return raw if raw[:4] == b"RIFF" else _wav(raw)
    return None


def _from_generate(js: dict) -> Optional[bytes]:
    for cand in js.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data") or {}
            if inline.get("data"):
                raw = base64.b64decode(inline["data"])
                if raw[:4] == b"RIFF":
                    return raw
                mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
                rate = 24000
                if "rate=" in mime:
                    try:
                        rate = int(mime.split("rate=")[1].split(";")[0])
                    except ValueError:
                        pass
                return _wav(raw, rate)
    return None


def _attempts(text: str, voice: str):
    for model in MODELS:
        yield ("interactions", model, f"{BASE}/interactions", {
            "model": model,
            "input": [{"type": "text", "text": text}],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": voice}]},
        })
        yield ("generate", model, f"{BASE}/models/{model}:generateContent", {
            "contents": [{"role": "user", "parts": [{"text": STYLE + text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
            },
        })


def synth(text: str, voice: Optional[str] = None) -> Optional[bytes]:
    """WAV bytes for `text`, or None (see last_error())."""
    _local.err = None
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    text = " ".join((text or "").split())[:MAX_CHARS]
    if not key or not text:
        _local.err = "no key" if not key else "empty text"
        return None
    voice = voice or os.environ.get("FEELABLE_TTS_VOICE", "").strip() or DEFAULT_VOICE
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    kwargs = {"timeout": TIMEOUT_S}
    if _TRANSPORT is not None:
        kwargs["transport"] = _TRANSPORT
    tries = list(_attempts(text, voice))
    if voice != FALLBACK_VOICE:
        tries += list(_attempts(text, FALLBACK_VOICE))
    pref = _working.get("best")
    if pref:
        tries.sort(key=lambda t: 0 if (t[0], t[1]) == pref else 1)
    errors = []
    import time
    deadline = time.monotonic() + 45.0
    with httpx.Client(**kwargs) as client:
        for shape, model, url, body in tries:
            if time.monotonic() > deadline:
                errors.append("time budget used up")
                break
            try:
                r = client.post(url, headers=headers, json=body)
            except httpx.TimeoutException:
                errors.append(f"{shape}/{model}: timeout")
                continue
            except httpx.HTTPError as e:
                errors.append(f"{shape}/{model}: {type(e).__name__}")
                break                                   # network problem: don't hammer
            if r.status_code == 200:
                try:
                    js = r.json()
                except ValueError:
                    errors.append(f"{shape}/{model}: bad json")
                    continue
                audio = _from_interactions(js) if shape == "interactions" else _from_generate(js)
                if audio and len(audio) > 1000:
                    _working["best"] = (shape, model)
                    return audio
                errors.append(f"{shape}/{model}: no audio")
                continue
            if r.status_code in (401, 403):
                errors.append(f"{shape}/{model}: key rejected ({r.status_code})")
                break
            errors.append(f"{shape}/{model}: {r.status_code}")
            if r.status_code == 429:
                continue
    _local.err = "; ".join(errors)[:400] or "no audio"
    log.warning("TTS failed: %s", _local.err)
    return None
