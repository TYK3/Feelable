"""Natural-voice TTS: response parsing for both Gemini shapes, fallback, and endpoint guards."""
import base64

import httpx
import pytest

from engine import tts


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    tts._working.clear()
    yield
    tts._TRANSPORT = None


def test_interactions_shape(key):
    wav = tts._wav(b"\x00\x01" * 2000)
    tts._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(200, json={
        "steps": [{"type": "model_output", "content": [{"type": "audio", "data": base64.b64encode(wav).decode()}]}]}))
    assert tts.synth("Left ventricle.") == wav


def test_falls_back_to_generate_content_pcm(key):
    pcm = b"\x10\x00" * 3000

    def handler(r):
        if r.url.path.endswith("/interactions"):
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": base64.b64encode(pcm).decode()}}]}}]})

    tts._TRANSPORT = httpx.MockTransport(handler)
    out = tts.synth("Nucleus.")
    assert out[:4] == b"RIFF" and out.endswith(pcm)


def test_failure_returns_none_and_never_leaks_key(key):
    tts._TRANSPORT = httpx.MockTransport(lambda r: httpx.Response(403, json={"error": {"message": "bad test-key"}}))
    assert tts.synth("Heart.") is None
    assert "test-key" not in (tts.last_error() or "")


def test_endpoint_guards(client, monkeypatch):
    r = client.post("/api/jobs", files={"file": ("d.png", __import__("conftest").png_bytes(), "image/png")},
                    data={"use_ai": "false"})
    job = r.json()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert client.post(f"/api/jobs/{job['id']}/tts", json={"text": job["parts"][0]["name"]}).status_code == 503
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert client.post(f"/api/jobs/{job['id']}/tts", json={"text": "say something rude"}).status_code == 403
    monkeypatch.setattr(tts, "synth", lambda text, voice=None: tts._wav(b"\x00" * 4000))
    r = client.post(f"/api/jobs/{job['id']}/tts", json={"text": job["parts"][0]["name"]})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
