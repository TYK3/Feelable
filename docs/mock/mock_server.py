"""Mock Feelable API implementing CONTRACT-v2 §2 (for frontend work only — NOT the real backend).

Run:  PORT=8204 DIST=web/dist python3 docs/mock/mock_server.py
      PORT    default 8204
      DIST    optional folder with a built SPA to serve at / (so Playwright can open http://127.0.0.1:PORT/)
      MOCK_AI "1" (default) pretends Gemini is configured; "0" = AI off (health provider null, ai-describe 503)

Behaviour
- GET  /api/health                      limits, retention, ai.sends
- GET  /api/samples                     the real samples/manifest.json (thumbs served from /samples/)
- POST /api/jobs (multipart, async=1)   202 {id, edit_token, status} then stages advance over ~4 s
       file name tricks for testing:     *busy*  -> 503 busy      *blank* -> blank page (blocked)
                                         *slow*  -> stages take ~12 s   non-image bytes -> error code "decode"
                                         > 15 MB -> 413 too_big
- POST /api/jobs (no async)             synchronous result + edit_token
- POST /api/jobs/from-sample/{id}       synchronous clone (heart sample -> heart template, others -> plant cell)
- GET  /api/jobs/{id}                   404 not_found; id "expired" -> 404 expired
- GET  /api/jobs/{id}/status            queued|running|done|error|cancelled with stage + elapsed
- POST /api/jobs/{id}/cancel            needs X-Edit-Token
- PATCH /api/jobs/{id}                  needs X-Edit-Token; limits, base_version (409), split/merge/approved
- POST /api/jobs/{id}/ai-describe       needs X-Edit-Token; 503 ai_off when MOCK_AI=0
- GET  /files/{id}/{name}               the template's files
"""
import copy, io, json, os, secrets, time, uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SAMPLES = os.path.join(REPO, "samples")
AI_ON = os.environ.get("MOCK_AI", "1") != "0"
LIMITS = {"name": 60, "description": 400, "title": 80}
RETENTION_H = 24

app = FastAPI()


def err(status, code, detail):
    return JSONResponse({"detail": detail, "code": code}, status_code=status)


# ---------------------------------------------------------------- templates
METHODS = ["leader", "inside", "leader", "nearest", "leader", "inside", "leader", "nearest", "leader"]


def load_template(name):
    j = json.load(open(os.path.join(HERE, "jobs", name, "result.json")))
    for i, p in enumerate(j["parts"]):
        p.setdefault("label_text", p["name"])
        p.setdefault("match_method", METHODS[i % len(METHODS)])
        p.setdefault("description_source", "auto")
    j["tactile"].setdefault("stl_size_mm", [274.8, 198.8, 3.0])
    return j


TEMPLATES = {"heart": load_template("heart"), "plant-cell": load_template("plant-cell")}
# make the heart a realistic "needs review" case: one unnamed part + an unmatched label
_h = TEMPLATES["heart"]["parts"][0]
_h.update(name="part 1", label_text=None, match_method="none", description="Part 1.")
TEMPLATES["heart"]["_unmatched"] = ["myocardium"]

JOBS: dict = {}  # id -> state


def now_iso(dt=None):
    return (dt or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_readiness(j, approved):
    inc = [p for p in j["parts"] if p["include"]]
    issues = []
    if not inc:
        issues.append({"code": "no_parts", "severity": "block", "message": "No parts are switched on, so there is nothing to feel."})
    if j.get("input_kind") in ("text_page", "blank"):
        issues.append({"code": "no_diagram", "severity": "block", "action": "retake",
                       "message": "This photo doesn't seem to contain a diagram. Take a photo of the diagram itself, filling the frame."})
    for p in inc:
        if p["name"].lower().startswith("part "):
            issues.append({"code": "unnamed", "severity": "warn", "action": "rename", "part_id": p["id"],
                           "message": f"“{p['name']}” has no name yet. Give it the name a student should hear."})
    for lab in j.get("_unmatched", []):
        issues.append({"code": "unmatched_label", "severity": "warn",
                       "message": f"The label “{lab}” wasn't matched to any part. Rename a part if it belongs to one."})
    if len(inc) > 8:
        issues.append({"code": "too_many", "severity": "warn", "action": "reduce",
                       "message": f"{len(inc)} parts is a lot to read by touch. Switch off the least important ones (8 or fewer is best)."})
    for c in j["checks"]:
        if not c["passed"] and not c["rule"].startswith("8 parts"):
            issues.append({"code": "geometry", "severity": "warn", "message": f"{c['rule']}: {c['detail']}"})
    if j.get("_ai_failed"):
        issues.append({"code": "ai_failed", "severity": "warn", "message": "AI descriptions failed: Gemini rate limit — try again in a minute."})
    if j.get("input_kind") in ("photo_object", "photo_scene"):
        issues.append({"code": "photo", "severity": "info", "message": "Simplified from a photo: check that the outline matches the object."})
    blocked = any(i["severity"] == "block" for i in issues)
    passed = sum(1 for c in j["checks"] if c["passed"])
    return {
        "status": "blocked" if blocked else ("ready" if approved else "needs_review"),
        "approved": approved and not blocked,
        "geometry": {"passed": passed, "total": len(j["checks"])},
        "issues": issues,
        "unmatched_labels": list(j.get("_unmatched", [])),
    }


def public(st):
    j = copy.deepcopy(st["result"])
    j["readiness"] = build_readiness(j, st["approved"])
    j["version"] = st["version"]
    for k in [k for k in j if k.startswith("_")]:
        del j[k]
    v = st["version"]
    jid = j["id"]
    tpl = st["template"]
    def f(name):
        return f"/files/{jid}/{name}?v={v}"
    j["source"]["url"] = f("original.png")
    j["tactile"].update(url=f("tactile.png"), print_url=f("tactile_print.png"), legend_url=f("legend.png"))
    j["files"] = {"pdf": f("tactile.pdf"), "stl": f("relief.stl"), "png": f("tactile.png"),
                  "print_png": f("tactile_print.png"), "original": f("original.png"), "legend_png": f("legend.png")}
    return j


def new_job(template, *, use_ai, title=None, kind="diagram", stages_s=None, fail=None):
    jid = uuid.uuid4().hex[:10]
    res = copy.deepcopy(TEMPLATES[template])
    res["id"] = jid
    res["created_at"] = now_iso()
    res["input_kind"] = kind
    res["notes"] = []
    res["retention_hours"] = RETENTION_H
    res["expires_at"] = now_iso(datetime.now(timezone.utc) + timedelta(hours=RETENTION_H))
    if title:
        res["title"] = title
    if kind == "blank":
        res["parts"] = []
        res["title"] = "untitled diagram"
        res["warnings"] = ["We couldn't find a diagram in this photo."]
        res["_unmatched"] = []
    if use_ai and AI_ON:
        res["ai_used"] = True
        for p in res["parts"]:
            if p["match_method"] != "none":
                p["description"] = ai_text(p)
                p["description_source"] = "ai"
    st = {"result": res, "token": secrets.token_hex(16), "version": 1, "approved": False, "template": template,
          "t0": time.time(), "use_ai": use_ai and AI_ON, "stages_s": stages_s or 0.0, "fail": fail, "cancelled": False}
    JOBS[jid] = st
    return st


def ai_text(p):
    n = p["name"]
    return f"The {n}. It sits {['near the top', 'in the middle', 'on the left', 'on the right', 'at the bottom'][hash(n) % 5]} of the diagram; follow its raised outline to feel its shape."


def check_token(req: Request, st):
    if req.headers.get("x-edit-token") != st["token"]:
        return err(403, "view_only", "This is a view-only link — open the teacher link to edit.")
    return None


def get_state(jid):
    if jid == "expired":
        return None, err(404, "expired", "This lesson has expired and was removed from the demo server.")
    if jid == "mock" and "mock" not in JOBS:  # old fixed id
        st = new_job("plant-cell", use_ai=False)
        JOBS.pop(st["result"]["id"])
        st["result"]["id"] = "mock"
        JOBS["mock"] = st
    st = JOBS.get(jid)
    if not st:
        return None, err(404, "not_found", "We can't find that job.")
    return st, None


# ---------------------------------------------------------------- stages
def stage_plan(st):
    stages = ["read", "find"] + (["ai"] if st["use_ai"] else []) + ["layout", "files"]
    slow = st["stages_s"] or 0.8
    plan, t = [("queued", 0.6)], 0.6
    for s in stages:
        t += slow * (1.3 if s == "ai" else 1.0)
        plan.append((s, t))
    return stages, plan


def status_of(st):
    stages, plan = stage_plan(st)
    el = time.time() - st["t0"]
    jid = st["result"]["id"]
    base = {"id": jid, "stages": stages, "elapsed": round(el, 1), "error": None}
    if st["cancelled"]:
        return {**base, "status": "cancelled", "stage": "queued", "error": {"code": "cancelled", "message": "Cancelled."}}
    for name, end in plan:
        if el < end:
            if st["fail"] and name == "read":
                return {**base, "status": "error", "stage": "read", "error": st["fail"]}
            return {**base, "status": "queued" if name == "queued" else "running", "stage": name}
    return {**base, "status": "done", "stage": "done"}


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0-mock",
            "ai": {"provider": "gemini" if AI_ON else None, "model": "gemini-3.5-flash" if AI_ON else None,
                   "sends": "A downscaled copy of the image and the detected label text is sent to Google Gemini to write names and descriptions. Nothing is stored by Feelable beyond 24 h."},
            "limits": LIMITS, "retention_hours": RETENTION_H}


@app.get("/api/samples")
def samples():
    man = json.load(open(os.path.join(SAMPLES, "manifest.json")))
    return [{"id": s["id"], "title": s["title"], "thumb_url": f"/samples/{s['file']}", "kind": s.get("kind", "colour")} for s in man]


@app.get("/samples/{name}")
def sample_file(name: str):
    p = os.path.join(SAMPLES, os.path.basename(name))
    return FileResponse(p) if os.path.isfile(p) else err(404, "not_found", "No such sample")


@app.post("/api/jobs")
async def create(file: UploadFile = File(...), use_ai: str = Form("true"), paper: str = Form("a4"), async_: str = Form("0", alias="async")):
    data = await file.read()
    name = (file.filename or "").lower()
    if len(data) > 15 * 1024 * 1024:
        return err(413, "too_big", "That photo is over 15 MB.")
    if "busy" in name:
        return err(503, "busy", "The server is busy with other diagrams. Try again in a moment.")
    fail = None
    try:
        from PIL import Image
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        fail = {"code": "decode", "message": "We couldn't read that image. The file may be damaged or in an unsupported format."}
    kind = "blank" if "blank" in name else "diagram"
    st = new_job("heart", use_ai=use_ai == "true", kind=kind, stages_s=2.4 if "slow" in name else 0.0, fail=fail)
    if async_ in ("1", "true"):
        return JSONResponse({"id": st["result"]["id"], "edit_token": st["token"], "status": "queued"}, status_code=202)
    if fail:
        return err(400, "decode", fail["message"])
    time.sleep(2.0)
    st["t0"] -= 100
    return JSONResponse({**public(st), "edit_token": st["token"]}, status_code=201)


@app.post("/api/jobs/from-sample/{sid}")
def from_sample(sid: str, use_ai: str = "true", paper: str = "a4"):
    time.sleep(0.6)
    man = {s["id"]: s for s in json.load(open(os.path.join(SAMPLES, "manifest.json")))}
    if sid not in man:
        return err(404, "not_found", "No such sample")
    st = new_job("heart" if sid == "heart" else "plant-cell", use_ai=use_ai == "true")
    st["t0"] -= 100
    return JSONResponse({**public(st), "edit_token": st["token"]}, status_code=201)


@app.get("/api/jobs/{jid}")
def get(jid: str):
    st, e = get_state(jid)
    if e:
        return e
    s = status_of(st)
    if s["status"] != "done":
        return err(409, "not_ready", "This job is still being made.")
    return public(st)


@app.get("/api/jobs/{jid}/status")
def status(jid: str):
    st, e = get_state(jid)
    return e or status_of(st)


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str, req: Request):
    st, e = get_state(jid)
    if e:
        return e
    bad = check_token(req, st)
    if bad:
        return bad
    st["cancelled"] = True
    return {"id": jid, "status": "cancelled"}


@app.patch("/api/jobs/{jid}")
async def patch(jid: str, req: Request):
    st, e = get_state(jid)
    if e:
        return e
    bad = check_token(req, st)
    if bad:
        return bad
    edit = await req.json()
    if "base_version" in edit and edit["base_version"] != st["version"]:
        return err(409, "conflict", "Someone else changed this lesson. Reload to see the latest version.")
    res = st["result"]
    if len(edit.get("title", "")) > LIMITS["title"]:
        return err(400, "too_long", f"Title is over {LIMITS['title']} characters.")
    for pe in edit.get("parts", []):
        if len(pe.get("name", "")) > LIMITS["name"]:
            return err(400, "too_long", f"Part names can be up to {LIMITS['name']} characters.")
        if len(pe.get("description", "")) > LIMITS["description"]:
            return err(400, "too_long", f"Descriptions can be up to {LIMITS['description']} characters.")
    time.sleep(0.7)
    nxt = copy.deepcopy(res)
    changed = False
    if "title" in edit:
        nxt["title"] = edit["title"]; changed = True
    for pe in edit.get("parts", []):
        for p in nxt["parts"]:
            if p["id"] == pe["id"]:
                if "name" in pe:
                    p["name"] = pe["name"]; p["match_method"] = "teacher"
                    nxt["_unmatched"] = [u for u in nxt.get("_unmatched", []) if u.lower() != pe["name"].lower()]
                if "description" in pe:
                    p["description"] = pe["description"]; p["description_source"] = "teacher"
                if "include" in pe:
                    p["include"] = pe["include"]
                changed = True
    if edit.get("split"):
        p = next((x for x in nxt["parts"] if x["id"] == edit["split"]), None)
        if not p or p["count"] < 2 or len(p["source_polygons"]) < 2:
            return err(400, "bad_split", "That part is a single shape, so it can't be split.")
        idx = nxt["parts"].index(p)
        new = []
        n = len(p["source_polygons"])
        used = {x["id"] for x in nxt["parts"]}
        k = 1
        for i in range(n):
            q = copy.deepcopy(p)
            if i:
                while f"p{len(used) + k}" in used:
                    k += 1
                q["id"] = f"p{len(used) + k}"; used.add(q["id"])
                q["key"] = (p["key"][0] + str(i + 1))
            q["name"] = f"{p['name']} {i + 1}"
            q["count"] = 1
            q["source_polygons"] = [p["source_polygons"][i]]
            q["tactile_polygons"] = [p["tactile_polygons"][i]] if i < len(p["tactile_polygons"]) else []
            xs = [pt[0] for pt in q["source_polygons"][0]]; ys = [pt[1] for pt in q["source_polygons"][0]]
            q["source_bbox"] = [min(xs), min(ys), max(xs), max(ys)]
            q["match_method"] = "teacher"
            new.append(q)
        nxt["parts"][idx:idx + 1] = new
        changed = True
    if edit.get("merge"):
        ids = edit["merge"]
        keep = next((x for x in nxt["parts"] if x["id"] == ids[0]), None)
        if not keep or len(ids) < 2:
            return err(400, "bad_merge", "Choose at least two parts to merge.")
        for oid in ids[1:]:
            o = next((x for x in nxt["parts"] if x["id"] == oid), None)
            if o:
                keep["source_polygons"] += o["source_polygons"]; keep["tactile_polygons"] += o["tactile_polygons"]
                keep["count"] += o["count"]; keep["area"] += o["area"]
                nxt["parts"].remove(o)
        keep["match_method"] = "teacher"
        changed = True
    if nxt["parts"] and not any(p["include"] for p in nxt["parts"]):
        return err(400, "no_parts", "Keep at least one part switched on.")
    st["result"] = nxt
    if changed:
        st["approved"] = False
    if edit.get("approved") is True and not changed:
        st["approved"] = True
    if edit.get("approved") is False:
        st["approved"] = False
    st["version"] += 1
    return public(st)


@app.post("/api/jobs/{jid}/ai-describe")
async def ai_describe(jid: str, req: Request):
    st, e = get_state(jid)
    if e:
        return e
    bad = check_token(req, st)
    if bad:
        return bad
    if not AI_ON:
        return err(503, "ai_off", "AI descriptions aren't set up on this server.")
    try:
        body = await req.json()
    except Exception:
        body = {}
    time.sleep(1.6)
    for p in st["result"]["parts"]:
        if p["include"] and (p["description_source"] != "teacher" or body.get("overwrite_teacher")):
            p["description"] = ai_text(p)
            p["description_source"] = "ai"
    st["result"]["ai_used"] = True
    st["approved"] = False
    st["version"] += 1
    return public(st)


@app.get("/files/{jid}/{name}")
def files(jid: str, name: str):
    st = JOBS.get(jid)
    tpl = st["template"] if st else "plant-cell"
    p = os.path.join(HERE, "jobs", tpl, os.path.basename(name))
    return FileResponse(p) if os.path.isfile(p) else err(404, "not_found", "No such file")


# compat with the old mock paths
@app.get("/files/mock/{name}")
def files_mock(name: str):
    return files("mock", name)


DIST = os.environ.get("DIST")
if DIST:
    @app.get("/{path:path}")
    def spa(path: str):
        f = os.path.join(DIST, path)
        return FileResponse(f) if path and os.path.isfile(f) else FileResponse(os.path.join(DIST, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8204)))
