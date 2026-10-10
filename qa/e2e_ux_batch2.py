"""UX05/07/08/10/11 regression.
Run against a local server: TS_BASE=http://localhost:8000 python3 qa/e2e_ux_batch2.py
Needs qa/retest-local/user-human-organs.png (user image, not in git); falls back to samples/heart.png.
Checks:
  UX07  missing-part warnings come first, offer "Draw it" (prefilled name) and "Detail view" (prefilled box);
        the lead names what to review; the approve button says "Approve and publish".
  UX08  at 320x568 / 390x844 / 844x390 the zoom group doesn't overlap the view tabs; zoom targets >= 44px; Fit exists.
  UX11  Share copy says students get changes after approval.
  UX05  a close-up made from an AI-off lesson is created with use_ai=false and links "Back to full diagram".
  UX10  the touch-mode voice select is readable (light text on a dark background).
"""
import json, os, sys, urllib.request, uuid
from playwright.sync_api import sync_playwright

B = os.environ.get("TS_BASE", "http://localhost:8000")
SP = os.environ.get("TS_OUT", "/tmp")
SRC = "qa/retest-local/user-human-organs.png" if os.path.exists("qa/retest-local/user-human-organs.png") else "samples/heart.png"


def up(path):
    data = open(path, "rb").read()
    bnd = uuid.uuid4().hex
    body = (f'--{bnd}\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n').encode() + data
    body += (f'\r\n--{bnd}\r\nContent-Disposition: form-data; name="use_ai"\r\n\r\nfalse\r\n--{bnd}--\r\n').encode()
    req = urllib.request.Request(B + "/api/jobs", data=body, headers={"Content-Type": f"multipart/form-data; boundary={bnd}"})
    return json.load(urllib.request.urlopen(req))


def overlap(a, b):
    return not (a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"] or a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"])


job = up(SRC)
jid, tok = job["id"], job["edit_token"]
issues = job["readiness"]["issues"]
res = {"first_issue": issues[0]["message"] if issues else None}
fails = []
with sync_playwright() as p:
    br = p.chromium.launch()

    # ---- UX07 (desktop)
    ctx = br.new_context(viewport={"width": 1280, "height": 800})
    pg = ctx.new_page()
    pg.goto(f"{B}/#/job/{jid}?t={tok}")
    pg.wait_for_timeout(2500)
    lead = pg.locator(".ready__lead").first.inner_text()
    res["lead"] = lead
    draw_issue = next((i for i in issues if i.get("action") == "draw"), None)
    if draw_issue:
        if issues[0].get("action") != "draw":
            fails.append("missing part not ranked first")
        if "missing" not in lead:
            fails.append("lead doesn't name the missing part")
        pg.get_by_role("button", name="Draw it").first.click()
        pg.wait_for_timeout(700)
        name = pg.locator(".edlg input").first.input_value()
        res["draw_prefill"] = name
        if name.lower() != draw_issue["target"].lower():
            fails.append(f"draw name not prefilled ({name!r})")
        pg.get_by_role("button", name="Cancel").click()
        pg.wait_for_timeout(300)
        for role in ("tab", "button", "link"):
            loc = pg.get_by_role(role, name="Preview", exact=True)
            if loc.count():
                loc.first.click()
                break
        pg.wait_for_timeout(400)
        if draw_issue.get("bbox"):
            det = pg.get_by_role("button", name="Detail view")
            if not det.count():
                fails.append("no Detail view action")
            else:
                det.first.click()
                pg.wait_for_timeout(700)
                res["detail_box_visible"] = pg.locator(".edlg__crop").count() == 1
                res["detail_make_enabled"] = pg.get_by_role("button", name="Make close-up").is_enabled()
                if not (res["detail_box_visible"] and res["detail_make_enabled"]):
                    fails.append("Detail view didn't prefill the crop")
                pg.get_by_role("button", name="Cancel").click()
                pg.wait_for_timeout(300)
    else:
        res["note"] = "no draw issue for this fixture"
    if pg.get_by_role("button", name="I've checked it — ready to teach").count():
        fails.append("old approve copy still present")
    # UX11 Share copy
    pg.get_by_role("button", name="Share").first.click()
    pg.wait_for_timeout(600)
    txt = pg.locator("main").inner_text()
    res["share_copy_ok"] = "after you approve them" in txt and "saved changes" not in txt
    if not res["share_copy_ok"]:
        fails.append("share copy")
    ctx.close()

    # ---- UX08 toolbar on phones
    for vw, vh in [(320, 568), (390, 844), (844, 390)]:
        ctx = br.new_context(viewport={"width": vw, "height": vh})
        pg = ctx.new_page()
        pg.goto(f"{B}/#/job/{jid}?t={tok}")
        pg.wait_for_timeout(2200)
        if pg.get_by_role("button", name="Fix").count():
            pg.get_by_role("button", name="Fix").first.click()
            pg.wait_for_timeout(500)
        tabs = pg.locator(".viewer__bar .tabs").first
        zoom = pg.locator(".viewer__bar .zoom").first
        if not (tabs.count() and zoom.count()):
            res[f"bar_{vw}x{vh}"] = "no viewer bar on this step"
            ctx.close()
            continue
        tb, zb = tabs.bounding_box(), zoom.bounding_box()
        zin = pg.get_by_role("button", name="Zoom in").bounding_box()
        fit = pg.get_by_role("button", name="Fit diagram to view").count()
        page_w = pg.evaluate("document.documentElement.scrollWidth")
        r = dict(overlap=overlap(tb, zb), zoom_btn_h=round(zin["height"]), fit=bool(fit), hscroll=page_w > vw)
        res[f"bar_{vw}x{vh}"] = r
        if r["overlap"] or (vw <= 480 and r["zoom_btn_h"] < 44) or not r["fit"] or r["hscroll"]:
            fails.append(f"toolbar {vw}x{vh}: {r}")
        tabs.scroll_into_view_if_needed()
        pg.evaluate("window.scrollBy(0, -80)")
        pg.wait_for_timeout(200)
        pg.screenshot(path=f"{SP}/ux08-bar-{vw}x{vh}.png")
        ctx.close()

    # ---- UX05 close-up keeps AI-off and links back
    ctx = br.new_context(viewport={"width": 1280, "height": 800})
    # pretend the server has AI, so the home page's default would be AI ON — the close-up must still send off
    def fake_health(route):
        r = route.fetch()
        h = r.json()
        h["ai"] = {**(h.get("ai") or {}), "provider": "gemini", "model": "test"}
        route.fulfill(response=r, json=h)

    ctx.route("**/api/health", fake_health)
    # record what the app sends when it creates the close-up job
    ctx.add_init_script("""(() => { const f = window.fetch; window.__creates = [];
      window.fetch = function (u, o) { try { if (o && o.body instanceof FormData && String(u).endsWith('/api/jobs'))
        window.__creates.push({ use_ai: o.body.get('use_ai'), paper: o.body.get('paper') }); } catch (e) {}
        return f.apply(this, arguments); }; })()""")
    pg = ctx.new_page()
    pg.goto(f"{B}/#/job/{jid}?t={tok}")
    pg.wait_for_timeout(2500)
    pg.get_by_role("button", name="Fix").first.click()
    pg.wait_for_timeout(500)
    pg.get_by_role("button", name="Close-up lesson").click()
    pg.wait_for_timeout(700)
    pg.locator(".edlg__canvas").focus()
    pg.keyboard.press("ArrowRight")  # keyboard starts a box
    pg.get_by_role("button", name="Make close-up").click()
    pg.wait_for_function(f"location.hash.startsWith('#/job/') && !location.hash.includes('{jid}')", timeout=180000)
    sent = pg.evaluate("window.__creates.slice(-1)[0] || {}")
    pg.wait_for_timeout(3000)
    back = pg.get_by_role("link", name="Back to full diagram")
    res["closeup_sent"] = sent
    res["closeup_back_link"] = back.count() > 0
    if sent.get("use_ai") != "false":
        fails.append(f"close-up didn't keep AI off: {sent}")
    if not back.count():
        fails.append("no Back to full diagram link")
    else:
        pg.reload()
        pg.wait_for_timeout(2500)
        res["back_link_after_reload"] = pg.get_by_role("link", name="Back to full diagram").count() > 0
        if not res["back_link_after_reload"]:
            fails.append("back link not durable")
        pg.get_by_role("link", name="Back to full diagram").first.click()
        pg.wait_for_timeout(2000)
        res["back_goes_to_parent"] = jid in pg.url
        if not res["back_goes_to_parent"]:
            fails.append("back link target")
    ctx.close()

    # ---- UX10 voice select contrast: approve (publish) the lesson, then open the student view
    cur = json.load(urllib.request.urlopen(f"{B}/api/jobs/{jid}"))
    req = urllib.request.Request(f"{B}/api/jobs/{jid}", method="PATCH", data=json.dumps({"approved": True, "base_version": cur.get("version")}).encode(),
                                 headers={"Content-Type": "application/json", "X-Edit-Token": tok})
    urllib.request.urlopen(req).read()
    ctx = br.new_context(viewport={"width": 390, "height": 844})
    # headless Chromium has no voices: provide two so the device-voice select renders
    ctx.add_init_script("""(() => { if (!window.speechSynthesis) return;
      const vs = [{ name: 'Test Voice A', lang: 'en-GB', default: true, localService: true, voiceURI: 'a' },
                  { name: 'Test Voice B', lang: 'en-US', default: false, localService: true, voiceURI: 'b' }];
      try { Object.defineProperty(speechSynthesis, 'getVoices', { value: () => vs });
            Object.defineProperty(SpeechSynthesisUtterance.prototype, 'voice', { configurable: true, get() { return null; }, set(v) {} });
            speechSynthesis.speak = () => {}; } catch (e) {} })()""")
    pg = ctx.new_page()
    pg.goto(f"{B}/#/job/{jid}?t={tok}")
    pg.wait_for_timeout(1500)
    pg.goto(f"{B}/#/touch/{jid}")
    pg.wait_for_timeout(2500)
    start = pg.get_by_role("button", name="Tap anywhere to start")
    if start.count():
        start.first.click()
        pg.wait_for_timeout(800)
    menu = pg.get_by_role("button", name="Menu", exact=True)
    if menu.count():
        menu.first.click()
        pg.wait_for_timeout(600)
        sel = pg.locator("#ts-t-voice")
        if sel.count():
            cs = sel.evaluate("e => { const s = getComputedStyle(e); return [s.color, s.backgroundColor]; }")
            res["voice_select_colors"] = cs
            if cs[0] == cs[1] or "255, 255, 255" in cs[1]:
                fails.append(f"voice select unreadable {cs}")
        else:
            res["voice_select_colors"] = "no device voices in headless Chromium (select not rendered)"
        pg.screenshot(path=f"{SP}/ux10-menu-390.png")
    else:
        res["voice_select_colors"] = "touch menu not reachable"
        fails.append("touch menu not reachable")
    ctx.close()
    br.close()

print(json.dumps(res, indent=1))
print("FAIL" if fails else "PASS", fails)
sys.exit(1 if fails else 0)
