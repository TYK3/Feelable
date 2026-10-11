"""Browser checks for the handover fixes (local or live). Usage:
    python qa/e2e_fixes.py BASE_URL SHOTS_DIR [ORGANS_IMAGE]
Covers: featured gallery (ten lessons + extended library), phone widths without sideways scroll,
curated skull and mixed circuit, Remove part + Undo, Add missing part (tap -> Find outline ->
Save part), approval with open checks, close-up opening on the pointer target, student touch."""
import json
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

B, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
ORGANS = sys.argv[3] if len(sys.argv) > 3 else None
res = []


def check(name, ok, detail=""):
    res.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def api(path):
    with urllib.request.urlopen(B + path, timeout=120) as r:
        return json.loads(r.read())


def no_hscroll(pg):
    return pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")


def open_sample(pg, title):
    pg.goto(B)
    pg.wait_for_selector(".gallery", timeout=60000)
    more = pg.locator(".gallery__more")
    if more.count() and more.get_attribute("aria-expanded") == "false":
        more.click()
    pg.locator(".sample", has_text=title).first.click()
    pg.wait_for_url("**/#/job/**", timeout=120000)
    pg.wait_for_selector(".ready", timeout=120000)
    return pg.url.split("/job/")[1].split("?")[0]


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))

    # gallery
    pg.goto(B)
    pg.wait_for_selector(".gallery", timeout=60000)
    n_feat = pg.locator("ul[aria-label='Featured lessons'] .sample").count()
    pg.locator(".gallery__more").click()
    n_all = pg.locator(".gallery .sample").count()
    check("gallery: 10 featured lessons + extended library", n_feat == 10 and n_all == 18, f"{n_feat}/{n_all}")
    check("gallery cards say prepared + objective", pg.locator(".sample__prep").count() == n_all and pg.locator(".sample__obj").count() == n_all)
    pg.screenshot(path=f"{OUT}/01-gallery-desktop.png", full_page=True)

    # curated skull: nasal bone is the nose, mandible exists, no missing-part suggestions
    jid = open_sample(pg, "Skull")
    job = api(f"/api/jobs/{jid}")
    names = {q["name"]: q for q in job["parts"] if q["include"]}
    nb, md = names.get("nasal bone"), names.get("mandible")
    ok = nb and md and (nb["source_bbox"][3] - nb["source_bbox"][1]) < 0.12 * job["source"]["height"] \
        and md["source_bbox"][1] > nb["source_bbox"][3]
    check("skull: nasal bone is the nose only; mandible is its own part below it", ok,
          f"nasal {nb and nb['source_bbox']} mandible {md and md['source_bbox']}")
    miss = [i for i in job["readiness"]["issues"] if i["code"] in ("dropped_label", "unmatched_label", "unnamed")]
    check("skull: zero unresolved missing targets (27 labels)", not miss and len(names) >= 27, f"{len(names)} parts")
    pg.screenshot(path=f"{OUT}/02-skull-review.png")

    # Remove part + Undo (preview panel)
    pg.locator(".glance__chip", has_text="Mandible").first.click()
    pg.get_by_role("button", name="Remove part").first.click()
    removed = pg.locator(".peek .tag--off", has_text="Removed").count() == 1
    pg.get_by_role("button", name="Undo remove").first.click()
    restored = pg.locator(".peek .tag--off").count() == 0
    check("Remove part shows Removed, Undo restores it", removed and restored)

    # approval with open checks says so
    pg.get_by_role("button", name="Approve and publish").last.click()
    pg.wait_for_selector(".ready__approved", timeout=60000)
    title = pg.locator("#ready-title").inner_text()
    pub = api(f"/api/jobs/{jid}/published")
    check("approve with open checks: 'Approved — N checks still open', student snapshot published",
          title.startswith("Approved") and "still open" in title and pub["version"] == api(f"/api/jobs/{jid}")["version"], title)

    # Add missing part: tap -> Find outline -> Save part
    pg.locator(f"#step-fix").click()
    pg.get_by_role("button", name="Add missing part").click()
    pg.wait_for_selector(".draw__canvas", timeout=10000)
    pg.locator(".edlg input").first.fill("test outline")
    box = pg.locator(".draw__canvas").bounding_box()
    W, H = job["source"]["width"], job["source"]["height"]
    # tap inside the frontal bone (source 760,190) through the SVG transform
    xy = pg.evaluate("""([x, y]) => { const s = document.querySelector('.draw__canvas');
        const pt = s.createSVGPoint(); pt.x = x; pt.y = y; const q = pt.matrixTransform(s.getScreenCTM()); return [q.x, q.y]; }""", [760, 190])
    pg.mouse.click(xy[0], xy[1])
    pg.get_by_role("button", name="Find outline").click()
    pg.wait_for_function("document.querySelector('.draw__status').textContent.includes('Suggested outline') || document.querySelector('.draw__status').textContent.includes(\"couldn't\")", timeout=30000)
    status = pg.locator(".draw__status").inner_text()
    check("Find outline proposes an outline from one tap", "Suggested outline" in status, status[:60])
    pg.screenshot(path=f"{OUT}/03-assisted-outline.png")
    pg.locator(".edlg button", has_text="Save part").click()
    # it covers the frontal bone: an explicit choice is offered
    pg.wait_for_selector(".draw__choice", timeout=20000)
    check("covering another part asks before changing it", "frontal bone" in pg.locator(".draw__choice").inner_text())
    pg.locator(".draw__choice button", has_text="Keep both").click()
    pg.wait_for_selector(".draw__canvas", state="detached", timeout=60000)
    after = api(f"/api/jobs/{jid}")
    fb = next((q for q in after["parts"] if q["name"] == "frontal bone"), None)
    check("saved part exists, frontal bone kept (Keep both)",
          any(q["name"] == "test outline" for q in after["parts"]) and fb and fb["include"])
    check("no page errors (teacher)", not errs, "; ".join(errs)[:200])

    # mixed circuit: student touch says resistor values
    cid = open_sample(pg, "Mixed circuit")
    cj = api(f"/api/jobs/{cid}")
    check("mixed circuit: battery, R1, R2, R3 with source values",
          sorted(q["name"] for q in cj["parts"] if q["include"]) == ["R1, 1.00 ohms", "R2, 6.00 ohms", "R3, 13.00 ohms", "battery, 12.0 volts"],
          str([q["name"] for q in cj["parts"]]))
    pg.get_by_role("button", name="Approve and publish").last.click()
    pg.wait_for_selector(".ready__approved", timeout=60000)
    S = br.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
    sp = S.new_page()
    sp.goto(f"{B}/#/touch/{cid}")
    sp.wait_for_selector("text=Tap anywhere to start", timeout=60000)
    sp.locator("text=Tap anywhere to start").click()
    sp.wait_for_timeout(1200)
    r2 = next(q for q in cj["parts"] if q["name"].startswith("R2"))
    poly = r2["tactile_polygons"][0]
    cx = sum(v[0] for v in poly) / len(poly)
    cy = sum(v[1] for v in poly) / len(poly)
    xy = sp.evaluate(f"window.__feelableMap && window.__feelableMap({cx},{cy})")
    said = ""
    if xy:
        sp.touchscreen.tap(xy[0], xy[1])
        sp.wait_for_timeout(600)
        said = sp.evaluate("window.__feelableLastSpoken") or ""
    check("student touch on R2 says its value", "6.00 ohms" in said.lower() or "r2" in said.lower(), said[:60])
    check("student page has no sideways scroll at 390px", no_hscroll(sp))
    sp.screenshot(path=f"{OUT}/04-student-circuit.png")

    # close-up opens on the pointer target (fresh upload with a missing pancreas)
    if ORGANS:
        pg.goto(B)
        pg.locator("input[type=file]").first.set_input_files(ORGANS)
        pg.wait_for_url("**/#/job/**", timeout=240000)
        pg.wait_for_selector(".ready", timeout=240000)
        more = pg.locator(".issues__more")
        if more.count() and more.get_attribute("aria-expanded") == "false":
            more.click()
        btn = pg.get_by_role("button", name="Make a close-up")
        if btn.count():
            btn.first.click()
            pg.wait_for_selector(".edlg__hint", timeout=10000)
            hint = pg.locator(".edlg__hint").inner_text()
            check("close-up starts at the pointer target and says so", "pointer" in hint, hint[:80])
            pg.screenshot(path=f"{OUT}/05-closeup-target.png")
            pg.keyboard.press("Escape")
        else:
            check("close-up offered for a missing part", False, "no Make a close-up button")

    # phone widths
    P = br.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
    pp = P.new_page()
    pp.goto(B)
    pp.wait_for_selector(".gallery", timeout=60000)
    check("home: no sideways scroll at 390px", no_hscroll(pp))
    pp.screenshot(path=f"{OUT}/06-phone-home.png", full_page=True)
    open_sample(pp, "Heart: chambers")
    check("complex review: no sideways scroll at 390px", no_hscroll(pp))
    pp.set_viewport_size({"width": 320, "height": 740})
    pp.wait_for_timeout(600)
    check("complex review: no sideways scroll at 320px", no_hscroll(pp))
    pp.screenshot(path=f"{OUT}/07-phone320-review.png")
    br.close()

n = sum(r[1] for r in res)
print(f"\n{n}/{len(res)} checks passed")
json.dump(res, open(f"{OUT}/e2e-results.json", "w"), indent=1)
sys.exit(0 if n == len(res) else 1)
