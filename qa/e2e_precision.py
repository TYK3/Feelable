"""Browser checks for the precision-audit fixes (local or live). Usage:
    python qa/e2e_precision.py BASE_URL SHOTS_DIR
Covers: ten featured lessons (5 basic / 5 complex, filters, extended library), skull counts and
student chips, drawing tool (Undo/Redo status, zoom, full-screen on phones), title Braille blocker,
flowchart / transit descriptions, phone touch targets, no sideways scroll, student preview."""
import json
import sys
import urllib.request

from playwright.sync_api import sync_playwright

B, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
res = []


def check(name, ok, detail=""):
    res.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def api(path, data=None, method=None, headers=None):
    req = urllib.request.Request(B + path, data=None if data is None else json.dumps(data).encode(),
                                 method=method, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def open_sample(pg, title):
    pg.goto(B)
    pg.wait_for_selector(".gallery", timeout=60000)
    if not pg.locator(".sample", has_text=title).count():
        pg.locator(".gallery__more").click()
    pg.locator(".sample", has_text=title).first.click()
    pg.wait_for_url("**/#/job/**", timeout=120000)
    pg.wait_for_selector(".ready", timeout=120000)
    return pg.url.split("/job/")[1].split("?")[0]


def svg_xy(pg, x, y):
    return pg.evaluate("""([x, y]) => { const s = document.querySelector('.draw__canvas');
        const pt = s.createSVGPoint(); pt.x = x; pt.y = y; const q = pt.matrixTransform(s.getScreenCTM()); return [q.x, q.y]; }""", [x, y])


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))

    # ---- gallery: ten featured lessons, filters, extended library
    pg.goto(B)
    pg.wait_for_selector(".gallery .sample", timeout=60000)
    n_all = pg.locator(".gallery ul[aria-label='Featured lessons'] .sample").count()
    pg.locator(".gallery__filter button", has_text="Basic").click()
    n_basic = pg.locator(".gallery ul[aria-label='Featured lessons'] .sample").count()
    pg.locator(".gallery__filter button", has_text="Complex").click()
    n_complex = pg.locator(".gallery ul[aria-label='Featured lessons'] .sample").count()
    titles_complex = pg.locator(".gallery ul[aria-label='Featured lessons'] .sample__title").all_inner_texts()
    pg.locator(".gallery__filter button", has_text="All").click()
    pg.locator(".gallery__more").click()
    n_ext = pg.locator(".gallery ul[aria-label='Extended library'] .sample").count()
    ext_note = pg.locator("ul[aria-label='Extended library'] .sample__prep").first.inner_text() if n_ext else ""
    check("gallery: 10 featured (5 basic + 5 complex), extended library separate",
          (n_all, n_basic, n_complex) == (10, 5, 5) and n_ext == 8, f"{n_all}/{n_basic}/{n_complex}/{n_ext}")
    check("complex lessons span subjects (skull, cell, circuit, flowchart, map)",
          len(titles_complex) == 5 and any("bus" in t.lower() for t in titles_complex)
          and any("robot" in t.lower() for t in titles_complex), "; ".join(titles_complex))
    check("extended library says it was not re-checked", "not re-checked" in ext_note, ext_note)
    pg.screenshot(path=f"{OUT}/p01-gallery.png", full_page=True)

    # ---- skull: counts, chips, sutures as boundaries
    jid = open_sample(pg, "Skull")
    job = api(f"/api/jobs/{jid}")
    inc = [q for q in job["parts"] if q["include"]]
    meta = pg.locator(".rv-meta li").first.inner_text()
    chips = pg.locator(".glance__chip").count()
    check("skull header counts included parts (not candidates)", meta.startswith(f"{len(inc)} included"), meta)
    check("student chips list only included parts", chips == len(inc), f"{chips} chips / {len(inc)} included")
    sut = [q for q in inc if "suture" in q["name"]]
    check("skull sutures are boundary targets traced along their course",
          sut and all(q["target_type"] == "boundary" and q["extent"] == "full" for q in sut),
          ", ".join(f"{q['name']}:{q['target_type']}/{q['extent']}" for q in sut))
    miss = [i for i in job["readiness"]["issues"] if i["code"] in ("dropped_label", "unmatched_label", "unnamed")]
    check("skull: no unresolved missing / unnamed targets", not miss, "; ".join(i["message"][:50] for i in miss))
    unres = [i for i in job["readiness"]["issues"] if i["code"] == "extent_unresolved"]
    check("spots with unknown extent are reported as 'location found'", all("location found" in i["message"] for i in unres),
          f"{len(unres)} reported")

    # ---- drawing tool: Undo / Redo status, zoom
    pg.locator("#step-fix").click()
    pg.get_by_role("button", name="Add missing part").click()
    pg.wait_for_selector(".draw__canvas", timeout=10000)
    pg.locator(".edlg input").first.fill("test outline")
    x, y = svg_xy(pg, 760, 190)
    pg.mouse.click(x, y)
    pg.get_by_role("button", name="Find outline").click()
    pg.wait_for_function("document.querySelector('.draw__status').textContent.includes('Suggested outline') || document.querySelector('.draw__status').textContent.includes(\"couldn't\")", timeout=30000)
    found = pg.locator(".draw__status").inner_text()
    pg.get_by_role("button", name="Undo").click()
    after_undo = pg.locator(".draw__status").inner_text()
    pg.get_by_role("button", name="Redo").click()
    after_redo = pg.locator(".draw__status").inner_text()
    check("Undo after Find outline says the outline was removed (no stale 'Suggested')",
          "Suggested outline" in found and "removed" in after_undo.lower() and "Suggested outline —" not in after_undo,
          f"{after_undo[:50]!r}")
    check("Redo says the outline was restored", "restored" in after_redo.lower(), after_redo[:50])
    vb0 = pg.locator(".draw__canvas").get_attribute("viewBox")
    pg.get_by_role("button", name="Zoom in").click()
    vb1 = pg.locator(".draw__canvas").get_attribute("viewBox")
    pg.get_by_role("button", name="Fit").click()
    vb2 = pg.locator(".draw__canvas").get_attribute("viewBox")
    check("draw tool zooms in and fits back", vb0 != vb1 and vb2 == vb0, f"{vb0} -> {vb1} -> {vb2}")
    pg.screenshot(path=f"{OUT}/p02-draw-desktop.png")
    pg.get_by_role("button", name="Cancel").click()

    # ---- title Braille blocker (API edit, as the editor would send)
    tok = pg.evaluate(f"(JSON.parse(localStorage.getItem('ts.tokens') || '{{}}'))['{jid}']")
    try:
        r = api(f"/api/jobs/{jid}", {"title": "H₂O café 心脏"}, "PATCH", {"X-Edit-Token": tok or ""})
        codes = [i["code"] for i in r["readiness"]["issues"]]
        check("unsupported title characters block publication", "braille_unsupported_title" in codes
              and r["readiness"]["status"] == "blocked", ", ".join(codes))
        api(f"/api/jobs/{jid}", {"title": "skull, right side"}, "PATCH", {"X-Edit-Token": tok or ""})
    except Exception as e:  # noqa: BLE001
        check("unsupported title characters block publication", False, f"edit failed: {e}")

    # ---- flowchart and transit descriptions
    fj = api(f"/api/jobs/{open_sample(pg, 'Plant watering')}")
    dec = next((q for q in fj["parts"] if "soil dry" in q["name"]), None)
    check("flowchart decision describes both branches", dec and "decision" in dec["description"].lower()
          and "yes:" in dec["description"].lower() and "no:" in dec["description"].lower(), dec and dec["description"])
    tj = api(f"/api/jobs/{open_sample(pg, 'bus routes')}")
    central = next((q for q in tj["parts"] if q["name"] == "central"), None)
    check("map station knows its interchange", central and "interchange" in central["description"], central and central["description"])
    check("no page errors (desktop)", not errs, "; ".join(errs)[:200])

    # ---- phones: touch targets, no sideways scroll, full-screen drawing
    for wpx in (390, 320):
        m = br.new_context(viewport={"width": wpx, "height": 780}, is_mobile=True, has_touch=True)
        q = m.new_page()
        perrs = []
        q.on("pageerror", lambda e: perrs.append(str(e)))
        jid2 = open_sample(q, "Skull")
        hs = q.evaluate("""() => [...document.querySelectorAll('.tabs__tab, .glance__chip, .zoom__btn')]
            .filter(e => e.offsetParent).map(e => Math.round(e.getBoundingClientRect().height))""")
        check(f"{wpx}px: tabs, chips and zoom buttons are at least 44 px tall", hs and min(hs) >= 44, f"min {min(hs) if hs else None}")
        check(f"{wpx}px: no sideways scroll on review", q.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))
        tabs_visible = q.evaluate("""() => [...document.querySelectorAll('.tabs__tab')].every(e => {
            const r = e.getBoundingClientRect(); return r.right <= window.innerWidth + 1 && r.left >= -1; })""")
        check(f"{wpx}px: every view tab is on screen", tabs_visible)
        q.locator("#step-fix").click()
        q.get_by_role("button", name="Add missing part").click()
        q.wait_for_selector(".draw__canvas", timeout=10000)
        box = q.locator(".edlg__box").bounding_box()
        check(f"{wpx}px: drawing opens full screen", box and box["width"] >= wpx - 2, str(box))
        q.screenshot(path=f"{OUT}/p03-draw-{wpx}.png")
        q.get_by_role("button", name="Cancel").click()
        check(f"{wpx}px: no page errors", not perrs, "; ".join(perrs)[:200])
        m.close()

    # ---- student preview: back to editor, never a silent blank sheet
    s = br.new_context(viewport={"width": 390, "height": 780}, is_mobile=True, has_touch=True)
    q = s.new_page()
    jid3 = open_sample(q, "Four flat shapes")
    q.goto(f"{B}/#/touch/{jid3}")
    q.wait_for_selector(".ts-t-start, .ts-t-explore-shell", timeout=60000)
    if q.locator(".ts-t-start").count():
        q.locator(".ts-t-start-btn").click()
    q.wait_for_selector(".ts-t-explore", timeout=30000)
    q.wait_for_timeout(1500)
    painted = q.evaluate("""() => { const c = document.querySelector('.ts-t-canvas'); const g = c.getContext('2d');
        const d = g.getImageData(0, 0, c.width, c.height).data; let dark = 0;
        for (let i = 0; i < d.length; i += 64) if (d[i] < 128) dark++; return dark; }""")
    check("student canvas is painted (not a blank sheet)", painted > 50, f"{painted} dark samples")
    check("loading notice gone once the picture is in", q.locator(".ts-t-loading").count() == 0)
    q.screenshot(path=f"{OUT}/p04-student-390.png")
    s.close()
    br.close()

print(f"\n{sum(ok for _, ok, _ in res)}/{len(res)} checks passed")
json.dump([{"check": n, "ok": ok, "detail": d} for n, ok, d in res], open(f"{OUT}/e2e-precision.json", "w"), indent=1)
sys.exit(0 if all(ok for _, ok, _ in res) else 1)
