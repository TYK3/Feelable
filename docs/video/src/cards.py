"""Render the video's title, story, stats, results and closing cards (1920x1080 PNG)."""
from playwright.sync_api import sync_playwright

NM = "file:///home/claude/feelable/web/node_modules"
CSS = f"""
@font-face {{ font-family: Brico; src: url({NM}/@fontsource-variable/bricolage-grotesque/files/bricolage-grotesque-latin-wght-normal.woff2); font-weight: 200 800; }}
@font-face {{ font-family: Inter; src: url({NM}/@fontsource/inter/files/inter-latin-400-normal.woff2); font-weight: 400; }}
@font-face {{ font-family: Inter; src: url({NM}/@fontsource/inter/files/inter-latin-600-normal.woff2); font-weight: 600; }}
* {{ box-sizing: border-box; margin: 0; }}
body {{ width: 1920px; height: 1080px; overflow: hidden; font-family: Inter; color: #17171C; background: #FFF9EA; }}
.dark {{ background: #17171C; color: #FFF9EA; }}
.wrap {{ position: absolute; inset: 0; padding: 120px 160px; display: flex; flex-direction: column; justify-content: center; }}
h1 {{ font-family: Brico; font-weight: 800; font-size: 150px; line-height: .95; letter-spacing: -4px; }}
h2 {{ font-family: Brico; font-weight: 800; font-size: 96px; line-height: 1.02; letter-spacing: -2px; }}
p.lead {{ font-size: 44px; line-height: 1.35; margin-top: 36px; max-width: 1400px; }}
.y {{ color: #FFD23F; }} .mark {{ background: #FFD23F; color: #17171C; padding: 0 14px; border-radius: 12px; }}
.small {{ font-size: 28px; opacity: .7; margin-top: 40px; }}
.row {{ display: flex; gap: 40px; margin-top: 70px; }}
.tile {{ flex: 1; background: #fff; border: 3px solid #17171C; border-radius: 28px; padding: 44px 40px; box-shadow: 0 10px 0 #17171C; }}
.dark .tile {{ background: #23232A; border-color: #FFD23F; box-shadow: 0 10px 0 #FFD23F; }}
.tile b {{ display: block; font-family: Brico; font-weight: 800; font-size: 92px; line-height: 1; }}
.tile span {{ display: block; font-size: 30px; margin-top: 18px; line-height: 1.3; }}
.tile i {{ display: block; font-style: normal; font-size: 22px; opacity: .6; margin-top: 14px; }}
.logo {{ display: inline-flex; align-items: center; gap: 22px; font-family: Brico; font-weight: 800; font-size: 56px; margin-bottom: 60px; }}
.logo em {{ width: 84px; height: 84px; border-radius: 22px; background: #FFD23F; display: grid; place-items: center; font-style: normal; font-size: 54px; color: #17171C; }}
.badge {{ display: inline-block; margin-top: 56px; font-size: 30px; font-weight: 600; padding: 16px 28px; border: 3px solid currentColor; border-radius: 999px; }}
.icons {{ font-size: 76px; }}
"""

CARDS = {
    "c1_title": ("", """<div class="wrap"><div class="logo"><em>F</em>Feelable</div>
        <h1>Every diagram,<br>feelable.</h1>
        <p class="lead">Snap a textbook diagram. Get a picture a blind student can <span class="mark">feel</span> and <span class="mark">hear</span>.</p>
        <div><span class="badge">Banana Hacks 2026 · image AI</span></div></div>"""),
    "c2_story": ("dark", """<div class="wrap"><h2>Image AI makes amazing pictures.</h2>
        <p class="lead">Comics from stories. Outfit try-ons. Room redesigns.<br>Every one of them is made <span class="y">for people who can see.</span></p></div>"""),
    "c3_question": ("dark", """<div class="wrap" style="align-items:center;text-align:center"><p class="lead" style="opacity:.75;margin:0 0 30px">So we asked the opposite question:</p>
        <h1 style="font-size:132px">Who does image AI<br><span class="y">leave out?</span></h1></div>"""),
    "c4_stats": ("", """<div class="wrap"><h2>Blind students learn from the same textbooks.<br>Every diagram is remade <span class="mark">by hand.</span></h2>
        <div class="row">
          <div class="tile"><b>hours</b><span>to hand-make one complex tactile diagram</span><i>TactileNet, 2025</i></div>
          <div class="tile"><b>~300 h</b><span>to adapt a single textbook</span><i>ASEE</i></div>
          <div class="tile"><b>weeks</b><span>that students wait for their materials</span><i>Carleton University</i></div>
        </div></div>"""),
    "c5_idea": ("dark", """<div class="wrap"><h2>Use image AI not to make a new picture for the eyes,<br><span class="y">but to remake a picture for the fingers.</span></h2>
        <div class="row">
          <div class="tile"><div class="icons">🖨️</div><b style="font-size:48px;margin-top:20px">Swell-paper PDF</b><span>Raised lines, textures, braille labels and a key</span></div>
          <div class="tile"><div class="icons">🧊</div><b style="font-size:48px;margin-top:20px">3D relief</b><span>A watertight, printable STL</span></div>
          <div class="tile"><div class="icons">📱</div><b style="font-size:48px;margin-top:20px">Talking touch mode</b><span>Touch a part, hear its name</span></div>
        </div></div>"""),
    "c6_phonebg": ("dark", """<div class="wrap" style="padding-right:820px"><h2>No printer?<br><span class="y">A phone that talks.</span></h2>
        <p class="lead" style="font-size:40px">Touch a part to hear its name.<br>Double-tap to hear more.<br>Feel a buzz at every edge.</p>
        <p class="small">In this video, speech is shown as the yellow caption on the phone.</p></div>
        <div style="position:absolute;right:236px;top:40px;width:476px;height:1000px;border-radius:64px;background:#000;border:6px solid #3a3a44;box-shadow:0 30px 80px rgba(0,0,0,.5)"></div>"""),
    "c7_results": ("", """<div class="wrap"><h2>Measured, not guessed.</h2>
        <p class="lead" style="font-size:34px;margin-top:20px">Scored against targets we annotated by hand from the source images: does each outline land on the right part?</p>
        <div class="row">
          <div class="tile"><b>67 → 104</b><span>of 142 targets valid<br>(critical errors 16 → 10)</span></div>
          <div class="tile"><b>11 / 11</b><span>flowchart arrows read with direction and Yes / No</span></div>
          <div class="tile"><b>15 / 15</b><span>bus-map stops matched to their routes</span></div>
        </div>
        <p class="small">Our own annotations, on images we developed with: not a held-out accuracy claim. 304 backend tests pass.</p></div>"""),
    "c8_close": ("dark", """<div class="wrap" style="align-items:center;text-align:center"><div class="logo"><em>F</em>Feelable</div>
        <h1>Every diagram,<br><span class="y">feelable.</span></h1>
        <p class="lead" style="margin-top:50px">Try it: <b>feelable.onrender.com</b><br><span style="opacity:.7;font-size:34px">github.com/TYK3/feelable</span></p></div>"""),
}

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    for name, (cls, html) in CARDS.items():
        fp = f"/tmp/claude-0/vid/cards/{name}.html"
        open(fp, "w").write(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body class='{cls}'>{html}</body></html>")
        pg.goto("file://" + fp); pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(400)
        pg.screenshot(path=f"/tmp/claude-0/vid/cards/{name}.png")
        print(name)
    b.close()
