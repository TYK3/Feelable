"""UX02-04 regression: Draw/Close-up coordinates, phone layout and focus containment.
Run against a local server: TS_BASE=http://localhost:8000 python3 qa/e2e_repair_dialogs.py
Needs qa/retest-local/user-human-organs.png (user image, not in git); falls back to samples/heart.png.
"""
import json, urllib.request, uuid, sys
from playwright.sync_api import sync_playwright
import os; B=os.environ.get("TS_BASE","http://localhost:8000"); SP=os.environ.get("TS_OUT","/tmp")
def up(path):
    data=open(path,'rb').read(); bnd=uuid.uuid4().hex
    body=(f'--{bnd}\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n').encode()+data+(f'\r\n--{bnd}\r\nContent-Disposition: form-data; name="use_ai"\r\n\r\nfalse\r\n--{bnd}--\r\n').encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(B+"/api/jobs",data=body,headers={"Content-Type":f"multipart/form-data; boundary={bnd}"})))
job=up('qa/retest-local/user-human-organs.png' if os.path.exists('qa/retest-local/user-human-organs.png') else 'samples/heart.png'); jid,tok=job["id"],job["edit_token"]
res={}
with sync_playwright() as p:
    b=p.chromium.launch()
    for vw,vh in [(1280,720),(390,844),(320,568),(844,390)]:
        ctx=b.new_context(viewport={"width":vw,"height":vh}); pg=ctx.new_page()
        pg.goto(f"{B}/#/job/{jid}?t={tok}"); pg.wait_for_timeout(2500)
        pg.get_by_role("button", name="Fix").first.click(); pg.wait_for_timeout(600)
        pg.get_by_role("button", name="Draw a missing part").click(); pg.wait_for_timeout(700)
        svg=pg.locator(".edlg__canvas"); sb=svg.bounding_box(); bb=pg.locator(".edlg__canvas image").bounding_box()
        pad_clicks=0
        if bb["x"]-sb["x"]>6:   # click in letterbox padding: must be ignored
            pg.mouse.click(sb["x"]+3, sb["y"]+sb["height"]/2); pg.wait_for_timeout(150)
            pad_clicks=pg.locator(".edlg__pt").count()
        svg.scroll_into_view_if_needed(); pg.wait_for_timeout(150)
        bb=pg.locator(".edlg__canvas image").bounding_box()
        foot=pg.locator(".edlg__foot").bounding_box(); body=pg.locator(".edlg__body").bounding_box()
        vis_top=max(bb["y"], body["y"]); vis_bot=min(bb["y"]+bb["height"], foot["y"])
        cx,cy=bb["x"]+bb["width"]*0.42, vis_top+(vis_bot-vis_top)*0.5
        print(vw,vh,"click at",round(cx),round(cy),"image",[round(v) for v in (bb["x"],bb["y"],bb["width"],bb["height"])])
        pg.mouse.click(cx,cy); pg.wait_for_timeout(200)
        pt=pg.locator(".edlg__pt").first.bounding_box()
        off=((pt["x"]+pt["width"]/2-cx)**2+(pt["y"]+pt["height"]/2-cy)**2)**0.5
        vis=lambda name: (lambda r: r is not None and r["y"]>=0 and r["y"]+r["height"]<=vh+0.5)(pg.get_by_role("button", name=name).first.bounding_box())
        close_ok=vis("Close Draw a missing part"); cancel_ok=vis("Cancel"); add_ok=vis("Add part")
        # focus containment: tab 15 times, record whether focus ever leaves dialog
        esc=0
        for i in range(15):
            pg.keyboard.press("Shift+Tab" if i%2 else "Tab")
            esc += pg.evaluate("!document.querySelector('.edlg__box').contains(document.activeElement)")
        pg.screenshot(path=f"{SP}/ux03-draw-{vw}x{vh}.png")
        pg.get_by_role("button", name="Cancel").click(); pg.wait_for_timeout(300)
        restored=pg.evaluate("document.activeElement && document.activeElement.textContent")
        # close-up crop check
        pg.get_by_role("button", name="Close-up lesson").click(); pg.wait_for_timeout(700)
        pg.locator(".edlg__canvas").scroll_into_view_if_needed(); pg.wait_for_timeout(150)
        bb=pg.locator(".edlg__canvas image").bounding_box()
        foot=pg.locator(".edlg__foot").bounding_box()
        bb={"x":bb["x"],"y":bb["y"],"width":bb["width"],"height":min(bb["height"], foot["y"]-bb["y"]-4)}
        x0,y0=bb["x"]+bb["width"]*0.3, bb["y"]+bb["height"]*0.3; x1,y1=bb["x"]+bb["width"]*0.7, bb["y"]+bb["height"]*0.6
        pg.mouse.move(x0,y0); pg.mouse.down(); pg.mouse.move(x1,y1,steps=6); pg.mouse.up(); pg.wait_for_timeout(200)
        cr=pg.locator(".edlg__crop").bounding_box()
        crop_err=max(abs(cr["x"]-x0),abs(cr["y"]-y0),abs(cr["x"]+cr["width"]-x1),abs(cr["y"]+cr["height"]-y1))
        cu_cancel=vis("Cancel"); cu_make=vis("Make close-up")
        pg.screenshot(path=f"{SP}/ux03-closeup-{vw}x{vh}.png")
        res[f"{vw}x{vh}"]=dict(pad_click_points=pad_clicks, marker_offset_px=round(off,2), close=close_ok, cancel=cancel_ok, add=add_ok, focus_escapes=esc, focus_restored_to=restored.strip()[:30] if restored else None, crop_edge_err_px=round(crop_err,2), closeup_cancel=cu_cancel, closeup_make=cu_make)
        ctx.close()
    b.close()
print(json.dumps(res, indent=1))
bad=[k for k,v in res.items() if v['marker_offset_px']>2 or v['crop_edge_err_px']>2 or v['focus_escapes'] or not all((v['close'],v['cancel'],v['add'],v['closeup_cancel'],v['closeup_make'])) or v['pad_click_points']]
print('FAIL' if bad else 'PASS', bad); sys.exit(1 if bad else 0)
