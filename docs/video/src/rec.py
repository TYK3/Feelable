"""Record Feelable scenes as high-quality frame sequences (CDP screencast) + caption timings."""
import base64, json, os, time

CURSOR_JS = r"""
(() => {
  if (window.__fcur) return; window.__fcur = 1;
  const add = () => {
    const c = document.createElement('div'); c.id = '__cur';
    Object.assign(c.style, {position:'fixed', left:'-50px', top:'-50px', width:'26px', height:'26px', margin:'-13px 0 0 -13px',
      borderRadius:'50%', background:'rgba(255,210,63,.55)', border:'3px solid #17171C', zIndex:2147483647,
      pointerEvents:'none', transition:'transform .12s', boxShadow:'0 2px 8px rgba(0,0,0,.25)'});
    document.documentElement.appendChild(c);
    const mv = e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; };
    addEventListener('mousemove', mv, true); addEventListener('pointermove', mv, true);
    addEventListener('mousedown', e => { mv(e); c.style.transform = 'scale(.7)';
      const r = document.createElement('div');
      Object.assign(r.style, {position:'fixed', left:e.clientX+'px', top:e.clientY+'px', width:'14px', height:'14px', margin:'-7px 0 0 -7px',
        borderRadius:'50%', border:'3px solid #FFD23F', zIndex:2147483646, pointerEvents:'none', transition:'all .5s ease-out', opacity:'1'});
      document.documentElement.appendChild(r);
      requestAnimationFrame(() => { Object.assign(r.style, {width:'70px', height:'70px', margin:'-35px 0 0 -35px', opacity:'0'}); });
      setTimeout(() => r.remove(), 600); }, true);
    addEventListener('mouseup', () => { c.style.transform = 'scale(1)'; }, true);
  };
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', add); else add();
})();
"""

TOUCH_JS = r"""
(() => {
  if (window.__ftouch) return; window.__ftouch = 1;
  const ring = (x, y) => { const r = document.createElement('div');
    Object.assign(r.style, {position:'fixed', left:x+'px', top:y+'px', width:'34px', height:'34px', margin:'-17px 0 0 -17px',
      borderRadius:'50%', background:'rgba(255,210,63,.6)', border:'3px solid #17171C', zIndex:2147483647, pointerEvents:'none',
      transition:'all .6s ease-out'});
    document.documentElement.appendChild(r);
    requestAnimationFrame(() => Object.assign(r.style, {width:'80px', height:'80px', margin:'-40px 0 0 -40px', opacity:'0'}));
    setTimeout(() => r.remove(), 700); };
  addEventListener('touchstart', e => { for (const t of e.changedTouches) ring(t.clientX, t.clientY); }, true);
})();
"""


class Rec:
    def __init__(self, page, name, out, max_w, max_h):
        self.page, self.name, self.dir = page, name, os.path.join(out, name)
        os.makedirs(self.dir, exist_ok=True)
        self.frames, self.caps, self.t0 = [], [], None
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._frame)
        self.max_w, self.max_h = max_w, max_h

    def _frame(self, ev):
        ts = ev["metadata"]["timestamp"]
        if self.t0 is None:
            self.t0 = ts
        fn = os.path.join(self.dir, f"f{len(self.frames):05d}.jpg")
        with open(fn, "wb") as f:
            f.write(base64.b64decode(ev["data"]))
        self.frames.append((fn, ts))
        try:
            self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
        except Exception:
            pass

    def start(self):
        self.wall0 = time.time()
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": self.max_w,
                                               "maxHeight": self.max_h, "everyNthFrame": 1})
        self.page.wait_for_timeout(300)

    def cap(self, text, sub=""):
        self.caps.append((time.time() - self.wall0, text, sub))

    def stop(self, tail=0.6):
        self.page.wait_for_timeout(int(tail * 1000))
        end = time.time() - self.wall0
        self.cdp.send("Page.stopScreencast")
        self.page.wait_for_timeout(200)
        # wall-clock offset of first frame vs start
        first = self.frames[0][1]
        lst = os.path.join(self.dir, "list.txt")
        with open(lst, "w") as f:
            for i, (fn, ts) in enumerate(self.frames):
                nxt = self.frames[i + 1][1] if i + 1 < len(self.frames) else first + end
                f.write(f"file '{fn}'\nduration {max(0.001, nxt - ts):.4f}\n")
            f.write(f"file '{self.frames[-1][0]}'\n")
        json.dump({"caps": self.caps, "dur": end, "first": 0}, open(os.path.join(self.dir, "meta.json"), "w"))
        print(f"{self.name}: {len(self.frames)} frames, {end:.1f}s, {len(self.caps)} captions", flush=True)
