"""Print each pointer line's label and the point it ends on (curation aid).
    python qa/tips.py JOB_DIR"""
import os
import pickle
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
a = pickle.load(open(os.path.join(sys.argv[1], "analysis.pkl"), "rb"))["analysis"]
lbs = {lb.text: lb for lb in a.labels}
for L in (a.trace or {}).get("leaders", []):
    lb, pts = lbs.get(L["label"]), L["points"]
    if lb is None or len(pts) < 2:
        continue
    cx, cy = (lb.bbox[0] + lb.bbox[2]) / 2, (lb.bbox[1] + lb.bbox[3]) / 2
    e = pts[0] if (pts[0][0] - cx) ** 2 + (pts[0][1] - cy) ** 2 > (pts[-1][0] - cx) ** 2 + (pts[-1][1] - cy) ** 2 else pts[-1]
    at = [p.name for p in a.parts if p.mask[max(0, e[1] - 3):e[1] + 4, max(0, e[0] - 3):e[0] + 4].any()]
    print(f"{L['label'][:30]:30} tip={e} on={at}")
