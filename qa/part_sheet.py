"""Contact sheet: one tile per part, the part highlighted in yellow on a dimmed original, so every
selection can be checked by eye (handover: "inspect every original/tactile selection").
    python qa/part_sheet.py JOB_DIR OUT.jpg [COLS]"""
import pickle
import sys

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
import cv2
import numpy as np

jd, out = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 6
a = pickle.load(open(f"{jd}/analysis.pkl", "rb"))["analysis"]
base = (a.image.astype(np.float32) * 0.45 + 140).astype(np.uint8)
tw = 300
tiles = []
for p in a.parts:
    t = base.copy()
    m = p.mask > 0
    t[m] = (0.35 * a.image[m] + 0.65 * np.array([255, 212, 59])).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(t, cnts, -1, (20, 20, 20), max(2, a.width // 300))
    s = tw / a.width
    t = cv2.resize(t, (tw, int(a.height * s)), interpolation=cv2.INTER_AREA)
    bar = np.full((26, tw, 3), 255 if p.include else 200, np.uint8)
    cv2.putText(bar, f"{p.id} {'' if p.include else '(off) '}{p.name}"[:38], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    tiles.append(np.vstack([bar, t]))
h = max(t.shape[0] for t in tiles)
tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 4), (0, 0)), constant_values=255) for t in tiles]
while len(tiles) % cols:
    tiles.append(np.full_like(tiles[0], 255))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
