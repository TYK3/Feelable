#!/usr/bin/env python3
"""Original artwork for three featured lessons (precision audit, "five basic and five complex"):

    shapes.png       Basic 5   - four flat shapes (mathematics)
    flowchart.png    Complex 4 - a labelled decision flowchart (computing)
    transit-map.png  Complex 5 - a town bus network (geography and transport)

Drawn here from scratch by the Feelable team (no third-party source), released CC0 with the
other generated samples. Uses the same Canvas helpers as make_samples.py. Does NOT rewrite
manifest.json (that file holds reviewed corrections).

    python3 samples/make_featured.py [shapes|flowchart|transit-map]
"""
from __future__ import annotations

import os
import sys

from PIL import ImageDraw
from shapely.geometry import Point, Polygon, box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_samples import HERE, INK, PAGE, S, Canvas, arrow, darker, font  # noqa: E402


def make_shapes():
    c = Canvas(1400, 950, PAGE)
    c.title("Fig. 2.1   Four flat shapes")
    shapes = [
        ("Circle", Point(300, 380).buffer(150, resolution=96), (66, 133, 214), (300, 380), "below"),
        ("Triangle", Polygon([(700, 230), (860, 530), (540, 530)]), (232, 134, 40), (700, 420), "below"),
        ("Rectangle", box(960, 260, 1300, 470), (52, 160, 98), (1130, 365), "below"),
        ("Square", box(1000, 620, 1230, 850), (150, 98, 196), (1115, 735), "left"),
    ]
    for name, g, col, (px, py), side in shapes:
        c.shadow(g, 6, 8, 5, alpha=30)
        c.shape(g, col, darker(col, 0.6), 2.5)
    c.label("Circle", 300, 580, 300, 470, side="below", size=26)
    c.label("Triangle", 700, 590, 700, 490, side="below", size=26)
    c.label("Rectangle", 1130, 530, 1130, 430, side="below", size=26)
    c.label("Square", 900, 735, 1060, 735, side="left", size=26)
    c.caption("A circle has no corners. A triangle has 3, a rectangle and a square have 4.", y=900)
    return c.final()


def _rbox(c: Canvas, x0, y0, x1, y1, fill, r=16):
    d = c.d
    d.rounded_rectangle((x0 * S, y0 * S, x1 * S, y1 * S), radius=r * S, fill=fill, outline=(20, 20, 26), width=3 * S)


def _diamond(c: Canvas, cx, cy, w, h, fill):
    c.poly([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)], fill=fill,
           outline=(20, 20, 26), width=3)


def make_flowchart():
    c = Canvas(1400, 1000, PAGE)
    c.title("Fig. 8.2   Plant watering robot")
    A = (198, 222, 247)     # actions share one colour on purpose: colour is style, not identity
    D = (255, 226, 120)     # decisions
    T = (210, 210, 216)     # start / stop
    nodes = {
        "Start": ((130, 120, 360, 190), T, "box"),
        "Read soil sensor": ((130, 250, 360, 320), A, "box"),
        "Soil dry?": ((245, 445), D, "diamond"),
        "Open valve": ((560, 410, 790, 480), A, "box"),
        "Wait": ((560, 560, 790, 630), A, "box"),
        "Close valve": ((560, 710, 790, 780), A, "box"),
        "Log reading": ((130, 710, 360, 780), A, "box"),
        "Stop": ((130, 860, 360, 930), T, "box"),
    }
    for name, (geo, col, kind) in nodes.items():
        if kind == "box":
            _rbox(c, *geo, col)
            cx, cy = (geo[0] + geo[2]) / 2, (geo[1] + geo[3]) / 2
        else:
            cx, cy = geo
            _diamond(c, cx, cy, 260, 130, col)
        c.text(cx, cy, name, 24, "Medium", anchor="mm")
    K = (22, 22, 28)
    arrow(c, [(245, 190), (245, 248)], 3, 16, K)
    arrow(c, [(245, 320), (245, 378)], 3, 16, K)
    arrow(c, [(375, 445), (558, 445)], 3, 16, K)           # yes
    arrow(c, [(675, 480), (675, 558)], 3, 16, K)
    arrow(c, [(675, 630), (675, 708)], 3, 16, K)
    arrow(c, [(560, 745), (362, 745)], 3, 16, K)
    arrow(c, [(245, 510), (245, 708)], 3, 16, K)           # no
    arrow(c, [(245, 780), (245, 858)], 3, 16, K)
    c.text(460, 425, "Yes", 22, "SemiBold", anchor="mm")
    c.text(275, 600, "No", 22, "SemiBold", anchor="lm")
    c.caption("Every morning the robot checks the soil and waters only when it is dry.", y=965)
    return c.final()


def make_transit_map():
    c = Canvas(1400, 1000, PAGE)
    c.title("Fig. 6.3   Riverside bus routes")
    st = {"Library": (150, 260), "Park": (420, 260), "Central": (700, 480), "Museum": (980, 480),
          "School": (1250, 700), "Hospital": (150, 700), "Market": (420, 700), "Pool": (980, 260),
          "Depot": (1250, 260), "Stadium": (420, 150), "Garden": (700, 860), "River": (980, 860)}
    routes = [
        ("Red route", (190, 30, 45), ["Library", "Park", "Central", "Museum", "School"], None),
        ("Blue route", (22, 81, 164), ["Hospital", "Market", "Central", "Pool", "Depot"], (22, 10)),
        ("Green route", (35, 115, 58), ["Stadium", "Park", "Market", "Garden", "River"], None),
    ]
    d = ImageDraw.Draw(c.im)
    for name, col, stops, dash in routes:
        pts = [st[s] for s in stops]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if dash is None:
                d.line((x0 * S, y0 * S, x1 * S, y1 * S), fill=col, width=9 * S)
            else:                                        # a dashed route: identity is not colour alone
                L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                on, off = dash
                t = 0.0
                while t < L:
                    a, b = t / L, min(L, t + on) / L
                    d.line(((x0 + (x1 - x0) * a) * S, (y0 + (y1 - y0) * a) * S,
                            (x0 + (x1 - x0) * b) * S, (y0 + (y1 - y0) * b) * S), fill=col, width=9 * S)
                    t += on + off
                for (ax, ay), (bx, by) in (((x0, y0), (x1, y1)), ((x1, y1), (x0, y0))):
                    k = 18 / L                           # every dashed run reaches its stops
                    d.line((ax * S, ay * S, (ax + (bx - ax) * k) * S, (ay + (by - ay) * k) * S), fill=col, width=9 * S)
    # each name sits clear of every line so a finger (and the reader) finds it beside its stop
    place = {"Library": (0, 38), "Park": (-70, -30), "Central": (0, 40), "Museum": (0, -36), "School": (0, 40),
             "Hospital": (0, 40), "Market": (-80, 34), "Pool": (0, -36), "Depot": (0, 40), "Stadium": (80, 0),
             "Garden": (0, 40), "River": (0, 40)}
    for name, (x, y) in st.items():
        r = 13
        d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=(255, 255, 255),
                  outline=(15, 15, 18), width=4 * S)
        dx, dy = place[name]
        c.text(x + dx, y + dy, name, 24, "Medium", anchor="mm")
    for i, (name, col, _s, dash) in enumerate(routes):
        y = 880 + i * 34
        c.text(60, y, name, 24, "SemiBold", fill=col, anchor="lm")
    c.caption("Change buses where two routes meet: Park, Central and Market.", y=975)
    return c.final()


ART = {"shapes": make_shapes, "flowchart": make_flowchart, "transit-map": make_transit_map}

if __name__ == "__main__":
    only = set(sys.argv[1:]) or set(ART)
    for k in only:
        im = ART[k]()
        p = os.path.join(HERE, f"{k}.png")
        im.save(p, optimize=True)
        print("wrote", p, im.size)
