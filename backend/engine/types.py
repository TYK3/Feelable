"""Shared data types for the Feelable engine.

OWNER: Engine team (CONTRACT-v2 §0). New fields always get defaults so old pickles still load.

Coordinate systems
------------------
* SOURCE coords: pixels of the *working image* (input resized so the long side
  is exactly WORKING_LONG_SIDE px, after optional page flattening).
  Analysis / Part.mask / Part.polygons / Label.bbox use SOURCE coords.
* TACTILE coords: pixels of the tactile canvas (page 1), PX_PER_MM px per mm.
  TactileResult.* uses TACTILE coords.
All images are numpy arrays: RGB uint8 (H, W, 3) or masks uint8 (H, W) with 0/255.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Literal, Optional

import numpy as np

WORKING_LONG_SIDE = 1200          # px, long side of the working image
PX_PER_MM = 5.0                   # tactile canvas resolution
PAPER_MM = {"a4": (297.0, 210.0), "letter": (279.4, 215.9)}   # landscape (w, h)
PAGE_MARGIN_MM = 12.0

# Braille geometry (BANA / UEB standard-ish), millimetres
BRAILLE_DOT_DIAMETER_MM = 1.5
BRAILLE_DOT_SPACING_MM = 2.5      # centre-to-centre inside a cell
BRAILLE_CELL_SPACING_MM = 6.2     # cell origin to next cell origin
BRAILLE_LINE_SPACING_MM = 10.0

MIN_LINE_MM = 1.0                 # thinnest raised line
OUTLINE_MM = 1.5                  # standard part outline
MIN_PART_MM = 6.0                 # smallest feature dimension after enlargement
MAX_RECOMMENDED_PARTS = 8

TextureName = Literal["none", "dots", "diag", "horiz", "vert", "cross", "solid", "grid"]


def _fill_missing_defaults(obj, state: dict) -> None:
    """Unpickling an object saved before a field existed: give that field its default."""
    obj.__dict__.update(state)
    for f in fields(obj):
        if f.name in obj.__dict__:
            continue
        if f.default is not MISSING:
            obj.__dict__[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            obj.__dict__[f.name] = f.default_factory()  # type: ignore[misc]


@dataclass
class Part:
    id: str                                   # "p1", "p2", ... stable across edits
    name: str                                 # e.g. "nucleus"; "part 3" if unknown
    description: str                          # one short spoken sentence, e.g. "Nucleus. Controls the cell."
    key: str                                  # braille key, 1-2 lowercase letters a-z, unique per diagram
    color: tuple[int, int, int]               # representative RGB colour in the source
    mask: np.ndarray                          # SOURCE coords, uint8 0/255, region with holes FILLED
    polygons: list[list[tuple[int, int]]]     # SOURCE coords, simplified outer contours (one per component)
    bbox: tuple[int, int, int, int]           # SOURCE coords x0, y0, x1, y1
    area: int                                 # pixel count of mask
    count: int = 1                            # connected components (e.g. 5 chloroplasts -> 5)
    include: bool = True                      # teacher can exclude a part
    kind: Literal["region", "line"] = "region"
    label_text: Optional[str] = None          # the OCR label this part was matched to (source text)
    match_method: str = "none"                # "leader" | "inside" | "nearest" | "ai" | "teacher" | "none"
    description_source: str = "auto"          # "auto" | "ai" | "teacher"

    def __setstate__(self, state):
        _fill_missing_defaults(self, state)


@dataclass
class Label:
    text: str
    bbox: tuple[int, int, int, int]           # SOURCE coords
    confidence: float
    part_id: Optional[str] = None             # part this label names, if matched
    role: Literal["part_label", "title", "caption", "other"] = "other"
    raw: Optional[str] = None                 # what OCR first read, when the zoomed re-read / spelling
                                              # repair changed it (e.g. "Parktatbone" -> "Parietal bone")

    def __setstate__(self, state):
        _fill_missing_defaults(self, state)


@dataclass
class Analysis:
    width: int                                # working image size (SOURCE)
    height: int
    image: np.ndarray                         # working RGB image
    title: str                                # e.g. "plant cell"
    diagram_type: str                         # e.g. "biology", "chart", "circuit", "map", "other"
    parts: list[Part]
    labels: list[Label]
    linework: np.ndarray                      # SOURCE coords uint8 0/255: meaningful strokes NOT in any part
                                              # (arrows, axes, wires). Text & leader lines REMOVED.
    dropped: list[str] = field(default_factory=list)    # human readable, e.g. "140 small dots (too small to feel)"
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    ai_used: bool = False
    page_flattened: bool = False
    input_kind: str = "diagram"               # "diagram" | "chart" | "photo_object" | "photo_scene" | "text_page" | "blank"
    unmatched_labels: list[str] = field(default_factory=list)   # label texts that look like part names
                                                                # but went to no part
    rotation: int = 0                         # degrees CLOCKWISE applied by auto-orientation (0/90/180/270)
    notes: list[str] = field(default_factory=list)              # info-level messages
    trace: dict = field(default_factory=dict)  # how the picture was read, for the "What Feelable saw"
                                               # view: {"leaders": [{"label", "points"}], "steps": {...}}

    def __setstate__(self, state):
        _fill_missing_defaults(self, state)


@dataclass
class TactileOptions:
    paper: Literal["a4", "letter"] = "a4"
    title_in_braille: bool = True


@dataclass
class Check:
    rule: str                                 # e.g. "Lines at least 1 mm"
    passed: bool
    detail: str


@dataclass
class TactileResult:
    width: int                                # canvas px (page 1)
    height: int
    px_per_mm: float
    ink: np.ndarray                           # (H, W) uint8, 255 = raised. Page 1 (drawing + keys + title)
    legend_ink: np.ndarray                    # (H, W) uint8, page 2: braille legend "key = name"
    print_layer: np.ndarray                   # (H, W, 3) uint8 RGB: page 1 with grey print labels for the
                                              # sighted teacher (raised ink black, print text grey)
    part_polygons: dict[str, list[list[tuple[int, int]]]]   # part id -> polygons in TACTILE coords
    key_positions: dict[str, tuple[int, int]]                # part id -> top-left of braille key, TACTILE
    textures: dict[str, str]                                 # part id -> TextureName
    legend: list[dict]                                       # [{"part_id","key","name"}] in legend order
    checks: list[Check]
    transform: tuple[float, float, float]                    # (scale, offset_x, offset_y): tactile = source*scale + offset
    legend_pages: list = field(default_factory=list)         # F03: further key pages (H, W) when one is not enough
                                                             # (before per-part enlargement)
