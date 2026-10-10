"""OCR spelling repair for diagram labels.

Small or blurry label text (a phone screenshot of a textbook figure) comes back from OCR as
"Parktatbone", "Witroouschamber" or "Occpitai bone". Diagram labels come from a fairly small
vocabulary, so each reading is snapped to the closest known term when — and only when — it is
already close to one:

1. whole phrase: "parktatbone" ~ "parietalbone" -> "parietal bone";
2. word by word, with glued words split by a small dynamic programme over the vocabulary
   ("witroouschamber" -> "vitreous" + "chamber").

Readings made of words we already know (or plain English glue words) are returned unchanged, so a
clean label is never "corrected". The vocabulary lives in engine/lexicon_terms.py.
"""
from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from engine.lexicon_terms import TERMS

# words that are fine on a label even though they're not science terms
GLUE = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "at", "for", "with", "from", "by", "into",
    "contains", "containing", "attached", "called", "also", "known", "as", "is", "are", "its", "it",
    "this", "that", "these", "those", "female", "male", "gum", "humor", "humour", "blind", "spot",
    "part", "parts", "area", "region", "layer", "layers", "inner", "outer", "main", "primary",
    "secondary", "tertiary", "small", "large", "big", "little", "upper", "lower", "top", "bottom",
    "front", "back", "side", "new", "old", "one", "two", "three", "four", "five", "six", "ii", "iii",
    "human", "anatomy", "system", "systems", "structure", "structures", "organs", "major", "minor",
    "image", "picture", "figure", "fig", "diagram", "labelled", "labeled", "simple", "basic",
    "animal", "plant", "cells", "typical", "generalized", "generalised", "model", "parts", "shapes",
}

_SPLIT = re.compile(r"[a-z]+")


def _common_english() -> set[str]:
    import pathlib
    try:
        txt = (pathlib.Path(__file__).parent / "data" / "common_english.txt").read_text(encoding="utf-8")
    except OSError:
        return set()
    return {ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")}


ENGLISH = _common_english()      # real words: never corrected (but never a correction target either)
PHRASES = sorted({t.strip() for t in TERMS.splitlines() if t.strip()})
WORDS = sorted({w for p in PHRASES for w in _SPLIT.findall(p) if len(w) >= 2} | GLUE)
_KNOWN = set(WORDS)
_SQUASHED = {re.sub(r"[^a-z]", "", p): p for p in PHRASES}
_SQ_KEYS = list(_SQUASHED)
_BY_LEN: dict[int, list[str]] = {}
for _w in WORDS:
    _BY_LEN.setdefault(len(_w), []).append(_w)


def _allowed(n: int) -> int:
    """Edits allowed when snapping a chunk of n letters to a vocabulary word."""
    if n <= 3:
        return 0
    if n <= 5:
        return 1
    if n <= 9:
        return 2
    return 3


@lru_cache(maxsize=4096)
def _nearest_word(chunk: str) -> tuple[str, int] | None:
    lim = _allowed(len(chunk))
    if chunk in _KNOWN:
        return chunk, 0
    if lim == 0:
        return None
    cands: list[str] = []
    for n in range(len(chunk) - lim, len(chunk) + lim + 1):
        cands.extend(_BY_LEN.get(n, ()))
    if not cands:
        return None
    hits = process.extract(chunk, cands, scorer=Levenshtein.distance, score_cutoff=lim, limit=None)
    # two or more edits only when the first letter agrees (OCR rarely gets the first letter wrong
    # *and* more besides); ties go to the same first letter, the same length, the longer prefix
    hits = [(w, int(d)) for w, d, _ in hits if d <= 1 or w[0] == chunk[0]]
    if not hits:
        return None
    w, d = min(hits, key=lambda h: (h[1], h[0][0] != chunk[0], abs(len(h[0]) - len(chunk)), -_prefix(h[0], chunk)))
    return w, d


def _prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _segment(token: str) -> list[str] | None:
    """Split a glued, misread token into vocabulary words (fewest edits, then fewest words)."""
    n = len(token)
    INF = (10 ** 9, 0)
    best: list[tuple[int, int]] = [INF] * (n + 1)
    back: list[tuple[int, str] | None] = [None] * (n + 1)
    best[0] = (0, 0)
    for i in range(1, n + 1):
        for j in range(max(0, i - 16), i - 1):
            if best[j] == INF:
                continue
            hit = _nearest_word(token[j:i])
            if hit is None:
                continue
            short = 1 if len(hit[0]) <= 3 and hit[0] not in ("ear", "eye", "rib", "jaw", "lip", "toe", "gum") else 0
            cost = (best[j][0] + hit[1] + short, best[j][1] + 1)
            if cost < best[i]:
                best[i], back[i] = cost, (j, hit[0])
    if best[n] == INF or best[n][0] > max(1, round(0.3 * n)):
        return None
    out: list[str] = []
    i = n
    while i > 0:
        j, w = back[i]  # type: ignore[misc]
        out.append(w)
        i = j
    return out[::-1]


def _known(t: str) -> bool:
    return t in _KNOWN or t in ENGLISH or (len(t) > 3 and t.endswith("s") and t[:-1] in _KNOWN) or (t.endswith("es") and t[:-2] in _KNOWN)


def _cap_like(src: str, dst: str) -> str:
    return dst[:1].upper() + dst[1:] if src[:1].isupper() else dst


def correct(text: str) -> str:
    """Return `text` with misread science terms repaired; unchanged when nothing is close."""
    raw = text.strip()
    low = raw.lower()
    tokens = _SPLIT.findall(low)
    if not tokens:
        return text
    if all(_known(t) or len(t) <= 2 for t in tokens):
        return text
    # 1. the whole label is one known term
    sq = "".join(tokens)
    if len(sq) >= 5:
        lim = max(1, int(0.36 * len(sq)))
        hits = process.extract(sq, _SQ_KEYS, scorer=Levenshtein.distance, score_cutoff=lim, limit=8)
        ok = []
        for key, d, _ in hits:
            if d > max(1, int(0.36 * max(len(key), len(sq)))) or (key[0] != sq[0] and d > int(0.25 * len(sq))):
                continue
            phrase = _SQUASHED[key]
            # never trade a word we can already read for a different one
            kept = [t.rstrip("s") for t in tokens if _known(t) and len(t) >= 4]
            if all(t in phrase for t in kept):
                ok.append((d, abs(len(key) - len(sq)), -_prefix(key, sq), phrase))
        if ok:
            return _cap_like(raw, min(ok)[3])
    # 2. word by word
    out: list[str] = []
    changed = False
    for t in tokens:
        if _known(t) or len(t) <= 3:
            out.append(t)
            continue
        hit = _nearest_word(t)
        if hit is not None:
            out.append(hit[0])
            changed = changed or hit[0] != t
            continue
        seg = _segment(t) if len(t) >= 7 else None
        if seg:
            out.extend(seg)
            changed = True
        else:
            out.append(t)
    if not changed:
        return text
    return _cap_like(raw, " ".join(out))


def is_known(text: str) -> bool:
    """Every word of `text` is a vocabulary or glue word (so it reads as a real label)."""
    toks = _SPLIT.findall(text.lower())
    return bool(toks) and all(_known(t) or len(t) <= 2 for t in toks)


def is_term(text: str) -> bool:
    """`text` is one complete vocabulary term ("large intestine", "pancreas")."""
    sq = "".join(_SPLIT.findall(text.lower()))
    return bool(sq) and (sq in _SQUASHED or (sq.endswith("s") and sq[:-1] in _SQUASHED))
