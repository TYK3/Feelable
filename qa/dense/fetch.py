"""Fetch the OpenStax figures for the dense benchmark and build the low-res variants.
usage: python qa/dense/fetch.py   (from the repo root). User images must already be in qa/retest-local/.
Figures: OpenStax Anatomy & Physiology (CC BY 4.0), via github.com/philschatz/anatomy-book."""
import io, json, pathlib, urllib.request
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = "https://raw.githubusercontent.com/philschatz/anatomy-book/master/resources/"
OX = {
    "ox-lateral-view-of-skull.jpg": "705_Lateral_View_of_Skull-01.jpg",
    "ox-parts-of-skull.jpg": "703_Parts_of_Skull-01.jpg",
    "ox-structure-of-the-eye.jpg": "1413_Structure_of_the_Eye.jpg",
    "ox-the-structures-of-the-ear.jpg": "1404_The_Structures_of_the_Ear.jpg",
    "ox-the-endocrine-system.jpg": "1801_The_Endocrine_System.jpg",
    "ox-major-respiratory-organs.jpg": "2301_Major_Respiratory_Organs.jpg",
    "ox-gross-anatomy-of-the-lungs.jpg": "2312_Gross_Anatomy_of_the_Lungs.jpg",
    "ox-components-of-the-digestive-system.jpg": "2401_Components_of_the_Digestive_System.jpg",
    "ox-the-kidney.jpg": "2610_The_Kidney.jpg",
    "ox-lobes-of-cerebral-cortex.jpg": "1306_Lobes_of_Cerebral_CortexN.jpg",
    "ox-tooth.jpg": "2409_Tooth.jpg",
    "ox-anatomy-of-nose-pharynx-mouth-larynx.jpg": "2303_Anatomy_of_Nose-Pharynx-Mouth-Larynx.jpg",
}


def main() -> None:
    out = ROOT / "qa" / "retest-local" / "ox"
    out.mkdir(parents=True, exist_ok=True)
    for name, src in OX.items():
        p = out / name
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(RAW + urllib.request.quote(src)).read())
    cases = {c["id"]: c for c in json.loads((ROOT / "qa/dense/manifest.json").read_text())["cases"]}
    for c in cases.values():
        if c["kind"] != "lowres":
            continue
        src = Image.open(ROOT / cases[c["source"]]["file"]).convert("RGB")
        s = 420 / src.width
        src.resize((420, round(src.height * s)), Image.LANCZOS).save(ROOT / c["file"], quality=70)
    print("ok")


if __name__ == "__main__":
    main()
