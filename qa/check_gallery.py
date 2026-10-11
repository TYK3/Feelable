"""Release gate for the prepared gallery (precision audit: ten featured lessons - five basic, five
complex - plus an extended library;
zero unresolved missing targets). Builds every example exactly as the gallery cache does (fresh
analysis + reviewed corrections, strict) and checks each one.

    python ../qa/check_gallery.py OUT_DIR        (from backend/)

Writes OUT_DIR/gallery-report.json and a per-part contact sheet per example. Exit code 1 on any
gate failure. Physical tactile suitability still needs a Braille reader / specialist teacher."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from curate import GATE, run  # noqa: E402

ROOT = os.path.dirname(HERE)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
manifest = json.load(open(os.path.join(ROOT, "samples", "manifest.json")))
report, ok = [], True
# precision audit: ten featured lessons, five basic and five complex; the rest is an extended library
feat = [s for s in manifest if s.get("featured")]
assert len(feat) == 10, f"there must be exactly 10 featured lessons, found {len(feat)}"
assert sum(s.get("group") == "complex" for s in feat) == 5, "five featured lessons must be complex"
assert sum(s.get("group") == "simple" for s in feat) == 5, "five featured lessons must be basic"
for s in manifest:
    featured = bool(s.get("featured"))
    # featured lessons are the release: reviewed corrections must all apply (strict). The extended
    # library was prepared before the precision audit; it is built and reported, not gated.
    r = run(s["id"], out, strict=featured)
    issues = r["readiness"]["issues"]
    bad = [i for i in issues if i["code"] in GATE]
    exp = [e.lower() for e in s.get("expected_parts", [])]
    names = [p["name"].lower() for p in r["parts"] if p["include"]]
    missing_exp = [e for e in exp if not any(e in n or n.startswith(e) for n in names)]
    entry = {"id": s["id"], "group": s.get("group"), "parts": len(names), "status": r["readiness"]["status"],
             "corrections": len(s.get("review") or []), "gate_failures": [f"{i['code']}: {i['message']}" for i in bad],
             "expected_missing": missing_exp,
             "open_checks": [f"{i['code']}: {i['message'][:90]}" for i in issues if i["severity"] == "warn" and i["code"] not in GATE]}
    entry["featured"] = featured
    if featured:
        ok &= not bad and not missing_exp
    report.append(entry)
    subprocess.run([sys.executable, os.path.join(HERE, "part_sheet.py"), os.path.join(out, "cache", "a4", s["id"]),
                    os.path.join(out, f"{s['id']}-parts.jpg"), "6"], check=False)
    verdict = ('PASS' if not bad and not missing_exp else 'FAIL') if featured else \
        ('ext-ok' if not bad and not missing_exp else 'ext-open')
    print(f"{verdict:8} {s['id']:24} {len(names):3} parts, "
          f"{entry['corrections']} corrections, gate {len(bad)}, expected missing {missing_exp}", flush=True)
json.dump(report, open(os.path.join(out, "gallery-report.json"), "w"), indent=1)
print("GALLERY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
