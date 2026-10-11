"""Release gate for the prepared gallery (handover: exactly 15 examples, five complex systems,
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
assert len(manifest) == 15, f"gallery must have exactly 15 examples, has {len(manifest)}"
assert sum(s.get("group") == "complex" for s in manifest) == 5, "gallery must have five complex systems"
for s in manifest:
    r = run(s["id"], out)
    issues = r["readiness"]["issues"]
    bad = [i for i in issues if i["code"] in GATE]
    exp = [e.lower() for e in s.get("expected_parts", [])]
    names = [p["name"].lower() for p in r["parts"] if p["include"]]
    missing_exp = [e for e in exp if not any(e in n or n.startswith(e) for n in names)]
    entry = {"id": s["id"], "group": s.get("group"), "parts": len(names), "status": r["readiness"]["status"],
             "corrections": len(s.get("review") or []), "gate_failures": [f"{i['code']}: {i['message']}" for i in bad],
             "expected_missing": missing_exp,
             "open_checks": [f"{i['code']}: {i['message'][:90]}" for i in issues if i["severity"] == "warn" and i["code"] not in GATE]}
    ok &= not bad and not missing_exp
    report.append(entry)
    subprocess.run([sys.executable, os.path.join(HERE, "part_sheet.py"), os.path.join(out, "cache", "a4", s["id"]),
                    os.path.join(out, f"{s['id']}-parts.jpg"), "6"], check=False)
    print(f"{'PASS' if not bad and not missing_exp else 'FAIL'} {s['id']:24} {len(names):3} parts, "
          f"{entry['corrections']} corrections, gate {len(bad)}, expected missing {missing_exp}", flush=True)
json.dump(report, open(os.path.join(out, "gallery-report.json"), "w"), indent=1)
print("GALLERY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
