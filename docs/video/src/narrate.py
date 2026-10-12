"""Narration for the Feelable demo: Kokoro neural voices (Apache-2.0), placed on the edit's timeline."""
import subprocess, sys
import numpy as np
import soundfile as sf
sys.path.insert(0, "/tmp/claude-0/tts")
from say import tts

V = "/tmp/claude-0/vid"
NARR, PHONE = 1, 9          # af_bella narrates; bm_george is the phone's own voice
SR = 24000


def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]))


# segment start times on the final timeline
starts, t = {}, 0.0
for line in open(f"{V}/final_list.txt"):
    p = line.split("'")[1]
    name = p.split("/")[-1].replace("_f.mp4", "").replace(".mp4", "")
    starts[name] = t
    t += dur(p)
TOTAL = t

L = [  # (segment, offset s, voice, text, speed)
    ("c1_title", 0.5, NARR, "This is Feelable. Every diagram, feelable.", 1.0),
    ("c2_story", 0.3, NARR, "Image ay eye makes amazing pictures. But only for people who can see.", 1.0),
    ("c3_question", 0.2, NARR, "So we asked: who does image ay eye leave out?", 1.0),
    ("c4_stats", 0.2, NARR, "Blind students. Every diagram in their textbooks is remade by hand. It takes hours, and they wait weeks.", 1.0),
    ("c5_idea", 0.3, NARR, "So Feelable uses image ay eye to remake a picture for the fingers.", 1.0),
    ("home", 0.4, NARR, "Touch a part of the example, and it tells you what it is.", 1.0),
    ("home", 6.8, NARR, "There are ten ready-made lessons, five basic and five complex, from biology to maps.", 1.0),
    ("upload", 0.4, NARR, "Now, a photo of a real worksheet.", 1.0),
    ("upload", 3.2, NARR, "This is real processing, not sped up. Feelable reads every label, finds each part, "
                          "and applies the rules for touch: raised outlines, a different texture for each neighbour, and braille.", 1.0),
    ("upload", 19.8, NARR, "Evaporation and rain become arrows you can follow with a finger.", 1.0),
    ("skull", 0.4, NARR, "Now the hard case: a textbook skull, with twenty-seven small labels, and bones of almost the same colour.", 1.0),
    ("skull", 7.1, NARR, "Scan details shows exactly what the ay eye saw. Every label it read, every pointer it followed, "
                         "and every bone split along a real edge. Even the sutures are traced where two bones meet.", 1.0),
    ("fix", 0.4, NARR, "The teacher stays in charge. Anything uncertain is listed first, with a fix.", 1.0),
    ("fix", 5.4, NARR, "Missing a part? Click inside it, and Feelable suggests the outline. "
                       "Students only see a lesson after the teacher approves it.", 1.0),
    ("graphs", 0.3, NARR, "It understands structure, too.", 1.0),
    ("graphs", 2.8, NARR, "A flowchart is spoken with its yes and no branches.", 1.0),
    ("graphs2", 0.3, NARR, "And on a bus map, it works out where the routes meet.", 1.0),
    ("share", 0.3, NARR, "Approve, and it's ready. Print it on swell paper, or share a Q R code.", 1.0),
    ("share", 8.2, NARR, "There's even a three-D printable relief of the same diagram.", 1.0),
    ("phone_f", 0.4, NARR, "And with no printer at all, a student explores on a phone.", 1.0),
    ("phone_f", 4.5, PHONE, "Nucleus.", 1.0),
    ("phone_f", 6.5, PHONE, "Nucleolus.", 1.0),
    ("phone_f", 8.5, PHONE, "Golgi body.", 1.0),
    ("phone_f", 10.5, PHONE, "Mitochondrion.", 1.0),
    ("phone_f", 12.55, PHONE, "Cell membrane.", 1.0),
    ("phone_f", 14.55, PHONE, "Ribosomes.", 1.0),
    ("phone_f", 16.8, PHONE, "Golgi body. Inside the cytoplasm.", 1.0),
    ("c7_results", 0.3, NARR, "We measure where every outline lands. Valid targets rose from sixty-seven to a hundred and four, out of a hundred and forty-two.", 1.08),
    ("c8_close", 0.4, NARR, "Feelable. Every diagram, feelable.", 1.0),
]

mix = np.zeros(int((TOTAL + 1) * SR), np.float32)
placed, ok = [], True
for seg, off, sid, text, sp in L:
    key = seg if seg in starts else seg.replace("_f", "")
    a = tts.generate(text, sid=sid, speed=sp)
    x = np.asarray(a.samples, np.float32)
    # trim leading / trailing silence
    nz = np.nonzero(np.abs(x) > 0.01)[0]
    x = x[max(0, nz[0] - 600): nz[-1] + 1800] if len(nz) else x
    if sid == PHONE:
        x = x * 0.9
    st = starts[key] + off
    en = st + len(x) / SR
    placed.append((st, en, seg, text))
    i = int(st * SR)
    mix[i:i + len(x)] += x[: len(mix) - i]
placed.sort()
seg_end = {}
names = list(starts)
for k, n in enumerate(names):
    seg_end[n] = starts[names[k + 1]] if k + 1 < len(names) else TOTAL
for k, (st, en, seg, text) in enumerate(placed):
    nxt = placed[k + 1][0] if k + 1 < len(placed) else TOTAL
    key = seg if seg in starts else seg.replace("_f", "")
    flag = ""
    if en > nxt - 0.15:
        flag += f"  OVERLAPS next by {en - nxt + 0.15:.2f}s"; ok = False
    if en > seg_end[key] + 0.4 and not flag:
        flag += f"  runs {en - seg_end[key]:.2f}s past its scene"
    print(f"{st:6.1f}-{en:6.1f} {seg:11s} {text[:58]}{flag}")
mix = mix[: int(TOTAL * SR)]
mix = mix / max(1e-6, np.abs(mix).max()) * 0.89
sf.write(f"{V}/narration.wav", mix, SR)
print("OK" if ok else "FIX OVERLAPS", f"total {TOTAL:.1f}s")

def _f(x):
    return f"{int(x//3600):02d}:{int(x%3600//60):02d}:{int(x%60):02d},{int(round((x%1)*1000)) % 1000:03d}"
with open(f"{V}/feelable-demo.srt", "w") as fh:
    for i, (st, en, seg, text) in enumerate(placed, 1):
        shown = text.replace("ay eye", "AI").replace("three-D", "3D").replace("Q R code", "QR code")
        if seg == "phone_f" and not shown.startswith("And"):
            shown = "[phone] " + shown
        fh.write(f"{i}\n{_f(st)} --> {_f(en + 0.25)}\n{shown}\n\n")
print("srt written")
