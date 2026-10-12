"""Turn recorded frame sequences into 1920x1080 clips with a caption band (ASS subtitles)."""
import json, os, subprocess, sys

REC = "/tmp/claude-0/vid/rec"
CLIPS = "/tmp/claude-0/vid/clips"
os.makedirs(CLIPS, exist_ok=True)

HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Inter SemiBold,44,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,80,80,{mv_main},1
Style: Sub,Inter,32,&H003FD2FF,&H003FD2FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,80,80,{mv_sub},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ts(t):
    t = max(0, t)
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def ass_for(caps, dur, path, has_sub_any):
    mv_main, mv_sub = (78, 26) if has_sub_any else (52, 26)
    out = HEAD.format(mv_main=mv_main, mv_sub=mv_sub)
    for i, (t, main, sub) in enumerate(caps):
        end = caps[i + 1][0] if i + 1 < len(caps) else dur
        mm = (78 if sub else 52)
        out += f"Dialogue: 0,{ts(t)},{ts(end)},Main,,0,0,{mm},,{{\\fad(180,120)}}{main}\n"
        if sub:
            out += f"Dialogue: 0,{ts(t)},{ts(end)},Sub,,0,0,26,,{{\\fad(180,120)}}{sub}\n"
    open(path, "w").write(out)


def clip(name, speed=1.0):
    d = os.path.join(REC, name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    caps = [(t / speed, a, b) for t, a, b in meta["caps"]]
    dur = meta["dur"] / speed
    ass = os.path.join(CLIPS, f"{name}.ass")
    ass_for(caps, dur, ass, any(c[2] for c in caps))
    vf = (f"setpts=PTS/{speed},fps=30,scale=1600:900:flags=lanczos,"
          f"pad=1920:1080:160:18:color=0x17171C,subtitles={ass}:fontsdir=/usr/share/fonts/opentype/inter")
    out = os.path.join(CLIPS, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(d, "list.txt"),
                    "-vf", vf, "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-r", "30", out], check=True)
    print("clip", out, f"{dur:.1f}s")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        n, _, s = a.partition("@")
        clip(n, float(s or 1))
