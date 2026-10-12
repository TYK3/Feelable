import json, os, subprocess
from PIL import Image, ImageDraw

V = "/tmp/claude-0/vid"
C = f"{V}/clips"
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30"]


def run(cmd):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + cmd, check=True)


def card(name, dur):
    out = f"{C}/{name}.mp4"
    run(["-loop", "1", "-t", f"{dur}", "-i", f"{V}/cards/{name}.png",
         "-vf", f"fps=30,format=yuv420p,fade=in:st=0:d=0.35,fade=out:st={dur - 0.35}:d=0.35"] + ENC + [out])
    return out


def faded(name, speed=1.0):
    """Re-time an app clip (already captioned) and add short fades."""
    src = f"{C}/{name}.mp4"
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src]))
    out = f"{C}/{name}_f.mp4"
    run(["-i", src, "-vf", f"fade=in:st=0:d=0.3,fade=out:st={dur - 0.3}:d=0.3"] + ENC + [out])
    return out


def phone():
    W, H, R = 456, 988, 56
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, W - 1, H - 1), radius=R, fill=255)
    m.save(f"{V}/phone_mask.png")
    meta = json.load(open(f"{V}/rec/phone/meta.json"))
    dur = meta["dur"]
    out = f"{C}/phone_f.mp4"
    fc = (f"[1:v]fps=30,scale={W}:{H}:flags=lanczos,format=rgba[ph];[2:v]format=gray,scale={W}:{H}[mk];"
          f"[ph][mk]alphamerge[phm];[0:v][phm]overlay=1218:46:shortest=1,format=yuv420p,"
          f"fade=in:st=0:d=0.35,fade=out:st={dur - 0.35}:d=0.35[v]")
    run(["-loop", "1", "-t", f"{dur:.2f}", "-i", f"{V}/cards/c6_phonebg.png",
         "-f", "concat", "-safe", "0", "-i", f"{V}/rec/phone/list.txt",
         "-loop", "1", "-i", f"{V}/phone_mask.png",
         "-filter_complex", fc, "-map", "[v]", "-t", f"{dur:.2f}"] + ENC + [out])
    return out


parts = [card("c1_title", 4.5), card("c2_story", 5), card("c3_question", 4), card("c4_stats", 7), card("c5_idea", 6.5),
         faded("home"), faded("upload"), faded("skull"), faded("fix"), faded("graphs"), faded("graphs2"), faded("share"),
         phone(), card("c7_results", 7.5), card("c8_close", 5)]
with open(f"{V}/final_list.txt", "w") as f:
    for p in parts:
        f.write(f"file '{p}'\n")
out = f"{V}/feelable-demo.mp4"
run(["-f", "concat", "-safe", "0", "-i", f"{V}/final_list.txt", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
     "-map", "0:v", "-map", "1:a", "-shortest"] + ENC + ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out])
d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out]))
print(f"{out}: {d:.1f}s, {os.path.getsize(out) / 1e6:.1f} MB")
