import sys, sherpa_onnx, soundfile as sf
M="/tmp/claude-0/tts/pkg/package/kokoro-int8-en-v0_19"
cfg=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(model=f"{M}/model.int8.onnx", voices=f"{M}/voices.bin", tokens=f"{M}/tokens.txt", data_dir=f"{M}/espeak-ng-data"),
    num_threads=4), max_num_sentences=2)
tts=sherpa_onnx.OfflineTts(cfg)
def say(text, sid, out, speed=1.0):
    a=tts.generate(text, sid=sid, speed=speed); sf.write(out, a.samples, a.sample_rate); return len(a.samples)/a.sample_rate
if __name__=="__main__":
    t="Blind students learn from the same textbooks as everyone else. But every diagram has to be remade by hand, and students wait weeks."
    for sid in [6,7,8,9,10,1,3]:
        print(sid, round(say(t, sid, f"s{sid}.wav"),1))
    print("speakers", tts.num_speakers)
