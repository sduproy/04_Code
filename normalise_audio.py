# normalise_audio.py — loudness-normalise FAV + DFE + corpus audio to a common target
import subprocess, pathlib, time
T0 = time.perf_counter()
from config import log_compute

BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
OUT  = BASE / "audio_norm"; OUT.mkdir(exist_ok=True)

# --- same clip_id -> file map as the audit ---
FAV = BASE / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
DFE = BASE / "Deepfake-Eval-2024" / "video-data"
COR = BASE / "Endorsement corpus"
SHORT = {"RealVideo-RealAudio":"rvra","FakeVideo-RealAudio":"fvra",
         "RealVideo-FakeAudio":"rvfa","FakeVideo-FakeAudio":"fvfa"}
index = {}
for p in FAV.rglob("*.mp4"):
    index[f"fav_{SHORT[p.relative_to(FAV).parts[0]]}_{p.parent.name}_{p.stem}"] = p
for p in DFE.glob("*.mp4"):
    index[f"dfe_{p.stem}"] = p
for lab in ["Fake", "Real"]:
    for p in (COR / lab).glob("*.mp4"):
        index[p.stem] = p

print(f"{len(index)} clips to normalise")
fail = []
for i, (cid, src) in enumerate(index.items(), 1):
    dst = OUT / f"{cid}.wav"
    if dst.exists():
        continue
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-af", "loudnorm=I=-23:TP=-2:LRA=7",
         "-ar", "16000", "-ac", "1", str(dst)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not dst.exists():
        fail.append(cid)
    if i % 100 == 0:
        print(f"{i}/{len(index)}  ({len(fail)} failed)")
print(f"done — {len(index)-len(fail)} written, {len(fail)} failed")
if fail:
    print("failed:", fail[:10])

log_compute("normalise_audio.py", T0, used_gpu=False)   # ffmpeg, CPU only