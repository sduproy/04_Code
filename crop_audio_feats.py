# crop_audio_feats.py — recompute a_ features on a centred CFG.audio_crop_secs window of the
# normalised wavs, so clip duration can't leak into the features
# FAV clips ~5 s, DFE/corpus ~40 s median; FAV reals run longer than FAV fakes
# clips shorter than the window keep their full length (~3% of FAV at 3 s)
import pathlib, time, wave
T0 = time.perf_counter()
import pandas as pd
import librosa
from config import CFG
from compute_log import log_compute
from frames_faces_audio import mfcc_feats

BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
NORM = BASE / "audio_norm"
SRC  = BASE / "all_features_norm.csv"          # v_ cols + loudness-normalised audio
DST  = BASE / "all_features_normcrop.csv"      # same, with a_ cols from the cropped window
WIN  = CFG.audio_crop_secs

def cropped_feats(wav):
    with wave.open(str(wav)) as f:                  # header only, no decode
        dur = f.getnframes() / f.getframerate()
    start = max(0.0, (dur - WIN) / 2)               # centred: the first-3 s window let
    y, sr = librosa.load(str(wav), sr=CFG.audio_sr, mono=True,   # leading silence act as a
                         offset=start, duration=WIN)             # corpus shortcut (AUC 0.635 alone)
    return mfcc_feats(y, sr), dur < WIN

df = pd.read_csv(SRC, low_memory=False)
AUDIO_SETS = {"FakeAVCeleb", "Deepfake_Eval_2024", "EndorsementCorpus_v1"}  # DF40 has no audio
mask = df["source_dataset"].isin(AUDIO_SETS) & (df["a_ok"] == 1)
todo = df.loc[mask, "clip_id"]
print(f"{len(todo)} rows to re-extract on a {WIN:g} s centred window")

new = {}; fail = []; short = 0
for i, cid in enumerate(todo, 1):
    wav = NORM / f"{cid}.wav"
    try:
        feats, is_short = cropped_feats(wav)
    except Exception as e:
        fail.append((cid, type(e).__name__)); continue
    if feats.get("a_ok") != 1:
        fail.append((cid, "under 1s audio")); continue
    new[cid] = feats; short += is_short
    if i % 1000 == 0:
        print(f"{i}/{len(todo)}  ({len(fail)} failed)")
print(f"re-extracted {len(new)}, {len(fail)} failed, {short} shorter than {WIN:g} s (kept whole)")
for cid, why in fail[:10]:
    print(f"  {cid}: {why}")

newdf = pd.DataFrame.from_dict(new, orient="index")
for col in newdf.columns:
    if not col.startswith("a_"):
        continue
    mapped = df.loc[mask, "clip_id"].map(newdf[col])
    df.loc[mask, col] = mapped.where(mapped.notna(), df.loc[mask, col])  # keep old where new missing

df.to_csv(DST, index=False)
print(f"wrote {DST}")

log_compute("crop_audio_feats.py", T0, used_gpu=False)   # librosa, CPU only
