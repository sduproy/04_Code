# reextract_audio.py — recompute a_ features from normalised wavs, merge by clip_id
import pathlib, time
T0 = time.perf_counter()
import pandas as pd
from frames_faces_audio import audio_feats
from config import log_compute

BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
NORM = BASE / "audio_norm"
SRC  = BASE / "all_features.csv"
DST  = BASE / "all_features_norm.csv"        # new file — baseline preserved

df = pd.read_csv(SRC, low_memory=False)
a_cols = [c for c in df.columns if c.startswith("a_")]
print(f"audio columns to refresh: {a_cols}")

AUDIO_SETS = {"FakeAVCeleb", "Deepfake_Eval_2024", "EndorsementCorpus_v1"}  # DF40 has no audio
todo = df[df["source_dataset"].isin(AUDIO_SETS)]
print(f"{len(todo)} rows to re-extract")

new = {}; fail = []
for i, cid in enumerate(todo["clip_id"], 1):
    wav = NORM / f"{cid}.wav"
    if not wav.exists():
        fail.append(cid); continue
    try:
        new[cid] = audio_feats(str(wav))
    except Exception:
        fail.append(cid)
    if i % 200 == 0:
        print(f"{i}/{len(todo)}  ({len(fail)} failed)")
print(f"re-extracted {len(new)}, {len(fail)} failed")

newdf = pd.DataFrame.from_dict(new, orient="index")
mask = df["source_dataset"].isin(AUDIO_SETS)
for col in newdf.columns:
    if not col.startswith("a_"):
        continue
    mapped = df.loc[mask, "clip_id"].map(newdf[col])
    df.loc[mask, col] = mapped.where(mapped.notna(), df.loc[mask, col])  # keep old where new missing

df.to_csv(DST, index=False)
print(f"wrote {DST}")

log_compute("reextract_audio.py", T0, used_gpu=False)   # librosa, CPU only