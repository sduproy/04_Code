import pathlib
import pandas as pd
from frames_faces_audio import visual_stats   # same import extract_df40.py used

BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
OUT  = BASE / "df40_reals_features.csv"
CDF  = BASE / "Celeb-DF-v2_real_data_for_DF40" / "Celeb-DF-v2" / "Celeb-real" / "frames"
FF   = BASE / "FaceForensics++_real_data_for_DF40" / "FaceForensics++" / "original_sequences" / "youtube" / "c23" / "frames"

def clips():
    for d in sorted(p for p in CDF.iterdir() if p.is_dir()):        # id0_0000/*.png
        yield f"df40_cdf_real_{d.name}", f"df40_cdf_{d.name.split('_')[0]}", d
    for d in sorted(p for p in FF.iterdir() if p.is_dir()):          # 000/*.png
        yield f"df40_ff_real_{d.name}", f"df40_ff_{d.name}", d

done = set(pd.read_csv(OUT, dtype=str, keep_default_na=False)["clip_id"]) if OUT.exists() else set()
n = 0
for clip_id, identity, folder in clips():
    if clip_id in done:
        continue
    frames = sorted(str(p) for p in folder.glob("*.png"))
    if not frames:
        print("no frames:", clip_id); continue
    row = {"clip_id": clip_id, "label": "real", "identity": identity,
           "source_dataset": "DF40_subset", "generator": "none", "a_ok": 0}
    try:
        row.update(visual_stats(frames))
    except Exception as e:
        print("fail", clip_id, type(e).__name__); continue
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    n += 1
    if n % 50 == 0: print(n, "extracted")
print("done", n)