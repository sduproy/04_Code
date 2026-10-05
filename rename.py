import pathlib, pandas as pd

P = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_registry.csv"
FOLDER = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\Endorsement corpus")

reg = pd.read_csv(P, dtype=str, keep_default_na=False)
from registry import sha256

# map current files by hash
by_hash = {}
for p in FOLDER.rglob("*.mp4"):
    by_hash[sha256(str(p))] = p

for r in reg.itertuples():
    src = by_hash.get(r.sha256)
    if src is None:
        print("NO FILE for", r.clip_id)
        continue
    dst = src.parent / f"{r.clip_id}.mp4"
    if src != dst:
        print(src.name, "->", dst.name)
        src.rename(dst)