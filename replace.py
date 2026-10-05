import pandas as pd
from registry import sha256

P = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_registry.csv"
CLIP_ID = "endo_r_00015"
NEW_PATH = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\Endorsement corpus\Real\endo_r_00015.mp4"

reg = pd.read_csv(P, dtype=str, keep_default_na=False)
mask = reg["clip_id"] == CLIP_ID
if not mask.any():
    raise SystemExit(f"{CLIP_ID} not found")

old = reg.loc[mask, "sha256"].iloc[0]
new = sha256(NEW_PATH)
reg.loc[mask, "sha256"] = new
reg.to_csv(P, index=False)
print(f"{CLIP_ID}: {old[:16]} -> {new[:16]}")