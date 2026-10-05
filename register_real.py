import pathlib
import pandas as pd
from collect_wild import register_real

OUT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_registry.csv")

row = register_real(
    clip_id="endo_r_00070",
    path=r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\Endorsement corpus\Real\endo_r_00070.mp4",
    origin_url="https://www.youtube.com/watch?v=80YVvwaLO8Q",
    identity="joe_rogan",
    source="EndorsementCorpus_v1",
    licence="YT-STD: YouTube Standard Licence, all rights reserved; JRE Clips (official clips channel, same rights holder as PowerfulJRE); no download licence granted; research use only, not redistributed; fetched 2026-08-29",
)
row["pair_id"] = "endo_00070"     # clip_id of the fake this matches
row["match_rule"] = "1"           # 1 = same persona, 2 = same content type

reg = pd.read_csv(OUT, dtype=str, keep_default_na=False)
if row["clip_id"] in set(reg["clip_id"]):
    raise SystemExit(f"{row['clip_id']} already registered")
reg = pd.concat([reg, pd.DataFrame([row])], ignore_index=True)
reg.to_csv(OUT, index=False)

print(len(reg), "rows |", (reg["label"] == "real").sum(), "real |", (reg["label"] == "fake").sum(), "fake")