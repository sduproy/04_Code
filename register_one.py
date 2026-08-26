import pathlib
import pandas as pd
from collect_wild import register_fake

OUT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_registry.csv")

row = register_fake(
    clip_id="endo_00070",
    path=r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\Endorsement corpus\joe_rogan_clip5.mp4",
    origin_url="https://www.reddit.com/r/youtube/comments/186b0lp/why_is_youtube_allowing_scam_ai_advertising_on/",
    evidence_url="https://www.reddit.com/r/youtube/comments/186b0lp/why_is_youtube_allowing_scam_ai_advertising_on/",
    identity="joe_rogan_5",
    source="EndorsementCorpus_v1",
    licence="mirror of ad — confirm handling with supervisor",
)

if OUT.exists():
    reg = pd.read_csv(OUT, dtype=str, keep_default_na=False)
    reg = pd.concat([reg, pd.DataFrame([row])], ignore_index=True)
else:
    reg = pd.DataFrame([row])

reg.to_csv(OUT, index=False)
print(len(reg), "rows")