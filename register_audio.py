import pathlib
import pandas as pd
from collect_wild import register_fake, register_real

DATA = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
OUT  = DATA / "audio_corpus_registry.csv"     # audio corpus only -- NOT corpus_registry.csv

# ---- edit these per clip ----
LABEL      = "fake"                           # "fake" or "real"
CLIP_ID    = "aud_00030"                      # fakes aud_00001..., reals aud_r_00001...
FILENAME   = "aud_00030.mp4"                  # file already saved in Audio corpus\Fake or \Real
ORIGIN_URL = "https://www.youtube.com/watch?v=5S-wdnVtQnU"
EVIDENCE   = "https://www.youtube.com/watch?v=5S-wdnVtQnU"                               # fakes only: link showing it's a deepfake
IDENTITY   = "lee_ermey"                               # e.g. joe_biden
LICENCE    = "YT-STD: YouTube Standard Licence, all rights reserved; uploader-disclosed AI-generated content (Vocal Synthesis); no download licence granted; research use only, not redistributed; fetched 2026-10-05"
PAIR_ID    = ""                               # reals only: clip_id of the fake this matches
MATCH_RULE = ""                               # reals only: 1 = same persona, 2 = same content type
# -----------------------------

path = DATA / "Audio corpus" / ("Fake" if LABEL == "fake" else "Real") / FILENAME
assert path.exists(), f"file not found: {path}"

if LABEL == "fake":
    row = register_fake(clip_id=CLIP_ID, path=str(path), origin_url=ORIGIN_URL,
                        evidence_url=EVIDENCE, identity=IDENTITY,
                        source="AudioCorpus_v1", licence=LICENCE)
else:
    row = register_real(clip_id=CLIP_ID, path=str(path), origin_url=ORIGIN_URL,
                        identity=IDENTITY, source="AudioCorpus_v1", licence=LICENCE)
    row["pair_id"] = PAIR_ID
    row["match_rule"] = MATCH_RULE

reg = pd.read_csv(OUT, dtype=str, keep_default_na=False)
if row["clip_id"] in set(reg["clip_id"]):
    raise SystemExit(f"{row['clip_id']} already registered")
if row["sha256"] in set(reg["sha256"]):
    raise SystemExit(f"same file already registered as {reg.loc[reg['sha256'] == row['sha256'], 'clip_id'].iloc[0]}")
reg = pd.concat([reg, pd.DataFrame([row])], ignore_index=True)
reg.to_csv(OUT, index=False)

print(len(reg), "rows |", (reg["label"] == "real").sum(), "real |", (reg["label"] == "fake").sum(), "fake")
