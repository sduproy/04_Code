import pathlib
import pandas as pd
from frames_faces_audio import frames, visual_stats, audio_feats

BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
REG = BASE / "corpus_registry.csv"
OUT = BASE / "corpus_features.csv"
CLIPS = BASE / "Endorsement corpus"

reg = pd.read_csv(REG, dtype=str, keep_default_na=False)

done = set()
if OUT.exists():
    done = set(pd.read_csv(OUT, dtype=str, keep_default_na=False)["clip_id"])
    print(f"resuming, {len(done)} already extracted")

for i, r in enumerate(reg.itertuples(index=False)):
    if r.clip_id in done:
        continue

    hits = list(CLIPS.rglob(f"{r.clip_id}.*"))
    if not hits:
        print(f"  MISSING FILE: {r.clip_id}")
        continue
    path = hits[0]

    print(f"{i+1}/{len(reg)}  {r.label:5}  {r.clip_id}")
    row = {"clip_id": r.clip_id, "label": r.label,
           "identity": r.identity, "source_dataset": r.source_dataset,
           "pair_id": getattr(r, "pair_id", ""),
           "generator": r.generator}
    try:
        fps = frames(str(path), "frames_tmp")
        row.update(visual_stats(fps))
    except Exception as e:
        print(f"  VIDEO FAIL: {e}")
        continue
    try:
        row.update(audio_feats(str(path)))
    except Exception as e:
        print(f"  audio fail, a_ok=0: {type(e).__name__}")
        row.update({"a_ok": 0})

    df = pd.DataFrame([row])
    df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)

print("done")