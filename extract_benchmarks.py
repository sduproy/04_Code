import pathlib, random
import pandas as pd
from frames_faces_audio import frames, visual_stats, audio_feats

BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
REG  = BASE / "registry.csv"
OUT  = BASE / "benchmark_features.csv"

FAV  = BASE / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
DFE  = BASE / "Deepfake-Eval-2024" / "video-data"

N_FAV, N_DFE = 25000, 3000 
random.seed(2026)

print("indexing files...")
SHORT = {"RealVideo-RealAudio":"rvra","FakeVideo-RealAudio":"fvra",
         "RealVideo-FakeAudio":"rvfa","FakeVideo-FakeAudio":"fvfa"}
index = {}
for p in FAV.rglob("*.mp4"):
    cat = p.relative_to(FAV).parts[0]
    index[f"fav_{SHORT[cat]}_{p.parent.name}_{p.stem}"] = p
for p in DFE.glob("*.mp4"):
    index[f"dfe_{p.stem}"] = p
print(f"{len(index)} files indexed")

reg = pd.read_csv(REG, dtype=str, keep_default_na=False)

fav = reg[reg.source_dataset == "FakeAVCeleb"]
fav["cat"] = fav["clip_id"].str.split("_").str[1]
per = max(1, N_FAV // fav["cat"].nunique())
fav_s = fav.groupby("cat", group_keys=False).apply(
    lambda g: g.sample(min(len(g), per), random_state=2026))

dfe = reg[reg.source_dataset == "Deepfake_Eval_2024"]
dfe_s = dfe.groupby("label", group_keys=False).apply(
    lambda g: g.sample(min(len(g), N_DFE // 2), random_state=2026))

sample = pd.concat([fav_s, dfe_s], ignore_index=True)
print(sample.groupby(["source_dataset", "label"]).size())

done = set()
if OUT.exists():
    done = set(pd.read_csv(OUT, dtype=str, keep_default_na=False)["clip_id"])
    print(f"resuming, {len(done)} done")

for i, r in enumerate(sample.itertuples(index=False), 1):
    if r.clip_id in done:
        continue
    path = index.get(r.clip_id)
    if path is None:
        print(f"  no file: {r.clip_id}")
        continue
    if i % 25 == 0:
        print(f"{i}/{len(sample)}")
    row = {"clip_id": r.clip_id, "label": r.label, "identity": r.identity,
           "source_dataset": r.source_dataset, "generator": r.generator}
    try:
        row.update(visual_stats(frames(str(path), "frames_tmp")))
    except Exception as e:
        print(f"  video fail {r.clip_id}: {type(e).__name__}")
        continue
    try:
        row.update(audio_feats(str(path)))
    except Exception:
        row.update({"a_ok": 0})
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)

print("done")