import pathlib, random
import pandas as pd
from frames_faces_audio import frames, visual_stats, audio_feats
from detect import train, p_fake

ROOT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")

def category(p):
    return p.relative_to(ROOT).parts[0]

random.seed(2026)
vids = list(ROOT.rglob("*.mp4"))
reals = [v for v in vids if category(v) == "RealVideo-RealAudio"]
fakes = [v for v in vids if category(v) != "RealVideo-RealAudio"]
print(f"{len(reals)} real, {len(fakes)} fake")

clips = random.sample(reals, 10) + random.sample(fakes, 10)

rows = []
for i, path in enumerate(clips):
    cat = category(path)
    label = "real" if cat == "RealVideo-RealAudio" else "fake"
    print(f"{i+1}/20  {label:5}  {cat:22}  {path.name}")
    fps = frames(str(path), "frames_tmp")
    row = {"clip_id": f"fav_{path.stem}", "label": label,
           "identity": f"fav_{path.parent.name}", "category": cat}
    row.update(visual_stats(fps))
    row.update(audio_feats(str(path)))
    rows.append(row)

df = pd.DataFrame(rows)
print("\nshape:", df.shape)
print("NaNs:", df.isna().sum().sum())
print(df[["clip_id", "label", "category", "a_ok", "v_face_rate"]].to_string())

model, cols = train(df)
df["p_fake"] = p_fake(model, cols, df)
print("\n", df[["clip_id", "label", "p_fake"]].to_string())

df.to_csv("smoke_features_fav.csv", index=False)
print("\nwrote smoke_features_fav.csv")