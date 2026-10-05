import pathlib
from frames_faces_audio import frames, visual_stats, audio_feats

ROOT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")

real = next((ROOT / "RealVideo-RealAudio").rglob("*.mp4"))
fake = next((ROOT / "FakeVideo-FakeAudio").rglob("*.mp4"))

rows = []
for path, label in ((real, "real"), (fake, "fake")):
    fps = frames(str(path), "frames_tmp")
    row = {"clip_id": f"fav_{path.stem}", "label": label}
    row.update(visual_stats(fps))
    row.update(audio_feats(str(path)))
    rows.append(row)

a, b = set(rows[0]), set(rows[1])
print("same columns:", a == b, "|", len(a), "cols")
print("diff:", (a - b) | (b - a))