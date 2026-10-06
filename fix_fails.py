# fix_fails.py — redo FAV clips the original extraction couldn't read
#   audio: a_ok == 0                      -> audio_feats on the raw mp4
#   video: v_sharpness_mean NaN (0 frames) -> frames + visual_stats
# patches all_features.csv in place, and the v_ cols in all_features_norm.csv
# (its a_ cols came from reextract_audio.py)
import pathlib, shutil, time
T0 = time.perf_counter()
import pandas as pd
from frames_faces_audio import audio_feats, frames, visual_stats
from compute_log import log_compute

BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
SRC  = BASE / "all_features.csv"
DST  = SRC                                    # fixed in place
NORM = BASE / "all_features_norm.csv"         # v_ cols patched in place
TMP  = pathlib.Path("frames_tmp") / "fix_fails"

# --- same clip_id -> file map as the audit (FAV only) ---
FAV = BASE / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
SHORT = {"RealVideo-RealAudio":"rvra","FakeVideo-RealAudio":"fvra",
         "RealVideo-FakeAudio":"rvfa","FakeVideo-FakeAudio":"fvfa"}
index = {}
for p in FAV.rglob("*.mp4"):
    index[f"fav_{SHORT[p.relative_to(FAV).parts[0]]}_{p.parent.name}_{p.stem}"] = p

def redo(todo, fn, ok):
    new, fail = {}, []
    for cid in todo:
        path = index.get(cid)
        if path is None:
            fail.append((cid, "no file")); continue
        try:
            feats = fn(str(path))
        except Exception as e:
            fail.append((cid, type(e).__name__)); continue
        if not ok(feats):
            fail.append((cid, "still empty")); continue
        new[cid] = feats
    return new, fail

def video_feats(path):
    fps = frames(path, str(TMP))
    return visual_stats(fps) if fps else {}

def patch(df, mask, new, prefix):
    newdf = pd.DataFrame.from_dict(new, orient="index")
    for col in newdf.columns:
        if not col.startswith(prefix):
            continue
        mapped = df.loc[mask, "clip_id"].map(newdf[col])
        df.loc[mask, col] = mapped.where(mapped.notna(), df.loc[mask, col])  # keep old where new missing

df = pd.read_csv(SRC, low_memory=False)
fav = df["source_dataset"] == "FakeAVCeleb"

# --- audio ---
a_mask = fav & (df["a_ok"] == 0)
print(f"audio: {a_mask.sum()} FAV rows with a_ok == 0")
a_new, a_fail = redo(df.loc[a_mask, "clip_id"], audio_feats,
                     lambda f: f.get("a_ok") == 1)          # a_ok 0 = under 1s audio
print(f"audio: fixed {len(a_new)}, {len(a_fail)} still failing")
for cid, why in a_fail:
    print(f"  {cid}: {why}")

# --- video ---
v_mask = fav & df["v_sharpness_mean"].isna()
print(f"video: {v_mask.sum()} FAV rows with no frames read")
TMP.mkdir(parents=True, exist_ok=True)
v_new, v_fail = redo(df.loc[v_mask, "clip_id"], video_feats,
                     lambda f: bool(f))
shutil.rmtree(TMP, ignore_errors=True)
print(f"video: fixed {len(v_new)}, {len(v_fail)} still failing")
for cid, why in v_fail:
    print(f"  {cid}: {why}")
if v_new:
    fr = pd.Series({c: f["v_face_rate"] for c, f in v_new.items()})
    print(f"video: face_rate of fixed clips - median {fr.median():.2f}, {(fr == 0).sum()} with no face")

# --- write ---
patch(df, a_mask, a_new, "a_")
patch(df, v_mask, v_new, "v_")
df.to_csv(DST, index=False)
print(f"wrote {DST}")

norm = pd.read_csv(NORM, low_memory=False)
n_mask = norm["clip_id"].isin(v_new.keys())
patch(norm, n_mask, v_new, "v_")
norm.to_csv(NORM, index=False)
print(f"patched v_ cols for {n_mask.sum()} rows in {NORM.name}")

for name, d in [(DST.name, df), (NORM.name, norm)]:
    f = d[d["source_dataset"] == "FakeAVCeleb"]
    print(f"{name}: FAV a_ok == 0: {(f['a_ok'] == 0).sum()}, "
          f"no frames: {f['v_sharpness_mean'].isna().sum()}, face_rate 0: {(f['v_face_rate'] == 0).sum()}")

log_compute("fix_fails.py", T0, used_gpu=False)   # librosa + MTCNN on CPU
