# run_robustness.py — Module 7: the benchmark rerun on re-encoded clips (Week 6 robustness table)
# trains once on clean FAV (same identity split as benchmark.run), then scores each test set
# clean and under every robustness.CONDITIONS re-encode, on the same clips, so deltas are paired.
# degrade() copies the audio stream (-c:a copy), so only the 4 v_ features can change:
# a_ cols come from the feature file, v_ cols are re-extracted from the re-encoded video.
# DF40 is PNG frames, nothing to re-encode, so it isn't in the table.
# python run_robustness.py [all_features_normcrop.csv]
import pathlib, shutil, sys, time
T0 = time.perf_counter()
import numpy as np
import pandas as pd
from config import CFG
from compute_log import log_compute
from detect import train, p_fake
from benchmark import identity_split, _scores
from frames_faces_audio import frames, visual_stats
from robustness import CONDITIONS, degrade, delta_table

BASE  = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
OUT   = BASE.parent / "05_Outputs"
CACHE = BASE / "robustness_features.csv"      # one row per (clip, condition) — resumable
TMP   = pathlib.Path("frames_tmp") / "robustness"

MAX_PER_LABEL = {"in_domain_held_out": 500,      # ~100 reals in held-out, so all reals + 500 fakes
                 "Deepfake_Eval_2024": 250,      # ~40 s clips, the slow set
                 CFG.corpus_name: None}          # None = every clip
MAX_CLIPS = None      # SMOKE TEST first with 5: prints s/clip per set, size the caps from that

V_COLS = ["v_sharpness_mean", "v_sharpness_std", "v_face_rate", "v_facebox_jitter"]

FEATURES = sys.argv[1] if len(sys.argv) > 1 else "all_features_normcrop.csv"   # default: norm + crop
TAG = FEATURES.removeprefix("all_features").removesuffix(".csv").strip("_") or "base"
print(f"robustness on {FEATURES} -> robustness_{TAG}.csv")
df_all = pd.read_csv(BASE / FEATURES, low_memory=False)

# --- test sets: same split and sets as benchmark.run, capped per label ---
tr, held = identity_split(df_all[df_all.source_dataset == CFG.train_set])
tests = {"in_domain_held_out": held}
for name in CFG.new_benchmarks + (CFG.corpus_name,):
    if name != "DF40_subset":                    # frames only, see top
        tests[name] = df_all[df_all.source_dataset == name]

def sample(df, cap):
    if cap is None:
        return df
    return pd.concat([g.sample(min(len(g), cap), random_state=CFG.seed)
                      for _, g in df.groupby("label")])

tests = {name: sample(te, MAX_PER_LABEL[name]) for name, te in tests.items()}
for name, te in tests.items():
    assert not set(te["identity"]) & set(tr["identity"]), f"identity leakage into {name}"
    print(f"{name}: {len(te)} clips {te['label'].value_counts().to_dict()}")

# --- clip_id -> media path (same map as audit_shortcuts.py) ---
FAV = BASE / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
DFE = BASE / "Deepfake-Eval-2024" / "video-data"
COR = BASE / "Endorsement corpus"
SHORT = {"RealVideo-RealAudio":"rvra","FakeVideo-RealAudio":"fvra",
         "RealVideo-FakeAudio":"rvfa","FakeVideo-FakeAudio":"fvfa"}
index = {}
for p in FAV.rglob("*.mp4"):
    index[f"fav_{SHORT[p.relative_to(FAV).parts[0]]}_{p.parent.name}_{p.stem}"] = p
for p in DFE.glob("*.mp4"):
    index[f"dfe_{p.stem}"] = p
for lab in ["Fake", "Real"]:
    for p in (COR / lab).glob("*.mp4"):
        index[p.stem] = p

# --- re-encode + re-extract v_ features ---
def degraded_v(src, cond):
    d = TMP / cond
    d.mkdir(parents=True, exist_ok=True)
    try:
        fps = frames(degrade(str(src), str(d), cond), str(d))
        if not fps:
            raise ValueError("no frames read")
        return visual_stats(fps)
    finally:
        shutil.rmtree(d, ignore_errors=True)     # re-encoded mp4 + its frames, nothing kept

done = set()
if CACHE.exists():
    c = pd.read_csv(CACHE, usecols=["clip_id", "cond"])
    done = set(zip(c["clip_id"], c["cond"]))
    print(f"resuming, {len(done)} clip-conditions done")

n = 0
for name, te in tests.items():
    secs = []
    for cid in te["clip_id"]:
        if all((cid, cond) in done for cond in CONDITIONS):
            continue
        src = index.get(cid)
        if src is None:
            print(f"  no file: {cid}")           # not cached, so retried next run
            continue
        t = time.perf_counter()
        rows = []
        for cond in CONDITIONS:
            if (cid, cond) in done:
                continue
            row = {"clip_id": cid, "cond": cond, "ok": 1}
            try:
                row.update(degraded_v(src, cond))
            except Exception as e:               # logged as ok=0, not retried
                row["ok"] = 0
                print(f"  {cond} fail {cid}: {type(e).__name__}")
            rows.append(row)
        pd.DataFrame(rows).reindex(columns=["clip_id", "cond", "ok", *V_COLS]).to_csv(
            CACHE, mode="a", header=not CACHE.exists(), index=False)
        secs.append(time.perf_counter() - t)
        n += 1
        if len(secs) <= 5 or len(secs) % 25 == 0:
            print(f"{name}: {len(secs)} clips this run, {np.mean(secs):.1f} s/clip")
        if MAX_CLIPS and n >= MAX_CLIPS:
            break
    if MAX_CLIPS and n >= MAX_CLIPS:
        break
shutil.rmtree(TMP, ignore_errors=True)

if MAX_CLIPS:
    log_compute("run_robustness.py", T0, used_gpu=False)
    raise SystemExit(f"smoke run done ({n} clips) — set MAX_CLIPS = None for the full run")

# --- score: clean vs each condition, paired on clips that survived every condition ---
cache = pd.read_csv(CACHE)
cache = cache[cache["ok"] == 1].drop_duplicates(["clip_id", "cond"], keep="last")
model, cols = train(tr)

rows = []
for name, te in tests.items():
    have = cache[cache["clip_id"].isin(te["clip_id"])]
    full = have.groupby("clip_id")["cond"].nunique()
    te = te[te["clip_id"].isin(full[full == len(CONDITIONS)].index)]
    y = (te["label"] == "fake").astype(int).to_numpy()
    print(f"{name}: {len(te)} clips with every condition {te['label'].value_counts().to_dict()}")
    if len(set(y)) < 2:
        print(f"  skip {name}: needs both labels")
        continue
    rows.append({"test_set": name, "cond": "clean", **_scores(y, p_fake(model, cols, te))})
    for cond in CONDITIONS:
        v = have[have["cond"] == cond].set_index("clip_id")[V_COLS]
        deg = te.copy()
        deg[V_COLS] = v.loc[deg["clip_id"]].to_numpy()
        rows.append({"test_set": name, "cond": cond, **_scores(y, p_fake(model, cols, deg))})

res = pd.DataFrame(rows)
delta = delta_table(res[res["cond"] == "clean"].drop(columns="cond"),
                    {c: res[res["cond"] == c].drop(columns="cond") for c in CONDITIONS})
print("\n" + res.round(3).to_string(index=False))
print("\n" + delta.round(3).to_string(index=False))

# Companion check: AUC should degrade monotonically with compression
crf = [c for c in CONDITIONS if c.startswith("crf")]
for _, r in delta.iterrows():
    d = [r[f"auc_delta_{c}"] for c in crf]
    if any(a < b for a, b in zip(d, d[1:])):
        print(f"  {r['test_set']}: AUC not monotonic in CRF {np.round(d, 3).tolist()} — inspect before reporting")

res.to_csv(OUT / f"robustness_{TAG}.csv", index=False)
delta.to_csv(OUT / f"robustness_delta_{TAG}.csv", index=False)
print(f"\nsaved robustness_{TAG}, robustness_delta_{TAG} to 05_Outputs")

log_compute("run_robustness.py", T0, used_gpu=False)   # ffmpeg + MTCNN on CPU
