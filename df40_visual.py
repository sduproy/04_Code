# df40_visual.py — DF40 scored by a visual-only detector, overall + per generator family
# DF40 has no audio, so the audio-visual GBM can't score it (flat 0.5).
# score_df40() is the swap point: today a GBM on the 4 v_ features (baseline),
# later a pretrained face-forgery detector (FF++-trained, not DF40-trained).
import re, time
T0 = time.perf_counter()
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from config import CFG, set_seeds
from compute_log import log_compute
from benchmark import identity_split, _scores

BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
OUT  = BASE + r"\..\05_Outputs\df40_visual_per_family.csv"
DETECTOR = "gbm_v_features_videolabel"               # name recorded in the output

def source_of(clip_id):                              # same as df40_per_family.py
    m = re.search(r"_(cdf|ff)_", clip_id)
    if m: return m.group(1)
    if "heygen" in clip_id: return "heygen"
    return "other"

def video_fake(clip_id):
    # FAV video-truth from the clip_id prefix: rvfa is real video (fake audio only)
    return int(clip_id.split("_")[1] in ("fvra", "fvfa"))

def score_df40(df_all, df40):
    # baseline: GBM on the v_ features, trained on the FAV identity split
    # with video-truth labels, not the clip label (which calls rvfa fake)
    set_seeds()
    tr, _ = identity_split(df_all[df_all.source_dataset == CFG.train_set])
    cols = [c for c in df_all.columns if c.startswith("v_")]
    m = HistGradientBoostingClassifier(random_state=CFG.seed)
    m.fit(tr[cols], tr["clip_id"].map(video_fake))
    return m.predict_proba(df40[cols])[:, 1]

df_all = pd.read_csv(BASE + r"\all_features.csv", low_memory=False)
df40 = df_all[df_all.source_dataset == "DF40_subset"].copy()
df40["src"] = df40["clip_id"].map(source_of)
df40["score"] = score_df40(df_all, df40)
if df40["score"].nunique() == 1:
    raise SystemExit("every DF40 clip got the same score; detector sees nothing")

def row(family, source, sub):
    y = (sub["label"] == "fake").astype(int).to_numpy()
    return {"detector": DETECTOR, "family": family, "source": source,
            "n_fake": int(y.sum()), "n_real": int(len(y) - y.sum()),
            **_scores(y, sub["score"].to_numpy())}

rows = [row("ALL", "all", df40)]
reals = df40[df40.label == "real"]
fakes = df40[df40.label == "fake"]
for (fam, src), g in fakes.groupby(["generator", "src"]):
    pool = reals[reals.src == src]                   # reals matched by source
    if len(g) < 30 or len(pool) < 10:
        print(f"skip {fam}:{src} (fake={len(g)}, real={len(pool)})")
        continue
    rows.append(row(fam, src, pd.concat([g, pool])))

res = pd.DataFrame(rows)
res = pd.concat([res.iloc[:1], res.iloc[1:].sort_values("auc")])
print(res.drop(columns="detector").round(3).to_string(index=False))
res.to_csv(OUT, index=False)
print(f"\nsaved {OUT}")

log_compute("df40_visual.py", T0, used_gpu=False)
