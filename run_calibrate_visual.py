# run_calibrate_visual.py — calibration for the visual-only detector, applied to DF40
# same recipe as run_calibrate.py: T fitted on a FAV val split only, never on DF40
# model = df40_visual.py baseline (GBM on v_ features, FAV video-truth labels)
import time
T0 = time.perf_counter()
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from config import CFG, set_seeds, log_compute
import calibrate

BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
OUT  = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\05_Outputs"

def to_logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def apply_T(p, T):
    return 1 / (1 + np.exp(-to_logit(p) / T))

def three_way(df):                                   # same splits as run_calibrate.py
    set_seeds()
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG.seed)
    rest_i, held_i = next(g1.split(df, groups=df["identity"]))
    rest, held = df.iloc[rest_i], df.iloc[held_i]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=CFG.seed)
    tr_i, val_i = next(g2.split(rest, groups=rest["identity"]))
    tr, val = rest.iloc[tr_i], rest.iloc[val_i]
    a, b, c = set(tr.identity), set(val.identity), set(held.identity)
    assert not (a & b) and not (a & c) and not (b & c), "identity leak across split"
    return tr, val, held

def video_fake(clip_id):                             # same as df40_visual.py
    return int(clip_id.split("_")[1] in ("fvra", "fvfa"))

df_all = pd.read_csv(BASE + r"\all_features.csv", low_memory=False)
cols = [c for c in df_all.columns if c.startswith("v_")]
tr, val, held = three_way(df_all[df_all.source_dataset == CFG.train_set])
df40 = df_all[df_all.source_dataset == "DF40_subset"]
print(f"train {len(tr)} / val {len(val)} / held {len(held)} | DF40 {len(df40)}")

set_seeds()
model = HistGradientBoostingClassifier(random_state=CFG.seed)
model.fit(tr[cols], tr["clip_id"].map(video_fake))
p = lambda d: model.predict_proba(d[cols])[:, 1]

y_val,  p_val  = val["clip_id"].map(video_fake).to_numpy(),  p(val)
y_held, p_held = held["clip_id"].map(video_fake).to_numpy(), p(held)
y_d,    p_d    = (df40["label"] == "fake").astype(int).to_numpy(), p(df40)   # DF40 label = video-truth

T = calibrate.fit_temperature(y_val, to_logit(p_val))     # val only
p_held_cal, p_d_cal = apply_T(p_held, T), apply_T(p_d, T)
print(f"T = {T:.3f}")
print(f"held-out ECE  before {calibrate.ece(y_held, p_held):.4f}  after {calibrate.ece(y_held, p_held_cal):.4f}")
print(f"DF40     ECE  before {calibrate.ece(y_d, p_d):.4f}  after {calibrate.ece(y_d, p_d_cal):.4f}")

calibrate.reliability(y_d, p_d).to_csv(OUT + r"\reliability_before_df40_visual.csv", index=False)
calibrate.reliability(y_d, p_d_cal).to_csv(OUT + r"\reliability_after_df40_visual.csv", index=False)

targets = CFG.op_precision_targets
op_held = calibrate.operating_points(y_held, p_held_cal, targets); op_held.insert(0, "set", "in_domain_held_out")
op_d    = calibrate.operating_points(y_d, p_d_cal, targets);       op_d.insert(0, "set", "DF40_subset")
print(f"\nbase rate  held-out fake(video) {y_held.mean():.3f} | DF40 fake {y_d.mean():.3f}")
op = pd.concat([op_held, op_d], ignore_index=True)
print(op.to_string(index=False))
op.to_csv(OUT + r"\operating_points_df40_visual.csv", index=False)
print("\nsaved reliability_before/after_df40_visual, operating_points_df40_visual to 05_Outputs")

log_compute("run_calibrate_visual.py", T0, used_gpu=False)
