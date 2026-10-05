# run_calibrate.py — reliability, ECE, temperature, operating points
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from config import CFG, set_seeds
from detect import train, p_fake
import calibrate

BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
OUT  = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\05_Outputs"

def to_logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def apply_T(p, T):
    return 1 / (1 + np.exp(-to_logit(p) / T))

def three_way(df):
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

def yp(model, cols, df):
    y = (df["label"] == "fake").astype(int).to_numpy()
    p = np.asarray(p_fake(model, cols, df), float)
    return y, p

import sys                                          # python run_calibrate.py all_features.csv
FEATURES = sys.argv[1] if len(sys.argv) > 1 else "all_features_normcrop.csv"   # default: norm + crop
TAG = FEATURES.removeprefix("all_features").removesuffix(".csv").strip("_") or "base"
print(f"calibrating on {FEATURES} -> *_{TAG}.csv")
df_all = pd.read_csv(BASE + "\\" + FEATURES, low_memory=False)
tr, val, held = three_way(df_all[df_all.source_dataset == CFG.train_set])
print(f"train {len(tr)} / val {len(val)} / held {len(held)} | "
      f"ids {tr.identity.nunique()}/{val.identity.nunique()}/{held.identity.nunique()}")

model, cols = train(tr)
y_val,  p_val  = yp(model, cols, val)
y_held, p_held = yp(model, cols, held)

T = calibrate.fit_temperature(y_val, to_logit(p_val))     # val only
p_held_cal = apply_T(p_held, T)
print(f"T = {T:.3f}")
print(f"held-out ECE  before {calibrate.ece(y_held, p_held):.4f}  "
      f"after {calibrate.ece(y_held, p_held_cal):.4f}")

calibrate.reliability(y_held, p_held).to_csv(OUT + rf"\reliability_before_{TAG}.csv", index=False)
calibrate.reliability(y_held, p_held_cal).to_csv(OUT + rf"\reliability_after_{TAG}.csv", index=False)

# operating points at CFG targets — held-out (signal, skewed base rate) AND corpus (balanced, chance)
targets = CFG.op_precision_targets
op_held = calibrate.operating_points(y_held, p_held_cal, targets); op_held.insert(0, "set", "in_domain_held_out")
y_c, p_c = yp(model, cols, df_all[df_all.source_dataset == CFG.corpus_name])
op_cor = calibrate.operating_points(y_c, apply_T(p_c, T), targets); op_cor.insert(0, "set", CFG.corpus_name)

print(f"\nbase rate  held-out fake {y_held.mean():.3f} | corpus fake {y_c.mean():.3f}")
op = pd.concat([op_held, op_cor], ignore_index=True)
print(op.to_string(index=False))
op.to_csv(OUT + rf"\operating_points_{TAG}.csv", index=False)
print(f"\nsaved reliability_before/after_{TAG}, operating_points_{TAG} to 05_Outputs")