# run_calibrate_5050.py — run_calibrate.py's operating points at the stimuli's prevalence (50/50),
# for the handoff pack. Same model, splits and temperature as run_calibrate.py.
# At the held-out's ~96% fake rate every precision target up to 0.95 is met by flagging
# everything, so those rows say nothing. Miss and false-alarm rates don't depend on
# prevalence, so precision and flag rate are recomputed at PI instead of subsampling:
#   precision = PI*TPR / (PI*TPR + (1-PI)*FPR),  flag_rate = PI*TPR + (1-PI)*FPR
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from config import CFG, set_seeds
from detect import train, p_fake
import calibrate

BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
OUT  = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\05_Outputs"
PI = CFG.stimulus_fake_rate
MIN_FLAGGED = 10                                    # a target met by a handful of clips isn't met

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

def yp(model, cols, df):
    y = (df["label"] == "fake").astype(int).to_numpy()
    p = np.asarray(p_fake(model, cols, df), float)
    return y, p

def operating_points_at(y, p, targets):
    """calibrate.operating_points, with precision and flag rate taken at PI"""
    rows = []
    for target in targets:
        best = None
        for t in np.unique(np.round(p, 3)):          # lowest threshold that meets the target
            flag = p >= t
            tpr, fpr = flag[y == 1].mean(), flag[y == 0].mean()
            den = PI * tpr + (1 - PI) * fpr
            if flag.sum() >= MIN_FLAGGED and den and PI * tpr / den >= target:
                best = t
                break
        if best is None:
            rows.append({"precision_target": target, "reachable": False})
            continue
        flag = p >= best
        tpr, fpr = flag[y == 1].mean(), flag[y == 0].mean()
        rows.append({"precision_target": target, "reachable": True,
                     "threshold": float(best),
                     "flag_rate": float(PI * tpr + (1 - PI) * fpr),
                     "miss_rate": float(1 - tpr),
                     "false_alarm_rate": float(fpr)})
    return pd.DataFrame(rows)

FEATURES = sys.argv[1] if len(sys.argv) > 1 else "all_features_normcrop.csv"   # default: norm + crop
TAG = FEATURES.removeprefix("all_features").removesuffix(".csv").strip("_") or "base"
print(f"calibrating on {FEATURES} -> operating_points_5050_{TAG}.csv")
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

targets = CFG.op_precision_targets
op_held = operating_points_at(y_held, p_held_cal, targets); op_held.insert(0, "set", "in_domain_held_out")
y_c, p_c = yp(model, cols, df_all[df_all.source_dataset == CFG.corpus_name])
op_cor = operating_points_at(y_c, apply_T(p_c, T), targets); op_cor.insert(0, "set", CFG.corpus_name)

print(f"\nbase rate  held-out fake {y_held.mean():.3f} | corpus fake {y_c.mean():.3f}"
      f"  -> precision and flag rate taken at {PI:.2f}")
op = pd.concat([op_held, op_cor], ignore_index=True)
print(op.to_string(index=False))
op.to_csv(OUT + rf"\operating_points_5050_{TAG}.csv", index=False)
print(f"\nsaved operating_points_5050_{TAG} to 05_Outputs")
