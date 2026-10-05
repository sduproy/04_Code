import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from config import CFG, set_seeds
from benchmark import identity_split
BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"

df = pd.read_csv(BASE + r"\all_features.csv", low_memory=False)
meta = {"clip_id","label","identity","source_dataset","pair_id"}
vis = [c for c in df.columns if not c.startswith("a_") and c not in meta and pd.api.types.is_numeric_dtype(df[c])]

def vislabel(cid):                       # FAV video-truth from clip_id prefix
    if not cid.startswith("fav_"):
        return np.nan                    # non-FAV shouldn't be in train set
    tok = cid.split("_")[1]
    return 0 if tok in ("rvra", "rvfa") else 1   # rvfa = real video labelled 0

set_seeds()
tr, _ = identity_split(df[df.source_dataset == CFG.train_set])
y_tr = tr["clip_id"].map(vislabel)
keep = y_tr.notna()                      # drop any non-FAV row defensively
m = HistGradientBoostingClassifier(random_state=CFG.seed).fit(tr.loc[keep, vis], y_tr[keep].astype(int))

d = df[df.source_dataset == "DF40_subset"]
s = m.predict_proba(d[vis])[:, 1]
y_d = (d["label"] == "fake").astype(int)   # DF40 label = video-truth already
print("visual-only DF40: unique", pd.Series(s).nunique(),
      "AUC", round(roc_auc_score(y_d, s), 4))

# sanity: how the video-truth relabel changed the training mix
print("train rows kept:", int(keep.sum()),
      "| fake(video) frac:", round(float(y_tr[keep].mean()), 4),
      "| vs clip-label fake frac:", round(float((tr['label']=='fake').mean()), 4))