# df40_per_family.py — DF40 per-generator-family AUC, reals matched by source
import re
import pandas as pd
from config import CFG
from detect import train, p_fake
from benchmark import identity_split, _scores

BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"

def source_of(clip_id):
    m = re.search(r"_(cdf|ff)_", clip_id)
    if m: return m.group(1)
    if "heygen" in clip_id: return "heygen"
    return "other"

df_all = pd.read_csv(BASE + r"\all_features_norm.csv", low_memory=False)

# train exactly as benchmark.run() does: on the inherited benchmark
tr, _ = identity_split(df_all[df_all.source_dataset == CFG.train_set])
model, cols = train(tr)

df40 = df_all[df_all.source_dataset == "DF40_subset"].copy()
df40["src"] = df40["clip_id"].map(source_of)
reals = df40[df40.label == "real"]
fakes = df40[df40.label == "fake"]

rows = []
for (fam, src), g in fakes.groupby(["generator", "src"]):
    pool = reals[reals.src == src]
    if len(g) < 30 or len(pool) < 10:
        print(f"skip {fam}:{src} (fake={len(g)}, real={len(pool)})")
        continue
    sub = pd.concat([g, pool])
    y = (sub["label"] == "fake").astype(int).to_numpy()
    rows.append({"family": fam, "source": src,
                 "n_fake": len(g), "n_real": len(pool),
                 **_scores(y, p_fake(model, cols, sub))})

res = pd.DataFrame(rows).sort_values("auc")
print(res.to_string(index=False))
res.to_csv(BASE + r"\..\05_Outputs\df40_per_family_anorm.csv", index=False)