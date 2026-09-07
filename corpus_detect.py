# import pandas as pd
# from sklearn.metrics import roc_auc_score, average_precision_score
# from benchmark import identity_split
# from detect import train, p_fake

# df = pd.read_csv(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_features.csv")
# print(f"{len(df)} clips, {df['identity'].nunique()} identities")

# tr, te = identity_split(df, test_frac=0.3)
# print(f"train {len(tr)} / test {len(te)}, no shared identities")

# model, cols = train(tr)
# print(f"{len(cols)} feature columns")

# te = te.copy()
# te["p_fake"] = p_fake(model, cols, te)
# y = (te["label"] == "fake").astype(int)

# print(f"\nAUC: {roc_auc_score(y, te['p_fake']):.3f}")
# print(f"AP:  {average_precision_score(y, te['p_fake']):.3f}")
# print(te[["clip_id", "identity", "label", "p_fake"]].sort_values("p_fake").to_string())


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import roc_auc_score
# from detect import train, p_fake

# df = pd.read_csv(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_features.csv")
# print(f"{len(df)} clips, {df['identity'].nunique()} identities")

# gkf = GroupKFold(n_splits=5)
# aucs = []
# for k, (tr_i, te_i) in enumerate(gkf.split(df, groups=df["identity"]), 1):
#     tr, te = df.iloc[tr_i], df.iloc[te_i]
#     y = (te["label"] == "fake").astype(int)
#     if y.nunique() < 2:
#         print(f"fold {k}: skipped, single class")
#         continue
#     m, cols = train(tr)
#     auc = roc_auc_score(y, p_fake(m, cols, te))
#     aucs.append(auc)
#     print(f"fold {k}: n={len(te):3d}  AUC={auc:.3f}")

# print(f"\nmean {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")


# from sklearn.inspection import permutation_importance
# m, cols = train(df)
# r = permutation_importance(m, df[cols], (df["label"] == "fake").astype(int),
#                            n_repeats=20, random_state=2026)
# print(pd.Series(r.importances_mean, index=cols).sort_values(ascending=False).head(15))


# import pandas as pd
# from sklearn.inspection import permutation_importance
# from benchmark import identity_split
# from detect import train, p_fake

# df = pd.read_csv(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_features.csv")

# tr, te = identity_split(df, test_frac=0.3)
# m, cols = train(tr)

# y_te = (te["label"] == "fake").astype(int)
# r = permutation_importance(m, te[cols], y_te, n_repeats=30,
#                            random_state=2026, scoring="roc_auc")

# imp = pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)
# print(imp.head(15))
# print("\nvisual only:")
# print(imp[[c for c in cols if c.startswith("v_")]])

import subprocess, pathlib, pandas as pd

BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\Endorsement corpus")
REG = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_registry.csv"

reg = pd.read_csv(REG, dtype=str, keep_default_na=False)

rows = []
for r in reg.itertuples(index=False):
    hits = list(BASE.rglob(f"{r.clip_id}.*"))
    if not hits:
        continue
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=sample_rate,channels,codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1", str(hits[0])],
        capture_output=True, text=True).stdout.split()
    rows.append({"clip_id": r.clip_id, "label": r.label, "probe": " ".join(out)})

d = pd.DataFrame(rows)
print(d.groupby(["label", "probe"]).size())