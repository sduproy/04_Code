# import pandas as pd
# BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"

# f = pd.read_csv(BASE + r"\benchmark_features.csv")
# print(f.groupby(["source_dataset", "label"], dropna=False).size())
# print("a_ok:", f["a_ok"].value_counts().to_dict())

# reg = pd.read_csv(BASE + r"\registry.csv", dtype=str, keep_default_na=False)
# print(reg[reg.source_dataset == "Deepfake_Eval_2024"]["label"].value_counts())

# reg2 = reg.set_index("clip_id")
# missing = f["label"].isna() | (f["label"] == "")
# f.loc[missing, "label"] = f.loc[missing, "clip_id"].map(reg2["label"])
# f.to_csv(BASE + r"\benchmark_features.csv", index=False)
# print(f.groupby(["source_dataset", "label"]).size())


# import pathlib
# from collections import Counter
# R = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\DF40\inswap")
# f = [p for p in R.rglob("*") if p.is_file()]
# print(len(f), Counter(p.suffix for p in f))
# for p in f[:10]: print(p.relative_to(R))

# import pathlib, numpy as np
# DF40 = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\DF40")
# for m in sorted(p for p in DF40.iterdir() if p.is_dir()):
#     pngs = list(m.rglob("*.png"))
#     clip_dirs = {p.parent for p in pngs}
#     s = next(iter(clip_dirs), None)
#     n = len(list(s.glob("*.png"))) if s else 0
#     print(f"{m.name:12} {len(pngs):>6} png  {len(clip_dirs):>5} clips  eg {s.relative_to(m) if s else '-'} ({n} frames)")

# print(np.load(next(DF40.glob('inswap/*/frames/*/000.npy'))).shape)  # identify the .npy

# import pathlib
# DF40 = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\DF40")
# for m in sorted(p for p in DF40.iterdir() if p.is_dir()):
#     l1 = sorted({q.name for q in m.iterdir() if q.is_dir()})
#     l2 = sorted({r.name for q in m.iterdir() if q.is_dir()
#                  for r in q.iterdir() if r.is_dir()})
#     print(f"{m.name:12} L1={l1}  L2={l2[:10]}")
# print("npy:", next(DF40.rglob('*.npy'), None))

# import pathlib
# DF40 = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\DF40")
# ok = badread = 0; samp = []
# for p in DF40.rglob("*.png"):
#     try:
#         with open(p,"rb") as f: f.read()
#         ok += 1
#     except OSError as e:
#         badread += 1
#         if len(samp) < 8: samp.append((e.errno, str(p)))
# print(f"ok={ok} badread={badread}")
# for s in samp: print(s)

# import pathlib
# root = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\Celeb-DF-v2_real_data_for_DF40")
# for p in list(root.rglob("*"))[:20]:
#     tag = "DIR " if p.is_dir() else "file"
#     print(tag, p.relative_to(root))

# import pathlib
# r = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\Celeb-DF-v2_real_data_for_DF40\Celeb-DF-v2\YouTube-real\frames\00000")
# print([p.name for p in list(r.iterdir())[:5]])

# import pandas as pd, pathlib
# BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
# f = pd.read_csv(BASE/"df40_features.csv", dtype=str, keep_default_na=False)
# f = f[f.label=="fake"]
# print(f["clip_id"].str.extract(r"_(cdf|ff)_")[0].value_counts(dropna=False))
# print(f.groupby("generator").size())

# r = pathlib.Path(str(BASE/"Celeb-DF-v2_real_data_for_DF40"/"Celeb-DF-v2"/"Celeb-real"/"frames"))
# for p in list(r.iterdir())[:8]: print(p.name)


# f = pd.read_csv(BASE/"df40_features.csv", dtype=str, keep_default_na=False)
# print(f[f.clip_id.str.contains("_ff_")][["clip_id","identity","generator"]].head(3).to_string())


# import pandas as pd, pathlib
# BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
# d = pd.read_csv(BASE/"df40_reals_features.csv", dtype=str, keep_default_na=False)
# print("rows:", len(d))
# print(d["clip_id"].str.extract(r"df40_(cdf|ff)_")[0].value_counts())     # cdf vs ff split
# print("cols:", d.shape[1])                                                # want 90, same as fakes
# print("dupe ids:", d["clip_id"].duplicated().sum())                       # want 0
# print("visual NaN:", d["v_sharpness_mean"].eq("").sum())                  # want 0 — blank = failed read

# import pandas as pd
# BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
# df_all = pd.read_csv(BASE + r"\all_features.csv", low_memory=False)
# print(df_all.assign(h=df_all["a_m0_mean"].notna()).groupby("source_dataset")["h"].mean())

# import pandas as pd, pathlib
# BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
# COR  = BASE / "Endorsement corpus"

# # 1. what's actually on disk (names + any subfolders + extension)
# for lab in ["Fake","Real"]:
#     items = list((COR/lab).iterdir())
#     print(f"== {lab}: {len(items)} items")
#     for p in items[:5]:
#         print("   ", "DIR " if p.is_dir() else "file", p.name)

# # 2. how the CSV names those same clips
# d = pd.read_csv(BASE/"corpus_features.csv", dtype=str, keep_default_na=False)
# print("\n== corpus clip_ids:")
# print(d[["clip_id","label"]].head(6).to_string(index=False))

# import pandas as pd
# BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
# o = pd.read_csv(BASE + r"\all_features.csv", low_memory=False)
# n = pd.read_csv(BASE + r"\all_features_norm.csv", low_memory=False)

# meta = {"clip_id","label","identity","source_dataset","pair_id"}
# feat  = [c for c in o.columns if c not in meta and pd.api.types.is_numeric_dtype(o[c])]
# aud   = [c for c in feat if c.startswith("a_")]
# vis   = [c for c in feat if not c.startswith("a_")]

# od = o[o.source_dataset=="DF40_subset"].reset_index(drop=True)
# nd = n[n.source_dataset=="DF40_subset"].reset_index(drop=True)

# print("rows:", len(od), len(nd))
# print("VISUAL identical orig vs norm:", od[vis].equals(nd[vis]))   # should be True
# print("norm DF40 visual all-NaN cols:",   sum(nd[c].isna().all() for c in vis))
# print("norm DF40 visual constant cols:",  sum(nd[c].nunique(dropna=True)<=1 for c in vis))
# print("orig DF40 visual constant cols:",   sum(od[c].nunique(dropna=True)<=1 for c in vis))

# import pandas as pd
# BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
# o = pd.read_csv(BASE + r"\all_features.csv", low_memory=False)
# n = pd.read_csv(BASE + r"\all_features_norm.csv", low_memory=False)

# meta = {"clip_id","label","identity","source_dataset","pair_id"}
# vis = [c for c in o.columns if not c.startswith("a_") and c not in meta and pd.api.types.is_numeric_dtype(o[c])]
# aud = [c for c in o.columns if c.startswith("a_")]

# od = o[o.source_dataset=="DF40_subset"].sort_values("clip_id").reset_index(drop=True)
# nd = n[n.source_dataset=="DF40_subset"].sort_values("clip_id").reset_index(drop=True)

# print("same clip_ids:", set(od.clip_id)==set(nd.clip_id))
# print("VISUAL identical after sort-by-clip_id:", od[vis].equals(nd[vis]))   # True = was just reordered
# print("orig DF40 audio non-null cells:", int(od[aud].notna().sum().sum()))  # expect ~0
# print("norm DF40 audio non-null cells:", int(nd[aud].notna().sum().sum()))  # if large -> DF40 audio got filled
# print("norm DF40 audio constant cols:", sum(nd[c].nunique(dropna=True)<=1 for c in aud))

# import pandas as pd, numpy as np
# BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
# o = pd.read_csv(BASE + r"\all_features.csv", low_memory=False)
# n = pd.read_csv(BASE + r"\all_features_norm.csv", low_memory=False)
# meta = {"clip_id","label","identity","source_dataset","pair_id"}
# vis = [c for c in o.columns if not c.startswith("a_") and c not in meta and pd.api.types.is_numeric_dtype(o[c])]

# od = o[o.source_dataset=="DF40_subset"].sort_values("clip_id").reset_index(drop=True)
# nd = n[n.source_dataset=="DF40_subset"].sort_values("clip_id").reset_index(drop=True)
# A, B = od[vis].values, nd[vis].values
# print("dtype flips:", [c for c in vis if o[c].dtype != n[c].dtype])
# print("nan-position mismatches:", int((np.isnan(A) != np.isnan(B)).sum()))
# print("max abs value diff:", np.nanmax(np.abs(A - B)))

# import pandas as pd, numpy as np
# from config import CFG, set_seeds
# from detect import train, p_fake
# from benchmark import identity_split
# BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"

# def look(csv, tag):
#     df = pd.read_csv(csv, low_memory=False)
#     set_seeds()
#     tr, _ = identity_split(df[df.source_dataset == CFG.train_set])
#     model, cols = train(tr)
#     s = np.asarray(p_fake(model, cols, df[df.source_dataset=="DF40_subset"]))
#     print(f"\n=== {tag} ===")
#     print(f"DF40 scores: n={len(s)} unique={pd.Series(s).nunique()} std={s.std():.6f}")
#     try:
#         imp = pd.Series(model.feature_importances_, index=cols)
#         aud = imp[[c for c in cols if c.startswith('a_')]].sum()/imp.sum()
#         print(f"audio share of importance: {aud:.1%}")
#         print("top 8:", list(imp.sort_values(ascending=False).head(8).index))
#     except Exception as e:
#         print("importance n/a:", e)

# look(BASE + r"\all_features.csv", "ORIGINAL")
# look(BASE + r"\all_features_norm.csv", "NORM")


import pandas as pd, numpy as np
from config import CFG, set_seeds
from detect import train, p_fake
from benchmark import identity_split
BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"

df = pd.read_csv(BASE + r"\all_features.csv", low_memory=False)   # current ORIGINAL
meta = {"clip_id","label","identity","source_dataset","pair_id"}
vis = [c for c in df.columns if not c.startswith("a_") and c not in meta and pd.api.types.is_numeric_dtype(df[c])]
d = df[df.source_dataset=="DF40_subset"]

print("DF40 rows:", len(d))
print("visual all-NaN cols:",  sum(d[c].isna().all() for c in vis))
print("visual constant cols:", sum(d[c].nunique(dropna=True)<=1 for c in vis), "/", len(vis))
print("visual NaN fraction:",  round(float(d[vis].isna().mean().mean()), 4))

set_seeds()
tr,_ = identity_split(df[df.source_dataset==CFG.train_set])
model, cols = train(tr)
s = np.asarray(p_fake(model, cols, d))
print("DF40 unique scores:", pd.Series(s).nunique(), "std:", round(float(s.std()),6))