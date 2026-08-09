#benchmark.py Train once, face four kinds of reality
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from config import CFG, set_seeds
from detect import feature_cols, p_fake, train

def identity_split(df, test_frac=0.2):
    set_seeds()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac,
                            random_state=CFG.seed)
    tr, te = next(gss.split(df, groups=df["identity"]))
    a, b = df.iloc[tr], df.iloc[te]
    assert not set(a["identity"]) & set(b["identity"])
    return a, b

def _scores(y, p, n_boot=2000):
    set_seeds()
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    idx = np.arange(len(y))
    boots = []
    for _ in range(n_boot):
        s = np.random.choice(idx, len(idx))
        if len(set(y[s])) == 2:
            boots.append(roc_auc_score(y[s], p[s]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": float(auc), "ap": float(ap),
            "auc_lo": float(lo), "auc_hi": float(hi), "n": len(y)}

def run(df_all) -> pd.DataFrame:
    train_df = df_all[df_all.source_dataset == CFG.train_set]
    tr, held = identity_split(train_df)
    model, cols = train(tr)
    rows = []
    tests = {"in_domain_held_out": held}
    for name in CFG.new_benchmarks + (CFG.corpus_name,):
        tests[name] = df_all[df_all.source_dataset == name]
    for name, te in tests.items():
        wild = set(te["identity"]) & set(tr["identity"])
        assert not wild, f"identity leakage into {name}"
        y = (te["label"] == "fake").astype(int).to_numpy()
        rows.append({"test_set": name,
                     **_scores(y, p_fake(model, cols, te))})
        for fam, g in te.groupby("generator"):
            yg = (g["label"] == "fake").astype(int).to_numpy()
            if len(set(yg)) == 2 and len(g) >= 50:
                rows.append({"test_set": f"{name}:{fam}"},
                            **_scores(yg, p_fake(model, cols, g)))
    return pd.DataFrame(rows)