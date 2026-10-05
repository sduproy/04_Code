# audit_labels.py — is each recording artifact a shortcut, and is the model using it?
#   label_rho : artifact vs LABEL  -> is the shortcut available in this set?
#   score_rho : artifact vs SCORE  -> does the score track it? (as in audit_shortcuts.py)
#   fake_rho / real_rho : score vs artifact WITHIN one label -> is the model using it,
#               or just detecting fakes that happen to differ on it?
# python audit_labels.py base | norm | normcrop   (reads shortcut_raw_<tag>.csv)
import pathlib, sys
import pandas as pd
from scipy.stats import spearmanr

OUT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\05_Outputs")
TAG = sys.argv[1] if len(sys.argv) > 1 else "normcrop"      # match audit_shortcuts.py default
d = pd.read_csv(OUT / f"shortcut_raw_{TAG}.csv")
d["y"] = (d["label"] == "fake").astype(int)

def rho(a, b, min_n=20):
    s = pd.concat([a, b], axis=1).dropna()
    if len(s) < min_n or s.iloc[:, 0].nunique() < 2 or s.iloc[:, 1].nunique() < 2:
        return None, None, len(s)
    r, p = spearmanr(s.iloc[:, 0], s.iloc[:, 1])
    return round(float(r), 3), round(float(p), 4), len(s)

rows = []
for name, g in d.groupby("set"):
    for v in ["loudness", "duration", "bitrate", "sample_rate"]:
        lr, lp, n = rho(g["y"], g[v])
        sr, sp, _ = rho(g["score"], g[v])
        fr, fp, nf = rho(g.loc[g.y == 1, "score"], g.loc[g.y == 1, v])
        rr, rp, nr = rho(g.loc[g.y == 0, "score"], g.loc[g.y == 0, v])
        rows.append({"set": name, "var": v, "n": n,
                     "label_rho": lr, "label_p": lp,
                     "score_rho": sr, "score_p": sp,
                     "fake_rho": fr, "fake_p": fp, "n_fake": nf,
                     "real_rho": rr, "real_p": rp, "n_real": nr})

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(OUT / f"shortcut_labels_{TAG}.csv", index=False)
print(f"\nsaved shortcut_labels_{TAG}.csv")
