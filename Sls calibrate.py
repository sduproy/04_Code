# sls_calibrate.py - Step 6: per-set Module 6 calibration on XLSR+SLS scores.
# Separate from run_calibrate.py (GBM). Reads score CSVs only: no model, no SLS env.
# Save in 04_Code (next to calibrate.py). Run with your normal Python 3.11, NOT the SLS env.
import sys, pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import reliability, ece, fit_temperature, operating_points
import matplotlib
matplotlib.use("Agg")             # write PNGs, no window
import matplotlib.pyplot as plt

OUTD = HERE.parent / "05_Outputs"
SAVE = OUTD / "sls_calibration"   # own folder, so it never overwrites the GBM's calibration files
SEED = 2026
VAL_FRAC = 0.5                    # half to fit (temperature + thresholds), half to report on
EPS = 1e-6                        # clip p so logit(1.0) isn't infinite

try:
    from config import CFG
    TARGETS = tuple(CFG.op_precision_targets)
except Exception as e:
    TARGETS = (0.90, 0.95)
    print("CFG precision targets not found, using", TARGETS, "-", e)

SETS = {
    "FakeAVCeleb":              ("scores_fakeavceleb.csv", "identity"),
    "Deepfake_Eval_2024_audio": ("scores_dfe_audio.csv",   "stratified"),
    "Corpus":                   ("scores_corpus.csv",      "stratified"),  # CHECK filename
    "Audio_corpus":             ("scores_audio_corpus.csv", "stratified"),
}
RUN = list(SETS)        # all sets; missing / one-class sets are skipped in main()


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def split(df, how, rng):
    """Val / held split. FAV: by identity (no person in both). Others: stratified by label."""
    if how == "identity":
        ident = df["clip_id"].str.split("_").str[2]          # fav_<cat>_<idXXXXX>_<stem>
        ids = ident.unique()
        rng.shuffle(ids)
        val_ids = set(ids[: int(len(ids) * VAL_FRAC)])
        is_val = ident.isin(val_ids).to_numpy()
        print(f"  identities: {len(ids)} total, {len(val_ids)} in val")
    else:
        is_val = np.zeros(len(df), bool)
        y = df["label"].to_numpy()
        for lab in (0, 1):
            idx = np.flatnonzero(y == lab)
            rng.shuffle(idx)
            is_val[idx[: int(len(idx) * VAL_FRAC)]] = True
    return df[is_val], df[~is_val]


def op_points(y_val, p_val, y_held, p_held, targets):
    """Thresholds picked on VAL by the Companion's operating_points(), then applied to HELD.
    Columns match the Companion; precision_achieved is the one extra (held precision)."""
    picked = operating_points(y_val, p_val, targets)
    rows = []
    for _, r in picked.iterrows():
        if not r["reachable"]:
            rows.append({"precision_target": r["precision_target"], "reachable": False})
            continue
        flag = p_held >= r["threshold"]
        rows.append({"precision_target": r["precision_target"], "reachable": True,
                     "threshold": r["threshold"],
                     "flag_rate": float(flag.mean()),
                     "miss_rate": float(1 - flag[y_held == 1].mean()),
                     "false_alarm_rate": float(flag[y_held == 0].mean()),
                     "precision_achieved": float(y_held[flag].mean()) if flag.sum() else np.nan})
    return pd.DataFrame(rows)


def plot_reliability(rb, ra, name, T, e0, e1, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(rb["confidence"], rb["accuracy"], "o-", label=f"before (ECE {e0:.3f})")
    ax.plot(ra["confidence"], ra["accuracy"], "s-", label=f"after, T = {T:.1f} (ECE {e1:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted p_fake",
           ylabel="fraction actually fake", title=f"XLSR+SLS reliability: {name} (held-out)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    for name in RUN:
        fname, how = SETS[name]
        if not (OUTD / fname).exists():
            print(f"\n== {name}: {fname} missing, skipped"); continue
        df = pd.read_csv(OUTD / fname)
        if df["label"].nunique() < 2:
            print(f"\n== {name}: one class only ({len(df)} clips), skipped -- needs reals and fakes"); continue
        print(f"\n== {name}: {len(df)} clips, "
              f"{int((df.label == 1).sum())} fake / {int((df.label == 0).sum())} real")

        val, held = split(df, how, np.random.default_rng(SEED))
        print(f"  val {len(val)} / held {len(held)}")
        yv, yh = val.label.to_numpy(int), held.label.to_numpy(int)
        zv, zh = logit(val.p_fake.to_numpy(float)), logit(held.p_fake.to_numpy(float))

        T = fit_temperature(yv, zv, grid=list(np.linspace(0.5, 50.0, 496)))   # 0.1 steps; default caps at 5
        pv_after = sigmoid(zv / T)
        p_before, p_after = sigmoid(zh), sigmoid(zh / T)
        e0, e1 = ece(yh, p_before), ece(yh, p_after)
        print(f"  T = {T:.3f} | held ECE {e0:.4f} -> {e1:.4f}")

        rb, ra = reliability(yh, p_before), reliability(yh, p_after)
        rb.to_csv(SAVE / f"reliability_before_{name}.csv", index=False)
        ra.to_csv(SAVE / f"reliability_after_{name}.csv", index=False)
        plot_reliability(rb, ra, name, T, e0, e1, SAVE / f"reliability_{name}.png")
        op = op_points(yv, pv_after, yh, p_after, TARGETS)
        op.to_csv(SAVE / f"operating_points_{name}.csv", index=False)
        print(op.to_string(index=False))

        pd.DataFrame([{"set": name, "n_val": len(val), "n_held": len(held), "T": T,
                       "ece_before": e0, "ece_after": e1, "seed": SEED}]
                     ).to_csv(SAVE / f"summary_{name}.csv", index=False)
    print("\nsaved to", SAVE)


if __name__ == "__main__":
    main()