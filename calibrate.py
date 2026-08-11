#calibrate.py Reliability, temperature, operating points.
import numpy as np
import pandas as pd

def reliability(y, p, bins=10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append({"bin_mid": (lo + hi)/ 2,
                         "confidence": float(p[m].mean()),
                         "accuracy": float(y[m].mean()),
                         "n": int(m.sum())})
    return pd.DataFrame(rows)

def ece(y, p, bins=10) -> float:
    r = reliability(y, p, bins)
    w = r["n"] / r["n"].sum()
    return float((w * (r["confidence"] - r["accuracy"]).abs()).sum())

def fit_temperature(y_val, logit_val, grid=None) -> float:
    """Fit on validation only; apply everywhere after."""
    grid = grid or np.linspace(0.5, 5.0, 46)
    def nll(t):
        p = 1 / (1 + np.exp(-logit_val / t))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y_val * np.log(p)
        + (1 - y_val) * np.log(1-p))
    return float(min(grid, key=nll))

def operating_points(y, p, targets) -> pd.DataFrame:
    """The table the experiment quotes"""
    rows = []
    for target in targets:
        best = None
        for t in np.unique(np.round(p, 3)):
            flag = p >= t
            if flag.sum() == 0:
                continue
            prec = float(y[flag].mean())
            if prec >= target:
                best = t
                break
        if best is None:
            rows.append({"precision_target": target,
                        "reachable": False})
            continue
        flag = p >= best
        rows.append({"precision_target": target, "reachable": True,
                     "threshold" : float(best),
                     "flag_rate": float(flag.mean()),
                     "miss_rate": float(1 - flag[y == 1].mean()),
                     "false_alarm_rate":
                        float(flag[y==0].mean())})
    return pd.DataFrame(rows)