#stimuli.py Matched, standardized, audited, overlaid, piloted.
import pathlib, subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from sklearn.metrics import cohen_kappa_score
from config import CFG

def standardize(in_path: str, out_path: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-i", in_path,
                    "t", str(CFG.stimulus_secs),
                    "-vf", "scale=1280:720:force_original_aspect_ratio="
                    "decrease, pad=1280:720:(ow-iw)/2:(oh-ih)/2",
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-r", "30", "out_path"], check=True)

def cue_audit(bank: pd.DataFrame) -> pd.DataFrame:
    """Superficial cues must not be seperate real from fake."""
    rows = []
    for cue in ["duration_s", "brightness", "loudness"]:
        a = bank[bank.label == "real"][cue]
        b = bank[bank.label == "fake"][cue]
        t = ttest_ind(a, b, equal_var=False)
        rows.append({"cue": cue, "p_value": float(t.pvalue),
                     "passes": t.pvalue > 0.10})
    return pd.DataFrame(rows)

TEXT = {"generic_ai": "AI-generated content",
        "provenance": "No content credentials: origin unverified",
        "accuracy_disclosed":
            "Flagged by AI detector (about 9 in 10 flags correct)"}

def overlay(in_path: str, out_path: str, condition: str) -> None:
    if condition == "none":
        subprocess.run(["ffmpeg", "-y", "-i", in_path, "-c", "copy", 
                        out_path], check=True)
        return
    text = TEXT[condition]
    subprocess.run(["ffmpeg", "-y", "-i", in_path, "-vf",
                    "drawbox=y=0:h=54:c=black@0.65:t=fill,"
                    f"drawtext=text='{text}':x=16:y=14:fontsize=28:"
                    "fontcolor=white",
                    "-c:a", "copy", out_path], check=True)

def pilot_sheet(bank: pd.DataFrame, rater:str) -> pd.DataFrame:
    """Blind: no labels, no machine scores, shuffled order."""
    s = bank.sample(frac=1.0, random_state=CFG.seed)
    return pd.DataFrame({"clip_id": s["clip_ip"], "rater":rater,
                         "realism_0_10": "", "guess_real_fake": ""})

def pilot_agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    j = a.merge(b, on="clip_id", suffixes=("_a", "_b"))
    kappa = cohen_kappa_score(j["guess_real_fake_a",
                                j["guess_real_fake_b"]])
    rho = spearmanr(j["realism_0_10_a"].astype(float),
                    j["realism_0_10_b"].astype(float)).statistic
    return {"n": len(j), "kappa_guess": float(kappa),
            "rho_realism": float(rho)}