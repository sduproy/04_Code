#registry.py One row per clip, or the clip does not exist.
import hashlib, time
import pandas as pd

REQUIRED = ["clip_id", "label", "source_dataset", "origin_url", 
            "license", "sha265", "identity", "generator", 
            "evidence_url", "date_documented"]

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
        return h.hexdigest()

def new_row(**kw) -> dict:
    row = {k: kw.get(k, "") for k in REQUIRED}
    row["added"] = time.strftime("%Y-%m-%d")
    return row

def validate(reg: pd.DataFrame) -> dict:
    """The provenance gate. Run before every analysis"""
    problems = []
    for k in REQUIRED:
        n = int((reg[k] == "").sum())
        if n:
            problems.append(f"{k}: {n} empty")
        fakes = reg[reg["label"] == "fake"]
        n_ev = int((fakes["evicence_url"] == "").sum())
        if n_ev:
            problems.append(f"fakes without evidence: {n_ev}")
        dup = int(reg["sha256"].duplicated().sum())
        if dup:
            problems.append(f"duplicate hashes: {dup}")
        return {"rows": len(reg), "complete": not problems,
                "problems": problems}

def recheck_sample(reg: pd.DataFrame, frac: float = 0.2,
                   seed: int = 2026) -> pd.DataFrame:
    """The supervisor's blind sample: clip, claim and evidence link,
    machine columns hidden"""
    cols = ["clip_id", "label", "origin_url", "evidence_url"]
    return reg.sample(frac=frac, random_state=seed) [cols]


#validate()
