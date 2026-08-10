#handoff.py The pack that powers Stage D
import json
import pandas as pd

def stimulus_manifest(bank: pd.DataFrame, path: str) -> None:
    cols = ["clip_id", "pair_id", "label", "identity",
            "source_dataset", "evidence_url", "sha256",
            "detector_p_fake", "pilot_realism_mean",
            "pilot_difficulty"] + [f"file_c{c}" for c in 
            ("none", "generic_ai", "provenance",
             "accuracy_disclosed")]
    bank[cols].to_csv(path, index=False)

def prereg_skeleton(path: str, op_table: pd.DataFrame) -> None:
    skel = {
        "design": "label regine (4) x content (real, fake),"
            "between-subjects on regime, within on content",
        "condition": ["none", "generic_ai", "provenance",
                      "accuracy_disclosed"],
        "error_rates_from": op_table.to_dict("records"),
        "outcomes": ["detection d_prime", "criterion c",
                     "trust in ad", "purchase intention",
                     "report intention"],
        "hypotheses": [
            "H1: labels shift criterion, not only sensitivity",
            "H2: implied truth: unlabeled fakes judged more real"
            "under partial labeling than under no labeling",
            "H3: false alarms reduce trust in authentic ads"
            "(the spillover cost)"],
        "exclusions": "attention and playback checks, prereg'd",
    }
    json.dump(skel, open(path, "w"), indent=2)