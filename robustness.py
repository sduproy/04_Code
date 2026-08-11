#robustness.py Social-media conditions 
import pathlib, subprocess 
import pandas as pd

CONDITIONS ={"crf23": ["-crf", "23"],
             "crf30": ["-crf", "30"],
             "crf37": ["-crf", "37"],
             "p480": ["-vf", "scale=-2:480", "-crf", "27"]}

def degrade(video_path: str, out_dir: str, cond:str) -> str:
    out = str(pathlib.Path(out_dir)
              / f"{pathlib.Path(video_path).stem}_{cond}.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", video_path,
                    "-c:v", "libx264", *CONDITIONS[cond],
                    "-c:a", "copy", out], check=True)
    return out 

def delta_table(base: pd.DataFrame, degraded: dict) -> pd.DataFrame:
    rows = []
    for cond, table in degraded.items():
        j = base.merge(table, on="test_set",
                       suffixes=("_base", f"_{cond}"))
        j[f"auc_delta_{cond}"] = (j[f"auc_{cond}"] - j["auc_base"])
        rows.append(j[["test_set", f"auc_delta_{cond}"]])
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on="test_set")
    return out