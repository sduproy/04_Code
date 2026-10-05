# extract_df40.py — frames-only feature extractor for DF40
# Writes df40_features.csv with the SAME columns as benchmark_features.csv:
#   clip_id, the 4 v_ columns (computed), a_ok=0, every a_ column = NaN (DF40 has no audio).
# Skips-and-logs unreadable clips (won't crash), and is resumable.

import pathlib
import numpy as np
import pandas as pd
from frames_faces_audio import visual_stats      # Module 3 — reuse so columns match exactly

DF40     = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\DF40")
DATA     = DF40.parent                            # ...\03_Data
FEAT_REF = DATA / "benchmark_features.csv"        # existing features — column template (fix name if different)
OUT      = DATA / "df40_features.csv"
DROP_MIDJOURNEY = True     # match the registry (the 8,136 rows excluded MidJourney)
MAX_CLIPS = None             # SMOKE TEST first with 5; set to None for the full run
FLUSH_EVERY = 100         # write to disk every N clips so a crash loses almost nothing

canon = list(pd.read_csv(FEAT_REF, nrows=0).columns)          # take a_ names from the real file, no guessing
assert "clip_id" in canon and "a_ok" in canon, f"unexpected cols in {FEAT_REF.name}: {canon[:6]}"
_reg = pd.read_csv(DATA / "df40_registry.csv", dtype=str, keep_default_na=False)
REG  = _reg.set_index("clip_id")[["label","identity","source_dataset","generator"]].to_dict("index")


done = set(pd.read_csv(OUT, usecols=["clip_id"])["clip_id"]) if OUT.exists() else set()
if done: print(f"resuming — {len(done)} clips already done")

buffer, failed, n = [], [], 0
def flush():
    global buffer
    if not buffer: return
    pd.DataFrame(buffer).reindex(columns=canon).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    buffer = []

for md in sorted(p for p in DF40.iterdir() if p.is_dir()):
    method = md.name
    if method == "MidJourney" and DROP_MIDJOURNEY: continue
    for cd in sorted({p.parent for p in md.rglob("*.png")}):
        clip_id = "df40_" + "_".join(cd.relative_to(DF40).parts)   # identical scheme to the registry
        if clip_id in done: continue
        frames = [str(p) for p in sorted(cd.glob("*.png"))]
        try:
            v = visual_stats(frames)
        except Exception as e:                                     # bad/None frame -> log, don't die
            failed.append((clip_id, repr(e))); continue
        buffer.append({"clip_id": clip_id, **REG[clip_id], "a_ok": 0, **v})       # a_ cols absent -> NaN on reindex
        n += 1
        if n % FLUSH_EVERY == 0: flush(); print(f"{n} clips (last: {method})", flush=True)
        if MAX_CLIPS and n >= MAX_CLIPS: break
    if MAX_CLIPS and n >= MAX_CLIPS: break

flush()
print(f"done: {n} new clips, {len(failed)} failed")
for cid, err in failed[:20]: print("FAIL", cid, err)