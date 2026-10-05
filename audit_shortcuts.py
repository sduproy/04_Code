# audit_shortcuts.py — does the detector's score track recording artifacts?
import subprocess, json, re, time
T0 = time.perf_counter()
import numpy as np, pandas as pd, pathlib
from scipy.stats import spearmanr
from config import CFG, set_seeds, log_compute
from detect import train, p_fake
from benchmark import identity_split

BASE = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data")
OUT  = BASE.parent / "05_Outputs"

# --- clip_id -> media path ---
FAV = BASE / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
DFE = BASE / "Deepfake-Eval-2024" / "video-data"
COR = BASE / "Endorsement corpus"
SHORT = {"RealVideo-RealAudio":"rvra","FakeVideo-RealAudio":"fvra",
         "RealVideo-FakeAudio":"rvfa","FakeVideo-FakeAudio":"fvfa"}
index = {}
for p in FAV.rglob("*.mp4"):
    index[f"fav_{SHORT[p.relative_to(FAV).parts[0]]}_{p.parent.name}_{p.stem}"] = p
for p in DFE.glob("*.mp4"):
    index[f"dfe_{p.stem}"] = p
for lab in ["Fake", "Real"]:
    for p in (COR / lab).glob("*.mp4"):
        index[p.stem] = p

def probe(path):                                    # ffprobe: instant, no decode
    r = subprocess.run(["ffprobe","-v","quiet","-print_format","json",
                        "-show_format","-show_streams",str(path)],
                       capture_output=True, text=True)
    j = json.loads(r.stdout or "{}")
    fmt = j.get("format", {})
    a = next((s for s in j.get("streams",[]) if s.get("codec_type")=="audio"), {})
    return {"duration": float(fmt.get("duration", np.nan)),
            "bitrate":  float(fmt.get("bit_rate", np.nan)),
            "sample_rate": float(a.get("sample_rate", np.nan))}

def loudness(path):                                 # ffmpeg ebur128: needs a decode
    r = subprocess.run(["ffmpeg","-i",str(path),"-af","ebur128","-f","null","-"],
                       capture_output=True, text=True)
    m = re.findall(r"I:\s*(-?\d+\.?\d*)\s*LUFS", r.stderr)
    return float(m[-1]) if m else np.nan

import sys                                          # python audit_shortcuts.py all_features.csv
FEATURES = sys.argv[1] if len(sys.argv) > 1 else "all_features_normcrop.csv"   # default: norm + crop
TAG = FEATURES.removeprefix("all_features").removesuffix(".csv").strip("_") or "base"
df_all = pd.read_csv(BASE / FEATURES, low_memory=False)
print(f"auditing {FEATURES} -> *_{TAG}.csv")

# media measurements come from the raw mp4s, so they don't depend on FEATURES:
# measure once, cache by clip_id, reuse on every later run
CACHE = OUT / "shortcut_media_cache.csv"
cache = (pd.read_csv(CACHE).set_index("clip_id").to_dict("index")
         if CACHE.exists() else {})

def media(cid, path):
    if cid not in cache:
        cache[cid] = {"loudness": loudness(path), **probe(path)}
    return cache[cid]

# resolution check — every corpus row should find its file before we decode anything
cids = set(df_all[df_all.source_dataset == CFG.corpus_name]["clip_id"])
print("corpus resolved:", sum(c in index for c in cids), "of", len(cids))

set_seeds()
tr, _ = identity_split(df_all[df_all.source_dataset == CFG.train_set])
model, cols = train(tr)

N = 300                                             # clips/set for the loudness decode
rows = []
for name in ["FakeAVCeleb", "Deepfake_Eval_2024", CFG.corpus_name]:
    sub = df_all[df_all.source_dataset == name]
    sub = sub[~sub["identity"].isin(set(tr["identity"]))]     # no training clips
    if not len(sub): continue
    sub = sub.sample(min(N, len(sub)), random_state=CFG.seed)
    sub = sub.assign(score=p_fake(model, cols, sub))
    for r in sub.itertuples(index=False):
        path = index.get(r.clip_id)
        if path is None: continue
        rec = {"set": name, "clip_id": r.clip_id,
               "label": r.label, "score": float(r.score)}
        rec.update(media(r.clip_id, path))
        rows.append(rec)

pd.DataFrame.from_dict(cache, orient="index").rename_axis("clip_id").to_csv(CACHE)
d = pd.DataFrame(rows)
d.to_csv(OUT / f"shortcut_raw_{TAG}.csv", index=False)

out = []
for name, g in d.groupby("set"):
    for var in ["loudness","duration","bitrate","sample_rate"]:
        v = g[["score",var]].dropna()
        if len(v) >= 20:
            rho, pval = spearmanr(v["score"], v[var])
            out.append({"set":name,"var":var,"rho":round(float(rho),3),
                        "p":round(float(pval),4),"n":len(v)})
res = pd.DataFrame(out)
print(res.to_string(index=False))
res.to_csv(OUT / f"shortcut_correlations_{TAG}.csv", index=False)
print(f"\nsaved shortcut_raw_{TAG} + shortcut_correlations_{TAG} to 05_Outputs")

log_compute("audit_shortcuts.py", T0, used_gpu=False)   # GBM + ffmpeg, CPU only