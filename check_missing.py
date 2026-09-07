# import pathlib, pandas as pd

# ROOT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")
# P = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\registry.csv"
# SHORT = {"RealVideo-RealAudio": "rvra", "FakeVideo-RealAudio": "fvra",
#          "RealVideo-FakeAudio": "rvfa", "FakeVideo-FakeAudio": "fvfa"}

# ids = {f"fav_{SHORT[p.relative_to(ROOT).parts[0]]}_{p.parent.name}_{p.stem}"
#        for p in ROOT.rglob("*.mp4")}
# reg = pd.read_csv(P, dtype=str, keep_default_na=False)
# have = set(reg[reg.source_dataset == "FakeAVCeleb"]["clip_id"])

# print(len(ids), "on disk |", len(have), "registered")
# missing = sorted(ids - have)
# print(len(missing), "missing")
# for m in missing[:27]:
#     print(" ", m)


####


# import pathlib, pandas as pd
# from registry import sha256

# ROOT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")
# P = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\registry.csv"
# SHORT = {"RealVideo-RealAudio": "rvra", "FakeVideo-RealAudio": "fvra",
#          "RealVideo-FakeAudio": "rvfa", "FakeVideo-FakeAudio": "fvfa"}

# reg = pd.read_csv(P, dtype=str, keep_default_na=False)
# have = set(reg[reg.source_dataset == "FakeAVCeleb"]["clip_id"])
# by_hash = dict(zip(reg["sha256"], reg["clip_id"]))

# for p in ROOT.rglob("*.mp4"):
#     cid = f"fav_{SHORT[p.relative_to(ROOT).parts[0]]}_{p.parent.name}_{p.stem}"
#     if cid in have:
#         continue
#     h = sha256(str(p))
#     print(f"{cid}\n   -> {'DUPLICATE of ' + by_hash[h] if h in by_hash else 'NOT IN REGISTRY'}")


import pathlib
from collections import defaultdict
from registry import sha256

ROOT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")

groups = defaultdict(list)
for p in ROOT.rglob("*.mp4"):
    groups[sha256(str(p))].append(p)

n = 0
for h, paths in groups.items():
    if len(paths) > 1:
        n += 1
        print(f"\n[{n}] {h[:16]}")
        for p in paths:
            print(p)

print(f"\n{n} duplicate groups")