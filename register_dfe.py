# import pathlib
# import pandas as pd
# from registry import new_row, sha256, validate

# ROOT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\Deepfake-Eval-2024")
# VID = ROOT / "video-data"
# OUT = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\registry.csv")

# meta = pd.read_csv(ROOT / "video-metadata-publish-with-links.csv")

# rows = []
# for i, r in enumerate(meta.itertuples(index=False)):
#     fn = r.Filename
#     path = VID / fn
#     if not path.exists():
#         continue
#     stem = pathlib.Path(fn).stem
#     rows.append(new_row(
#         clip_id=f"dfe_{stem}",
#         label="fake" if getattr(r, "_2") == "Fake" else "real",
#         source_dataset="Deepfake_Eval_2024",
#         origin_url=str(getattr(r, "Media", "")) or "not recorded",
#         licence="repository terms; evaluation only; takedowns respected; access filed 01_Admin",
#         sha256=sha256(str(path)),
#         identity=f"dfe_{stem}",
#         generator="unknown_wild",
#         evidence_url="dataset card: Deepfake-Eval-2024 (nuriachandra/Deepfake-Eval-2024)",
#         date_documented="2026-08-11",
#     ))
#     if i % 100 == 0:
#         print(i)

# reg = pd.DataFrame(rows)
# print(validate(reg))
# reg.to_csv(OUT, index=False)
# print(len(reg), "rows written")



# import pandas as pd

# reg = pd.read_csv(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\registry.csv", dtype=str, keep_default_na=False)
# dups = reg[reg["sha256"].duplicated(keep=False)].sort_values("sha256")
# print(dups[["clip_id", "label", "sha256"]].to_string())


import pandas as pd
from registry import validate

P = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\registry.csv"

reg = pd.read_csv(P, dtype=str, keep_default_na=False)
before = len(reg)
reg = reg.drop_duplicates(subset="sha256", keep="first")
reg.to_csv(P, index=False)

print(f"{before} -> {len(reg)}")
print(validate(reg))