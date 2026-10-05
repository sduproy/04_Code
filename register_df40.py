import pathlib, re, hashlib
import pandas as pd
from registry import new_row, validate  # Module 1

DF40 = pathlib.Path(r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\DF40")
ID_RE = re.compile(r"(id\d+)", re.IGNORECASE)

def clip_digest(clip_dir):
    h = hashlib.sha256()
    for p in sorted(clip_dir.glob("*.png")):
        st = p.stat()
        h.update(p.name.encode()); h.update(str(st.st_size).encode())
    return h.hexdigest()

def label_of(parts):
    return "real" if any(p.lower() == "real" for p in parts) else "fake"

def identity_of(parts, clip_name, method):
    # namespaced so DF40's bare idNN can't collide with FakeAVCeleb's idNN.
    # groups a Celeb-DF/FF person across methods; unique fallback for synthetic faces.
    src = "cdf" if "cdf" in parts else ("ff" if "ff" in parts else "na")
    m = ID_RE.search(clip_name)
    return f"df40_{src}_{m.group(1).lower()}" if m else f"df40_{method}_{clip_name}"

def register_df40(root=DF40, drop_midjourney=True):
    rows = []
    for md in sorted(p for p in root.iterdir() if p.is_dir()):
        method = md.name

        if method == "MidJourney":                      # shape E
            if drop_midjourney:
                continue
            for png in sorted(md.rglob("*.png")):
                parts = png.relative_to(root).parts
                rows.append(new_row(
                    clip_id=f"df40_MidJourney_{png.stem}", label=label_of(parts),
                    source_dataset="DF40_subset",
                    origin_url="https://github.com/YZY-stack/DF40",
                    licence="research-only, DF40 agreement filed 01_Admin",
                    sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                    identity=f"df40_MidJourney_{png.stem}", generator="MidJourney",
                    evidence_url="dataset card: DF40", date_documented="2026-09-08"))
            continue

        clip_dirs = {p.parent for p in md.rglob("*.png")}   # shapes A–D
        for cd in sorted(clip_dirs):
            parts = cd.relative_to(root).parts
            rows.append(new_row(
                clip_id="df40_" + "_".join(parts),          # full path -> collision-proof
                label=label_of(parts), source_dataset="DF40_subset",
                origin_url="https://github.com/YZY-stack/DF40",
                licence="research-only, DF40 agreement filed 01_Admin",
                sha256=clip_digest(cd),
                identity=identity_of(parts, cd.name, method), generator=method,
                evidence_url="dataset card: DF40", date_documented="2026-09-08"))
    return pd.DataFrame(rows)

reg_df40 = register_df40()
print(reg_df40.groupby(["generator", "label"]).size())
print(validate(reg_df40))

reg_df40.to_csv(DF40.parent / "df40_registry.csv", index=False)
print("saved", len(reg_df40), "rows")