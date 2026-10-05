import pandas as pd

P = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_registry.csv"
reg = pd.read_csv(P, dtype=str, keep_default_na=False)

def lic(row):
    u, d = row["origin_url"], row["date_documented"]
    if row["label"] == "real":
        return row["licence"]                      # leave reals alone
    if "reddit.com" in u:
        p = ("RD-STD: Reddit User Agreement — uploader retains copyright, no "
             "third-party licence granted")
    elif "x.com" in u or "twitter.com" in u:
        p = ("X-STD: X ToS — uploader retains copyright, no third-party licence; "
             "scraping prohibited without written consent")
    elif "youtube.com" in u or "youtu.be" in u:
        p = ("YT-STD: YouTube Standard Licence, all rights reserved; no download "
             "licence granted")
    elif "facebook.com/ads/library" in u:
        p = ("META-ADLIB: retained in Meta Ad Library; Meta ToS, no third-party "
             "licence granted")
    elif "facebook.com" in u:
        p = "META-STD: Meta ToS — no third-party licence granted"
    elif "threads." in u:
        p = "META-STD: Threads/Meta ToS — no third-party licence granted"
    else:
        p = "OTHER: no third-party licence granted"
    return (f"{p}; fraudulent impersonation content, no legitimate rightsholder "
            f"identified; underlying footage may be third-party copyright; "
            f"research use only, not redistributed; fetched {d}")

reg["licence"] = reg.apply(lic, axis=1)
reg.to_csv(P, index=False)
print(reg[reg.label == "fake"]["licence"].str.split(":").str[0].value_counts())