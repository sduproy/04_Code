import pandas as pd

BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"

a = pd.read_csv(BASE + r"\benchmark_features.csv")
b = pd.read_csv(BASE + r"\corpus_features.csv")
c = pd.read_csv(BASE + r"\df40_features.csv")
d = pd.read_csv(BASE + r"\df40_reals_features.csv")     # add this
df_all = pd.concat([a, b, c, d], ignore_index=True)     # add d here

print(df_all.groupby(["source_dataset", "label"]).size())
print(f"{df_all['clip_id'].nunique()} unique ids of {len(df_all)}")
print(f"identities: {df_all['identity'].nunique()}")
print(f"columns: {df_all.shape[1]}")

df_all.to_csv(BASE + r"\all_features.csv", index=False)