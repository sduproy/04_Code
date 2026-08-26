import pandas as pd
from registry import validate

P = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data\corpus_registry.csv"

reg = pd.read_csv(P, dtype=str, keep_default_na=False)
print(len(reg), "rows")
print(validate(reg))
print(sorted(reg["identity"].unique()))