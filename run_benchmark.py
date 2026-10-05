# import pandas as pd
# import benchmark

# BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026"

# df_all = pd.read_csv(BASE + r"\03_Data\all_features.csv")
# res = benchmark.run(df_all)
# print(res.to_string())
# res.to_csv(BASE + r"\05_Outputs\benchmark_results.csv", index=False)

# import pandas as pd
# from config import CFG

# BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
# df_all = pd.read_csv(BASE + r"\all_features.csv")

# print(df_all["source_dataset"].value_counts())
# print("train_set:", repr(CFG.train_set))
# print("new_benchmarks:", CFG.new_benchmarks)
# print("corpus_name:", repr(CFG.corpus_name))


import pandas as pd
from config import CFG
from benchmark import identity_split

BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
df_all = pd.read_csv(BASE + r"\all_features_normcrop.csv")

train_df = df_all[df_all.source_dataset == CFG.train_set]
tr, held = identity_split(train_df)

tests = {"in_domain_held_out": held}
for name in CFG.new_benchmarks + (CFG.corpus_name,):
    tests[name] = df_all[df_all.source_dataset == name]

for name, te in tests.items():
    print(f"{name}: {len(te)} rows | {te['label'].value_counts().to_dict()}")


import pandas as pd
import benchmark

BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026"

df_all = pd.read_csv(BASE + r"\03_Data\all_features_normcrop.csv", low_memory=False)
res = benchmark.run(df_all)
print(res.to_string())
res.to_csv(BASE + r"\05_Outputs\normcrop_benchmark_results.csv", index=False)

# import pandas as pd
# from config import CFG
# from benchmark import identity_split
# from detect import train, p_fake

# BASE = r"C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\03_Data"
# df = pd.read_csv(BASE + r"\all_features.csv", low_memory=False)

# tr, _ = identity_split(df[df.source_dataset == CFG.train_set])
# m, cols = train(tr)

# corpus = df[df.source_dataset == CFG.corpus_name].copy()
# corpus["p_fake"] = p_fake(m, cols, corpus)

# print(corpus.groupby("label")["p_fake"].describe()[["mean", "50%", "min", "max"]])
# print()
# print(corpus[["clip_id", "identity", "label", "p_fake"]]
#       .sort_values("p_fake", ascending=False).to_string())