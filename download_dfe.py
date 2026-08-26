import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nuriachandra/Deepfake-Eval-2024",
    repo_type="dataset",
    local_dir=r"C:\data\Deepfake-Eval-2024",
    token=os.environ["HF_TOKEN"],
)
print("done")