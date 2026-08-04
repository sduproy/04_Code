#config.py Nothing else hard-codes a number, a path, a model name
#or a threshold. Change things here, nowhere else






from dataclasses import dataclass, asdict
from import json, hashlib, platform, random, subprocess, sys
import numpy as np

@dataclass(frozen=True)
class Config:
    seed: int = 2026
    data_root: str = "/kaggle/working"
    store_title: str = "Deepfake Detection S2 2026 Store"
    raw_video_root: str = "03_Data/raw"     #licensed videos live here
    #Frames, faces, audio                   #and never enter the Store
    frames_fps: float = 2.0
    face_size: int = 224
    audio_sr: int = 16000
    n_mfcc: int = 20
    #The datasets in play, each under its own licence 
    train_set: str = "FakeAVCeleb"          #the inherited domain
    new_benchmarks: tuple = ("DF40_subset", "Deepfake_Eval_2024")
    corpus_name: str = "EndorsementCorpus_v1"
    #Models
    detector_features: str = "gbm"          #the guaranteed path
    detector_pretrained: str = ""           #optional checkopint id; if
                                            #set, verify its card first
    #The bridge to the experiment
    op_precision_targets: tuple = (0.90, 0.95, 0.99)
    label_conditions: tuple = ("none", "generic_ai", "provenance",
                               "accuracy_disclosed")
    stimulus_secs: int = 20

CFG = Config()

def secret(name: str) -> str:
    """Kaggle Secrets first, local environment second. Never print."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        import os
        return os.environ[name]

def set_seeds(seed: int = CFG.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def run_manifest(stage: str, path: str) -> dict:
    pkgs = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                          capture_output=True, text=True).stdout
    m = {"stage": stage, "config": asdict(CFG),
         "python": platform.python_version(),
         "packages_sha256": hashlib.sha256(pkgs.encode()).hexdigest()}
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    m["git_commit"] = git.stdout.strip() or "not a git repository"
    json.dump(m, open(path, "w"), indent=2)
    return m