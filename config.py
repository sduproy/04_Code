#config.py Nothing else hard-codes a number, a path, a model name
#or a threshold. Change things here, nowhere else

#Run this once 
#pip install pandas pyarrow opencv-python facenet-pytorch librosa \
#    torch scikit-learn scipy matplotlib kaggle yt-dlp


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#I personally ran 
#python -m venv C:\dfd-env
#C:\dfd-env\Scripts\Activate.ps1
#cd "C:\Users\seand\OneDrive - UWA\Machine_Learning_Models_for_Deepfake_Detection_S2_2026\04_Code"
#pip install pandas pyarrow opencv-python facenet-pytorch librosa torch scikit-learn scipy matplotlib kaggle yt-dlp
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

from dataclasses import dataclass, asdict
import json, hashlib, platform, random, subprocess, sys
import numpy as np

@dataclass(frozen=True)
class Config:
    seed: int = 2026
    data_root: str = "/kaggle/working"
    store_title: str = "Deepfake Detection S2 2026 Store"
    raw_video_root: str = "03_Data/raw"     #licensed videos live here
    #Frames, faces, audio                   #and never enter the Store
    frame_fps: float = 2.0
    face_size: int = 224
    audio_sr: int = 16000
    n_mfcc: int = 20
    audio_crop_secs: float = 3.0            #centred window for crop_audio_feats.py
    stimulus_fake_rate: float = 0.5         #prevalence of the Stage D stimuli; run_calibrate_5050.py
    #The datasets in play, each under its own licence 
    train_set: str = "FakeAVCeleb"          #the inherited domain
    new_benchmarks: tuple = ("Deepfake_Eval_2024", "DF40_subset")
    corpus_name: str = "EndorsementCorpus_v1"
    #Models
    detector_features: str = "gbm"          #the guaranteed path
    #detector_pretrained: str = ""           #optional checkopint id; if
                                            #set, verify its card first
    detector_pretrained: str = "XLSR+SLS_MM2024"   #primary audio detector
    detector_repo: str = "https://github.com/QiShanZhang/SLSforASVspoof-2021-DF"
    detector_commit: str = "89a09ac4404c5687d96d6c123fcf6db20e4e4b38"
    detector_sls_sha256: str = "0d315184aa8e6f017ea72c4d2458c11bae8f07fd743fe860a3aa932e36135fa6"
    detector_xlsr_sha256: str = "b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9"
    detector_licence: str = "SLS repo: no LICENSE/model card; XLS-R front-end: Meta fairseq (terms unverified)"

    #The bridge to the experiment
    #More targets added for completeness
    op_precision_targets: tuple = (0.70, 0.80, 0.90, 0.95, 0.99)
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



# set_seeds()
# m = run_manifest("smoke", "manifest_smoke.json")