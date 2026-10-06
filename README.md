# Deepfake Detection S2 2026: code

This code trains a deepfake detector on FakeAVCeleb, then tests how well it holds up on real-world material: Deepfake-Eval-2024, a DF40 subset, and our own **Endorsement Corpus** of fake celebrity endorsement ads. It then calibrates the detector's scores into operating points (thresholds with known miss and false-alarm rates) for the Stage D human labelling experiment.

The project folders sit next to this one:

| Folder | What's in it |
|---|---|
| `03_Data/` | Raw datasets, registries (`*registry.csv`) and feature tables (`*features*.csv`) |
| `04_Code/` | This repo |
| `05_Outputs/` | Results: benchmark tables, calibration, operating points, audits, `compute_log.csv` |
| `06_Reports/` | Write-ups (Benchmark, Audio Benchmark, Central Measurement reports) |
| `01_Admin/` | Licences, dataset access proofs, collection log |

---

## Setup

```powershell
python -m venv C:\dfd-env
C:\dfd-env\Scripts\Activate.ps1
pip install pandas pyarrow opencv-python facenet-pytorch librosa torch scikit-learn scipy matplotlib kaggle yt-dlp av
```

You also need `ffmpeg` and `ffprobe` on PATH.

- `av` fixes ffmpeg read errors (see `edits.txt`).
- Paths are hard-coded to `C:\Users\seand\OneDrive - UWA\...` in most scripts. On another machine, find and replace that prefix.
- Run every script from inside `04_Code/`. Several of them write to the relative path `frames_tmp/`.

---

## Core modules

These are imported by the scripts; you don't run them directly.

| Module | Role |
|---|---|
| `config.py` | Every setting lives in `CFG`: seed, frame rate, MFCC count, dataset names, precision targets, SLS detector provenance. Also has `set_seeds()` and `run_manifest()`. |
| `compute_log.py` | `log_compute(script, T0, used_gpu)` appends one row per run (wall and GPU hours) to `05_Outputs/compute_log.csv`. Python 3.7-safe, so the XLSR+SLS scorers in the SLS env import it too. |
| `registry.py` | One row per clip, with provenance. `new_row()`, `sha256()`, and `validate()` (the provenance gate: no empty fields, every fake has evidence, no duplicate hashes). |
| `collect_wild.py` | `register_fake()` / `register_real()` for clips collected by hand, plus the rules for matching reals to fakes. |
| `frames_faces_audio.py` | Feature extraction, identical for every dataset. `frames()` samples at 2 fps, `visual_stats()` computes the 4 `v_` features (sharpness, face rate, face-box jitter) and `audio_feats()` / `mfcc_feats()` compute 80 `a_` MFCC features. |
| `detect.py` | The baseline detector: a gradient-boosted tree (GBM) on the `v_` + `a_` features. `train()` and `p_fake()`. |
| `benchmark.py` | `identity_split()` (no person in both train and test), `_scores()` (AUC, AP, bootstrap CI) and `run()`, which trains on FakeAVCeleb and scores every test set. |
| `calibrate.py` | `reliability()`, `ece()`, `fit_temperature()`, `operating_points()`. |

---

## Datasets

| `source_dataset` | Role | Notes |
|---|---|---|
| `FakeAVCeleb` | Training set + in-domain held-out | Clip label is `fake` if **either** audio or video is fake. The `clip_id` prefix tells you which (`rvra`, `rvfa`, `fvra`, `fvfa`). |
| `Deepfake_Eval_2024` | Real-world benchmark | Up to 1,500 real + 1,500 fake sampled |
| `DF40_subset` | Face-swap/reenactment benchmark | **Video only, no audio.** Reals come from Celeb-DF v2 and FF++. MidJourney is excluded. |
| `EndorsementCorpus_v1` | Our own corpus: fake celebrity ads, each paired with a matched real | `03_Data/Endorsement corpus/{Fake,Real}`, registry `corpus_registry.csv` |
| `AudioCorpus_v1` | Our own audio-deepfake corpus, **in progress** | `03_Data/Audio corpus/`, registry `audio_corpus_registry.csv` |

---

## Pipeline, in run order

### 1. Register clips → `03_Data/*registry.csv`

| Script | Does |
|---|---|
| `register_fakeav` *(no `.py`; run it with `python register_fakeav`)* | FakeAVCeleb → `registry.csv` |
| `register_dfe.py` | Deepfake-Eval-2024 → `registry.csv` (`download_dfe.py` fetches it from HuggingFace) |
| `register_df40.py` | DF40 → `df40_registry.csv` |
| `register_one.py`, `register_real.py` | Add **one** fake or real clip to `corpus_registry.csv`. Edit the values at the top, then run. |
| `register_audio.py` | Add one clip to `audio_corpus_registry.csv`. Edit the block at the top, then run. |
| `check_corpus.py` | Runs `validate()` on the corpus registry |

**Registry upkeep (one-off fixes):** `rename.py` renames corpus files to their `clip_id` (matched by hash). `replace.py` updates one clip's hash after its file is swapped. `edit_license.py` rewrites licence strings for corpus fakes based on the source platform. `check_missing.py` finds FakeAVCeleb files that are missing from the registry or duplicated.

### 2. Extract features → `03_Data/all_features.csv`

| Script | Output |
|---|---|
| `extract_benchmarks.py` | `benchmark_features.csv` (FakeAVCeleb + Deepfake-Eval-2024 sample; resumable) |
| `extract_corpus.py` | `corpus_features.csv` |
| `extract_df40.py` | `df40_features.csv` (frames only, `a_ok=0`) |
| `register_df40_reals.py` | `df40_reals_features.csv`. Despite the name, this **extracts features** for the DF40 reals. |
| `join_features.py` | Stacks the four tables above → `all_features.csv` |

### 3. Audio fixes → three versions of the feature table

The audits (step 6) showed the detector could tell fakes from reals by **loudness** and **clip length** instead of real deepfake artefacts. Each fix produces its own feature table, so the results can be compared:

| Run | Output | Tag |
|---|---|---|
| (after step 2) | `all_features.csv` | `base` |
| `normalise_audio.py` → `reextract_audio.py` | `audio_norm/*.wav` → `all_features_norm.csv` (audio loudness-normalised) | `norm` |
| `fix_fails.py` | Fixes FakeAVCeleb clips whose audio or video failed to extract, in **both** `all_features.csv` and `all_features_norm.csv`. Run it after `reextract_audio.py`. | - |
| `crop_audio_feats.py` | `all_features_normcrop.csv`: audio features taken from a centred 3 s window, so clip length can't leak | `normcrop` |

**`normcrop` is the current default.** Most scripts below take the feature file as an argument and name their outputs `*_<tag>.csv`.

### 4. Benchmark → `05_Outputs/*benchmark_results.csv`

- `run_benchmark.py`: trains on FakeAVCeleb and reports AUC, AP and 95% CI per test set and per generator. Currently reads `all_features_normcrop.csv` and writes `normcrop_benchmark_results.csv`. The commented-out blocks are earlier runs.

### 5. Calibration and operating points → `05_Outputs/`

| Script | Detector | Output |
|---|---|---|
| `run_calibrate.py [features.csv]` | GBM | `reliability_before/after_<tag>.csv`, `operating_points_<tag>.csv`. Temperature is fitted on a FakeAVCeleb validation split only. |
| `run_calibrate_5050.py [features.csv]` | GBM | `operating_points_5050_<tag>.csv`: the same model with precision recomputed at **50% fakes**, the prevalence of the Stage D stimuli. Use these for the handoff. |
| `Sls calibrate.py` *(the space is in the name; run `python "Sls calibrate.py"`)* | XLSR+SLS | `05_Outputs/sls_calibration/` (per-set reliability, plots, operating points) |

**About XLSR+SLS:** it's the primary audio detector, with the GBM as a second opinion. It's run **outside this repo** in its own environment. This repo only reads its scores from `05_Outputs/scores_*.csv`. Its provenance (repo, commit, checkpoint hashes, licence) is in `config.py` and `03_Data/run_manifest.md`.

### 6. Shortcut audits → `05_Outputs/shortcut_*`

| Script | Does |
|---|---|
| `audit_shortcuts.py [features.csv]` | Measures loudness, duration, bitrate and sample rate for every clip, then correlates each with the detector score → `shortcut_raw_<tag>.csv`, `shortcut_correlations_<tag>.csv` |
| `audit_labels.py <tag>` | Reads `shortcut_raw_<tag>.csv`. For each artefact: does it predict the **label** (is the shortcut available)? Does the score track it **within** a label (is the model using it)? → `shortcut_labels_<tag>.csv` |

### 7. DF40 visual-only detector

DF40 has no audio, so the audio-visual GBM can't score it. These scripts use a GBM trained on the `v_` features only, with FakeAVCeleb **video-truth** labels: `rvfa` (fake audio, real video) counts as real.

| Script | Output |
|---|---|
| `df40_visual.py` | `df40_visual_per_family.csv`: AUC overall and per generator, reals matched by source. `score_df40()` is where a pretrained face-forgery model would plug in. |
| `run_calibrate_visual.py` | `reliability_*_df40_visual.csv`, `operating_points_df40_visual.csv` |
| `df40_per_family.py` | `df40_per_family_anorm.csv`: the **audio-visual** GBM per DF40 family, kept for comparison |
| `df_visual_only.py` | Prints one visual-only DF40 AUC (quick check) |

### 8. Not run yet (Stage D)

- `stimuli.py`: standardises stimulus clips, audits them for surface cues, overlays the 4 label conditions, and produces pilot rating sheets.
- `handoff.py`: stimulus manifest + preregistration skeleton.
- `robustness.py`: re-encodes clips at social-media quality levels and compares AUC.

---

## Checks and scratch

| File | What it is |
|---|---|
| `smoketest.py` | 20 FakeAVCeleb clips end to end (frames → features → train) → `smoke_features_fav.csv` |
| `module_3_check.py` | Checks that a real clip and a fake clip produce the same feature columns |
| `corpus_detect.py` | Probes audio codecs and sample rates in the corpus; earlier corpus experiments are commented out |
| `misc.py` | Commented-out one-off queries |
| `smoke_features*.csv`, `manifest_smoke.json` | Smoke-test outputs |
| `edits.txt` | Environment note |

---

## Things to know

- **Splits are always by identity.** No person appears in both train and test, and `benchmark.run()` asserts it.
- **Temperature is fitted on FakeAVCeleb validation data only**, never on a test set.
- **The FakeAVCeleb held-out set is about 96% fake**, so its raw operating points look better than they are. Quote the `5050` table for anything with balanced prevalence.
- **`frames_tmp/` is safe to delete** once extraction is done; nothing reads it afterwards. It's in `.gitignore` because it holds ~480k JPEGs.
- **Scripts that rewrite files in place:** `fix_fails.py`, `edit_license.py`, `rename.py` (renames video files) and `replace.py`. Back up the target first.
