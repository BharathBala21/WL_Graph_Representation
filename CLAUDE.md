# CLAUDE.md: project handoff (WL-LeafKernel, final-year project, Team 49)

Research question: how graph construction influences Weisfeiler–Lehman (WL) graph kernels for
plant-stress classification from OLID-I leaf photos (38 handcrafted measurements → feature graph → WL
kernel → SVM). Main study = **Levels 1–3 (healthy vs stressed)**. The 57-class task is set aside (future work).

## Layout

| Path | What |
|---|---|
| `full_pipeline/` | research notebooks NB08–NB18 (features, graphs, WL engine, 57-class), `olid_full_features.csv`, `cv_fold_assignments.csv` (fixed folds; never regenerate) |
| `full_pipeline/decomposition/` | NB21 Level 1, NB22 Level 2 (per crop), NB23 Level 3 (crop routing → crop model) + results |
| `Application/` | React/Vite web app (PlantCare AI) + `backend/` FastAPI (diagnosis, accounts, SQLite) |
| `Application/backend/app/` | `wl.py` WL engine (verified equal to research), `features.py` (verbatim NB08; change only together with NB08), `ml.py` predictor + per-step `trace`, `lesions.py` spot check, `main.py` API |
| `New_Models_Baselines/03_Benchmark_Suite/` | **final multi-dimensional benchmark**: start with its `README.md` |
| `New_Models_Baselines/04_CNN_Frozen_Features/` | frozen MobileNetV3 / EfficientNet-B0 + logistic regression baseline (`cache/`, `embeddings/` git-ignored) |
| `Application/IMPROVEMENTS.txt` | app performance / feature suggestions (plain text) |
| `Research_Paper/`, `ppt/` | LaTeX paper, slides |

## Current state (Nov 2026)

* **App:** step-by-step pipeline walkthrough (7 steps incl. spot check and reliability check),
  "Unsure" verdict when a photo is unlike the training photos (`metadata.json` → `familiarity`, from
  `train.py --calibrate-only`), spot detector warning (tuned: 1/320 healthy false alarms).
* **Benchmark (done):** 13 feature models + 5 nested-CV tuned + WL (fixed and tuned) + 2 frozen CNNs on
  Levels 1–3; robustness (240 photos × 7 changes), learning curves, speed, size, memory, statistics.
* **Headline results (macro-F1):** Level 3 full pipeline: CNN EfficientNet-B0 0.739, tuned Extra Trees
  0.729, WL (best research setting) 0.706, WL (nested-tuned, fair) 0.683, always-"stressed" 0.456.
  WL = the most robust *feature-based* model (mean F1 change −0.053; frozen CNNs ≈ 0). Frozen CNN is ~8×
  faster end to end (feature pipeline segments the full 3,024 px photo, ~1.06 s).
* **Agreed framing:** WL is competitive and the most robust/interpretable feature-based model; it is NOT
  the most accurate, fastest, or most data-efficient; do not claim it beats CNNs.

## Rules for this project

* **Accuracy is misleading:** 84% of leaves are stressed; always report macro-F1, balanced accuracy,
  specificity, and the "always stressed" reference (accuracy 83.8%).
* OLID-I has **no severity labels** (no healthy/moderate/severe); do not claim severity grading.
* Fair comparisons: same folds for all models; tune baselines and WL the same way (nested CV) when claiming a ranking.
* NB14's 57-class WL number (0.391) used old labels (no feature identity); corrected = 0.423.
* **Windows gotcha:** import LightGBM before scikit-learn (`bench.py` does this), or it crashes with an
  access violation. Scripts must `import bench` first.
* Memory is tight (16 GB, other processes); full-size photos (3,024 px) can hit MemoryError, so free memory before image-heavy runs.
* Token cost: for long jobs use ONE background command that notifies at the end; no per-step monitors
  (the user can watch logs with `Get-Content <log> -Wait`).

## Run

```bash
# backend (from Application/backend):  uvicorn app.main:app --port 8000
# frontend (from Application):         npm run dev
# backend tests (repo root):           python -m pytest Application/backend/tests -q
# benchmark (from New_Models_Baselines/03_Benchmark_Suite): see its README "How to reproduce"
```

## Open items

* CNN fine-tuning (option B, ~6–8 h overnight) if the guide wants the strongest CNN.
* Update the research paper/README claims (e.g. root README says deep learning "requires extensive labelled
  data"; the frozen CNN was the most data-efficient model here).
* **Done in the app (57-class decision):** 57-class shown only as a stated limitation (ML Pipeline card,
  portal tab "57-class (limitation)", Diagnostics/Stress Library labels); condition shown as a top-3 shortlist
  with tested hit rate (83–98% per crop); reliability = balanced accuracy + healthy found (`metadata.json`
  `stress_balanced_accuracy`, `stress_healthy_found`, `always_stressed_accuracy`); "weak for <crop>" chip when
  balanced accuracy < 0.6 (ridge gourd 0.53).
* Stress type (healthy / nutrient / pest / disease) with WL was tested: mean macro-F1 0.66, below the
  majority guess for ash gourd. Not added to the app; mention as a tested option.
* **Done in the PPT** (`ppt/new_presentation/presentation.tex`, compile on Overleaf; no LaTeX locally):
  results slide = Level 1 only + "always stressed" row, honest key box; new slides "Fair Benchmark: Where
  WL Stands" and "Limitation: The Exact Condition (57 Classes)" (top-3 shortlist table); updated next
  steps and conclusion. `backup_application_template_deck` not changed.
* **Next:** the same framing for the research paper (`Research_Paper/main.tex`) and the root README.
* Segmentation bug B7b (black-background JPGs) in `Application/IMPROVEMENTS.txt`: not fixed (would change research features).
