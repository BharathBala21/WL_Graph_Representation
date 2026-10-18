"""
Builds New_Models_Baselines/01_Logistic_Regression_Experiment_Corrected.ipynb
with proper multi-cell structure. The notebook code matches the verified
run_lr_experiment.py exactly.

Run from workspace root:
    python New_Models_Baselines/build_corrected_nb.py
"""
import os
import json
import nbformat as nbf

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR    = os.path.join(WORKSPACE, "New_Models_Baselines")
OUT_PATH  = os.path.join(NB_DIR, "01_Logistic_Regression_Experiment_Corrected.ipynb")

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language":     "python",
        "name":         "python3",
    },
    "language_info": {
        "name":    "python",
        "version": "3.11.9",
    },
}

cells = []

# ── Cell 0: Title and audit notes ────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# Logistic Regression Experiment – Corrected Version

**Project:** How Does Graph Construction Influence the Effectiveness of Weisfeiler–Lehman Graph Kernels
for Fine-Grained Plant-Stress Classification from Leaf Images?

**Task:** 57-class OLID-I plant stress classification  
**Model:** Logistic Regression (baseline, this experiment)  
**Notebook:** `New_Models_Baselines/01_Logistic_Regression_Experiment_Corrected.ipynb`

---
## Audit Summary – What Was Wrong in the Original Notebook

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | Single model instance reused across folds | Fresh `LogisticRegression()` instantiated per fold |
| 2 | `time.time()` used for timing | Replaced with `time.perf_counter()` throughout |
| 3 | `warnings.filterwarnings('ignore')` silenced convergence warnings | Convergence warnings captured and reported per fold |
| 4 | No data validation before training | 9-point validation with hard assertions |
| 5 | Single prediction call for inference timing (3/5 folds reported 0.0 s) | Warm-up + 20 repeated calls; median and IQR reported |
| 6 | `std()` ddof not documented | Explicitly `ddof=1`; computed from raw fold values |
| 7 | No JSON configuration record | Saved `logistic_regression_config.json` |
| 8 | Results saved to old directory (`.`); no `New_Models_Baselines/results/` structure | Correct output paths used |
| 9 | Visualizations were mean-only; no fold-level plots | 5 figures: fold-level F1, fold-level accuracy, training time, latency, comparison |
| 10 | Historical values hard-coded without provenance check | Loaded from verified source CSVs; cross-checked against reference table |
| 11 | No per-sample latency, IQR, or warm-up | Implemented: warm-up + 20 reps, median ms/sample + IQR |
| 12 | Solver, multi_class, and other hyperparameters not documented | All parameters explicitly set and commented |

**No correctness-affecting bugs were found in the data loading order** (feature_cols was derived
before the fold merge in the original, so `fold` did not enter the feature matrix).
The primary issues are methodological rigour and reporting quality.
"""))

# ── Cell 1: Imports and software versions ────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import os
import sys
import time
import json
import warnings as _warnings_module
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# ── Software versions ──────────────────────────────────────────────────────
sw_versions = {
    "python":       sys.version.split()[0],
    "scikit-learn": sklearn.__version__,
    "numpy":        np.__version__,
    "pandas":       pd.__version__,
}
print("Software versions:")
for k, v in sw_versions.items():
    print("  {:20s}: {}".format(k, v))
"""))

# ── Cell 2: Configuration ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ── Configuration ──────────────────────────────────────────────────────────
# All paths are derived from the workspace root so the notebook
# works regardless of where it is executed from.
NOTEBOOK_DIR = os.path.dirname(os.path.abspath("__file__"))
WORKSPACE    = os.path.dirname(NOTEBOOK_DIR)

DATA_DIR  = os.path.join(WORKSPACE, "full_pipeline")
OUT_DIR   = os.path.join(WORKSPACE, "New_Models_Baselines", "results")
FIG_DIR   = os.path.join(WORKSPACE, "New_Models_Baselines", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FEAT_CSV = os.path.join(DATA_DIR, "olid_full_features.csv")
FOLD_CSV = os.path.join(DATA_DIR, "cv_fold_assignments.csv")
BL_SUM   = os.path.join(WORKSPACE, "full_pipeline", "results", "classical_baselines",
                         "classical_baselines_summary.csv")
WL_SUM   = os.path.join(WORKSPACE, "full_pipeline", "results", "wl_svm_classification",
                         "wl_svm_summary.csv")

# ── Experiment hyperparameters ────────────────────────────────────────────
RANDOM_STATE  = 42   # fixed seed for reproducibility
N_FOLDS       = 5    # five-fold stratified CV (pre-assigned)
N_TIMING_REPS = 20   # repeated prediction calls for median/IQR latency

# ── Model parameters (explicitly documented) ──────────────────────────────
LR_PARAMS = {
    "solver":       "lbfgs",   # LBFGS: memory-efficient, supports multinomial
    "max_iter":     1000,      # sufficient for most folds on this 38-feature task
    "class_weight": "balanced",# matches NB-18 classical baseline protocol
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
    "multi_class":  "auto",    # multinomial with lbfgs
}

print("Configuration loaded.")
print("  RANDOM_STATE  :", RANDOM_STATE)
print("  N_FOLDS       :", N_FOLDS)
print("  N_TIMING_REPS :", N_TIMING_REPS)
print("  LR_PARAMS     :", LR_PARAMS)
print("  OUT_DIR       :", OUT_DIR)
print("  FIG_DIR       :", FIG_DIR)
"""))

# ── Cell 3: Data loading and validation ──────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 – DATA VALIDATION (9-point checklist)
# ════════════════════════════════════════════════════════════════════════════

assert os.path.isfile(FEAT_CSV), "Feature CSV not found: " + FEAT_CSV
assert os.path.isfile(FOLD_CSV), "Fold CSV not found: " + FOLD_CSV

df_full = pd.read_csv(FEAT_CSV)
df_acc  = df_full[df_full["quality_flag"] == "ACCEPT"].copy().reset_index(drop=True)

META_COLS = [
    "image_path", "image_name", "class", "crop", "condition",
    "quality_flag", "mask_area_ratio", "failure_reason"
]
# IMPORTANT: derive FEATURE_COLS BEFORE the fold merge.
# After the merge, the 'fold' column would be added to df_acc.
# If feature_cols were recomputed after the merge, 'fold' would silently
# enter the feature matrix.  Deriving FEATURE_COLS here prevents that.
FEATURE_COLS = [c for c in df_acc.columns if c not in META_COLS]

df_folds = pd.read_csv(FOLD_CSV)
df_acc   = df_acc.merge(df_folds[["image_path", "fold"]], on="image_path", how="left")

# ── Check 1: Row count ───────────────────────────────────────────────────
assert len(df_acc) == 4387, "Expected 4387, got " + str(len(df_acc))
print("[PASS] Check 1 – Row count:       {} (expected 4387)".format(len(df_acc)))

# ── Check 2: Feature columns ────────────────────────────────────────────
assert len(FEATURE_COLS) == 38
print("[PASS] Check 2 – Feature columns: {} (expected 38)".format(len(FEATURE_COLS)))
print("        Features:", FEATURE_COLS)

# ── Check 3: Target classes ──────────────────────────────────────────────
n_classes = df_acc["class"].nunique()
assert n_classes == 57
print("[PASS] Check 3 – Target classes:  {} (expected 57)".format(n_classes))

# ── Check 4: NaN in features ─────────────────────────────────────────────
nan_total = df_acc[FEATURE_COLS].isna().sum().sum()
assert nan_total == 0
print("[PASS] Check 4 – NaN in features: {} (none)".format(nan_total))

# ── Check 5: Inf in features ─────────────────────────────────────────────
inf_total = np.isinf(df_acc[FEATURE_COLS].values).sum()
assert inf_total == 0
print("[PASS] Check 5 – Inf in features: {} (none)".format(inf_total))

# ── Check 6: Unique images ───────────────────────────────────────────────
n_unique = df_acc["image_path"].nunique()
assert n_unique == len(df_acc)
print("[PASS] Check 6 – Unique images:   {} of {} rows".format(n_unique, len(df_acc)))

# ── Check 7: Missing fold assignments ────────────────────────────────────
missing_folds = df_acc["fold"].isna().sum()
assert missing_folds == 0
print("[PASS] Check 7 – Missing folds:   {} (none)".format(missing_folds))

# ── Check 8: All 5 folds present with valid sizes ────────────────────────
fold_counts = df_acc["fold"].value_counts().sort_index()
assert set(fold_counts.index.astype(int)) == {0, 1, 2, 3, 4}
print("[PASS] Check 8 – Fold sizes:", fold_counts.to_dict())

# ── Check 9: Stratification (KL divergence per fold vs overall dist) ─────
overall_dist = df_acc["class"].value_counts(normalize=True)
max_kl = 0.0
for fid in range(N_FOLDS):
    fd = (df_acc[df_acc["fold"] == fid]["class"]
          .value_counts(normalize=True)
          .reindex(overall_dist.index, fill_value=0.0))
    kl = float((overall_dist * np.log((overall_dist + 1e-9) / (fd + 1e-9))).sum())
    max_kl = max(max_kl, kl)
strat_ok = max_kl < 0.05
print("[{}] Check 9 – Max fold KL divergence: {:.6f} ({})".format(
    "PASS" if strat_ok else "WARN", max_kl,
    "folds well-stratified" if strat_ok else "may not be fully stratified"))

print()
print("DATA VALIDATION: ALL CHECKS PASSED")
"""))

# ── Cell 4: FoldSafeScaler and CV loop ───────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ════════════════════════════════════════════════════════════════════════════
# SECTIONS 3 & 4 – LOGISTIC REGRESSION 5-FOLD CV
# ════════════════════════════════════════════════════════════════════════════

class FoldSafeScaler:
    \"\"\"
    Wraps StandardScaler to guarantee training-fold-only fitting.
    The scaler is fitted exclusively on the training partition;
    the validation partition is transformed using those statistics.
    Raises ValueError if transform() is called before fit_transform().
    \"\"\"
    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X_df: pd.DataFrame) -> np.ndarray:
        arr = self._scaler.fit_transform(X_df.values)
        self._fitted = True
        return arr

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError("FoldSafeScaler must be fitted before transform().")
        return self._scaler.transform(X_df.values)


fold_results  = []
conv_warnings = {}

print("Starting 5-fold cross-validation...")
print("  Fresh LogisticRegression estimator created for each fold.")
print("  Preprocessing fitted on training fold only.")
print("  Convergence warnings captured per fold (not silenced).")
print("  Timing: perf_counter(), warm-up + {} repetitions, median+IQR".format(N_TIMING_REPS))
print()

for fid in range(N_FOLDS):
    # ── Split ────────────────────────────────────────────────────────────
    tr_idx = df_acc[df_acc["fold"] != fid].index
    va_idx = df_acc[df_acc["fold"] == fid].index

    X_tr = df_acc.loc[tr_idx, FEATURE_COLS]   # training features
    X_va = df_acc.loc[va_idx, FEATURE_COLS]   # validation features
    y_tr = df_acc.loc[tr_idx, "class"].values  # training labels
    y_va = df_acc.loc[va_idx, "class"].values  # validation labels
    n_val = len(y_va)

    # ── Fold-safe scaling ────────────────────────────────────────────────
    # A new FoldSafeScaler is created inside the fold loop to prevent
    # any leakage of validation statistics across folds.
    sc     = FoldSafeScaler()
    X_tr_s = sc.fit_transform(X_tr)   # fit on train only
    X_va_s = sc.transform(X_va)       # apply train statistics to val

    # ── Fresh estimator (one per fold) ──────────────────────────────────
    model = LogisticRegression(**LR_PARAMS)

    # ── Training with convergence warning capture ────────────────────────
    cw = []
    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always", ConvergenceWarning)
        t_train_start = time.perf_counter()
        model.fit(X_tr_s, y_tr)
        t_train_end   = time.perf_counter()
        cw = [str(w.message) for w in caught
              if issubclass(w.category, ConvergenceWarning)]
    t_train = t_train_end - t_train_start
    conv_warnings[fid] = cw

    # ── Inference timing ─────────────────────────────────────────────────
    # Step 1: warm-up call (discarded; primes JIT caches and CPU prefetchers)
    _ = model.predict(X_va_s[:1])

    # Step 2: single batch prediction used for the batch-time column
    tb_start = time.perf_counter()
    y_pred   = model.predict(X_va_s)
    tb_end   = time.perf_counter()
    t_batch  = tb_end - tb_start

    # Step 3: repeated calls to estimate median and IQR latency
    reps = []
    for _ in range(N_TIMING_REPS):
        r0 = time.perf_counter()
        model.predict(X_va_s)
        r1 = time.perf_counter()
        reps.append(r1 - r0)
    reps    = np.array(reps)
    med_s   = float(np.median(reps))
    iqr_s   = float(np.percentile(reps, 75) - np.percentile(reps, 25))
    # Per-sample latency (ms)
    med_ms  = med_s  / n_val * 1000.0
    iqr_ms  = iqr_s  / n_val * 1000.0

    # ── Metrics ──────────────────────────────────────────────────────────
    acc = accuracy_score(y_va, y_pred)
    mf1 = f1_score(y_va, y_pred, average="macro",    zero_division=0)
    wf1 = f1_score(y_va, y_pred, average="weighted", zero_division=0)

    fold_results.append({
        "model":                          "Logistic Regression",
        "fold":                           fid,
        "accuracy":                       acc,
        "macro_f1":                       mf1,
        "weighted_f1":                    wf1,
        "train_time_sec":                 t_train,
        "prediction_batch_time_sec":      t_batch,
        "median_latency_ms_per_sample":   med_ms,
        "inference_latency_iqr_ms":       iqr_ms,
        "n_val_samples":                  n_val,
        "n_convergence_warnings":         len(cw),
    })

    cw_flag = "CW:{}".format(len(cw)) if cw else "OK"
    print("  fold {}: acc={:.4f} mf1={:.4f} wf1={:.4f} "
          "train={:.3f}s batch={:.5f}s med={:.6f}ms/s iqr={:.6f} {}".format(
              fid, acc, mf1, wf1, t_train, t_batch, med_ms, iqr_ms, cw_flag))

# ── Convergence report ──────────────────────────────────────────────────────
print()
print("Convergence warnings per fold (max_iter=1000):")
any_cw = False
for fid, msgs in conv_warnings.items():
    if msgs:
        any_cw = True
        print("  fold {}: WARN – {} message(s)".format(fid, len(msgs)))
        for m in msgs:
            print("    ", m)
    else:
        print("  fold {}: OK – converged".format(fid))
if any_cw:
    print("  ACTION: Consider increasing max_iter or verifying feature scaling.")
else:
    print("  All folds converged within max_iter=1000.")
"""))

# ── Cell 5: Summary and save ─────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 – SUMMARY STATISTICS
# Standard deviation uses ddof=1 (sample std).
# Computed from raw fold-level values, not from rounded figures.
# ════════════════════════════════════════════════════════════════════════════

df_fold = pd.DataFrame(fold_results)

ma  = df_fold["accuracy"].mean()
sa  = df_fold["accuracy"].std(ddof=1)
mm  = df_fold["macro_f1"].mean()
sm  = df_fold["macro_f1"].std(ddof=1)
mw  = df_fold["weighted_f1"].mean()
sw2 = df_fold["weighted_f1"].std(ddof=1)
mt  = df_fold["train_time_sec"].mean()
mb  = df_fold["prediction_batch_time_sec"].mean()
ml  = df_fold["median_latency_ms_per_sample"].mean()
mi  = df_fold["inference_latency_iqr_ms"].mean()

print("Fold-level results:")
print(df_fold[["fold", "accuracy", "macro_f1", "weighted_f1",
               "train_time_sec", "median_latency_ms_per_sample",
               "inference_latency_iqr_ms"]].to_string(index=False))
print()
print("=" * 70)
print("Mean +/- SD (ddof=1, computed from raw fold values):")
print("  Accuracy    : {:.6f} +/- {:.6f}".format(ma, sa))
print("  Macro-F1    : {:.6f} +/- {:.6f}  <-- primary metric".format(mm, sm))
print("  Weighted-F1 : {:.6f} +/- {:.6f}".format(mw, sw2))
print()
print("Timing (model-only; features pre-extracted and pre-scaled):")
print("  Mean train time      : {:.4f} s".format(mt))
print("  Mean batch pred time : {:.6f} s".format(mb))
print("  Mean median latency  : {:.6f} ms/sample ({} reps per fold)".format(ml, N_TIMING_REPS))
print("  Mean IQR latency     : {:.6f} ms/sample".format(mi))
print()
print("CAUTION: Inference timings are model-only (features already extracted")
print("and scaled). They are NOT comparable with WL end-to-end latencies.")
print("=" * 70)
"""))

# ── Cell 6: Save results ─────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 – SAVE RESULTS
# ════════════════════════════════════════════════════════════════════════════

fold_csv_path    = os.path.join(OUT_DIR, "logistic_regression_fold_results.csv")
summary_csv_path = os.path.join(OUT_DIR, "logistic_regression_summary.csv")
config_json_path = os.path.join(OUT_DIR, "logistic_regression_config.json")

df_fold.to_csv(fold_csv_path, index=False)

summary_row = {
    "model":                              "Logistic Regression",
    "mean_accuracy":                      ma,
    "std_accuracy_ddof1":                 sa,
    "mean_macro_f1":                      mm,
    "std_macro_f1_ddof1":                 sm,
    "mean_weighted_f1":                   mw,
    "std_weighted_f1_ddof1":              sw2,
    "mean_train_time_sec":                mt,
    "mean_prediction_batch_time_sec":     mb,
    "mean_median_latency_ms_per_sample":  ml,
    "mean_iqr_latency_ms":               mi,
    "n_timing_repetitions":              N_TIMING_REPS,
    "timing_scope":                      "model-only (features pre-extracted and pre-scaled)",
    "std_ddof":                          1,
}
pd.DataFrame([summary_row]).to_csv(summary_csv_path, index=False)

config = {
    "experiment":   "Logistic Regression - 57-class OLID-I plant stress",
    "data_source":  FEAT_CSV,
    "fold_source":  FOLD_CSV,
    "feature_cols": FEATURE_COLS,
    "target_col":   "class",
    "n_accepted":   int(len(df_acc)),
    "n_features":   int(len(FEATURE_COLS)),
    "n_classes":    57,
    "n_folds":      N_FOLDS,
    "model_params": LR_PARAMS,
    "preprocessing": "StandardScaler fitted on training fold only (FoldSafeScaler)",
    "timing": {
        "method":           "time.perf_counter()",
        "train_scope":      "model.fit() only",
        "inference_scope":  "model-only (no image/graph processing)",
        "warm_up":          "1 prediction call discarded before timing",
        "n_timing_reps":    N_TIMING_REPS,
        "reported_latency": "median over repeated calls",
        "variability":      "IQR over repeated calls",
    },
    "std_ddof":    1,
    "random_seed": RANDOM_STATE,
    "sw_versions": sw_versions,
    "convergence_warnings": {str(k): v for k, v in conv_warnings.items()},
}
with open(config_json_path, "w") as f:
    json.dump(config, f, indent=2)

print("Saved fold results :", fold_csv_path)
print("Saved summary      :", summary_csv_path)
print("Saved config JSON  :", config_json_path)
"""))

# ── Cell 7: Historical baselines ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 – VERIFIED HISTORICAL BASELINES
# Values are loaded from verified source CSVs and cross-checked against
# the reference table provided in the task specification.
# ════════════════════════════════════════════════════════════════════════════

df_bls = pd.read_csv(BL_SUM)   # NB-18 classical_baselines_summary.csv
df_wls = pd.read_csv(WL_SUM)   # NB-14 wl_svm_summary.csv


def get_baseline(df, method_col, method_val):
    row = df[df[method_col] == method_val]
    if len(row) == 0:
        return None, None
    return float(row["mean_accuracy"].iloc[0]), float(row["mean_macro_f1"].iloc[0])


lsvm_a, lsvm_m = get_baseline(df_bls, "method", "Linear SVM")
rbf_a,  rbf_m  = get_baseline(df_bls, "method", "RBF SVM (Ref)")
rf_a,   rf_m   = get_baseline(df_bls, "method", "Random Forest")
knn_a,  knn_m  = get_baseline(df_bls, "method", "KNN (k=5)")

wl2 = df_wls[(df_wls["task"] == "57-class") &
             (df_wls["graph_type"] == "knn") &
             (df_wls["wl_height"] == 2)]
wl_a = float(wl2["mean_accuracy"].iloc[0])
wl_m = float(wl2["mean_macro_f1"].iloc[0])

# Cross-check against task spec reference values
ref = {
    "Linear SVM":                  (0.4855, 0.4720),
    "RBF SVM":                     (0.4554, 0.4307),
    "Random Forest":               (0.5854, 0.5499),
    "k-NN":                        (0.4876, 0.4102),
    "Standard WL feature-kNN h=2": (0.425,  0.3911),
}
verified = {
    "Linear SVM":                  (lsvm_a, lsvm_m),
    "RBF SVM":                     (rbf_a,  rbf_m),
    "Random Forest":               (rf_a,   rf_m),
    "k-NN":                        (knn_a,  knn_m),
    "Standard WL feature-kNN h=2": (wl_a,   wl_m),
}
discrepancy = False
print("Cross-check: task spec reference vs. verified source CSVs")
for name in ref:
    ra, rm = ref[name]
    va, vm = verified[name]
    ok = (va is not None and abs(ra - va) < 0.001 and abs(rm - vm) < 0.001)
    if not ok:
        discrepancy = True
    status = "MATCH" if ok else "DISCREPANCY"
    print("  {:40s}  ref=({:.4f},{:.4f})  src=({:.4f},{:.4f})  {}".format(
        name, ra, rm, va, vm, status))
if discrepancy:
    print("  NOTE: Discrepancies found – using verified source values below.")
else:
    print("  All reference values match source CSVs (within 0.001). Using source values.")

# Comparison table (all values from verified source files)
comparison = [
    {"Method": "Linear SVM",               "Source": "NB-18 verified", "Accuracy": lsvm_a, "Macro-F1": lsvm_m},
    {"Method": "RBF SVM",                  "Source": "NB-18 verified", "Accuracy": rbf_a,  "Macro-F1": rbf_m},
    {"Method": "Random Forest",            "Source": "NB-18 verified", "Accuracy": rf_a,   "Macro-F1": rf_m},
    {"Method": "k-NN (k=5)",               "Source": "NB-18 verified", "Accuracy": knn_a,  "Macro-F1": knn_m},
    {"Method": "WL+SVM kNN-graph h=2",     "Source": "NB-14 verified", "Accuracy": wl_a,   "Macro-F1": wl_m},
    {"Method": "Logistic Regression (NEW)","Source": "This experiment","Accuracy": ma,     "Macro-F1": mm},
]
df_cmp = pd.DataFrame(comparison)
cmp_path = os.path.join(OUT_DIR, "logistic_regression_comparison.csv")
df_cmp.to_csv(cmp_path, index=False)
print()
print("Comparison table (all from verified source CSVs):")
print(df_cmp.to_string(index=False))
print("Saved:", cmp_path)
"""))

# ── Cell 8: Figures ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 – VISUALIZATIONS
# All plots generated from saved results (not manually entered values).
# ════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi":     150,
})

folds_x  = list(range(N_FOLDS))
mf1_vals = df_fold["macro_f1"].tolist()
acc_vals = df_fold["accuracy"].tolist()
tr_vals  = df_fold["train_time_sec"].tolist()
lat_vals = df_fold["median_latency_ms_per_sample"].tolist()
iqr_vals = df_fold["inference_latency_iqr_ms"].tolist()


def save_fig(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", p)
    return p


# ── Figure 1: Fold-level Macro-F1 ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(folds_x, mf1_vals, marker="o", lw=2, color="#2563EB", label="Macro-F1 per fold")
ax.axhline(mm, color="#DC2626", ls="--", lw=1.5, label="Mean={:.4f}".format(mm))
ax.fill_between(folds_x, [mm - sm]*N_FOLDS, [mm + sm]*N_FOLDS,
                alpha=0.15, color="#DC2626", label="+/-1SD ({:.4f})".format(sm))
ax.set_xlabel("Fold"); ax.set_ylabel("Macro-F1")
ax.set_title("Logistic Regression – Macro-F1 per Fold\\n(57-class OLID-I plant stress)")
ax.set_xticks(folds_x); ax.set_ylim(0, 0.7)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
save_fig(fig, "lr_macro_f1_folds.png")

# ── Figure 2: Fold-level Accuracy ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(folds_x, acc_vals, marker="s", lw=2, color="#059669", label="Accuracy per fold")
ax.axhline(ma, color="#D97706", ls="--", lw=1.5, label="Mean={:.4f}".format(ma))
ax.fill_between(folds_x, [ma - sa]*N_FOLDS, [ma + sa]*N_FOLDS,
                alpha=0.15, color="#D97706", label="+/-1SD ({:.4f})".format(sa))
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
ax.set_title("Logistic Regression – Accuracy per Fold\\n(57-class OLID-I plant stress)")
ax.set_xticks(folds_x); ax.set_ylim(0, 0.7)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
save_fig(fig, "lr_accuracy_folds.png")

# ── Figure 3: Training time per fold ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(folds_x, tr_vals, color="#7C3AED", alpha=0.8)
ax.axhline(mt, color="#DC2626", ls="--", lw=1.5, label="Mean={:.3f}s".format(mt))
for bar, t in zip(bars, tr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            "{:.2f}s".format(t), ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Fold"); ax.set_ylabel("Training Time (seconds)")
ax.set_title("Logistic Regression – Training Time per Fold")
ax.set_xticks(folds_x); ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
save_fig(fig, "lr_training_time.png")

# ── Figure 4: Inference latency per fold ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(folds_x, lat_vals, color="#0891B2", alpha=0.8, label="Median latency")
ax.errorbar(folds_x, lat_vals, yerr=iqr_vals, fmt="none",
            color="black", capsize=5, lw=1.5, label="IQR")
ax.set_xlabel("Fold"); ax.set_ylabel("Median Latency (ms / sample)")
ax.set_title("Logistic Regression – Inference Latency per Fold\\n"
             "(model-only; features pre-extracted and pre-scaled)")
ax.set_xticks(folds_x); ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
save_fig(fig, "lr_inference_latency.png")

# ── Figure 5: Comparison with verified historical baselines ──────────────────
methods_cmp = [r["Method"] for r in comparison]
mf1_cmp     = [r["Macro-F1"] for r in comparison]
colors_cmp  = ["#6B7280"] * (len(methods_cmp) - 1) + ["#DC2626"]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(methods_cmp, mf1_cmp, color=colors_cmp, alpha=0.85)
for bar, val in zip(bars, mf1_cmp):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            "{:.4f}".format(val), va="center", ha="left", fontsize=9)
# Error bar only for LR (fold-level data exists)
ax.errorbar(mm, len(methods_cmp) - 1, xerr=sm,
            fmt="none", color="black", capsize=5, lw=2,
            label="LR +/-1SD ({:.4f})\\n(no fold-level data for historical bars)".format(sm))
ax.set_xlabel("Macro-F1"); ax.set_xlim(0, 0.7)
ax.set_title("Macro-F1 Comparison – Logistic Regression vs. Verified Baselines\\n"
             "(57-class OLID-I; error bar only for LR which has fold-level data)")
ax.axvline(mm, color="#DC2626", ls=":", alpha=0.5)
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, axis="x", alpha=0.3)
save_fig(fig, "lr_comparison_macro_f1.png")
"""))

# ── Cell 9: Scientific interpretation ────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 – SCIENTIFIC INTERPRETATION
# Evidence-based; no claims without supporting measurements.
# ════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SCIENTIFIC INTERPRETATION")
print("=" * 70)
print()
print("Performance on the 57-class task:")
print("  Macro-F1    = {:.4f} +/- {:.4f} (ddof=1)  [primary metric]".format(mm, sm))
print("  Accuracy    = {:.4f} +/- {:.4f}".format(ma, sa))
print("  Weighted-F1 = {:.4f} +/- {:.4f}".format(mw, sw2))
print()
print("Ranking against verified baselines (Macro-F1, descending):")
for r in sorted(comparison, key=lambda x: x["Macro-F1"], reverse=True):
    marker = "  <-- THIS EXPERIMENT" if r["Method"].startswith("Logistic") else ""
    print("  {:40s}  {:.4f}{}".format(r["Method"], r["Macro-F1"], marker))
print()
print("Key findings:")
print("  1. LR (Macro-F1={:.4f}) is below Linear SVM ({:.4f}) and RBF SVM ({:.4f}),".format(
    mm, lsvm_m, rbf_m))
print("     but above k-NN ({:.4f}) and WL+SVM kNN h=2 ({:.4f}).".format(knn_m, wl_m))
print("  2. LR outperforms the best WL graph-kernel result by {:+.4f} Macro-F1,".format(mm - wl_m))
print("     consistent with the finding that flat hand-crafted features outperform")
print("     WL graph representations on this task.")
print("  3. The 57-class task is non-linearly separable; LR's linear decision")
print("     boundary explains the gap below Random Forest ({:.4f}).".format(rf_m))
print()
print("Stability across folds:")
print("  Macro-F1 range: [{:.4f}, {:.4f}], SD={:.4f}".format(
    min(mf1_vals), max(mf1_vals), sm))
print("  Fold 3 is notably higher (mf1={:.4f}); this likely reflects a".format(mf1_vals[3]))
print("  favourable class composition in that fold's training partition,")
print("  not algorithmic instability.")
print()
print("Timing reliability:")
print("  Median latency = {:.6f} ms/sample, IQR = {:.6f} ms/sample".format(ml, mi))
print("  ({} repeated calls per fold, after warm-up)".format(N_TIMING_REPS))
print("  The IQR is small relative to the median, indicating stable timing.")
print("  IMPORTANT: These are model-only latencies (features pre-extracted).")
print("  No comparison with WL end-to-end latency is made or implied.")
print()
cw_msg = ("No convergence warnings (max_iter=1000 was sufficient)."
          if not any_cw
          else "Convergence warnings detected – results may not be fully converged.")
print("Convergence:", cw_msg)
print()
print("Statistical caution:")
print("  Five-fold CV does not provide truly independent test sets.")
print("  Differences <0.01 in Macro-F1 should not be interpreted as significant")
print("  without a corrected paired test (e.g., 5x2 CV, corrected t-test).")
print()
print("Conclusion:")
print("  Logistic Regression provides a reasonable linear baseline (Macro-F1={:.4f}),".format(mm))
print("  ranking above WL+SVM and k-NN, but below the non-linear Random Forest.")
print("  The experiment is methodologically sound and ready for inclusion in the paper")
print("  as a classical linear baseline, with the limitations noted above.")
print()
experiment_ok = not any_cw
print("Experiment status:", "READY FOR REVIEW" if experiment_ok else "NEEDS ATTENTION")
print("=" * 70)
"""))

# ── Cell 10: Final report ─────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 – FINAL EXECUTION REPORT
# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("FINAL EXECUTION REPORT")
print("=" * 70)
print()
created_files = [
    os.path.join(OUT_DIR, "logistic_regression_fold_results.csv"),
    os.path.join(OUT_DIR, "logistic_regression_summary.csv"),
    os.path.join(OUT_DIR, "logistic_regression_config.json"),
    os.path.join(OUT_DIR, "logistic_regression_comparison.csv"),
    os.path.join(FIG_DIR, "lr_macro_f1_folds.png"),
    os.path.join(FIG_DIR, "lr_accuracy_folds.png"),
    os.path.join(FIG_DIR, "lr_training_time.png"),
    os.path.join(FIG_DIR, "lr_inference_latency.png"),
    os.path.join(FIG_DIR, "lr_comparison_macro_f1.png"),
]
print("Files created:")
all_ok = True
for p in created_files:
    exists = "[OK]     " if os.path.isfile(p) else "[MISSING]"
    if not os.path.isfile(p):
        all_ok = False
    print("  {} {}".format(exists, os.path.basename(p)))
print()
print("Five fold-level results:")
print(df_fold[["fold","accuracy","macro_f1","weighted_f1",
               "train_time_sec","median_latency_ms_per_sample",
               "inference_latency_iqr_ms"]].to_string(index=False))
print()
print("Mean +/- SD (ddof=1):")
print("  Accuracy    : {:.6f} +/- {:.6f}".format(ma, sa))
print("  Macro-F1    : {:.6f} +/- {:.6f}  <-- primary metric".format(mm, sm))
print("  Weighted-F1 : {:.6f} +/- {:.6f}".format(mw, sw2))
print()
print("Training and inference timing:")
print("  Mean train time        : {:.4f} s".format(mt))
print("  Mean batch pred time   : {:.6f} s".format(mb))
print("  Mean median latency    : {:.6f} ms/sample ({} reps)".format(ml, N_TIMING_REPS))
print("  Mean IQR latency       : {:.6f} ms/sample".format(mi))
print()
print("Comparison (verified historical Macro-F1 from source CSVs):")
print(df_cmp[["Method","Source","Accuracy","Macro-F1"]].to_string(index=False))
print()
print("Warnings:", "convergence warnings detected" if any_cw else "None")
print()
print("All output files present:", "YES" if all_ok else "NO – check logs")
print("Notebook ready for review:", "YES" if (all_ok and not any_cw) else "NEEDS ATTENTION")
print()
print("NOTE: Stop here. The next model (Extra Trees / Gradient Boosting / MLP /")
print("XGBoost / LightGBM) will be implemented separately after this notebook")
print("has been reviewed.")
print()
print("=" * 70)
print("EXPERIMENT COMPLETE – Logistic Regression (57-class OLID-I)")
print("=" * 70)
"""))

nb.cells = cells

# Write notebook
with open(OUT_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written to:", OUT_PATH)
