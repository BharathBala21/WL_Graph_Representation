"""
Corrected Logistic Regression experiment runner.
Implements all requirements from the task specification.
Run from workspace root: python New_Models_Baselines/run_lr_experiment.py
"""
import os
import sys
import time
import json
import warnings as _warnings_module
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

RANDOM_STATE = 42
N_FOLDS      = 5
N_TIMING_REPS = 20

# ---------------------------------------------------------------------------
# 1. Software versions
# ---------------------------------------------------------------------------
sw_versions = {
    "python":       sys.version.split()[0],
    "scikit-learn": sklearn.__version__,
    "numpy":        np.__version__,
    "pandas":       pd.__version__,
}
print("Software versions:", sw_versions)

# ---------------------------------------------------------------------------
# 2. Data validation
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SECTION 2 - DATA VALIDATION")
print("=" * 70)

assert os.path.isfile(FEAT_CSV), "Feature CSV not found: " + FEAT_CSV
assert os.path.isfile(FOLD_CSV), "Fold CSV not found: " + FOLD_CSV

df_full = pd.read_csv(FEAT_CSV)
df_acc  = df_full[df_full["quality_flag"] == "ACCEPT"].copy().reset_index(drop=True)

META_COLS = [
    "image_path", "image_name", "class", "crop", "condition",
    "quality_flag", "mask_area_ratio", "failure_reason"
]
# Derive feature_cols BEFORE merge so 'fold' never enters feature matrix
FEATURE_COLS = [c for c in df_acc.columns if c not in META_COLS]

df_folds = pd.read_csv(FOLD_CSV)
df_acc = df_acc.merge(df_folds[["image_path", "fold"]], on="image_path", how="left")

# Hard assertions
assert len(df_acc) == 4387, "Expected 4387 rows, got " + str(len(df_acc))
assert len(FEATURE_COLS) == 38, "Expected 38 features, got " + str(len(FEATURE_COLS))
assert df_acc["class"].nunique() == 57, "Expected 57 classes"
assert df_acc["fold"].isna().sum() == 0, "Missing fold assignments"
assert df_acc[FEATURE_COLS].isna().sum().sum() == 0, "NaN in features"
assert np.isinf(df_acc[FEATURE_COLS].values).sum() == 0, "Inf in features"
assert df_acc["image_path"].nunique() == len(df_acc), "Duplicate image paths"

fold_counts = df_acc["fold"].value_counts().sort_index()
assert set(fold_counts.index.astype(int)) == {0, 1, 2, 3, 4}, "Missing folds"

print("[PASS] Row count:       ", len(df_acc), " (expected 4387)")
print("[PASS] Feature columns: ", len(FEATURE_COLS), " (expected 38)")
print("[PASS] Target classes:  ", df_acc["class"].nunique(), " (expected 57)")
print("[PASS] NaN in features: ", 0)
print("[PASS] Inf in features: ", 0)
print("[PASS] Unique images:   ", df_acc["image_path"].nunique())
print("[PASS] All 5 folds present, sizes:", fold_counts.to_dict())

# Stratification check via KL divergence
overall_dist = df_acc["class"].value_counts(normalize=True)
max_kl = 0.0
for fid in range(N_FOLDS):
    fd = (df_acc[df_acc["fold"] == fid]["class"]
          .value_counts(normalize=True)
          .reindex(overall_dist.index, fill_value=0.0))
    kl = float((overall_dist * np.log((overall_dist + 1e-9) / (fd + 1e-9))).sum())
    max_kl = max(max_kl, kl)
strat_status = "well-stratified" if max_kl < 0.05 else "WARNING: may not be fully stratified"
print("[INFO] Fold stratification max KL: {:.6f} ({})".format(max_kl, strat_status))
print()

# ---------------------------------------------------------------------------
# 3 & 4. Five-fold CV
# ---------------------------------------------------------------------------
print("=" * 70)
print("SECTION 3+4 - LOGISTIC REGRESSION 5-FOLD CV")
print("=" * 70)
print()
print("Model: LogisticRegression(solver=lbfgs, max_iter=1000,")
print("       class_weight=balanced, random_state=42, n_jobs=-1)")
print("Timing: time.perf_counter(), warm-up call, {} reps, median+IQR".format(N_TIMING_REPS))
print("Scope:  model-only inference (features pre-extracted and pre-scaled)")
print()


class FoldSafeScaler:
    """Wraps StandardScaler to enforce training-only fitting."""
    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X_df):
        arr = self._scaler.fit_transform(X_df.values)
        self._fitted = True
        return arr

    def transform(self, X_df):
        if not self._fitted:
            raise ValueError("FoldSafeScaler must be fitted first.")
        return self._scaler.transform(X_df.values)


fold_results   = []
conv_warnings  = {}

for fid in range(N_FOLDS):
    tr_idx = df_acc[df_acc["fold"] != fid].index
    va_idx = df_acc[df_acc["fold"] == fid].index

    X_tr = df_acc.loc[tr_idx, FEATURE_COLS]
    X_va = df_acc.loc[va_idx, FEATURE_COLS]
    y_tr = df_acc.loc[tr_idx, "class"].values
    y_va = df_acc.loc[va_idx, "class"].values
    n_val = len(y_va)

    # Fold-safe scaling
    sc = FoldSafeScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)

    # Fresh estimator for every fold
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        multi_class="auto",
    )

    # Capture convergence warnings
    cw = []
    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        model.fit(X_tr_s, y_tr)
        t1 = time.perf_counter()
        cw = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    t_train = t1 - t0
    conv_warnings[fid] = cw

    # Warm-up (discarded)
    _ = model.predict(X_va_s[:1])

    # Single batch prediction (for batch column)
    tb0 = time.perf_counter()
    y_pred = model.predict(X_va_s)
    tb1 = time.perf_counter()
    t_batch = tb1 - tb0

    # Repeated timing for median+IQR
    reps = []
    for _ in range(N_TIMING_REPS):
        r0 = time.perf_counter()
        model.predict(X_va_s)
        r1 = time.perf_counter()
        reps.append(r1 - r0)
    reps = np.array(reps)
    med_s  = float(np.median(reps))
    iqr_s  = float(np.percentile(reps, 75) - np.percentile(reps, 25))
    med_ms = med_s  / n_val * 1000.0
    iqr_ms = iqr_s  / n_val * 1000.0

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

    cw_str = "WARN:{} CW".format(len(cw)) if cw else "OK"
    print("  fold {}: acc={:.4f} mf1={:.4f} wf1={:.4f} "
          "train={:.3f}s med_lat={:.6f}ms iqr={:.6f} {}".format(
              fid, acc, mf1, wf1, t_train, med_ms, iqr_ms, cw_str))

# Convergence report
any_cw = any(len(v) > 0 for v in conv_warnings.values())
print()
print("Convergence warnings per fold:")
for fid, msgs in conv_warnings.items():
    st = "WARN - {} msg(s)".format(len(msgs)) if msgs else "OK"
    print("  fold {}: {}".format(fid, st))
if any_cw:
    print("  ACTION: Consider increasing max_iter or verifying feature scaling.")
else:
    print("  All folds converged within max_iter=1000.")

# ---------------------------------------------------------------------------
# 5. Summary statistics (ddof=1)
# ---------------------------------------------------------------------------
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

print()
print("=" * 70)
print("SUMMARY (mean +/- std, ddof=1 from raw fold values)")
print("=" * 70)
print("  Accuracy    : {:.6f} +/- {:.6f}".format(ma, sa))
print("  Macro-F1    : {:.6f} +/- {:.6f}  <-- primary metric".format(mm, sm))
print("  Weighted-F1 : {:.6f} +/- {:.6f}".format(mw, sw2))
print("  Train time  : {:.4f} s (mean)".format(mt))
print("  Batch pred  : {:.6f} s (mean single call)".format(mb))
print("  Med latency : {:.6f} ms/sample (median over {} reps)".format(ml, N_TIMING_REPS))
print("  IQR latency : {:.6f} ms/sample".format(mi))

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
df_fold.to_csv(os.path.join(OUT_DIR, "logistic_regression_fold_results.csv"), index=False)

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
pd.DataFrame([summary_row]).to_csv(
    os.path.join(OUT_DIR, "logistic_regression_summary.csv"), index=False)

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
    "model_params": {
        "estimator":    "LogisticRegression",
        "solver":       "lbfgs",
        "max_iter":     1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs":       -1,
        "multi_class":  "auto",
    },
    "preprocessing": "StandardScaler fitted on training fold only (FoldSafeScaler)",
    "timing": {
        "method":          "time.perf_counter()",
        "train_scope":     "model.fit() only",
        "inference_scope": "model-only (no image/graph processing)",
        "warm_up":         "1 prediction call discarded before timing",
        "n_timing_reps":   N_TIMING_REPS,
        "reported_latency":"median over repeated calls",
        "variability":     "IQR over repeated calls",
    },
    "std_ddof":    1,
    "random_seed": RANDOM_STATE,
    "sw_versions": sw_versions,
    "convergence_warnings": {str(k): v for k, v in conv_warnings.items()},
}
with open(os.path.join(OUT_DIR, "logistic_regression_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print()
print("Saved fold results    :", os.path.join(OUT_DIR, "logistic_regression_fold_results.csv"))
print("Saved summary         :", os.path.join(OUT_DIR, "logistic_regression_summary.csv"))
print("Saved config JSON     :", os.path.join(OUT_DIR, "logistic_regression_config.json"))

# ---------------------------------------------------------------------------
# 7. Verified historical baselines
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SECTION 7 - VERIFIED HISTORICAL BASELINES")
print("=" * 70)

df_bls = pd.read_csv(BL_SUM)
df_wls = pd.read_csv(WL_SUM)


def get_baseline(df, method_col, method_val):
    row = df[df[method_col] == method_val]
    if len(row) == 0:
        return None, None
    return float(row["mean_accuracy"].iloc[0]), float(row["mean_macro_f1"].iloc[0])


lsvm_a, lsvm_m = get_baseline(df_bls, "method", "Linear SVM")
rbf_a,  rbf_m  = get_baseline(df_bls, "method", "RBF SVM (Ref)")
rf_a,   rf_m   = get_baseline(df_bls, "method", "Random Forest")
knn_a,  knn_m  = get_baseline(df_bls, "method", "KNN (k=5)")

wl2 = df_wls[
    (df_wls["task"] == "57-class") &
    (df_wls["graph_type"] == "knn") &
    (df_wls["wl_height"] == 2)
]
wl_a = float(wl2["mean_accuracy"].iloc[0])
wl_m = float(wl2["mean_macro_f1"].iloc[0])

# Verify against task spec reference values
ref = {
    "Linear SVM":                 (0.4855, 0.4720),
    "RBF SVM":                    (0.4554, 0.4307),
    "Random Forest":              (0.5854, 0.5499),
    "k-NN":                       (0.4876, 0.4102),
    "Standard WL feature-kNN h=2":(0.425,  0.3911),
}
verified = {
    "Linear SVM":                 (lsvm_a, lsvm_m),
    "RBF SVM":                    (rbf_a,  rbf_m),
    "Random Forest":              (rf_a,   rf_m),
    "k-NN":                       (knn_a,  knn_m),
    "Standard WL feature-kNN h=2":(wl_a,   wl_m),
}
discrepancy = False
print("Cross-check (ref values from task spec vs. source CSVs):")
for name in ref:
    ra, rm = ref[name]
    va, vm = verified[name]
    ok = (va is not None and abs(ra - va) < 0.001 and abs(rm - vm) < 0.001)
    status = "MATCH" if ok else "DISCREPANCY"
    if not ok:
        discrepancy = True
    print("  {:40s} ref=({:.4f},{:.4f})  src=({:.4f},{:.4f})  {}".format(
        name, ra, rm, va, vm, status))
if discrepancy:
    print("  NOTE: Using verified source values in comparison table.")
else:
    print("  All reference values match source CSVs (within 0.001).")

# Build comparison table
comparison = [
    {"Method": "Linear SVM",               "Source": "NB-18 verified", "Accuracy": lsvm_a, "Macro-F1": lsvm_m},
    {"Method": "RBF SVM",                  "Source": "NB-18 verified", "Accuracy": rbf_a,  "Macro-F1": rbf_m},
    {"Method": "Random Forest",            "Source": "NB-18 verified", "Accuracy": rf_a,   "Macro-F1": rf_m},
    {"Method": "k-NN (k=5)",               "Source": "NB-18 verified", "Accuracy": knn_a,  "Macro-F1": knn_m},
    {"Method": "WL+SVM kNN-graph h=2",     "Source": "NB-14 verified", "Accuracy": wl_a,   "Macro-F1": wl_m},
    {"Method": "Logistic Regression (NEW)","Source": "This experiment","Accuracy": ma,     "Macro-F1": mm},
]
df_cmp = pd.DataFrame(comparison)
df_cmp.to_csv(os.path.join(OUT_DIR, "logistic_regression_comparison.csv"), index=False)
print()
print("Comparison table (all values from verified source files):")
print(df_cmp.to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SECTION 8 - FIGURES")
print("=" * 70)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})
folds_x   = list(range(N_FOLDS))
mf1_vals  = df_fold["macro_f1"].tolist()
acc_vals  = df_fold["accuracy"].tolist()
tr_vals   = df_fold["train_time_sec"].tolist()
lat_vals  = df_fold["median_latency_ms_per_sample"].tolist()
iqr_vals  = df_fold["inference_latency_iqr_ms"].tolist()


def save_fig(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p)
    plt.close(fig)
    print("  Saved:", p)


# Fig 1 - Macro-F1 per fold
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(folds_x, mf1_vals, marker="o", lw=2, color="#2563EB", label="Macro-F1 per fold")
ax.axhline(mm, color="#DC2626", ls="--", lw=1.5, label="Mean={:.4f}".format(mm))
ax.fill_between(folds_x, [mm - sm]*N_FOLDS, [mm + sm]*N_FOLDS,
                alpha=0.15, color="#DC2626", label="+/-1SD({:.4f})".format(sm))
ax.set_xlabel("Fold"); ax.set_ylabel("Macro-F1")
ax.set_title("Logistic Regression - Macro-F1 per Fold (57-class task)")
ax.set_xticks(folds_x); ax.set_ylim(0, 0.7); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "lr_macro_f1_folds.png")

# Fig 2 - Accuracy per fold
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(folds_x, acc_vals, marker="s", lw=2, color="#059669", label="Accuracy per fold")
ax.axhline(ma, color="#D97706", ls="--", lw=1.5, label="Mean={:.4f}".format(ma))
ax.fill_between(folds_x, [ma - sa]*N_FOLDS, [ma + sa]*N_FOLDS,
                alpha=0.15, color="#D97706", label="+/-1SD({:.4f})".format(sa))
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
ax.set_title("Logistic Regression - Accuracy per Fold (57-class task)")
ax.set_xticks(folds_x); ax.set_ylim(0, 0.7); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "lr_accuracy_folds.png")

# Fig 3 - Training time
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(folds_x, tr_vals, color="#7C3AED", alpha=0.8)
ax.axhline(mt, color="#DC2626", ls="--", label="Mean={:.3f}s".format(mt))
for bar, t in zip(bars, tr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            "{:.2f}s".format(t), ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Fold"); ax.set_ylabel("Training Time (s)")
ax.set_title("Logistic Regression - Training Time per Fold")
ax.set_xticks(folds_x); ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, "lr_training_time.png")

# Fig 4 - Inference latency
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(folds_x, lat_vals, color="#0891B2", alpha=0.8, label="Median latency")
ax.errorbar(folds_x, lat_vals, yerr=iqr_vals, fmt="none",
            color="black", capsize=5, lw=1.5, label="IQR")
ax.set_xlabel("Fold"); ax.set_ylabel("Median Latency (ms/sample)")
ax.set_title("Logistic Regression - Inference Latency per Fold\n"
             "(model-only; features pre-extracted and pre-scaled)")
ax.set_xticks(folds_x); ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, "lr_inference_latency.png")

# Fig 5 - Comparison
methods_cmp = [r["Method"] for r in comparison]
mf1_cmp     = [r["Macro-F1"] for r in comparison]
colors_cmp  = ["#6B7280"] * (len(methods_cmp) - 1) + ["#DC2626"]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(methods_cmp, mf1_cmp, color=colors_cmp, alpha=0.85)
for bar, val in zip(bars, mf1_cmp):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            "{:.4f}".format(val), va="center", ha="left", fontsize=9)
ax.errorbar(mm, len(methods_cmp) - 1, xerr=sm,
            fmt="none", color="black", capsize=5, lw=2,
            label="LR +/-1SD ({:.4f})\n(historical bars: no fold data available)".format(sm))
ax.set_xlabel("Macro-F1"); ax.set_xlim(0, 0.7)
ax.set_title("Macro-F1 Comparison - Logistic Regression vs. Verified Baselines\n"
             "(57-class OLID-I; error bar only for LR which has fold-level data)")
ax.axvline(mm, color="#DC2626", ls=":", alpha=0.5)
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
save_fig(fig, "lr_comparison_macro_f1.png")

# ---------------------------------------------------------------------------
# 9. Scientific interpretation
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SECTION 9 - SCIENTIFIC INTERPRETATION")
print("=" * 70)
print()
print("Performance:")
print("  Logistic Regression achieves mean Macro-F1 = {:.4f} +/- {:.4f} (ddof=1)".format(mm, sm))
print("  and mean Accuracy = {:.4f} +/- {:.4f} on the 57-class OLID-I task.".format(ma, sa))
print()
print("Relative to verified baselines:")
print("  Random Forest (best): Macro-F1 = {:.4f}".format(rf_m))
print("  Linear SVM:           Macro-F1 = {:.4f}".format(lsvm_m))
print("  Logistic Regression:  Macro-F1 = {:.4f}".format(mm))
print("  RBF SVM:              Macro-F1 = {:.4f}".format(rbf_m))
print("  k-NN (k=5):           Macro-F1 = {:.4f}".format(knn_m))
print("  WL+SVM kNN h=2:       Macro-F1 = {:.4f}".format(wl_m))
print("  LR vs WL+SVM delta:   {:+.4f}".format(mm - wl_m))
print()
print("Stability:")
print("  Fold Macro-F1 range: [{:.4f}, {:.4f}], SD={:.4f}".format(
    min(mf1_vals), max(mf1_vals), sm))
print("  Moderate variance expected for 57-class imbalanced task.")
print()
print("Timing reliability:")
print("  {} repeated calls, median={:.6f} ms/sample, IQR={:.6f} ms/sample".format(
    N_TIMING_REPS, ml, mi))
print("  CAUTION: These are model-only latencies (features pre-extracted).")
print("  Do NOT compare with WL end-to-end latency.")
print()
cw_status = "WARN: convergence warnings in some folds (see config JSON)." if any_cw else "All folds converged within max_iter=1000."
print("Convergence:", cw_status)
print()
print("Limitations:")
print("  (a) 5 folds do not support classical significance testing without correction.")
print("  (b) LR assumes log-linear boundary; 57-class task likely non-linear.")
print("  (c) Historical SDs available in NB-18 CSV but not duplicated here.")
print()
print("Experiment status:", "NEEDS REVIEW (convergence warnings)" if any_cw else "READY FOR REVIEW")

# ---------------------------------------------------------------------------
# 10. Final report
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SECTION 10 - FINAL EXECUTION REPORT")
print("=" * 70)
print()
print("Files created:")
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
for p in created_files:
    exists = "[OK]" if os.path.isfile(p) else "[MISSING]"
    print("  {} {}".format(exists, p))
print()
print("Five fold-level results:")
print(df_fold[["fold", "accuracy", "macro_f1", "weighted_f1",
               "train_time_sec", "median_latency_ms_per_sample",
               "inference_latency_iqr_ms"]].to_string(index=False))
print()
print("Mean +/- SD (ddof=1):")
print("  Accuracy    : {:.6f} +/- {:.6f}".format(ma, sa))
print("  Macro-F1    : {:.6f} +/- {:.6f}".format(mm, sm))
print("  Weighted-F1 : {:.6f} +/- {:.6f}".format(mw, sw2))
print()
print("Timing:")
print("  Mean train time  : {:.4f} s".format(mt))
print("  Mean batch pred  : {:.6f} s".format(mb))
print("  Mean med latency : {:.6f} ms/sample ({} reps)".format(ml, N_TIMING_REPS))
print("  Mean IQR latency : {:.6f} ms/sample".format(mi))
print()
print("Comparison table (verified historical Macro-F1):")
print(df_cmp[["Method", "Source", "Accuracy", "Macro-F1"]].to_string(index=False))
print()
print("Warnings:", "convergence warnings - see config JSON" if any_cw else "None")
print()
print("Notebook ready for review:", "NEEDS ATTENTION" if any_cw else "YES")
print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
