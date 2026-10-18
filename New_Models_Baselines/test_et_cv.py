"""
Test script: 5-fold cross-validation for ExtraTreesClassifier.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV  = os.path.join(WORKSPACE, "full_pipeline", "olid_full_features.csv")
FOLD_CSV  = os.path.join(WORKSPACE, "full_pipeline", "cv_fold_assignments.csv")

df_full = pd.read_csv(FEAT_CSV)
df_acc  = df_full[df_full["quality_flag"] == "ACCEPT"].copy().reset_index(drop=True)

META_COLS = [
    "image_path", "image_name", "class", "crop", "condition",
    "quality_flag", "mask_area_ratio", "failure_reason"
]
FEATURE_COLS = [c for c in df_acc.columns if c not in META_COLS]
df_folds = pd.read_csv(FOLD_CSV)
df_acc   = df_acc.merge(df_folds[["image_path", "fold"]], on="image_path", how="left")

print(f"Accepted samples: {len(df_acc)}")
print(f"Features: {len(FEATURE_COLS)}")
print(f"Classes: {df_acc['class'].nunique()}")

ET_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

N_FOLDS = 5
N_TIMING_REPS = 20
N_SINGLE_TIMING_REPS = 50

results = []
print("\nStarting 5-Fold Cross-Validation...")

for fid in range(N_FOLDS):
    tr_idx = df_acc[df_acc["fold"] != fid].index
    va_idx = df_acc[df_acc["fold"] == fid].index

    X_tr = df_acc.loc[tr_idx, FEATURE_COLS].values
    X_va = df_acc.loc[va_idx, FEATURE_COLS].values
    y_tr = df_acc.loc[tr_idx, "class"].values
    y_va = df_acc.loc[va_idx, "class"].values
    n_val = len(y_va)

    clf = ExtraTreesClassifier(**ET_PARAMS)

    t_tr0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    t_train = time.perf_counter() - t_tr0

    # Warm-up batch prediction
    _ = clf.predict(X_va[:1])

    # Single batch prediction for evaluation
    tb0 = time.perf_counter()
    y_pred = clf.predict(X_va)
    t_batch = time.perf_counter() - tb0

    # Repeated batch predictions for amortized time per sample
    batch_reps = []
    for _ in range(N_TIMING_REPS):
        r0 = time.perf_counter()
        clf.predict(X_va)
        r1 = time.perf_counter()
        batch_reps.append(r1 - r0)
    batch_reps = np.array(batch_reps)
    med_b = float(np.median(batch_reps))
    iqr_b = float(np.percentile(batch_reps, 75) - np.percentile(batch_reps, 25))
    amort_ms = med_b / n_val * 1000.0
    amort_iqr_ms = iqr_b / n_val * 1000.0

    # Dedicated single-sample prediction benchmark
    x_single = X_va[0:1]
    for _ in range(10):  # warm-up
        _ = clf.predict(x_single)
    single_reps = []
    for _ in range(N_SINGLE_TIMING_REPS):
        s0 = time.perf_counter()
        clf.predict(x_single)
        s1 = time.perf_counter()
        single_reps.append(s1 - s0)
    single_reps = np.array(single_reps) * 1000.0
    s_med = float(np.median(single_reps))
    s_iqr = float(np.percentile(single_reps, 75) - np.percentile(single_reps, 25))

    acc = accuracy_score(y_va, y_pred)
    mf1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
    wf1 = f1_score(y_va, y_pred, average="weighted", zero_division=0)

    results.append({
        "fold": fid,
        "accuracy": acc,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "train_time_sec": t_train,
        "prediction_batch_time_sec": t_batch,
        "amortized_batch_ms_per_sample": amort_ms,
        "amortized_batch_iqr_ms": amort_iqr_ms,
        "single_sample_median_ms": s_med,
        "single_sample_iqr_ms": s_iqr,
        "n_val_samples": n_val
    })
    print(f"Fold {fid}: Acc={acc:.4f}, Macro-F1={mf1:.4f}, Weighted-F1={wf1:.4f}, "
          f"Train={t_train:.3f}s, AmortBatch={amort_ms:.6f}ms, SingleSample={s_med:.4f}ms")

df_fold = pd.DataFrame(results)
print("\n" + "=" * 70)
print("EXTRA TREES SUMMARY RESULTS (ddof=1)")
print("=" * 70)
print(f"Mean Accuracy:    {df_fold['accuracy'].mean():.6f} +/- {df_fold['accuracy'].std(ddof=1):.6f}")
print(f"Mean Macro-F1:    {df_fold['macro_f1'].mean():.6f} +/- {df_fold['macro_f1'].std(ddof=1):.6f}")
print(f"Mean Weighted-F1: {df_fold['weighted_f1'].mean():.6f} +/- {df_fold['weighted_f1'].std(ddof=1):.6f}")
print(f"Mean Train Time:  {df_fold['train_time_sec'].mean():.4f} s")
print(f"Mean Amort Batch: {df_fold['amortized_batch_ms_per_sample'].mean():.6f} ms/sample")
print(f"Mean Single Samp: {df_fold['single_sample_median_ms'].mean():.4f} ms")
