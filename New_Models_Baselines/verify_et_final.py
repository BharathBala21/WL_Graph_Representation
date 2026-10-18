"""
Verification script for Extra Trees Experiment:
Verifies notebook cells and outputs, result CSVs, JSON config, and figures.
"""
import os
import json
import pandas as pd
import nbformat as nbf

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH   = os.path.join(WORKSPACE, "New_Models_Baselines", "02_Extra_Trees", "02_Extra_Trees_Experiment.ipynb")
RES_DIR   = os.path.join(WORKSPACE, "New_Models_Baselines", "results")
FIG_DIR   = os.path.join(WORKSPACE, "New_Models_Baselines", "figures")

print("=" * 70)
print("1. EXTRA TREES NOTEBOOK VERIFICATION")
print("=" * 70)
assert os.path.isfile(NB_PATH), f"Notebook not found: {NB_PATH}"
nb = nbf.read(NB_PATH, as_version=4)
print(f"Notebook file: {NB_PATH} ({os.path.getsize(NB_PATH):,} bytes)")
print(f"Total cells: {len(nb.cells)}")

for i, c in enumerate(nb.cells):
    n_out = len(c.get("outputs", []))
    print(f"  Cell {i:2d} [{c.cell_type:8s}]: {n_out} outputs")
    if c.cell_type == "code":
        assert n_out > 0, f"Cell {i} has 0 outputs!"
        for out in c.get("outputs", []):
            if "text" in out:
                lines = [l.strip() for l in out["text"].split("\n") if l.strip()]
                if lines:
                    print(f"     Preview: {lines[0][:80]}")
                    if len(lines) > 1:
                        print(f"              ... {lines[-1][:80]}")
            elif "data" in out:
                print(f"     Rich data: {list(out['data'].keys())}")

print()
print("=" * 70)
print("2. WORKSPACE ARTIFACT VERIFICATION (CSVs, JSON, PNGs)")
print("=" * 70)

tracked_files = [
    os.path.join(RES_DIR, "extra_trees_fold_results.csv"),
    os.path.join(RES_DIR, "extra_trees_summary.csv"),
    os.path.join(RES_DIR, "extra_trees_config.json"),
    os.path.join(RES_DIR, "extra_trees_comparison.csv"),
    os.path.join(FIG_DIR, "et_comparison_macro_f1.png"),
    os.path.join(FIG_DIR, "et_comparison_accuracy.png"),
    os.path.join(FIG_DIR, "et_comparison_weighted_f1.png"),
    os.path.join(FIG_DIR, "et_training_time.png"),
    os.path.join(FIG_DIR, "et_prediction_time.png"),
    os.path.join(FIG_DIR, "et_macro_f1_folds.png"),
    os.path.join(FIG_DIR, "et_accuracy_folds.png"),
]

for p in tracked_files:
    assert os.path.isfile(p), f"Missing file: {p}"
    print(f"  [EXISTS] {os.path.basename(p):32s} : {os.path.getsize(p):>8,d} bytes | {p}")

print()
print("=" * 70)
print("3. FOLD RESULTS CSV CONTENTS")
print("=" * 70)
df_folds = pd.read_csv(os.path.join(RES_DIR, "extra_trees_fold_results.csv"))
print(df_folds[["fold", "accuracy", "macro_f1", "weighted_f1", "train_time_sec",
                "amortized_batch_ms_per_sample", "single_sample_median_ms"]].to_string(index=False))

print()
print("=" * 70)
print("4. SUMMARY CSV CONTENTS")
print("=" * 70)
df_sum = pd.read_csv(os.path.join(RES_DIR, "extra_trees_summary.csv"))
for col in df_sum.columns:
    print(f"  {col:38s}: {df_sum[col].iloc[0]}")

print()
print("=" * 70)
print("5. COMPARISON CSV CONTENTS (PROVENANCE)")
print("=" * 70)
df_cmp = pd.read_csv(os.path.join(RES_DIR, "extra_trees_comparison.csv"))
print(df_cmp[["Method", "Source", "Accuracy", "Macro-F1", "Weighted-F1"]].to_string(index=False))

print()
print("=" * 70)
print("6. CONFIG JSON TIMING & PARAMS")
print("=" * 70)
with open(os.path.join(RES_DIR, "extra_trees_config.json")) as f:
    cfg = json.load(f)
print("Model params:")
print(json.dumps(cfg["model_params"], indent=2))
print("Timing config:")
print(json.dumps(cfg["timing"], indent=2))

print()
print("ALL EXTRA TREES VERIFICATIONS COMPLETED SUCCESSFULLY.")
