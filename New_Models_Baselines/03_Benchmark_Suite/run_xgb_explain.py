"""Experiment 7 (round 3): what XGBoost bases its decision on (TreeSHAP), Level 1.

XGBoost computes exact SHAP values for trees itself (`pred_contribs=True`), so no extra package is
needed. For every fixed fold the model is trained on the other folds and explained on the held-out
leaves; the mean |SHAP| per measurement (in log-odds of "stressed") is averaged over all leaves.

Usage (from this folder):  python run_xgb_explain.py
Writes results/xgboost_shap_level1.csv (per measurement) and results/xgboost_shap_family_level1.csv.
"""
import bench  # first: loads torch and LightGBM before scikit-learn (Windows)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from app.wl import FEATURE_FAMILIES

if __name__ == '__main__':
    df = bench.load()
    shap, n = np.zeros(len(bench.FEATURES)), 0
    signed = np.zeros(len(bench.FEATURES))
    for fold, Xtr, ytr, Xte, yte in bench.splits(df, 'level1'):
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method='hist', n_jobs=1,
                              random_state=bench.SEED, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
        c = m.get_booster().predict(xgb.DMatrix(Xte, feature_names=bench.FEATURES), pred_contribs=True)[:, :-1]
        shap += np.abs(c).sum(0)
        signed += (c * np.where(yte[:, None] == 1, 1, -1)).sum(0)  # >0: pushes towards the true class
        n += len(Xte)
    fam = {f: k for k, fl in FEATURE_FAMILIES.items() for f in fl}
    out = pd.DataFrame({'feature': bench.FEATURES, 'mean_abs_shap': shap / n,
                        'family': [fam.get(f, 'other') for f in bench.FEATURES]})
    out = out.sort_values('mean_abs_shap', ascending=False)
    out['share'] = out.mean_abs_shap / out.mean_abs_shap.sum()
    bench.save(out.round(5), 'xgboost_shap_level1.csv')
    by_fam = out.groupby('family')[['mean_abs_shap', 'share']].sum().sort_values('share', ascending=False)
    bench.save(by_fam.reset_index().round(4), 'xgboost_shap_family_level1.csv')
    print(out.head(10).to_string(index=False))
    print(by_fam.round(3))
