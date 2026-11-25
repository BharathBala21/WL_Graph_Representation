"""Experiment 4: Level 3, the full app pipeline (as in NB23 and the PlantCare app).

For every leaf: Stage 1 identifies the crop (Random Forest, 300 trees, the same for every model), then
Stage 2 sends the leaf to THAT crop's healthy/stressed model. Splits: the fixed folds in
cv_fold_assignments.csv. Every Stage-2 model family is trained per crop on the training fold only, so
the only thing that differs between rows is the Stage-2 stress model.

Also reported: the same pipeline with the TRUE crop ("oracle" routing), to show how much is lost when the
crop is identified wrongly.

Usage (from this folder):  python run_level3.py [--models all|default|tuned|round3]
Writes results/benchmark_level3_folds.csv (merged per model).
"""
import bench  # first: loads LightGBM before scikit-learn (Windows OpenMP clash)
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier



def router():
    return RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                  random_state=bench.SEED, n_jobs=1)


def level3(df, stage2, X=None, router_X=None, make_router=router, name='model'):
    """stage2(crop) -> unfitted model. Yields one result row per fold."""
    X = df[bench.FEATURES].values.astype(float) if X is None else X
    RX = X if router_X is None else router_X
    crops, y = df.crop.values, df.stressed.values
    for f in sorted(df.fold.unique()):
        tr, te = np.where(df.fold != f)[0], np.where(df.fold == f)[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t0 = time.perf_counter()
            r = make_router().fit(RX[tr], crops[tr])
            models = {}
            for c in sorted(set(crops)):
                rows = tr[crops[tr] == c]
                models[c] = clone(stage2(c)).fit(X[rows], y[rows])
            train_s = time.perf_counter() - t0
            routed = r.predict(RX[te])
            pred, oracle = np.empty(len(te), int), np.empty(len(te), int)
            for c, m in models.items():
                k = routed == c
                if k.any():
                    pred[k] = m.predict(X[te][k])
                k = crops[te] == c
                oracle[k] = m.predict(X[te][k])
            # one-leaf latency: router + the routed crop's model
            one = te[:1]
            lat = []
            for _ in range(20):
                t = time.perf_counter()
                c = r.predict(RX[one])[0]
                models[c].predict(X[one])
                lat.append(time.perf_counter() - t)
        q = bench.quality(y[te], pred)
        qo = bench.quality(y[te], oracle)
        yield {'fold': int(f), **q, 'routing_accuracy': float((routed == crops[te]).mean()),
               'oracle_macro_f1': qo['macro_f1'], 'oracle_balanced_accuracy': qo['balanced_accuracy'],
               'specificity_healthy': float((pred[y[te] == 0] == 0).mean()),
               'sensitivity_stressed': float((pred[y[te] == 1] == 1).mean()),
               'train_s': train_s, 'single_leaf_ms': float(np.median(lat) * 1000),
               'model_kb': (len(pickle.dumps(r)) + sum(len(pickle.dumps(m)) for m in models.values())) / 1024,
               'n_train': len(tr), 'n_test': len(te)}


def stage2_zoo(which):
    """{family name: function crop -> unfitted Stage-2 model}. Looked up by family, because WL's
    model name contains the crop's own graph setting."""
    first = sorted(bench.WL_LEVEL2)[0]
    out = {}
    round3 = lambda task, crop: bench.select(task, crop, 'round3')
    for source, use in ((bench.models, which in ('all', 'default')), (bench.tuned_models, which in ('all', 'tuned')),
                        (round3, which == 'round3')):
        if not use:
            continue
        for n in source('level2', first):
            fam = bench.family(n)
            out[fam] = lambda c, fam=fam, source=source: next(
                m for k, m in source('level2', c).items() if bench.family(k) == fam)
    return out


def save_merged(new):
    path = os.path.join(bench.RESULTS, 'benchmark_level3_folds.csv')
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[~old.model.isin(new.model.unique())], new], ignore_index=True)
    bench.save(new, 'benchmark_level3_folds.csv')


if __name__ == '__main__':
    which = sys.argv[sys.argv.index('--models') + 1] if '--models' in sys.argv else 'all'
    df = bench.load()
    rows = []
    path = os.path.join(bench.RESULTS, 'benchmark_level3_folds.csv')
    for fam, stage2 in stage2_zoo(which).items():
        if which == 'round3' and os.path.exists(path) and (pd.read_csv(path).model == fam).any():
            print(f'  level3 {fam}: already done', flush=True)  # resume after a disconnect
            continue
        t0 = time.time()
        for row in level3(df, stage2):
            rows.append({'task': 'level3', 'crop': 'all', 'model': fam, 'model_family': fam, **row})
        save_merged(pd.DataFrame([r for r in rows if r['model'] == fam]))  # keep results even if a later model fails
        last = [r for r in rows if r['model'] == fam]
        print(f'  level3 {fam:38s} macro-F1 {np.mean([r["macro_f1"] for r in last]):.4f}  oracle '
              f'{np.mean([r["oracle_macro_f1"] for r in last]):.4f}  routing {np.mean([r["routing_accuracy"] for r in last]):.3f}'
              f'  ({time.time() - t0:.0f}s)', flush=True)

