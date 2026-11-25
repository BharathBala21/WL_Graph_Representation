"""Experiment 2: data efficiency. How good is each model when it gets less training data?

For every fold, the training part is reduced to a stratified random fraction (10% ... 100%), the
model is trained on it and tested on the full, untouched test fold. Fractions below 100% are
repeated with different random subsets.

Usage (from this folder):  python run_learning_curves.py [level2] [level1] [--models round2|round3]
Writes results/learning_curve_<task>.csv (--models round2 runs only the round-2 models and merges them in).
Tuned (nested-CV) models are not included: re-tuning at every data fraction would take hours.
"""
import bench  # first: loads LightGBM before scikit-learn (Windows OpenMP clash)
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split


FRACTIONS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
REPEATS = {'level2': 3, 'level1': 2}


def subset(X, y, frac, seed):
    if frac >= 1.0:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=seed)
    return Xs, ys


def run(df, task, only=None):
    rows = []
    crops = sorted(bench.WL_LEVEL2) if task == 'level2' else [None]
    path = os.path.join(bench.RESULTS, f'learning_curve_{task}.csv')
    for crop in crops:
        if only and os.path.exists(path):  # resume: skip crops whose rows for these models are saved
            old = pd.read_csv(path)
            if all(((old.model == m) & (old.crop == (crop or 'all'))).any() for m in only):
                print(f'  {task} {crop or "all"}: already done', flush=True)
                continue
        t0 = time.time()
        start = len(rows)
        for fold, Xtr, ytr, Xte, yte in bench.splits(df, task, crop):
            for frac in FRACTIONS:
                for rep in range(1 if frac >= 1.0 else REPEATS[task]):
                    Xs, ys = subset(Xtr, ytr, frac, bench.SEED + rep)
                    if np.bincount(ys).min() < 2:
                        continue
                    for name, model in bench.models(task, crop).items():
                        if (only and name not in only) or (not only and name in bench.ROUND3):
                            continue
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            try:
                                pred = model.fit(Xs, ys).predict(Xte)
                            except ValueError as e:  # e.g. k-NN with fewer than 5 training leaves
                                print('   skip', name, frac, e)
                                continue
                        rows.append({'task': task, 'crop': crop or 'all', 'model': name,
                                     'model_family': bench.family(name),
                                     'fold': fold, 'fraction': frac, 'repeat': rep, 'n_train': len(ys),
                                     'macro_f1': f1_score(yte, pred, average='macro'),
                                     'balanced_accuracy': balanced_accuracy_score(yte, pred)})
        print(f'  {task} {crop or "all"} done ({time.time() - t0:.0f}s)', flush=True)
        if only:  # save after every crop: replace this crop's rows for these models
            new = pd.DataFrame(rows[start:])
            if os.path.exists(path):
                old = pd.read_csv(path)
                old = old[~(old.model.isin(only) & (old.crop == (crop or 'all')))]
                new = pd.concat([old, new], ignore_index=True)
            bench.save(new, f'learning_curve_{task}.csv')
    if not only:
        bench.save(pd.DataFrame(rows), f'learning_curve_{task}.csv')


if __name__ == '__main__':
    args = sys.argv[1:]
    which = args[args.index('--models') + 1] if '--models' in args else None
    only = {'round2': bench.ROUND2, 'round3': bench.ROUND3}.get(which)
    tasks = [a for a in args if a in ('level1', 'level2')] or ['level2', 'level1']
    df = bench.load()
    for task in tasks:
        print('==', task, flush=True)
        run(df, task, only)
