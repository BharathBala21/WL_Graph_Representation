"""Experiment 1: quality, speed and size of every model on the three tasks.

Usage (from this folder):
  python run_benchmark.py [tasks...] [--models all|round2|tuned]
    tasks: 57-class level1 level2 extraction   (default: all)
    --models round2 : only the classic models added in round 2 (Decision Tree, Naive Bayes, AdaBoost,
                      XGBoost, LightGBM); results are merged into the existing CSV files
    --models tuned  : only the nested-CV tuned models (Level 1 and Level 2)
    --models round3 : only the GNNs (GCN, GIN, GAT) and the round-3 tuned models (XGBoost, GIN); merged in
Writes results/benchmark_<task>_folds.csv (one row per model x fold [x crop]) and
results/feature_extraction_timing.csv.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import bench


def zoo(task, crop, which):
    return bench.select(task, crop, which)


def run_task(df, task, which='all'):
    rows = []
    path = os.path.join(bench.RESULTS, f'benchmark_{task}_folds.csv')
    crops = sorted(bench.WL_LEVEL2) if task == 'level2' else [None]
    for crop in crops:
        for name in zoo(task, crop, which):
            if which == 'round3' and os.path.exists(path):  # resume: skip (model, crop) runs already saved
                old = pd.read_csv(path)
                if ((old.model == name) & (old.crop == (crop or 'all'))).any():
                    print(f'  {task} {crop or ""} {name}: already done', flush=True)
                    continue
            t0 = time.time()
            done = []
            for fold, Xtr, ytr, Xte, yte in bench.splits(df, task, crop):
                row = bench.run_fold(zoo(task, crop, which)[name], Xtr, ytr, Xte, yte)
                if fold == 0 and which not in ('tuned', 'round3'):
                    # peak training memory, in a separate (slower) traced fit. Not for the GNNs: tracemalloc
                    # does not see PyTorch's allocations, so their parameter count is reported instead.
                    row['train_peak_mb'] = bench.run_fold(zoo(task, crop, which)[name], Xtr, ytr, Xte, yte,
                                                          measure_memory=True)['train_peak_mb']
                done.append({'task': task, 'crop': crop or 'all', 'model': name, 'model_family': bench.family(name),
                             'fold': fold, **row})
            rows += done
            print(f'  {task:8s} {crop or "":13s} {name:38s} macro-F1 {np.mean([r["macro_f1"] for r in done]):.4f}'
                  f'  ({time.time() - t0:.0f}s)', flush=True)
            if which == 'round3':  # save after every model, so a disconnect loses at most one model
                save_merged(path, pd.DataFrame(done), task)
    if which != 'round3' and rows:
        new = pd.DataFrame(rows)
        if which != 'all' and os.path.exists(path):  # merge: replace these models' rows, keep the others
            old = pd.read_csv(path)
            new = pd.concat([old[~old.model.isin(new.model.unique())], new], ignore_index=True)
        bench.save(new, f'benchmark_{task}_folds.csv')


def save_merged(path, new, task):
    """Replace the rows of these (model, crop) pairs in the task's CSV, keep everything else."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        key = set(zip(new.model, new.crop))
        old = old[[k not in key for k in zip(old.model, old.crop)]]
        new = pd.concat([old, new], ignore_index=True)
    bench.save(new, f'benchmark_{task}_folds.csv')


def run_extraction(df, n=12):
    """Time of the photo -> 38 measurements step (same for every feature-based model)."""
    import cv2
    from app.features import measure_leaf
    rows = []
    sample = df.groupby('crop', group_keys=False).apply(lambda g: g.sample(min(len(g), max(1, n // 8)), random_state=bench.SEED))
    for r in sample.itertuples():
        try:
            t = time.perf_counter()
            img = cv2.imread(r.image_path)
            t_read = time.perf_counter() - t
            t = time.perf_counter()
            measure_leaf(img)
            rows.append({'image': r.image_name, 'width': img.shape[1], 'height': img.shape[0],
                         'read_ms': t_read * 1000, 'segment_and_measure_ms': (time.perf_counter() - t) * 1000})
            del img
        except MemoryError:
            print('  skipped (out of memory):', r.image_name)
    out = pd.DataFrame(rows)
    bench.save(out, 'feature_extraction_timing.csv')
    print(out.describe().round(1))


if __name__ == '__main__':
    args = sys.argv[1:]
    which = 'all'
    if '--models' in args:
        i = args.index('--models')
        which = args[i + 1]
        args = args[:i] + args[i + 2:]
    tasks = args or ['level2', 'level1', '57-class', 'extraction']
    df = bench.load()
    for task in tasks:
        print('==', task, which, flush=True)
        if task == 'extraction':
            run_extraction(df)
        elif which in ('tuned', 'round3') and task == '57-class':
            print('  (tuned models are run on Level 1 and Level 2 only)')
        else:
            run_task(df, task, which)
