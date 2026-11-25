"""Experiment 6 (round 3): does graph construction matter for GNNs the way it does for WL?

Level 1 (healthy vs stressed, all crops, the fixed folds). For each graph construction (the five used
for WL, plus 'none' = no edges, which turns every GNN into a model of the 38 nodes on their own) we
train GCN, GIN and GAT with 3 layers, and WL + SVM with h = 3 on the same graph, on the same folds.

Usage (from this folder):  python run_graph_study.py [workers]
Writes results/graph_study_level1_folds.csv (one row per model x graph x fold); finished runs are
skipped, so the script can be restarted.
"""
import bench  # first: loads torch and LightGBM before scikit-learn (Windows)
import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from app.wl import GRAPH_NAMES
from gnn import ARCH_NAMES, GNNClassifier

OUT = os.path.join(bench.RESULTS, 'graph_study_level1_folds.csv')
GRAPHS = list(GRAPH_NAMES) + ['none']


def one(model_name, graph, fold):
    df = bench.load()
    Xtr = Xte = None
    for f, a, ya, b, yb in bench.splits(df, 'level1'):
        if f == fold:
            Xtr, ytr, Xte, yte = a, ya, b, yb
    if model_name == 'WL + SVM':
        model = bench.WLSVM(graph, 3)
    else:
        arch = {v: k for k, v in ARCH_NAMES.items()}[model_name]
        model = GNNClassifier(arch, graph)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t = time.perf_counter()
        model.fit(Xtr, ytr)
        train_s = time.perf_counter() - t
        pred = model.predict(Xte)
    row = {'model': model_name, 'graph': graph, 'fold': fold, 'train_s': train_s, **bench.quality(yte, pred, model, Xte)}
    if hasattr(model, 'n_parameters'):
        row.update({'n_parameters': model.n_parameters(), 'epochs_run': model.epochs_run_, 'device': model.device_})
    return row


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=['model', 'graph', 'fold'])
    have = set(zip(done.model, done.graph, done.fold.astype(int)))
    jobs = [(m, g, f) for m in ['WL + SVM'] + list(ARCH_NAMES.values()) for g in GRAPHS for f in range(5)
            if not (m == 'WL + SVM' and g == 'none') and (m, g, f) not in have]
    print(len(jobs), 'runs to do', flush=True)
    rows = [] if done.empty else done.to_dict('records')
    # 'spawn': CUDA cannot be used in forked worker processes (Linux / Colab)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as ex:
        futs = {ex.submit(one, *j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  {r['model']:9s} {r['graph']:12s} fold {r['fold']}  macro-F1 {r['macro_f1']:.4f}"
                  f"  ({r['train_s']:.0f}s)", flush=True)
    res = pd.DataFrame(rows).groupby(['model', 'graph']).macro_f1.mean().unstack('model').round(4)
    print(res)
