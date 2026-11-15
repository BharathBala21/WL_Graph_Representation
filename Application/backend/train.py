"""Train the models served by the PlantCare AI backend.

Reads the research feature table (full_pipeline/olid_full_features.csv, quality-accepted
rows only) and writes to backend/models/:

  crop_router.joblib   Random Forest that identifies the crop (as in NB23 Stage 1)
  <crop>.joblib        per crop: WL kernel SVM for healthy vs stressed (Level 2 setting),
                       WL kernel SVM ranking the specific conditions, and healthy reference
                       statistics used to explain a result
  metadata.json        configuration + 5-fold cross-validated reliability of every model

Each crop uses the graph type and WL depth that scored best for it in the Level 2
experiments (NB22). Reliability numbers are measured here with stratified 5-fold CV
and shown to users next to every prediction.

Usage (from Application/backend):  python train.py
                                   python train.py --calibrate-only   # only refresh the
                                   "familiarity" limits in metadata.json (no retraining)
"""
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from app.conditions import CONDITIONS, CROPS
from app.features import FEATURES
from app.wl import GRAPH_NAMES, WLFeaturizer, WLKernelSVC, cosine_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, '..', '..', 'full_pipeline', 'olid_full_features.csv'))
OUT = os.path.join(HERE, 'models')
SEED = 42

# Best graph type and WL depth per crop from the Level 2 experiments (NB22).
BEST_GRAPH = {
    'ash_gourd': ('family', 2), 'bitter_gourd': ('knn', 3), 'bottle_gourd': ('hybrid', 3),
    'cucumber': ('family', 2), 'eggplant': ('hybrid', 3), 'ridge_gourd': ('hybrid', 1),
    'snake_gourd': ('hybrid', 3), 'tomato': ('correlation', 2),
}


def topk_accuracy(proba, classes, y, k=3):
    top = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y[i] in classes[top[i]] for i in range(len(y))]))


def cv_crop_models(X, y_stress, y_cond, graph, h):
    """5-fold CV of both per-crop models, sharing the WL features within each fold."""
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    s_acc, s_f1, s_bal, s_spec, c_top1, c_top3 = [], [], [], [], [], []
    for tr, te in skf.split(X, y_cond):
        feat = WLFeaturizer(graph, h).fit(X[tr])
        Ptr, Pte = feat.transform(X[tr]), feat.transform(X[te])
        Ktr, Kte = cosine_kernel(Ptr), cosine_kernel(Ptr, Pte)
        s = SVC(kernel='precomputed', class_weight='balanced', random_state=SEED).fit(Ktr, y_stress[tr])
        ps = s.predict(Kte)
        s_acc.append(accuracy_score(y_stress[te], ps))
        s_f1.append(f1_score(y_stress[te], ps, average='macro'))
        s_bal.append(balanced_accuracy_score(y_stress[te], ps))
        s_spec.append(float((ps[y_stress[te] == 0] == 0).mean()))  # share of healthy leaves found
        c = SVC(kernel='precomputed', class_weight='balanced', probability=True, random_state=SEED).fit(Ktr, y_cond[tr])
        pr = c.predict_proba(Kte)
        c_top1.append(accuracy_score(y_cond[te], c.classes_[pr.argmax(1)]))
        c_top3.append(topk_accuracy(pr, c.classes_, y_cond[te], 3))
    m = lambda v: round(float(np.mean(v)), 4)
    return {'stress_accuracy': m(s_acc), 'stress_macro_f1': m(s_f1), 'stress_balanced_accuracy': m(s_bal),
            'stress_healthy_found': m(s_spec), 'always_stressed_accuracy': round(float(np.mean(y_stress)), 4),
            'condition_top1_accuracy': m(c_top1), 'condition_top3_accuracy': m(c_top3)}


def calibrate_familiarity(X, y_cond, graph, h):
    """What a *new* leaf of this crop normally looks like to the model, from 5-fold CV.

    For every held-out leaf: WL-kernel similarity to its most similar training leaf, and the number
    of measurements outside the training range (0.5th-99.5th percentile). The API flags a photo that
    is more unusual than 99% of these. (Unseen WL patterns were tried too, but even ordinary held-out
    leaves often have most of their deepest patterns unseen, so they do not separate odd photos.)
    """
    best_sim, out_of_range = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y_cond):
        feat = WLFeaturizer(graph, h).fit(X[tr])
        best_sim += list(cosine_kernel(feat.transform(X[tr]), feat.transform(X[te])).max(1))
        lo, hi = np.percentile(X[tr], 0.5, axis=0), np.percentile(X[tr], 99.5, axis=0)
        out_of_range += list(((X[te] < lo) | (X[te] > hi)).sum(1))
    return {
        'best_similarity_p1': round(float(np.percentile(best_sim, 1)), 4),
        'out_of_range_p99': float(np.percentile(out_of_range, 99)),
        'range_low': [round(float(v), 6) for v in np.percentile(X, 0.5, axis=0)],
        'range_high': [round(float(v), 6) for v in np.percentile(X, 99.5, axis=0)],
    }


def calibrate_only():
    """Add the familiarity limits and refreshed reliability numbers to models/metadata.json, without retraining."""
    df = pd.read_csv(DATA)
    df = df[df.quality_flag == 'ACCEPT'].reset_index(drop=True)
    path = os.path.join(OUT, 'metadata.json')
    with open(path, encoding='utf-8') as f:
        meta = json.load(f)
    for crop, (graph, h) in BEST_GRAPH.items():
        d = df[df.crop == crop].reset_index(drop=True)
        Xc = d[FEATURES].values.astype(float)
        meta['crops'][crop]['familiarity'] = calibrate_familiarity(Xc, d.condition.values, graph, h)
        # refresh the reliability numbers (same CV as training; the saved models are not changed)
        meta['crops'][crop].update(cv_crop_models(Xc, (d.condition != 'healthy').astype(int).values,
                                                  d.condition.values, graph, h))
        fam = meta['crops'][crop]['familiarity']
        print(f'  {crop:13s} best sim p1 {fam["best_similarity_p1"]:.3f}  '
              f'out-of-range p99 {fam["out_of_range_p99"]:.0f}')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    df = df[df.quality_flag == 'ACCEPT'].reset_index(drop=True)
    X = df[FEATURES].values.astype(float)
    print(f'Training on {len(df)} quality-accepted leaves, {len(FEATURES)} measurements each')

    # ---- crop router -------------------------------------------------------
    rf = lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                        random_state=SEED, n_jobs=-1)
    y_crop = df.crop.values
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y_crop):
        accs.append(accuracy_score(y_crop[te], rf().fit(X[tr], y_crop[tr]).predict(X[te])))
    router = rf().fit(X, y_crop)
    joblib.dump(router, os.path.join(OUT, 'crop_router.joblib'), compress=3)
    meta = {
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'training_rows': int(len(df)),
        'features': FEATURES,
        'crop_router': {'model': 'Random Forest (300 trees)', 'cv_accuracy': round(float(np.mean(accs)), 4)},
        'crops': {},
    }
    print(f'  crop router: CV accuracy {np.mean(accs):.3f}')

    # ---- per-crop models ---------------------------------------------------
    for crop, (graph, h) in BEST_GRAPH.items():
        d = df[df.crop == crop].reset_index(drop=True)
        Xc = d[FEATURES].values.astype(float)
        y_stress = (d.condition != 'healthy').astype(int).values
        y_cond = d.condition.values
        cv = cv_crop_models(Xc, y_stress, y_cond, graph, h)
        stress = WLKernelSVC(graph, h, probability=True).fit(Xc, y_stress)
        cond = WLKernelSVC(graph, h, probability=True).fit(Xc, y_cond)
        healthy = Xc[y_stress == 0]
        joblib.dump({
            'stress': stress, 'condition': cond, 'graph': graph, 'h': h,
            'healthy_mean': healthy.mean(0), 'healthy_std': healthy.std(0) + 1e-9,
        }, os.path.join(OUT, f'{crop}.joblib'), compress=3)
        meta['crops'][crop] = {
            'name': CROPS[crop], 'graph': graph, 'graph_name': GRAPH_NAMES[graph], 'wl_depth': h,
            'leaves': int(len(d)), 'healthy': int((y_stress == 0).sum()),
            'conditions': sorted(set(y_cond)), **cv,
            'familiarity': calibrate_familiarity(Xc, y_cond, graph, h),
        }
        print(f'  {crop:13s} {GRAPH_NAMES[graph]:24s} h={h}  stress F1 {cv["stress_macro_f1"]:.3f}  '
              f'condition top-1 {cv["condition_top1_accuracy"]:.3f} / top-3 {cv["condition_top3_accuracy"]:.3f}')

    meta['conditions'] = CONDITIONS
    with open(os.path.join(OUT, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Done in {time.time() - t0:.0f}s -> {OUT}')


if __name__ == '__main__':
    import sys
    calibrate_only() if '--calibrate-only' in sys.argv else main()
