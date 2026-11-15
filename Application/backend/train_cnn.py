"""Train the CNN classifiers used for the hybrid decision (see app/cnn.py).

EfficientNet-B0 stays frozen (ImageNet weights). Trained here, on its 1,280-number photo embeddings:
  router        logistic regression: which of the 8 crops
  stress[crop]  logistic regression per crop: healthy (0) vs stressed (1)
Regularisation C is chosen by inner 3-fold CV (macro-F1), as in the benchmark.

Reliability (5-fold CV, the same splits as train.py: StratifiedKFold(5, shuffle, 42), stratified on the
condition) is written to models/metadata.json under "cnn", so the app reports the CNN's own numbers.

Embeddings: reused from New_Models_Baselines/04_CNN_Frozen_Features/embeddings/efficientnet_b0.npy when it
exists and matches; otherwise computed from the photos (~4 min, CPU).

Usage (from Application/backend):  python train_cnn.py
"""
import json
import os
import time
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.cnn import BACKBONE, BACKBONE_NAME, SIDE, load_backbone, to_input
from train import DATA, OUT, SEED

HERE = os.path.dirname(os.path.abspath(__file__))
CACHED = os.path.abspath(os.path.join(HERE, '..', '..', 'New_Models_Baselines', '04_CNN_Frozen_Features',
                                      'embeddings', f'{BACKBONE}.npy'))


def head():
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED))
    return GridSearchCV(est, {'logisticregression__C': [0.001, 0.01, 0.1]}, scoring='f1_macro',
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1)


def embeddings(df):
    if os.path.exists(CACHED):
        E = np.load(CACHED)
        if len(E) == len(df):
            print(f'Using cached embeddings {CACHED} {E.shape}')
            return E
    import torch
    torch.set_num_threads(os.cpu_count())
    model = load_backbone()
    out, t0 = [], time.time()
    with torch.inference_mode():
        for i in range(0, len(df), 64):
            batch = [to_input(cv2.resize(cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4), (SIDE, SIDE),
                                         interpolation=cv2.INTER_AREA)) for p in df.image_path[i:i + 64]]
            out.append(model(torch.from_numpy(np.stack(batch))).numpy())
            print(f'  {min(i + 64, len(df))}/{len(df)}  ({time.time() - t0:.0f}s)', flush=True)
    return np.concatenate(out)


def main():
    t0 = time.time()
    df = pd.read_csv(DATA)
    df = df[df.quality_flag == 'ACCEPT'].reset_index(drop=True)
    E = embeddings(df)
    y_crop = df.crop.values
    y_stress = (df.condition != 'healthy').astype(int).values
    warnings.simplefilter('ignore')

    # ---- crop router: CV reliability, then fit on everything
    accs = [accuracy_score(y_crop[te], head().fit(E[tr], y_crop[tr]).predict(E[te]))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(E, y_crop)]
    router = head().fit(E, y_crop)
    meta_cnn = {'backbone': BACKBONE_NAME, 'trained_at': pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds'),
                'crop_router_cv_accuracy': round(float(np.mean(accs)), 4), 'crops': {}}
    print(f'  crop router: CV accuracy {np.mean(accs):.3f}')

    # ---- per-crop stress models
    stress = {}
    for crop in sorted(set(y_crop)):
        rows = np.where(y_crop == crop)[0]
        X, y, cond = E[rows], y_stress[rows], df.condition.values[rows]
        m = {'accuracy': [], 'macro_f1': [], 'balanced_accuracy': [], 'healthy_found': []}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, cond):
            p = head().fit(X[tr], y[tr]).predict(X[te])
            m['accuracy'].append(accuracy_score(y[te], p))
            m['macro_f1'].append(f1_score(y[te], p, average='macro'))
            m['balanced_accuracy'].append(balanced_accuracy_score(y[te], p))
            m['healthy_found'].append(float((p[y[te] == 0] == 0).mean()))
        stress[crop] = head().fit(X, y)
        meta_cnn['crops'][crop] = {f'stress_{k}': round(float(np.mean(v)), 4) for k, v in m.items()}
        print(f'  {crop:13s} macro-F1 {np.mean(m["macro_f1"]):.3f}  balanced {np.mean(m["balanced_accuracy"]):.3f}')

    joblib.dump({'backbone': BACKBONE, 'router': router, 'stress': stress}, os.path.join(OUT, 'cnn_heads.joblib'),
                compress=3)
    path = os.path.join(OUT, 'metadata.json')
    with open(path, encoding='utf-8') as f:
        meta = json.load(f)
    meta['cnn'] = meta_cnn
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Done in {time.time() - t0:.0f}s -> {OUT}/cnn_heads.joblib')


if __name__ == '__main__':
    main()
