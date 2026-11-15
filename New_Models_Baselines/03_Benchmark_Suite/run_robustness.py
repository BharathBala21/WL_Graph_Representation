"""Experiment 3: robustness. What happens when the photo is taken differently?

A stratified sample of the Level-1 test fold 0 photos is re-processed after realistic changes
(darker / brighter light, warm colour cast, blur, heavy JPEG compression, sensor noise, half
resolution). The 38 measurements are re-extracted with the exact research code, and every model
- trained on the clean measurements of folds 1-4 - is evaluated on them.

Reported per model and change: macro-F1, balanced accuracy, and the share of leaves whose
prediction FLIPS compared with the unchanged photo (lower = more stable).

Usage (from this folder):
  python run_robustness.py extract [n_images] [workers]   # slow: image processing, resumable
  python run_robustness.py evaluate
Writes results/robustness_features.csv and results/robustness_level1.csv
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import bench

PERTURBATIONS = ['clean', 'darker_25', 'brighter_25', 'warm_cast', 'blur', 'jpeg_q20', 'noise', 'half_resolution']
FEATURES_OUT = os.path.join(bench.RESULTS, 'robustness_features.csv')
TEST_FOLD = 0


def perturb(img, kind):
    import cv2
    f = img.astype(np.float32)
    if kind == 'clean':
        return img
    if kind == 'darker_25':
        return np.clip(f * 0.75, 0, 255).astype(np.uint8)
    if kind == 'brighter_25':
        return np.clip(f * 1.25, 0, 255).astype(np.uint8)
    if kind == 'warm_cast':  # BGR: less blue, more red (indoor / evening light)
        return np.clip(f * np.array([0.85, 1.0, 1.12], np.float32), 0, 255).astype(np.uint8)
    if kind == 'blur':
        return cv2.GaussianBlur(img, (0, 0), 4)
    if kind == 'jpeg_q20':
        return cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 20])[1], cv2.IMREAD_COLOR)
    if kind == 'noise':
        rng = np.random.default_rng(bench.SEED)
        return np.clip(f + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    if kind == 'half_resolution':
        return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    raise ValueError(kind)


def extract_one(path):
    import cv2
    from app.features import FEATURES, measure_leaf
    img = cv2.imread(path)
    out = []
    for kind in PERTURBATIONS:
        m = measure_leaf(perturb(img, kind))
        row = {'image_path': path, 'perturbation': kind, 'quality_flag': m['quality_flag']}
        if m['features']:
            row.update(dict(zip(FEATURES, m['features'])))
        out.append(row)
    return out


def extract(n=320, workers=3):
    df = bench.load()
    test = df[df.fold == TEST_FOLD]
    sample = test.groupby(['crop', 'stressed'], group_keys=False).apply(
        lambda g: g.sample(max(2, round(n * len(g) / len(test))), random_state=bench.SEED) if len(g) > 2 else g)
    done = set(pd.read_csv(FEATURES_OUT).image_path) if os.path.exists(FEATURES_OUT) else set()
    todo = [p for p in sample.image_path if p not in done]
    print(f'{len(sample)} sampled photos, {len(todo)} still to process with {workers} workers', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, rows in enumerate(ex.map(extract_one, todo, chunksize=1)):
            pd.DataFrame(rows).to_csv(FEATURES_OUT, mode='a', header=not os.path.exists(FEATURES_OUT), index=False)
            if (i + 1) % 20 == 0:
                print(f'  {i + 1}/{len(todo)}  {(time.time() - t0) / (i + 1):.1f}s per photo', flush=True)


def evaluate():
    from sklearn.metrics import balanced_accuracy_score, f1_score
    df = bench.load()
    feats = pd.read_csv(FEATURES_OUT).merge(df[['image_path', 'stressed', 'fold']], on='image_path')
    clean_csv = df.set_index('image_path').loc[feats[feats.perturbation == 'clean'].image_path, bench.FEATURES].values
    redo = feats[feats.perturbation == 'clean'][bench.FEATURES].values
    print('re-extracted clean features match the research CSV:', np.allclose(clean_csv, redo, rtol=1e-4, atol=1e-6))

    train = df[df.fold != TEST_FOLD]
    Xtr, ytr = train[bench.FEATURES].values.astype(float), train.stressed.values
    rows, preds = [], []
    for name, model in {**bench.models('level1'), **bench.tuned_models('level1')}.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(Xtr, ytr)
        base = None
        for kind in PERTURBATIONS:
            d = feats[feats.perturbation == kind].set_index('image_path')
            ok = d[d.quality_flag == 'ACCEPT']
            pred = pd.Series(model.predict(ok[bench.FEATURES].values.astype(float)), index=ok.index)
            if kind == 'clean':
                base = pred
            preds.append(pd.DataFrame({'model_family': bench.family(name), 'perturbation': kind,
                                       'image_path': pred.index, 'y_true': ok.stressed.values, 'y_pred': pred.values}))
            common = pred.index.intersection(base.index)
            rows.append({'model': name, 'model_family': bench.family(name), 'perturbation': kind,
                         'n_photos': len(d), 'rejected_by_qc': int((d.quality_flag != 'ACCEPT').sum()),
                         'macro_f1': f1_score(ok.stressed, pred, average='macro'),
                         'balanced_accuracy': balanced_accuracy_score(ok.stressed, pred),
                         'flip_rate': float((pred[common] != base[common]).mean())})
        print(f'  {name:38s} done', flush=True)
    bench.save(pd.DataFrame(rows), 'robustness_level1.csv')
    bench.save(pd.concat(preds), 'robustness_level1_predictions.csv')


if __name__ == '__main__':
    if sys.argv[1] == 'extract':
        extract(int(sys.argv[2]) if len(sys.argv) > 2 else 320, int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        evaluate()
