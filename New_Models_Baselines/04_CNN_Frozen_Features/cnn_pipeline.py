"""CNN baseline, option A: frozen ImageNet CNN features + a trained linear classifier ("linear probe").

The CNN is NOT trained on OLID-I. Each photo is passed once through an ImageNet-pretrained network,
its pooled embedding is stored, and a logistic-regression classifier is trained on the embeddings with
exactly the same folds, tasks and metrics as the feature-based benchmark (../03_Benchmark_Suite).
Results are merged into that benchmark's results/ so make_report.py compares everything together.

Steps (run from this folder, in order):
  python cnn_pipeline.py cache          # photos -> 224x224 JPEG cache (once, ~3 min)
  python cnn_pipeline.py robust-cache   # the 240 robustness photos x 8 changes, changed at full size
  python cnn_pipeline.py embed          # embeddings for both CNNs (downloads ImageNet weights once)
  python cnn_pipeline.py evaluate       # quality / speed / size on Level 1 and Level 2
  python cnn_pipeline.py level3         # full pipeline: CNN identifies the crop, then the crop's model
  python cnn_pipeline.py curves         # learning curves (Level 1, Level 2)
  python cnn_pipeline.py robust         # robustness on the changed photos
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(HERE, '..', '03_Benchmark_Suite')
sys.path.insert(0, SUITE)

import bench  # noqa: E402  (LightGBM-safe import order lives there)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CACHE = os.path.join(HERE, 'cache')
ROBUST_CACHE = os.path.join(CACHE, 'robustness')
EMB = os.path.join(HERE, 'embeddings')
SIDE = 224
BACKBONES = {'MobileNetV3-Large': 'mobilenet_v3_large', 'EfficientNet-B0': 'efficientnet_b0'}


def cache_name(path):
    return os.path.splitext(os.path.basename(path))[0].replace(' ', '_') + '.jpg'


# ---------------------------------------------------------------- 1. image caches
def _cache_one(path):
    import cv2
    out = os.path.join(CACHE, cache_name(path))
    if not os.path.exists(out):
        img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)  # JPEG decoded at 1/4 size: fast, then resized
        cv2.imwrite(out, cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out


def cache(workers=6):
    os.makedirs(CACHE, exist_ok=True)
    df = bench.load()
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(_cache_one, df.image_path, chunksize=16)):
            if (i + 1) % 500 == 0:
                print(f'  {i + 1}/{len(df)}  ({time.time() - t0:.0f}s)', flush=True)
    print('cached', len(df), 'photos in', f'{time.time() - t0:.0f}s')


def _robust_one(path):
    import cv2
    from run_robustness import PERTURBATIONS, perturb
    img = cv2.imread(path)  # full size: the change is applied exactly as for the feature models
    for kind in PERTURBATIONS:
        out = os.path.join(ROBUST_CACHE, kind, cache_name(path))
        if not os.path.exists(out):
            p = perturb(img, kind)
            cv2.imwrite(out, cv2.resize(p, (SIDE, SIDE), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def robust_cache(workers=3):
    from run_robustness import FEATURES_OUT, PERTURBATIONS
    for kind in PERTURBATIONS:
        os.makedirs(os.path.join(ROBUST_CACHE, kind), exist_ok=True)
    paths = sorted(pd.read_csv(FEATURES_OUT).image_path.unique())  # the same 240 photos as the feature models
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(_robust_one, paths)):
            if (i + 1) % 40 == 0:
                print(f'  {i + 1}/{len(paths)}  ({time.time() - t0:.0f}s)', flush=True)


# ---------------------------------------------------------------- 2. embeddings
def load_backbone(name):
    import torch
    import torchvision.models as tvm
    arch = BACKBONES[name]
    weights = tvm.get_model_weights(arch).DEFAULT
    model = tvm.get_model(arch, weights=weights).eval()
    model.classifier = torch.nn.Identity()  # keep the pooled embedding (960 / 1280 numbers per photo)
    return model


def to_tensor(paths):
    import cv2
    import torch
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    imgs = [(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - mean) / std for p in paths]
    return torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).copy())


def embed_paths(model, paths, batch=64):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            out.append(model(to_tensor(paths[i:i + batch])).numpy())
    return np.concatenate(out)


def embed():
    import torch
    from run_robustness import PERTURBATIONS
    torch.set_num_threads(os.cpu_count())
    os.makedirs(EMB, exist_ok=True)
    df = bench.load()
    paths = [os.path.join(CACHE, cache_name(p)) for p in df.image_path]
    rpaths = sorted(pd.read_csv(os.path.join(bench.RESULTS, 'robustness_features.csv')).image_path.unique())
    for name in BACKBONES:
        model = load_backbone(name)
        t0 = time.time()
        E = embed_paths(model, paths)
        np.save(os.path.join(EMB, f'{BACKBONES[name]}.npy'), E.astype(np.float32))
        print(f'{name}: {E.shape} in {time.time() - t0:.0f}s', flush=True)
        R = {kind: embed_paths(model, [os.path.join(ROBUST_CACHE, kind, cache_name(p)) for p in rpaths])
             for kind in PERTURBATIONS}
        np.savez(os.path.join(EMB, f'{BACKBONES[name]}_robustness.npz'), paths=np.array(rpaths), **R)


def load_embeddings(name):
    return np.load(os.path.join(EMB, f'{BACKBONES[name]}.npy'))


# ---------------------------------------------------------------- 3. classifier on the embeddings
def head():
    """Logistic regression on standardised embeddings; regularisation C chosen by inner 3-fold CV (nested)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced',
                                                             random_state=bench.SEED))
    return GridSearchCV(est, {'logisticregression__C': [0.001, 0.01, 0.1]}, scoring='f1_macro',
                        cv=StratifiedKFold(3, shuffle=True, random_state=bench.SEED), n_jobs=1)


def model_name(name):
    return f'CNN {name} (frozen) + LR'


def backbone_mb(name):
    model = load_backbone(name)
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20


def end_to_end_ms(name, n=20):
    """Photo file -> answer, for one leaf at a time: decode at 1/4 size, resize, CNN, classifier."""
    import cv2
    import torch
    torch.set_num_threads(os.cpu_count())
    model = load_backbone(name)
    df = bench.load().sample(n, random_state=bench.SEED)
    times = {'read_resize_ms': [], 'cnn_ms': []}
    with torch.inference_mode():
        model(to_tensor([os.path.join(CACHE, cache_name(df.image_path.iloc[0]))]))  # warm-up
        for p in df.image_path:
            t = time.perf_counter()
            img = cv2.resize(cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4), (SIDE, SIDE), interpolation=cv2.INTER_AREA)
            times['read_resize_ms'].append((time.perf_counter() - t) * 1000)
            x = torch.from_numpy(((cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0 - [0.485, 0.456, 0.406]) /
                                  [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None].copy())
            t = time.perf_counter()
            model(x)
            times['cnn_ms'].append((time.perf_counter() - t) * 1000)
    return {k: float(np.median(v)) for k, v in times.items()}


def evaluate():
    rows_by_task = {t: [] for t in ('level2', 'level1')}  # 57-class is set aside for now (future work)
    timing = []
    df = bench.load()
    for name in BACKBONES:
        E = load_embeddings(name)
        mb = backbone_mb(name)
        e2e = end_to_end_ms(name)
        timing.append({'model': model_name(name), 'backbone_mb': mb, **e2e})
        for task in rows_by_task:
            crops = sorted(bench.WL_LEVEL2) if task == 'level2' else [None]
            for crop in crops:
                t0 = time.time()
                for fold, Xtr, ytr, Xte, yte in bench.splits(df, task, crop, X=E):
                    row = bench.run_fold(head(), Xtr, ytr, Xte, yte)
                    # the frozen CNN is part of the model: add its size and its per-leaf time
                    row['head_kb'] = row['model_kb']
                    row['model_kb'] += mb * 1024
                    row['head_single_leaf_ms'] = row['single_leaf_ms']
                    row['single_leaf_ms'] += e2e['cnn_ms']
                    rows_by_task[task].append({'task': task, 'crop': crop or 'all', 'model': model_name(name),
                                               'model_family': model_name(name), 'fold': fold, **row})
                f1 = np.mean([r['macro_f1'] for r in rows_by_task[task][-5:]])
                print(f'  {task:8s} {crop or "":13s} {model_name(name):40s} macro-F1 {f1:.4f} ({time.time() - t0:.0f}s)',
                      flush=True)
    for task, rows in rows_by_task.items():
        merge(pd.DataFrame(rows), f'benchmark_{task}_folds.csv')
    bench.save(pd.DataFrame(timing), 'cnn_end_to_end_timing.csv')


def merge(new, fname):
    path = os.path.join(bench.RESULTS, fname)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[~old.model.isin(new.model.unique())], new], ignore_index=True)
    bench.save(new, fname)


# ---------------------------------------------------------------- 3b. Level 3 (full app pipeline)
def level3():
    """Pure CNN pipeline: the CNN embedding identifies the crop, then the crop's stress classifier."""
    from run_level3 import level3 as run
    df = bench.load()
    timing = pd.read_csv(os.path.join(bench.RESULTS, 'cnn_end_to_end_timing.csv')).set_index('model')
    rows = []
    for name in BACKBONES:
        E = load_embeddings(name)
        mn = model_name(name)
        mb = float(timing.loc[mn, 'backbone_mb'])
        for row in run(df, lambda c: head(), X=E, make_router=head):
            row['model_kb'] += mb * 1024                         # the frozen CNN is part of the system
            row['single_leaf_ms'] += float(timing.loc[mn, 'cnn_ms'])
            rows.append({'task': 'level3', 'crop': 'all', 'model': mn, 'model_family': mn, **row})
        print(f'  level3 {mn}: macro-F1 {np.mean([r["macro_f1"] for r in rows[-5:]]):.4f}', flush=True)
    merge(pd.DataFrame(rows), 'benchmark_level3_folds.csv')


# ---------------------------------------------------------------- 4. learning curves
def curves():
    from run_learning_curves import FRACTIONS, REPEATS, subset
    from sklearn.metrics import balanced_accuracy_score, f1_score
    df = bench.load()
    for task in ('level2', 'level1'):
        rows = []
        for name in BACKBONES:
            E = load_embeddings(name)
            for crop in (sorted(bench.WL_LEVEL2) if task == 'level2' else [None]):
                for fold, Xtr, ytr, Xte, yte in bench.splits(df, task, crop, X=E):
                    for frac in FRACTIONS:
                        for rep in range(1 if frac >= 1.0 else REPEATS[task]):
                            Xs, ys = subset(Xtr, ytr, frac, bench.SEED + rep)
                            if np.bincount(ys).min() < 3:
                                continue
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                pred = head().fit(Xs, ys).predict(Xte)
                            rows.append({'task': task, 'crop': crop or 'all', 'model': model_name(name),
                                         'model_family': model_name(name), 'fold': fold, 'fraction': frac,
                                         'repeat': rep, 'n_train': len(ys),
                                         'macro_f1': f1_score(yte, pred, average='macro'),
                                         'balanced_accuracy': balanced_accuracy_score(yte, pred)})
            print(f'  curves {task} {name} done', flush=True)
        merge(pd.DataFrame(rows), f'learning_curve_{task}.csv')


# ---------------------------------------------------------------- 5. robustness
def robust():
    from run_robustness import PERTURBATIONS, TEST_FOLD
    from sklearn.metrics import balanced_accuracy_score, f1_score
    df = bench.load()
    tr = np.where(df.fold != TEST_FOLD)[0]
    label = df.set_index('image_path').stressed
    rows, preds = [], []
    for name in BACKBONES:
        E = load_embeddings(name)
        R = np.load(os.path.join(EMB, f'{BACKBONES[name]}_robustness.npz'))
        paths = list(R['paths'])
        y = label.loc[paths].values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = head().fit(E[tr], df.stressed.values[tr])
        base = None
        for kind in PERTURBATIONS:
            pred = model.predict(R[kind])
            base = pred if kind == 'clean' else base
            mn = model_name(name)
            rows.append({'model': mn, 'model_family': mn, 'perturbation': kind, 'n_photos': len(paths),
                         'rejected_by_qc': 0,  # a CNN has no segmentation step that can reject a photo
                         'macro_f1': f1_score(y, pred, average='macro'),
                         'balanced_accuracy': balanced_accuracy_score(y, pred),
                         'flip_rate': float((pred != base).mean())})
            preds.append(pd.DataFrame({'model_family': mn, 'perturbation': kind, 'image_path': paths,
                                       'y_true': y, 'y_pred': pred}))
        print(f'  robustness {name} done', flush=True)
    merge(pd.DataFrame(rows), 'robustness_level1.csv')
    p = pd.concat(preds)
    path = os.path.join(bench.RESULTS, 'robustness_level1_predictions.csv')
    old = pd.read_csv(path)
    bench.save(pd.concat([old[~old.model_family.isin(p.model_family.unique())], p]), 'robustness_level1_predictions.csv')


if __name__ == '__main__':
    {'cache': cache, 'robust-cache': robust_cache, 'embed': embed, 'evaluate': evaluate, 'level3': level3,
     'curves': curves, 'robust': robust}[sys.argv[1]]()
