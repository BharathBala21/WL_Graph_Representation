"""Paired, routed binary stress experiment. Use --development before choosing a model.

Development excludes original fold 4 from all fits and scores. The default run evaluates
the frozen revision and its controls on each original outer fold, including fold 4.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'Application/backend'))
from app.wl import LeafMeasurementKernel, WLFeaturizer, WLKernelSVC, cosine_kernel, stress_classifier
from train import BEST_GRAPH, FEATURES


def data():
    leaves = pd.read_csv(ROOT / 'full_pipeline/olid_full_features.csv').query("quality_flag == 'ACCEPT'")
    folds = pd.read_csv(ROOT / 'Research_Paper/analysis/cv_fold_assignments.csv').rename(columns={'image_path': 'image_name'})
    data = leaves.merge(folds, on=['image_name', 'class'], how='left', validate='one_to_one', sort=False)
    if data.fold.isna().any() or set(data.fold) != set(range(5)) or len(data) != len(folds):
        raise ValueError('The fold manifest and accepted leaves must match exactly.')
    return data


class StressBins:
    """Learn univariate interval labels from the training fold's stress labels."""

    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit(self, X, y):
        self.bin_edges_ = []
        for column in X.T:
            tree = DecisionTreeClassifier(max_leaf_nodes=self.n_bins, min_samples_leaf=0.05,
                                          class_weight='balanced', random_state=42).fit(column[:, None], y).tree_
            cuts = np.sort(tree.threshold[tree.children_left != -1])
            self.bin_edges_.append(np.r_[-np.inf, cuts, np.inf])
        return self

    def transform(self, X):
        return np.column_stack([np.searchsorted(edges[1:-1], col, side='right')
                                for edges, col in zip(self.bin_edges_, X.T)])


@dataclass
class Candidate:
    name: str
    train_kernel: np.ndarray
    test_kernel: np.ndarray
    C: float = 1.
    weight_power: float = 1.


def development_kernels(train, target, test, crop):
    graph, h = BEST_GRAPH[crop]
    f = WLFeaturizer(graph, h).fit(train)
    P = f.transform(train)
    original_train, original_test = cosine_kernel(P), cosine_kernel(P, f.transform(test))
    yield Candidate('Original WL', original_train, original_test)
    trees = ExtraTreesClassifier(n_estimators=96, min_samples_leaf=4, max_features=1.0,
                                 class_weight='balanced', random_state=42, n_jobs=1).fit(train, target)
    enc = OneHotEncoder(handle_unknown='ignore')
    train_nodes = enc.fit_transform(trees.apply(train))
    test_nodes = enc.transform(trees.apply(test))
    K = (train_nodes @ train_nodes.T).toarray() / trees.n_estimators
    L = (test_nodes @ train_nodes.T).toarray() / trees.n_estimators
    yield Candidate('Tree proximity only', K, L)
    yield Candidate('Original WL + tree proximity', (K + original_train) / 2, (L + original_test) / 2)
    qt = QuantileTransformer(n_quantiles=min(100, len(train)), random_state=42).fit(train)
    a, b = qt.transform(train), qt.transform(test)
    R, S = [np.exp(-4 * cdist(x, a, 'cityblock') / a.shape[1]) for x in (a, b)]
    yield Candidate('Continuous only', R, S)
    yield Candidate('Original WL + continuous', (original_train + R) / 2, (original_test + S) / 2)
    cont = LeafMeasurementKernel().fit(train)
    A, B = cont.kernel(), cont.kernel(test)
    yield Candidate('Measurement contrasts only', A, B)
    yield Candidate('Original WL + measurement contrasts', (original_train + A) / 2, (original_test + B) / 2)
    f = WLFeaturizer('correlation', 1).fit(train)
    P = f.transform(train)
    K, L = cosine_kernel(P), cosine_kernel(P, f.transform(test))
    yield Candidate('Shallow correlation WL', K, L)
    mixed_train, mixed_test = (K + A) / 2, (L + B) / 2
    yield Candidate('Shallow WL + measurement contrasts', mixed_train, mixed_test)
    yield Candidate('Shallow WL + contrasts (moderate weights)', mixed_train, mixed_test, 1, 0.75)
    yield Candidate('Shallow WL + contrasts C4', mixed_train, mixed_test, 4)
    yield Candidate('Shallow WL + contrasts C4 (moderate weights)', mixed_train, mixed_test, 4, 0.75)
    family = WLFeaturizer('family', 1).fit(train)
    family_p = family.transform(train)
    yield Candidate('Family WL + contrasts C4', (cosine_kernel(family_p) + A) / 2, (cosine_kernel(family_p, family.transform(test)) + B) / 2, 4)
    for topology in ('spatial', 'correlation'):
        layers = []
        for n in (2, 4, 8):
            f = WLFeaturizer(topology, 1, n).fit(train)
            A = f.transform(train)
            layers.append((cosine_kernel(A), cosine_kernel(A, f.transform(test))))
        K, L = np.mean([k for k, _ in layers], axis=0), np.mean([l for _, l in layers], axis=0)
        yield Candidate(f'Multiscale {topology} WL', K, L)
        yield Candidate(f'Multiscale {topology} WL + continuous', (K + R) / 2, (L + S) / 2)
    for bins in (3, 6):
        f = WLFeaturizer('correlation', 1, bins).fit(train, discretizer=StressBins(bins).fit(train, target))
        P = f.transform(train)
        K, L = cosine_kernel(P), cosine_kernel(P, f.transform(test))
        yield Candidate(f'Stress bins {bins} WL', K, L)
        yield Candidate(f'Stress bins {bins} WL + continuous', (K + R) / 2, (L + S) / 2)


def fixed_predictions(train, target, test, crop):
    # The single-component controls match the revised SVM's C, graph and class weights.
    for name, model in [('Original WL', WLKernelSVC(*BEST_GRAPH[crop], probability=False)),
                        ('Revised WL + measurements', stress_classifier(probability=False)),
                        ('WL only C4', WLKernelSVC('correlation', h=1, C=4, probability=False))]:
        start = time.perf_counter()
        model.fit(train, target)
        fit_s = time.perf_counter() - start
        pred = model.predict(test)
        latencies = []
        for row in test[:10]:
            start = time.perf_counter()
            model.predict(row[None, :])
            latencies.append((time.perf_counter() - start) * 1000)
        yield name, pred, fit_s, len(pickle.dumps(model)), latencies
    start = time.perf_counter()
    cont = LeafMeasurementKernel().fit(train)
    svc = SVC(C=4, class_weight='balanced', kernel='precomputed').fit(cont.kernel(), target)
    fit_s = time.perf_counter() - start
    latencies = []
    for row in test[:10]:
        start = time.perf_counter()
        svc.predict(cont.kernel(row[None, :]))
        latencies.append((time.perf_counter() - start) * 1000)
    yield 'Measurements only C4', svc.predict(cont.kernel(test)), fit_s, len(pickle.dumps((cont, svc))), latencies


def classification_metrics(y, pred):
    return dict(macro_f1=f1_score(y, pred, average='macro'), accuracy=accuracy_score(y, pred),
                balanced_accuracy=balanced_accuracy_score(y, pred),
                healthy_recall=(pred[y == 0] == 0).mean(), stressed_recall=(pred[y == 1] == 1).mean())


def evaluate_unrouted():
    """Level 1 pooled model and Level 2 known-crop models on the original benchmark protocols."""
    df = data()
    rows = []
    for crop in ['all', *BEST_GRAPH]:
        part = df if crop == 'all' else df[df.crop == crop]
        X = part[FEATURES].to_numpy()
        y = (part.condition != 'healthy').to_numpy(int)
        if crop == 'all':
            splits = [(np.where(part.fold != f)[0], np.where(part.fold == f)[0]) for f in range(5)]
            graph, h = 'hybrid', 3
        else:
            splits = StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)
            graph, h = BEST_GRAPH[crop]
        for f, (tr, te) in enumerate(splits):
            for name, model in [('Original WL', WLKernelSVC(graph, h, probability=False)),
                                ('Revised WL + measurements', stress_classifier(probability=False))]:
                pred = model.fit(X[tr], y[tr]).predict(X[te])
                rows.append(dict(crop=crop, fold=f, model=name, leaves=len(te), **classification_metrics(y[te], pred)))
    return pd.DataFrame(rows)


def evaluate_robustness():
    """Route every perturbed image again and compare to its own clean prediction."""
    df = data()
    train = df[df.fold != 0]
    cached = pd.read_csv(HERE / 'results/robustness_features.csv')
    cached = cached.merge(df[['image_path', 'fold', 'condition']], on='image_path', how='left', validate='many_to_one')
    if not cached.fold.eq(0).all() or cached.duplicated(['image_path', 'perturbation']).any():
        raise ValueError('The perturbation cache must contain distinct fold-0 images per condition.')
    accepted = cached[cached.quality_flag == 'ACCEPT'].copy()
    router = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=42, n_jobs=1)
    routed = router.fit(train[FEATURES].to_numpy(), train.crop).predict(accepted[FEATURES].to_numpy())
    for crop, (graph, h) in BEST_GRAPH.items():
        part = train[train.crop == crop]
        take = routed == crop
        for name, model in [('Original WL', WLKernelSVC(graph, h, probability=False)),
                            ('Revised WL + measurements', stress_classifier(probability=False))]:
            model.fit(part[FEATURES].to_numpy(), (part.condition != 'healthy').astype(int))
            accepted.loc[take, name] = model.predict(accepted.loc[take, FEATURES].to_numpy())
    clean = accepted[accepted.perturbation == 'clean'].set_index('image_path')
    rows = []
    for perturbation, part in accepted.groupby('perturbation'):
        matched = clean.loc[part.image_path]
        truth = (part.condition != 'healthy').to_numpy(int)
        for name in ('Original WL', 'Revised WL + measurements'):
            pred, base = part[name].to_numpy(int), matched[name].to_numpy(int)
            rows.append(dict(perturbation=perturbation, model=name, accepted=len(part),
                             rejected=int((cached.perturbation == perturbation).sum()) - len(part),
                             flip_rate=(pred != base).mean(), delta_macro_f1=f1_score(truth, pred, average='macro')
                             - f1_score(truth, base, average='macro'), **classification_metrics(truth, pred)))
    return pd.DataFrame(rows)


def run(development):
    df = data()
    if development:
        df = df[df.fold != 4]
    X, crops = df[FEATURES].to_numpy(), df.crop.to_numpy()
    y = (df.condition != 'healthy').to_numpy(int)
    results = []
    leaf_predictions = []
    for fold in sorted(df.fold.unique()):
        tr, te = (df.fold != fold).to_numpy(), (df.fold == fold).to_numpy()
        router = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=42, n_jobs=1)
        routed = router.fit(X[tr], crops[tr]).predict(X[te])
        predictions = {}
        costs = {}
        for crop in BEST_GRAPH:
            train = tr & (crops == crop)
            take = routed == crop
            if not development:
                for name, pred, fit_s, size, latency in fixed_predictions(X[train], y[train], X[te][take], crop):
                    predictions.setdefault(name, np.empty(te.sum(), dtype=int))[take] = pred
                    total = costs.setdefault(name, [0., 0, []])
                    total[0] += fit_s
                    total[1] += size
                    total[2].extend(latency)
                continue
            for candidate in development_kernels(X[train], y[train], X[te][take], crop):
                weights = 'balanced'
                if candidate.weight_power != 1:
                    weights = dict(enumerate((train.sum() / (2 * np.bincount(y[train]))) ** candidate.weight_power))
                svc = SVC(kernel='precomputed', C=candidate.C, class_weight=weights).fit(candidate.train_kernel, y[train])
                predictions.setdefault(candidate.name, np.empty(te.sum(), dtype=int))[take] = svc.predict(candidate.test_kernel)
        truth = y[te]
        for name, p in predictions.items():
            cost = {}
            if not development:
                fit_s, size, latency = costs[name]
                cost = dict(stage2_fit_s=fit_s, stage2_mib=size / 2**20, stage2_leaf_ms=np.median(latency))
            results.append(dict(fold=int(fold), model=name, leaves=len(p), **classification_metrics(truth, p),
                                routing_accuracy=(routed == crops[te]).mean(), **cost))
        if not development:
            leaf_predictions.append(pd.DataFrame({'image': df.loc[te, 'image_name'], 'fold': fold,
                                                  'crop': crops[te], 'routed_crop': routed, 'stressed': truth, **predictions}))
    if leaf_predictions:
        pd.concat(leaf_predictions).to_csv(HERE / 'results/wl_revision_predictions.csv', index=False)
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--development', action='store_true')
    mode.add_argument('--unrouted', action='store_true', help='Also benchmark the pooled and known-crop tasks.')
    mode.add_argument('--robustness', action='store_true', help='Compare full routed models on cached fold-0 perturbations.')
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        if args.robustness:
            stage, rows = 'robustness', evaluate_robustness()
        elif args.unrouted:
            stage, rows = 'unrouted', evaluate_unrouted()
        else:
            stage = 'development' if args.development else 'outer'
            rows = run(args.development)
    rows.to_csv(HERE / 'results' / f'wl_revision_{stage}.csv', index=False)
    if stage == 'outer':
        metrics = ['macro_f1', 'accuracy', 'healthy_recall', 'stressed_recall']
        summary = rows.groupby('model', sort=False)[metrics].mean().round(6).reset_index()
        target = ROOT / 'Application/src/plantcare/data/stress_revision.json'
        target.write_text(json.dumps(summary.to_dict('records'), indent=2) + '\n', encoding='utf-8')
    groups = ['crop', 'model'] if args.unrouted else ['model']
    print(rows.drop(columns=['fold', 'leaves'], errors='ignore').groupby(groups).mean(numeric_only=True).round(4).to_string())
