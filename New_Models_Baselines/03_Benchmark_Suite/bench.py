"""Shared protocol for the multi-dimensional baseline benchmark.

Every model is run on the same data, the same folds and the same metrics as the research
notebooks, so the numbers are directly comparable:

  57-class  : crop x condition, fixed folds from full_pipeline/cv_fold_assignments.csv (NB13/14/18)
  level1    : healthy vs stressed over all crops, same fixed folds (NB21)
  level2    : healthy vs stressed inside each crop, StratifiedKFold(5, shuffle, seed 42) on the
              crop's binary label (NB22)

WL + SVM uses the backend engine (Application/backend/app/wl.py), which tests/test_wl_equivalence.py
checks against the research implementation. The WL setting for each task is the best one found in
the research (NB14 / NB21 / NB22); baselines use fixed, documented hyperparameters (no tuning),
as in NB18.

Fairness rules
  * Scaling, discretisation and graph building are fitted on the training fold only.
  * All models train on ONE CPU core (n_jobs=1), so training times are comparable.
  * Latency = model only (features already extracted). Feature extraction from the photo is the
    same for every feature-based model and is measured separately (see run_benchmark.py).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import tracemalloc
import warnings

# LightGBM must be imported before scikit-learn: on Windows, loading sklearn.neighbors first makes
# LightGBM crash with an access violation (clashing OpenMP runtimes).
from lightgbm import LGBMClassifier  # noqa: I001
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'Application', 'backend'))

from app.features import FEATURES  # noqa: E402
from app.wl import GRAPH_NAMES, WLFeaturizer, cosine_kernel  # noqa: E402

FEAT_CSV = os.path.join(ROOT, 'full_pipeline', 'olid_full_features.csv')
FOLD_CSV = os.path.join(ROOT, 'full_pipeline', 'cv_fold_assignments.csv')
RESULTS = os.path.join(HERE, 'results')
FIGURES = os.path.join(HERE, 'figures')
SEED = 42

# best WL settings from the research notebooks
WL_57 = ('knn', 2)            # NB14
WL_LEVEL1 = ('hybrid', 3)     # NB21
WL_LEVEL2 = {                 # NB22
    'ash_gourd': ('family', 2), 'bitter_gourd': ('knn', 3), 'bottle_gourd': ('hybrid', 3),
    'cucumber': ('family', 2), 'eggplant': ('hybrid', 3), 'ridge_gourd': ('hybrid', 1),
    'snake_gourd': ('hybrid', 3), 'tomato': ('correlation', 2),
}


# ---------------------------------------------------------------- WL + SVM estimator
class WLSVM(BaseEstimator, ClassifierMixin):
    """WL subtree kernel (cosine-normalised) + SVM, C=1, class-balanced, as in NB14/NB21/NB22."""

    def __init__(self, graph='hybrid', h=2, C=1.0):
        self.graph, self.h, self.C = graph, h, C

    def fit(self, X, y):
        self.feat_ = WLFeaturizer(self.graph, self.h).fit(X)
        self.P_ = self.feat_.transform(X)
        self.svc_ = SVC(kernel='precomputed', C=self.C, class_weight='balanced', random_state=SEED)
        self.svc_.fit(cosine_kernel(self.P_), y)
        self.classes_ = self.svc_.classes_
        return self

    def _k(self, X):
        return cosine_kernel(self.P_, self.feat_.transform(X))

    def predict(self, X):
        return self.svc_.predict(self._k(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._k(X))


# ---------------------------------------------------------------- model zoo
def models(task: str, crop: str | None = None) -> dict:
    """Fresh, unfitted models with fixed hyperparameters. All single-core."""
    if task == '57-class':
        graph, h = WL_57
    elif task == 'level1':
        graph, h = WL_LEVEL1
    else:
        graph, h = WL_LEVEL2[crop]
    scaled = lambda m: make_pipeline(StandardScaler(), m)
    return {
        f'WL + SVM ({GRAPH_NAMES[graph]}, h={h})': WLSVM(graph, h),
        'Logistic Regression': scaled(LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)),
        'Linear SVM': scaled(SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=SEED)),
        'RBF SVM': scaled(SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=SEED)),
        'k-NN (k=5)': scaled(KNeighborsClassifier(n_neighbors=5)),
        'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                                random_state=SEED, n_jobs=1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=1),
        'Gradient Boosting (HistGB)': HistGradientBoostingClassifier(class_weight='balanced', random_state=SEED),
        'MLP (128-64)': scaled(MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=500,
                                             random_state=SEED)),
        # classic baselines added in round 2
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=SEED),
        'Naive Bayes (Gaussian)': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, algorithm='SAMME', random_state=SEED),
        'XGBoost': Encoded(XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method='hist',
                                         n_jobs=1, random_state=SEED, verbosity=0), balance=True),
        'LightGBM': Encoded(LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight='balanced', n_jobs=1,
                                           random_state=SEED, verbose=-1)),
    }


ROUND2 = ['Decision Tree', 'Naive Bayes (Gaussian)', 'AdaBoost', 'XGBoost', 'LightGBM']


class Encoded(BaseEstimator, ClassifierMixin):
    """Lets XGBoost/LightGBM take string labels; optional class-balanced sample weights (like class_weight)."""

    def __init__(self, model=None, balance=False):
        self.model, self.balance = model, balance

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight
        self.enc_ = LabelEncoder().fit(y)
        self.classes_ = self.enc_.classes_
        self.model_ = clone(self.model)
        w = compute_sample_weight('balanced', y) if self.balance else None
        self.model_.fit(X, self.enc_.transform(y), sample_weight=w)
        return self

    def predict(self, X):
        return self.enc_.inverse_transform(np.asarray(self.model_.predict(X)).astype(int).ravel())

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# ---------------------------------------------------------------- tuned (nested CV) models
def tuned_models(task: str, crop: str | None = None) -> dict:
    """Hyperparameters chosen by an inner 3-fold CV on the training fold only (nested CV), macro-F1.

    Small, documented grids. WL + SVM is tuned the same way over graph type x WL depth, which removes
    the advantage of using the best research setting found on all the data.
    """
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    grid = lambda est, params: GridSearchCV(est, params, scoring='f1_macro', cv=inner, n_jobs=1, error_score=0.0)
    scaled = lambda m: make_pipeline(StandardScaler(), m)
    out = {
        'RBF SVM (tuned)': grid(scaled(SVC(kernel='rbf', class_weight='balanced', random_state=SEED)),
                                {'svc__C': [0.3, 1, 3, 10, 30], 'svc__gamma': ['scale', 0.01, 0.1]}),
        'Logistic Regression (tuned)': grid(scaled(LogisticRegression(max_iter=2000, class_weight='balanced',
                                                                      random_state=SEED)),
                                            {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}),
        'Random Forest (tuned)': grid(RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                                             random_state=SEED, n_jobs=1),
                                      {'max_features': ['sqrt', 0.5], 'min_samples_leaf': [1, 3, 5]}),
        'Extra Trees (tuned)': grid(ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=SEED,
                                                         n_jobs=1),
                                    {'max_features': ['sqrt', 0.5], 'min_samples_leaf': [1, 3, 5]}),
        'Gradient Boosting (HistGB, tuned)': grid(HistGradientBoostingClassifier(class_weight='balanced',
                                                                                 random_state=SEED),
                                                  {'learning_rate': [0.03, 0.1], 'max_leaf_nodes': [15, 31],
                                                   'l2_regularization': [0.0, 1.0]}),
    }
    if task == 'level2':
        out['WL + SVM (tuned)'] = grid(WLSVM(), {'graph': list(GRAPH_NAMES), 'h': [1, 2, 3]})
    return out


def is_wl(name: str) -> bool:
    return name.startswith('WL + SVM') and 'tuned' not in name


def family(name: str) -> str:
    """Model name used in reports (the per-task WL setting is folded into one 'WL + SVM' row)."""
    return 'WL + SVM' if is_wl(name) else name


# ---------------------------------------------------------------- data and splits
def load():
    df = pd.read_csv(FEAT_CSV)
    df = df[df.quality_flag == 'ACCEPT'].reset_index(drop=True)
    folds = pd.read_csv(FOLD_CSV)[['image_path', 'fold']]
    df = df.merge(folds, on='image_path', how='left')
    assert len(df) == 4387 and df.fold.notna().all() and len(FEATURES) == 38
    df['stressed'] = (df.condition != 'healthy').astype(int)
    return df


def split_indices(df: pd.DataFrame, task: str, crop: str | None = None):
    """Yields (fold, train_rows, test_rows, y) with row positions into df, so any representation of the
    same leaves (38 measurements, CNN embeddings, ...) gets exactly the same splits."""
    if task in ('57-class', 'level1'):
        y = df['class'].values if task == '57-class' else df.stressed.values
        for f in sorted(df.fold.unique()):
            yield int(f), np.where(df.fold != f)[0], np.where(df.fold == f)[0], y
    else:
        rows = np.where(df.crop == crop)[0]
        y = df.stressed.values
        yc = y[rows]
        n = min(5, int(np.bincount(yc).min()))
        for f, (tr, te) in enumerate(StratifiedKFold(n, shuffle=True, random_state=SEED).split(rows, yc)):
            yield f, rows[tr], rows[te], y


def splits(df: pd.DataFrame, task: str, crop: str | None = None, X: np.ndarray | None = None):
    """Yields (fold, X_train, y_train, X_test, y_test). X defaults to the 38 research measurements."""
    X = df[FEATURES].values.astype(float) if X is None else X
    for f, tr, te, y in split_indices(df, task, crop):
        yield f, X[tr], y[tr], X[te], y[te]


# ---------------------------------------------------------------- measuring one fit
def scores(model, X):
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:  # noqa: BLE001  (e.g. SVC without probability)
            pass
    return model.decision_function(X)


def quality(y, pred, model=None, X=None) -> dict:
    out = {'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, average='macro'),
           'weighted_f1': f1_score(y, pred, average='weighted'), 'balanced_accuracy': balanced_accuracy_score(y, pred)}
    if len(np.unique(y)) == 2 and model is not None:
        s = scores(model, X)
        out.update({'roc_auc': roc_auc_score(y, s), 'pr_auc_healthy': average_precision_score(1 - y, -s),
                    'sensitivity_stressed': recall_score(y, pred, pos_label=1),
                    'specificity_healthy': recall_score(y, pred, pos_label=0)})
    return out


def latency_ms(model, X, reps=30) -> tuple[float, float]:
    """(single-leaf median ms, batch ms per leaf) after a warm-up call."""
    one = X[:1]
    model.predict(one)
    single = []
    for _ in range(reps):
        t = time.perf_counter()
        model.predict(one)
        single.append(time.perf_counter() - t)
    t = time.perf_counter()
    model.predict(X)
    batch = (time.perf_counter() - t) / len(X)
    return float(np.median(single) * 1000), float(batch * 1000)


def run_fold(model, Xtr, ytr, Xte, yte, measure_memory=False) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if measure_memory:
            tracemalloc.start()
        t = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t
        peak = None
        if measure_memory:
            peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
            tracemalloc.stop()
        pred = model.predict(Xte)
        row = quality(yte, pred, model, Xte)
        single, batch = latency_ms(model, Xte)
    row.update({'train_s': fit_s, 'single_leaf_ms': single, 'batch_ms_per_leaf': batch,
                'model_kb': len(pickle.dumps(model)) / 1024, 'n_train': len(ytr), 'n_test': len(yte)})
    if peak is not None:
        row['train_peak_mb'] = peak
    return row


def save(df: pd.DataFrame, name: str):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    df.to_csv(path, index=False)
    print('saved', os.path.relpath(path, ROOT), f'({len(df)} rows)')
