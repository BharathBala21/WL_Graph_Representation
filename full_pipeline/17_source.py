import os, time, json, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

# ── Parameters (must match NB14 and NB16 exactly) ──────────────────────────────
RANDOM_STATE  = 42
CORR_THRESH   = 0.70
KNN_K         = 3
N_FOLDS       = 5
SVM_C         = 1.0
SVM_KERNEL    = 'rbf'
CLASS_WEIGHT  = 'balanced'

RESULTS_DIR = 'results/continuous_graph_svm'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Configuration:")
print(f"  RANDOM_STATE  = {RANDOM_STATE}")
print(f"  CORR_THRESH   = {CORR_THRESH}")
print(f"  KNN_K         = {KNN_K}")
print(f"  N_FOLDS       = {N_FOLDS}")
print(f"  SVM_C         = {SVM_C}")
print(f"  SVM_KERNEL    = {SVM_KERNEL}")
print(f"  CLASS_WEIGHT  = {CLASS_WEIGHT}")
print(f"  RESULTS_DIR   = {RESULTS_DIR}")
print()
print("Representation: Continuous Graph-Derived (Rep-A, 190-dim)")
print("Source:         Notebook 16")
print("Baseline:       Flat RBF SVM from Notebook 14 (Macro-F1=0.4307)")

csv_path = 'olid_full_features.csv'
df_full  = pd.read_csv(csv_path)
df_accepted = df_full[df_full['quality_flag'] == 'ACCEPT'].copy().reset_index(drop=True)

meta_cols = ['image_path', 'image_name', 'class', 'crop', 'condition',
             'quality_flag', 'mask_area_ratio', 'failure_reason']
feature_cols = [c for c in df_accepted.columns if c not in meta_cols]

assert len(df_accepted) == 4387, f"Expected 4387, got {len(df_accepted)}"
assert len(feature_cols) == 38,  f"Expected 38 features, got {len(feature_cols)}"

# Load authoritative fold assignments (same as NB14)
df_folds    = pd.read_csv('cv_fold_assignments.csv')
df_accepted = df_accepted.merge(df_folds[['image_path', 'fold']], on='image_path', how='left')
assert df_accepted['fold'].isna().sum() == 0, "Missing fold assignments!"

print(f"Accepted images   : {len(df_accepted)}")
print(f"Classes           : {df_accepted['class'].nunique()}")
print(f"Features          : {len(feature_cols)}")
print(f"Fold distribution :")
print(df_accepted['fold'].value_counts().sort_index().to_string())

SHAPE_FEATURES         = ['area', 'perimeter', 'aspect_ratio', 'circularity', 'extent',
                           'solidity', 'convex_area', 'major_axis_length',
                           'minor_axis_length', 'eccentricity']
COLOR_RGB_FEATURES     = ['mean_R', 'std_R', 'mean_G', 'std_G', 'mean_B', 'std_B']
COLOR_HSV_FEATURES     = ['mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V']
COLOR_LAB_FEATURES     = ['mean_L', 'std_L', 'mean_a', 'std_a', 'mean_b', 'std_b']
DERIVED_COLOR_FEATURES = ['green_ratio', 'yellow_ratio']
TEXTURE_FEATURES       = ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation']
SPATIAL_FEATURES       = ['core_mean_H', 'core_yellow_ratio', 'edge_mean_H', 'edge_yellow_ratio']

FEATURE_FAMILIES = {
    'Shape':        SHAPE_FEATURES,
    'RGB':          COLOR_RGB_FEATURES,
    'HSV':          COLOR_HSV_FEATURES,
    'Lab':          COLOR_LAB_FEATURES,
    'DerivedColor': DERIVED_COLOR_FEATURES,
    'Texture':      TEXTURE_FEATURES,
    'Spatial':      SPATIAL_FEATURES,
}

all_fam_feats = [f for fam in FEATURE_FAMILIES.values() for f in fam]
assert sorted(all_fam_feats) == sorted(feature_cols), "Feature family mismatch!"


class GraphConstructor:
    def __init__(self, features):
        self.features   = list(features)
        self.base_graph = nx.Graph()
        self.base_graph.add_nodes_from(self.features)

    def fit(self, X_train): raise NotImplementedError


class CorrelationGraph(GraphConstructor):
    def __init__(self, features, threshold=0.70):
        super().__init__(features)
        self.threshold = threshold

    def fit(self, X_train):
        corr = X_train[self.features].corr(method='pearson')
        self.base_graph.clear_edges()
        n = len(self.features)
        for i in range(n):
            for j in range(i+1, n):
                r = corr.iloc[i, j]
                if abs(r) >= self.threshold:
                    self.base_graph.add_edge(self.features[i], self.features[j],
                                             weight=float(r), source='correlation')
        return self


class FeatureSimilarityKNNGraph(GraphConstructor):
    def __init__(self, features, k=3):
        super().__init__(features)
        self.k      = k
        self.scaler = StandardScaler()

    def fit(self, X_train):
        X_scaled = self.scaler.fit_transform(X_train[self.features])
        X_t      = X_scaled.T
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean')
        nn.fit(X_t)
        _, indices = nn.kneighbors(X_t)
        self.base_graph.clear_edges()
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:
                self.base_graph.add_edge(self.features[i], self.features[j],
                                         source='feature_similarity')
        return self


class FeatureFamilyGraph(GraphConstructor):
    def __init__(self, features, families):
        super().__init__(features)
        self.families = families

    def fit(self, X_train):
        self.base_graph.clear_edges()
        corr = X_train[self.features].corr().abs()
        for fam_feats in self.families.values():
            vf = [f for f in fam_feats if f in self.features]
            for i in range(len(vf)):
                for j in range(i+1, len(vf)):
                    self.base_graph.add_edge(vf[i], vf[j], source='family')
        fams = list(self.families.keys())
        for i in range(len(fams)):
            for j in range(i+1, len(fams)):
                fa = [f for f in self.families[fams[i]] if f in self.features]
                fb = [f for f in self.families[fams[j]] if f in self.features]
                if fa and fb:
                    sub = corr.loc[fa, fb]
                    mx  = sub.max().max()
                    if pd.notna(mx):
                        ri, ci = np.where(sub.values == mx)
                        self.base_graph.add_edge(fa[ri[0]], fb[ci[0]],
                                                 source='family_bridge', weight=float(mx))
        return self


class SpatialGraph(GraphConstructor):
    def __init__(self, features, spatial_features, corr_threshold=0.60):
        super().__init__(features)
        self.spatial_features = spatial_features
        self.corr_threshold   = corr_threshold

    def fit(self, X_train):
        self.base_graph.clear_edges()
        core_feats   = [f for f in self.spatial_features if f.startswith('core_')]
        edge_feats   = [f for f in self.spatial_features if f.startswith('edge_')]
        global_feats = [f for f in self.features if f not in self.spatial_features]
        corr         = X_train[self.features].corr().abs()
        for i in range(len(core_feats)):
            for j in range(i+1, len(core_feats)):
                self.base_graph.add_edge(core_feats[i], core_feats[j], source='spatial_core')
        for i in range(len(edge_feats)):
            for j in range(i+1, len(edge_feats)):
                self.base_graph.add_edge(edge_feats[i], edge_feats[j], source='spatial_edge')
        for cf in core_feats:
            ef = 'edge_' + cf.replace('core_', '')
            if ef in edge_feats:
                self.base_graph.add_edge(cf, ef, source='spatial_bridge')
        for gf in global_feats:
            for sf in self.spatial_features:
                if corr.loc[gf, sf] >= self.corr_threshold:
                    self.base_graph.add_edge(gf, sf, source='spatial_global_link',
                                             weight=float(corr.loc[gf, sf]))
        return self


class HybridGraph(GraphConstructor):
    def __init__(self, features, corr_threshold=0.80, knn_k=3,
                 families=None, spatial_features=None, spatial_threshold=0.70):
        super().__init__(features)
        self._corr_t = corr_threshold
        self._knn_k  = knn_k
        self._fams   = families or FEATURE_FAMILIES
        self._spatf  = spatial_features or SPATIAL_FEATURES
        self._spat_t = spatial_threshold

    def fit(self, X_train):
        self.base_graph.clear_edges()
        _corr    = CorrelationGraph(self.features, self._corr_t).fit(X_train)
        _fam     = FeatureFamilyGraph(self.features, self._fams).fit(X_train)
        _spatial = SpatialGraph(self.features, self._spatf, self._spat_t).fit(X_train)
        for sub, src in [(_corr, 'correlation'), (_fam, 'family'), (_spatial, 'spatial')]:
            for u, v, d in sub.base_graph.edges(data=True):
                if self.base_graph.has_edge(u, v):
                    cur = self.base_graph[u][v].get('source', '')
                    if src not in cur:
                        self.base_graph[u][v]['source'] = 'multiple'
                else:
                    self.base_graph.add_edge(u, v, source=src)
        return self


GRAPH_BUILDERS = {
    'correlation': lambda feats: CorrelationGraph(feats, CORR_THRESH),
    'knn':         lambda feats: FeatureSimilarityKNNGraph(feats, KNN_K),
    'family':      lambda feats: FeatureFamilyGraph(feats, FEATURE_FAMILIES),
    'spatial':     lambda feats: SpatialGraph(feats, SPATIAL_FEATURES),
    'hybrid':      lambda feats: HybridGraph(feats),
}

GRAPH_DISPLAY_NAMES = {
    'correlation': 'Correlation',
    'knn':         'Feature-Similarity kNN',
    'family':      'Feature-Family',
    'spatial':     'Spatial/Regional',
    'hybrid':      'Hybrid',
}

print("Graph constructors loaded: correlation, knn, family, spatial, hybrid")

class FoldSafeScaler:
    """Wraps StandardScaler to enforce fold-safe training-only fitting."""
    def __init__(self):
        self._scaler   = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, X_train):
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        arr = self._scaler.fit_transform(X_train.values)
        self.is_fitted = True
        self.columns_  = list(X_train.columns)
        self.mean_     = self._scaler.mean_.copy()
        self.scale_    = self._scaler.scale_.copy()
        return pd.DataFrame(arr, columns=self.columns_, index=X_train.index)

    def transform(self, X_val):
        if not self.is_fitted:
            raise ValueError("FoldSafeScaler must be fitted first.")
        if not isinstance(X_val, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        arr = self._scaler.transform(X_val.values)
        return pd.DataFrame(arr, columns=self.columns_, index=X_val.index)


def build_rep_A_vector(G, scaled_row, feature_cols):
    """
    Representation A: 190-dimensional continuous graph-derived feature vector.

    For each node v (one phenotypic feature):
        self_value    : standardised continuous feature value
        nbr_mean      : mean of neighbour values (0 if isolated)
        nbr_std       : std  of neighbour values (0 if isolated)
        nbr_sum       : sum  of neighbour values (0 if isolated)
        nbr_mad       : mean absolute deviation from neighbours (0 if isolated)

    Returns a flat numpy array of length 5 * 38 = 190.
    Source: Notebook 16, Section 8.
    """
    vals = []
    for node in feature_cols:
        v_val    = float(scaled_row[node])
        nbrs     = list(G.neighbors(node))
        nbr_vals = np.array([float(scaled_row[n]) for n in nbrs])

        nbr_mean = float(np.mean(nbr_vals))              if len(nbr_vals) > 0 else 0.0
        nbr_std  = float(np.std(nbr_vals))               if len(nbr_vals) > 0 else 0.0
        nbr_sum  = float(np.sum(nbr_vals))               if len(nbr_vals) > 0 else 0.0
        nbr_mad  = float(np.mean(np.abs(nbr_vals - v_val))) if len(nbr_vals) > 0 else 0.0

        vals.extend([v_val, nbr_mean, nbr_std, nbr_sum, nbr_mad])

    arr = np.array(vals, dtype=np.float32)
    assert not np.isnan(arr).any(), "NaN in Rep-A vector!"
    assert not np.isinf(arr).any(), "Inf in Rep-A vector!"
    return arr


def build_rep_A_matrix(G, X_scaled_df, feature_cols):
    """Build Rep-A matrix for a set of images. Returns (n, 190) float32 array."""
    rows = []
    for _, row in X_scaled_df.iterrows():
        rows.append(build_rep_A_vector(G, row, feature_cols))
    M = np.stack(rows, axis=0)
    assert M.shape[1] == 5 * len(feature_cols), f"Expected 190 cols, got {M.shape[1]}"
    assert not np.isnan(M).any(), "NaN in Rep-A matrix!"
    assert not np.isinf(M).any(), "Inf in Rep-A matrix!"
    return M


# Quick self-check
_test_G = CorrelationGraph(feature_cols, CORR_THRESH)
_test_G.fit(df_accepted.loc[df_accepted['fold'] != 0, feature_cols])
_test_sc = FoldSafeScaler()
_test_sc_df = _test_sc.fit_transform(df_accepted.loc[df_accepted['fold'] != 0, feature_cols])
_test_vec = build_rep_A_vector(_test_G.base_graph, _test_sc_df.iloc[0], feature_cols)
assert len(_test_vec) == 190, f"Expected 190, got {len(_test_vec)}"
print(f"Rep-A vector length : {len(_test_vec)}  (5 x 38 = 190)  OK")
print("NaN / Inf check     : PASS")

print("Sanity Check: Correlation + Rep-A + Fold 0")
print("=" * 55)

fold_id = 0
tr_idx  = df_accepted[df_accepted['fold'] != fold_id].index
va_idx  = df_accepted[df_accepted['fold'] == fold_id].index
y_tr    = df_accepted.loc[tr_idx, 'class'].values
y_va    = df_accepted.loc[va_idx, 'class'].values
X_tr    = df_accepted.loc[tr_idx, feature_cols]
X_va    = df_accepted.loc[va_idx, feature_cols]

# Fold-safe scaling
sc = FoldSafeScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_va_sc = sc.transform(X_va)

# Graph fitted on training only
ctor = CorrelationGraph(feature_cols, CORR_THRESH)
ctor.fit(X_tr)
G = ctor.base_graph

# Build Rep-A matrices
M_tr = build_rep_A_matrix(G, X_tr_sc, feature_cols)
M_va = build_rep_A_matrix(G, X_va_sc, feature_cols)

# Assertions
assert M_tr.shape == (len(tr_idx), 190), f"Train matrix shape wrong: {M_tr.shape}"
assert M_va.shape == (len(va_idx), 190), f"Val matrix shape wrong: {M_va.shape}"
assert M_tr.shape[0] == len(y_tr),       "Train sample count mismatch"
assert M_va.shape[0] == len(y_va),       "Val sample count mismatch"
assert not np.isnan(M_tr).any(),          "NaN in training matrix"
assert not np.isinf(M_tr).any(),          "Inf in training matrix"
assert not np.isnan(M_va).any(),          "NaN in validation matrix"
assert not np.isinf(M_va).any(),          "Inf in validation matrix"

print(f"  Train shape         : {M_tr.shape}  -- 190 dims: PASS")
print(f"  Val   shape         : {M_va.shape}  -- correct samples: PASS")
print(f"  NaN / Inf           : PASS")
print(f"  Scaler fitted on training only : PASS")
print(f"  Graph fitted on training only  : PASS")

# SVM train + predict
t0    = time.time()
svm_s = SVC(kernel=SVM_KERNEL, C=SVM_C, class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE)
svm_s.fit(M_tr, y_tr)
t_train = time.time() - t0

t0      = time.time()
y_pred  = svm_s.predict(M_va)
t_pred  = time.time() - t0

assert len(y_pred) == len(y_va), "Prediction count != validation count!"

acc   = accuracy_score(y_va, y_pred)
mf1   = f1_score(y_va, y_pred, average='macro', zero_division=0)

print(f"  SVM trains without error       : PASS")
print(f"  Prediction count matches val   : PASS  ({len(y_pred)} predictions)")
print(f"  Accuracy (fold 0)              : {acc:.4f}")
print(f"  Macro-F1 (fold 0)              : {mf1:.4f}")
print(f"  Train time                     : {t_train:.2f}s")
print(f"  Prediction time                : {t_pred:.2f}s")
print()
print("Sanity check PASSED. Proceeding to full experiment.")

def run_experiment(method_key, graph_key, fold_id,
                   df_accepted, feature_cols, CORR_THRESH, KNN_K,
                   SVM_C, SVM_KERNEL, CLASS_WEIGHT, RANDOM_STATE):
    """
    Run one fold of a single method.

    method_key : 'flat' or a key in GRAPH_BUILDERS
    graph_key  : display name for the graph (or 'flat')
    fold_id    : 0..4

    Returns dict with metrics and OOF predictions.
    """
    tr_idx = df_accepted[df_accepted['fold'] != fold_id].index
    va_idx = df_accepted[df_accepted['fold'] == fold_id].index
    y_tr   = df_accepted.loc[tr_idx, 'class'].values
    y_va   = df_accepted.loc[va_idx, 'class'].values
    X_tr   = df_accepted.loc[tr_idx, feature_cols]
    X_va   = df_accepted.loc[va_idx, feature_cols]
    img_va = df_accepted.loc[va_idx, 'image_path'].values
    crop_va = df_accepted.loc[va_idx, 'crop'].values
    cond_va = df_accepted.loc[va_idx, 'condition'].values

    # Fold-safe scaling
    sc      = FoldSafeScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_va_sc = sc.transform(X_va)

    if method_key == 'flat':
        M_tr = X_tr_sc.values.astype(np.float32)
        M_va = X_va_sc.values.astype(np.float32)
    else:
        ctor = GRAPH_BUILDERS[method_key](feature_cols)
        ctor.fit(X_tr)
        G    = ctor.base_graph
        M_tr = build_rep_A_matrix(G, X_tr_sc, feature_cols)
        M_va = build_rep_A_matrix(G, X_va_sc, feature_cols)

    # SVM
    svm = SVC(kernel=SVM_KERNEL, C=SVM_C, class_weight=CLASS_WEIGHT,
              random_state=RANDOM_STATE)
    t0       = time.time()
    svm.fit(M_tr, y_tr)
    t_train  = time.time() - t0
    t0       = time.time()
    y_pred   = svm.predict(M_va)
    t_pred   = time.time() - t0

    acc  = accuracy_score(y_va, y_pred)
    mf1  = f1_score(y_va, y_pred, average='macro',    zero_division=0)
    mp   = precision_score(y_va, y_pred, average='macro',    zero_division=0)
    mr   = recall_score(y_va, y_pred, average='macro',       zero_division=0)
    wf1  = f1_score(y_va, y_pred, average='weighted', zero_division=0)

    oof_rows = []
    for i in range(len(y_va)):
        oof_rows.append({
            'image_path':   img_va[i],
            'crop':         crop_va[i],
            'condition':    cond_va[i],
            'true_label':   y_va[i],
            'pred_label':   y_pred[i],
            'fold':         fold_id,
            'method':       graph_key,
        })

    return {
        'fold':            fold_id,
        'method':          graph_key,
        'accuracy':        acc,
        'macro_precision': mp,
        'macro_recall':    mr,
        'macro_f1':        mf1,
        'weighted_f1':     wf1,
        'train_time':      t_train,
        'pred_time':       t_pred,
        'n_train':         len(y_tr),
        'n_val':           len(y_va),
        'feature_dim':     M_tr.shape[1],
        'oof_rows':        oof_rows,
    }


# ── Method definitions ──────────────────────────────────────────────────────────
METHODS = [
    ('flat',        'Flat RBF SVM'),
    ('correlation', 'Rep-A + Correlation'),
    ('knn',         'Rep-A + kNN'),
    ('family',      'Rep-A + Family'),
    ('spatial',     'Rep-A + Spatial'),
    ('hybrid',      'Rep-A + Hybrid'),
]

fold_results = []
all_oof      = []
total_runs   = len(METHODS) * N_FOLDS
done         = 0

print(f"Running {total_runs} experiments ({len(METHODS)} methods x {N_FOLDS} folds)")
print("=" * 70)
print(f"{'Method':<25} {'Fold':>5}  {'Acc':>7}  {'MacroF1':>9}  {'Train(s)':>9}")
print("-" * 60)

for method_key, method_name in METHODS:
    for fold_id in range(N_FOLDS):
        res = run_experiment(
            method_key, method_name, fold_id,
            df_accepted, feature_cols,
            CORR_THRESH, KNN_K, SVM_C, SVM_KERNEL, CLASS_WEIGHT, RANDOM_STATE
        )
        oof_rows = res.pop('oof_rows')
        fold_results.append(res)
        all_oof.extend(oof_rows)
        done += 1

        print(f"  {method_name:<23} fold={fold_id}  "
              f"{res['accuracy']:>7.4f}  {res['macro_f1']:>9.4f}  "
              f"{res['train_time']:>9.2f}s")

print()
print(f"All {total_runs} experiments complete.")

df_fold = pd.DataFrame(fold_results)
df_fold.to_csv(f'{RESULTS_DIR}/continuous_graph_svm_fold_results.csv', index=False)

summary_rows = []
for method_name in [m[1] for m in METHODS]:
    sub = df_fold[df_fold['method'] == method_name]
    summary_rows.append({
        'task':                   '57-class',
        'method':                 method_name,
        'feature_dim':            int(sub['feature_dim'].iloc[0]),
        'mean_accuracy':          sub['accuracy'].mean(),
        'std_accuracy':           sub['accuracy'].std(),
        'mean_macro_precision':   sub['macro_precision'].mean(),
        'std_macro_precision':    sub['macro_precision'].std(),
        'mean_macro_recall':      sub['macro_recall'].mean(),
        'std_macro_recall':       sub['macro_recall'].std(),
        'mean_macro_f1':          sub['macro_f1'].mean(),
        'std_macro_f1':           sub['macro_f1'].std(),
        'mean_weighted_f1':       sub['weighted_f1'].mean(),
        'std_weighted_f1':        sub['weighted_f1'].std(),
        'mean_train_time':        sub['train_time'].mean(),
        'std_train_time':         sub['train_time'].std(),
        'mean_pred_time':         sub['pred_time'].mean(),
        'std_pred_time':          sub['pred_time'].std(),
        'folds_completed':        len(sub),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f'{RESULTS_DIR}/continuous_graph_svm_summary.csv', index=False)

print("Aggregated Summary:")
print("=" * 80)
print(f"{'Method':<25} {'Dim':>5}  {'Acc':>8}  {'MacroF1':>9}  {'WgtF1':>8}  {'Train(s)':>9}")
print("-" * 70)
for _, row in df_summary.iterrows():
    print(f"  {row['method']:<23} {int(row['feature_dim']):>5}  "
          f"{row['mean_accuracy']:>8.4f}  {row['mean_macro_f1']:>9.4f}  "
          f"{row['mean_weighted_f1']:>8.4f}  {row['mean_train_time']:>9.2f}s")
print()
print(f"Saved: {RESULTS_DIR}/continuous_graph_svm_fold_results.csv")
print(f"Saved: {RESULTS_DIR}/continuous_graph_svm_summary.csv")

df_oof = pd.DataFrame(all_oof)
df_oof.to_csv(f'{RESULTS_DIR}/continuous_graph_svm_oof_predictions.csv', index=False)

print(f"OOF prediction records: {len(df_oof)}")
print(f"  Methods: {df_oof['method'].nunique()}")
print(f"  Images per method: {len(df_oof) // df_oof['method'].nunique()}")
print(f"Saved: {RESULTS_DIR}/continuous_graph_svm_oof_predictions.csv")

per_class_rows = []
for method_name in [m[1] for m in METHODS]:
    sub_oof = df_oof[df_oof['method'] == method_name]
    report  = classification_report(
        sub_oof['true_label'], sub_oof['pred_label'],
        output_dict=True, zero_division=0
    )
    for cls, metrics in report.items():
        if cls in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        per_class_rows.append({
            'method':    method_name,
            'class':     cls,
            'precision': metrics['precision'],
            'recall':    metrics['recall'],
            'f1':        metrics['f1-score'],
            'support':   metrics['support'],
        })

df_per_class = pd.DataFrame(per_class_rows)
df_per_class.to_csv(f'{RESULTS_DIR}/per_class_results.csv', index=False)
print(f"Per-class results saved: {len(df_per_class)} rows")
print(f"  Classes: {df_per_class['class'].nunique()}")
print(f"  Methods: {df_per_class['method'].nunique()}")
print(f"Saved: {RESULTS_DIR}/per_class_results.csv")

crop_rows = []
for method_name in [m[1] for m in METHODS]:
    sub = df_oof[df_oof['method'] == method_name]
    for crop_name, grp in sub.groupby('crop'):
        acc  = accuracy_score(grp['true_label'], grp['pred_label'])
        mf1  = f1_score(grp['true_label'], grp['pred_label'], average='macro', zero_division=0)
        crop_rows.append({
            'method': method_name, 'crop': crop_name,
            'n_samples': len(grp), 'accuracy': acc, 'macro_f1': mf1
        })

df_crop = pd.DataFrame(crop_rows)
df_crop.to_csv(f'{RESULTS_DIR}/crop_wise_results.csv', index=False)
print(f"Crop-wise results saved ({df_crop['crop'].nunique()} crops x {df_crop['method'].nunique()} methods)")
print()
# Print crop summary for flat baseline vs best Rep-A
pivot = df_crop.pivot(index='crop', columns='method', values='macro_f1').round(4)
print(pivot.to_string())
print(f"\nSaved: {RESULTS_DIR}/crop_wise_results.csv")

cond_rows = []
for method_name in [m[1] for m in METHODS]:
    sub = df_oof[df_oof['method'] == method_name]
    for cond_name, grp in sub.groupby('condition'):
        acc  = accuracy_score(grp['true_label'], grp['pred_label'])
        mf1  = f1_score(grp['true_label'], grp['pred_label'], average='macro', zero_division=0)
        cond_rows.append({
            'method': method_name, 'condition': cond_name,
            'n_samples': len(grp), 'accuracy': acc, 'macro_f1': mf1
        })

df_cond = pd.DataFrame(cond_rows)
df_cond.to_csv(f'{RESULTS_DIR}/condition_wise_results.csv', index=False)
print(f"Condition-wise results saved ({df_cond['condition'].nunique()} conditions)")
print(f"Saved: {RESULTS_DIR}/condition_wise_results.csv")

# Flat baseline fold-level Macro-F1 values
flat_fold_f1 = {}
for _, row in df_fold[df_fold['method'] == 'Flat RBF SVM'].iterrows():
    flat_fold_f1[row['fold']] = row['macro_f1']

paired_rows = []
rep_a_methods = [m[1] for m in METHODS if m[0] != 'flat']

for method_name in rep_a_methods:
    sub    = df_fold[df_fold['method'] == method_name].sort_values('fold')
    deltas = []
    for _, row in sub.iterrows():
        delta = row['macro_f1'] - flat_fold_f1[row['fold']]
        deltas.append(delta)
        paired_rows.append({
            'method':            method_name,
            'fold':              row['fold'],
            'rep_a_macro_f1':    row['macro_f1'],
            'flat_macro_f1':     flat_fold_f1[row['fold']],
            'delta_macro_f1':    delta,
        })

df_paired = pd.DataFrame(paired_rows)
df_paired.to_csv(f'{RESULTS_DIR}/paired_fold_comparison.csv', index=False)

print("Paired fold comparison (Rep-A vs Flat baseline):")
print(f"{'Method':<25}  {'MeanDelta':>10}  {'StdDelta':>9}  {'Folds+':>7}  {'Folds-':>7}")
print("-" * 65)
for method_name in rep_a_methods:
    sub     = df_paired[df_paired['method'] == method_name]
    deltas  = sub['delta_macro_f1'].values
    pos     = (deltas > 0).sum()
    neg     = (deltas <= 0).sum()
    print(f"  {method_name:<23}  {deltas.mean():>10.4f}  {deltas.std():>9.4f}  "
          f"{pos:>7}  {neg:>7}")
print()
print(f"Saved: {RESULTS_DIR}/paired_fold_comparison.csv")

flat_row = df_summary[df_summary['method'] == 'Flat RBF SVM'].iloc[0]
flat_f1  = flat_row['mean_macro_f1']

print()
print("=" * 70)
print("CONTINUOUS GRAPH SVM — FINAL RESULTS")
print("=" * 70)
print()
print(f"  Task                : 57-class OLID-I crop-condition classification")
print(f"  Representation      : Continuous graph-derived (Rep-A, 190-dim)")
print(f"  Classifier          : SVC(kernel=rbf, C=1.0, class_weight=balanced)")
print(f"  CV folds            : {N_FOLDS} (same as NB14)")
print(f"  Primary metric      : Macro-F1")
print()
print(f"  {'Method':<25} {'Dim':>5}  {'Macro-F1':>10}  {'Delta vs Flat':>14}  {'Accuracy':>9}")
print(f"  {'-'*70}")

best_rep_a_f1   = -1
best_rep_a_name = ''
for _, row in df_summary.iterrows():
    delta = row['mean_macro_f1'] - flat_f1
    delta_str = f'{delta:+.4f}'
    print(f"  {row['method']:<25} {int(row['feature_dim']):>5}  "
          f"{row['mean_macro_f1']:>10.4f}  {delta_str:>14}  {row['mean_accuracy']:>9.4f}")
    if row['method'] != 'Flat RBF SVM' and row['mean_macro_f1'] > best_rep_a_f1:
        best_rep_a_f1   = row['mean_macro_f1']
        best_rep_a_name = row['method']

print()
print(f"  Flat RBF Macro-F1                  : {flat_f1:.4f}")
print(f"  Highest-performing continuous graph-derived representation:")
print(f"    {best_rep_a_name} — Macro-F1 = {best_rep_a_f1:.4f}")
delta_best = best_rep_a_f1 - flat_f1
print(f"    Delta vs flat baseline = {delta_best:+.4f}")
print()
print("  NOTE: 'highest-performing continuous graph-derived representation'")
print("  refers to the Rep-A variant with the highest observed mean Macro-F1.")
print("  This does NOT constitute proven superiority over the flat baseline.")
print("  Statistical significance has not been evaluated here.")
print()
print("=" * 70)
print("REPRESENTATION STATUS")
print("=" * 70)
print()
print("  Flat continuous RBF SVM    : VALIDATED (reference baseline)")
print("  Representation A + 5 graphs: EVALUATED (continuous graph-derived)")
print("  Representation C           : NOT RUN   (exploratory; out of scope)")
print()
print("STOP. Do not create Notebook 18.")

config = {
    'notebook':          '17_OLID_Continuous_Graph_SVM',
    'representation':    'Rep-A continuous graph-derived (190-dim)',
    'source_notebook':   '16_OLID_Attribute_Aware_Graph_Representation',
    'task':              '57-class crop-condition classification',
    'n_accepted':        4387,
    'n_features':        38,
    'rep_a_dim':         190,
    'n_classes':         57,
    'n_folds':           N_FOLDS,
    'fold_source':       'cv_fold_assignments.csv (same as NB14)',
    'svm_kernel':        SVM_KERNEL,
    'svm_C':             SVM_C,
    'class_weight':      CLASS_WEIGHT,
    'corr_threshold':    CORR_THRESH,
    'knn_k':             KNN_K,
    'graph_types':       list(GRAPH_DISPLAY_NAMES.values()),
    'methods_run':       [m[1] for m in METHODS],
    'rep_c_run':         False,
    'hyperparameter_tuning': False,
    'new_cv_splits':     False,
}
with open(f'{RESULTS_DIR}/experiment_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f"Saved: {RESULTS_DIR}/experiment_config.json")
