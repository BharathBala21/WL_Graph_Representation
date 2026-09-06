"""
NB21 – OLID Level 1: Binary Stress Detection via WL Graph Kernel + SVM
=======================================================================
Research Question:
  "Does simplifying the OLID-I prediction task from 57-class joint
   crop-condition classification to binary stress detection improve the
   effectiveness of Weisfeiler–Lehman graph kernels, and does graph
   construction still influence performance?"

Pipeline Architecture:
  OLID-I image -> preprocessing/segmentation -> 38 handcrafted features
  -> graph construction (5 types) -> WL transformation (h=0..3)
  -> WL kernel -> SVM -> binary stress prediction

Flat baselines (RBF SVM, RF) are included for comparison only.
The primary experiment is the graph/WL architecture.

Author: Final Year Project – WL Graph Representation
"""

# ============================================================
# SECTION 0: Imports & Configuration
# ============================================================
import sys, io
# Force UTF-8 output on Windows to handle special characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
import scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore')

# -- Reproducibility -----------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -- WL / Graph parameters (same as NB12/NB13/NB14) ----------
N_BINS      = 3
BIN_LABELS  = ['Low', 'Medium', 'High']
CORR_THRESH = 0.70
KNN_K       = 3
WL_HEIGHTS  = [0, 1, 2, 3]
N_FOLDS     = 5

# -- SVM (same as NB14) ---------------------------------------
SVM_C            = 1.0
SVM_CLASS_WEIGHT = 'balanced'

# -- Output directories ----------------------------------------
RESULTS_DIR = 'results/level1_binary_wl'
PLOTS_DIR   = f'{RESULTS_DIR}/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 65)
print("NB21  Level 1: Binary Stress Detection – WL Graph/SVM")
print("=" * 65)
print(f"WL heights    : {WL_HEIGHTS}")
print(f"Folds         : {N_FOLDS}")
print(f"Corr threshold: {CORR_THRESH}")
print(f"kNN k         : {KNN_K}")
print(f"Results dir   : {RESULTS_DIR}")

# ============================================================
# SECTION 1: Data Loading & Verification
# ============================================================
csv_path    = '../olid_full_features.csv'
df_full     = pd.read_csv(csv_path)
df_accepted = df_full[df_full['quality_flag'] == 'ACCEPT'].copy().reset_index(drop=True)

meta_cols = ['image_path', 'image_name', 'class', 'crop', 'condition',
             'quality_flag', 'mask_area_ratio', 'failure_reason']
feature_cols = [c for c in df_accepted.columns if c not in meta_cols]

assert len(df_accepted) == 4387, f"Expected 4387 accepted, got {len(df_accepted)}"
assert len(feature_cols) == 38,  f"Expected 38 features, got {len(feature_cols)}"
print(f"\nAccepted images : {len(df_accepted)}")
print(f"Feature columns : {len(feature_cols)}")

# -- Load fold assignments (same as NB13/NB14) ---------------
df_folds    = pd.read_csv('../cv_fold_assignments.csv')
df_accepted = df_accepted.merge(df_folds[['image_path', 'fold']],
                                on='image_path', how='left')
assert df_accepted['fold'].isna().sum() == 0, "Missing fold assignments!"
print("Fold assignments: OK")

# -- Feature families (same as NB12) -------------------------
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

def get_feature_group(feat):
    for grp, feats in FEATURE_FAMILIES.items():
        if feat in feats:
            return grp
    return 'Unknown'

all_family_features = [f for fam in FEATURE_FAMILIES.values() for f in fam]
assert sorted(all_family_features) == sorted(feature_cols), "Feature family mismatch!"
print("Feature family assignment: OK")

# ============================================================
# SECTION 2: Condition -> Stressed / Non-Stressed Mapping
# ============================================================
print("\n" + "=" * 65)
print("SECTION 2: Binary Stress Label Construction")
print("=" * 65)

cond_counts = df_accepted['condition'].value_counts()
print(f"\nAll unique conditions ({len(cond_counts)} total):")
print(cond_counts.to_string())

# -- Scientifically justified mapping -------------------------
# Justification:
#   'healthy' is the only condition explicitly representing a non-stressed
#   control state. All remaining conditions are documented plant stresses:
#   - Diseases: DM (Downy Mildew), PM (Powdery Mildew), EB (Early Blight),
#               FB (Fusarium/other Blight), LS (Leaf Spot), LM (Leaf Miner),
#               PLEI (Pleiotropy)
#   - Pests/mites: MIT (Mite infestation), JAS (Jasmonates/mite damage)
#   - Nutrient deficiencies: N, K, Mg, N_K, N_Mg, K_Mg (Macronutrient)
#   - Compound stresses: JAS_MIT, MIT_EB, N_K, N_Mg, K_Mg, PLEI_IEM,
#                        PLEI_MIT, PC
#   All non-healthy conditions are biologically classified as STRESSED.
NON_STRESSED_CONDITIONS = {'healthy'}

def map_to_binary_stress(condition):
    """0 = NON-STRESSED (healthy), 1 = STRESSED (any pathology/deficiency)."""
    return 0 if condition in NON_STRESSED_CONDITIONS else 1

df_accepted['binary_label'] = df_accepted['condition'].apply(map_to_binary_stress)
assert df_accepted['binary_label'].isna().sum() == 0, "Missing binary labels!"

# -- Save mapping CSV ------------------------------------------
mapping_rows = []
for cond, count in cond_counts.items():
    bl = map_to_binary_stress(cond)
    if bl == 0:
        just = 'Explicitly labeled healthy control; no documented stress'
    else:
        just = 'Known pathology, nutrient deficiency, pest or compound stress'
    mapping_rows.append({
        'condition':           cond,
        'binary_stress_label': bl,
        'binary_name':         'NON-STRESSED' if bl == 0 else 'STRESSED',
        'sample_count':        count,
        'justification':       just,
    })

df_mapping = pd.DataFrame(mapping_rows).sort_values('binary_stress_label')
df_mapping.to_csv(f'{RESULTS_DIR}/condition_to_stress_mapping.csv', index=False)
print(f"\nMapping saved -> condition_to_stress_mapping.csv")
print(df_mapping[['condition', 'binary_name', 'sample_count']].to_string(index=False))

n_stress     = (df_accepted['binary_label'] == 1).sum()
n_nonstress  = (df_accepted['binary_label'] == 0).sum()
print(f"\nSTRESSED     : {n_stress}  ({n_stress/len(df_accepted)*100:.1f}%)")
print(f"NON-STRESSED : {n_nonstress}  ({n_nonstress/len(df_accepted)*100:.1f}%)")
print(f"Imbalance    : {n_stress/n_nonstress:.2f}:1")

# -- Verify every accepted sample has exactly one binary label -
assert df_accepted['binary_label'].isin([0, 1]).all(), "Not all labels are 0 or 1!"
print("Label integrity check: PASS (every sample has exactly one binary label)")

# ============================================================
# SECTION 3: WL Engine Components
#   Copied verbatim from NB12 (validated WL implementation).
#   DO NOT MODIFY graph topology or WL algorithm here.
# ============================================================

# -- Fold-safe discretizer -------------------------------------
class FoldSafeDiscretizer:
    """Quantile KBinsDiscretizer fitted on training data only."""
    def __init__(self, n_bins=3):
        self.disc      = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                                          strategy='quantile', subsample=None)
        self.labels    = ['Low', 'Medium', 'High']
        self.is_fitted = False

    def fit(self, X_train):
        if isinstance(X_train, pd.DataFrame):
            self.disc.fit(X_train.values)
        else:
            self.disc.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("FoldSafeDiscretizer must be fitted first.")
        cols  = list(X.columns) if isinstance(X, pd.DataFrame) else None
        idx   = X.index if isinstance(X, pd.DataFrame) else None
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_int = self.disc.transform(X_arr).astype(int)
        label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        df_lbl = pd.DataFrame(X_int, columns=cols, index=idx)
        for col in df_lbl.columns:
            df_lbl[col] = df_lbl[col].map(label_map)
        assert df_lbl.isna().sum().sum() == 0, "NaN after discretization!"
        return df_lbl


def fit_discretizer(X_train):
    d = FoldSafeDiscretizer(N_BINS)
    d.fit(X_train)
    return d

def transform_discretizer(X, disc):
    return disc.transform(X)


# -- Graph constructors (verbatim from NB12) -------------------
class GraphConstructor:
    def __init__(self, features):
        self.features   = list(features)
        self.base_graph = nx.Graph()
        self.base_graph.add_nodes_from(self.features)

    def fit(self, X_train):
        raise NotImplementedError

    def transform_single(self, label_row):
        G = self.base_graph.copy()
        for node in self.features:
            G.nodes[node]['label'] = f"{node}::{label_row[node]}"
        return G

    def transform(self, df_labels):
        graphs = []
        for _, row in df_labels.iterrows():
            g = self.transform_single(row)
            if g.number_of_edges() == 0:
                warnings.warn("Zero-edge graph detected.", RuntimeWarning)
            graphs.append(g)
        return graphs


class CorrelationGraph(GraphConstructor):
    def __init__(self, features, threshold=0.70):
        super().__init__(features)
        self.threshold = threshold

    def fit(self, X_train):
        corr = X_train[self.features].corr(method='pearson')
        self.base_graph.clear_edges()
        n = len(self.features)
        for i in range(n):
            for j in range(i + 1, n):
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
                for j in range(i + 1, len(vf)):
                    self.base_graph.add_edge(vf[i], vf[j], source='family')
        fams = list(self.families.keys())
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                fa = [f for f in self.families[fams[i]] if f in self.features]
                fb = [f for f in self.families[fams[j]] if f in self.features]
                if fa and fb:
                    sub = corr.loc[fa, fb]
                    mx  = sub.max().max()
                    if pd.notna(mx):
                        ri, ci = np.where(sub.values == mx)
                        self.base_graph.add_edge(fa[ri[0]], fb[ci[0]],
                                                 source='family_bridge',
                                                 weight=float(mx))
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
            for j in range(i + 1, len(core_feats)):
                self.base_graph.add_edge(core_feats[i], core_feats[j],
                                         source='spatial_core')
        for i in range(len(edge_feats)):
            for j in range(i + 1, len(edge_feats)):
                self.base_graph.add_edge(edge_feats[i], edge_feats[j],
                                         source='spatial_edge')
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

print("\nGraph constructors loaded:", list(GRAPH_DISPLAY_NAMES.values()))


# -- WL Label Mapper (verbatim from NB12) ---------------------
class WLLabelMapper:
    """
    Inductive WL label compression.
    Training: map_training_signature() — registers new signatures.
    Inference: map_inference_signature() — returns OOV token for unseen.
    OOV tokens contribute zero to the feature vector and do NOT alter
    the training vocabulary.
    """
    def __init__(self):
        self._maps    = {}
        self._counter = {}
        self._oov     = {}

    def _key(self, sig):
        label, neighbors = sig
        return f"{label}|{','.join(sorted(neighbors))}"

    def map_training_signature(self, iteration, sig):
        if iteration not in self._maps:
            self._maps[iteration]    = {}
            self._counter[iteration] = 0
            self._oov[iteration]     = set()
        k = self._key(sig)
        if k not in self._maps[iteration]:
            self._maps[iteration][k] = f"wl_{iteration}_{self._counter[iteration]}"
            self._counter[iteration] += 1
        return self._maps[iteration][k]

    def map_inference_signature(self, iteration, sig):
        if iteration not in self._maps:
            return f"__OOV__{iteration}"
        k = self._key(sig)
        if k not in self._maps[iteration]:
            self._oov[iteration].add(k)
            return f"__OOV__{iteration}"
        return self._maps[iteration][k]

    def oov_count(self, iteration):
        return len(self._oov.get(iteration, set()))


# -- WL Subtree Feature Function (verbatim from NB12) ---------
def wl_subtree_features(graphs_train, graphs_val, max_height=3, mapper=None):
    """
    Compute WL subtree feature matrices for training and validation graphs.
    Returns: Phi_train, Phi_val, feature_index, mapper
    """
    if mapper is None:
        mapper = WLLabelMapper()

    train_node_labels = [
        {n: G.nodes[n]['label'] for n in G.nodes()} for G in graphs_train
    ]
    val_node_labels = [
        {n: G.nodes[n]['label'] for n in G.nodes()} for G in graphs_val
    ]

    train_label_bags = [Counter() for _ in graphs_train]
    val_label_bags   = [Counter() for _ in graphs_val]

    def _compute_new_labels(graphs, node_labels, iteration, is_training):
        new_node_labels = []
        for g_idx, G in enumerate(graphs):
            cur_labels = node_labels[g_idx]
            new_labels = {}
            for node in G.nodes():
                nbr_labels = sorted([cur_labels[n] for n in G.neighbors(node)])
                sig = (cur_labels[node], tuple(nbr_labels))
                if is_training:
                    compressed = mapper.map_training_signature(iteration, sig)
                else:
                    compressed = mapper.map_inference_signature(iteration, sig)
                new_labels[node] = compressed
            new_node_labels.append(new_labels)
        return new_node_labels

    # h=0: initial node labels
    for g_idx in range(len(graphs_train)):
        for lbl in train_node_labels[g_idx].values():
            train_label_bags[g_idx][lbl] += 1
    for g_idx in range(len(graphs_val)):
        for lbl in val_node_labels[g_idx].values():
            val_label_bags[g_idx][lbl] += 1

    # h=1 … max_height
    for h in range(1, max_height + 1):
        train_node_labels = _compute_new_labels(
            graphs_train, train_node_labels, iteration=h - 1, is_training=True)
        val_node_labels = _compute_new_labels(
            graphs_val, val_node_labels, iteration=h - 1, is_training=False)
        for g_idx in range(len(graphs_train)):
            for lbl in train_node_labels[g_idx].values():
                train_label_bags[g_idx][lbl] += 1
        for g_idx in range(len(graphs_val)):
            for lbl in val_node_labels[g_idx].values():
                if not lbl.startswith('__OOV__'):
                    val_label_bags[g_idx][lbl] += 1

    # Build feature index from training only
    all_train_labels = set()
    for bag in train_label_bags:
        all_train_labels.update(bag.keys())
    feature_index = {lbl: idx for idx, lbl in enumerate(sorted(all_train_labels))}
    n_features    = len(feature_index)

    # Build sparse matrices
    def _bags_to_sparse(bags, feat_idx, n_feat):
        rows, cols, vals = [], [], []
        for g_idx, bag in enumerate(bags):
            for lbl, cnt in bag.items():
                if lbl in feat_idx:
                    rows.append(g_idx)
                    cols.append(feat_idx[lbl])
                    vals.append(cnt)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(bags), n_feat))

    Phi_train = _bags_to_sparse(train_label_bags, feature_index, n_features)
    Phi_val   = _bags_to_sparse(val_label_bags,   feature_index, n_features)

    return Phi_train, Phi_val, feature_index, mapper


# -- WL Kernel construction ------------------------------------
def build_kernel_matrix(Phi_train, Phi_val=None):
    """
    Compute the normalized WL kernel matrix.
    K(i,j) = Phi_i · Phi_j / sqrt(||Phi_i||^2 · ||Phi_j||^2)
    K_train: (n_train, n_train)
    K_val:   (n_val, n_train)  — rows=val, cols=train (for SVM predict)
    """
    Phi_tr = Phi_train.toarray().astype(np.float64)
    norm_tr = np.linalg.norm(Phi_tr, axis=1, keepdims=True)
    norm_tr[norm_tr == 0] = 1.0
    Phi_tr_n = Phi_tr / norm_tr
    K_train  = Phi_tr_n @ Phi_tr_n.T

    if Phi_val is None:
        return K_train, None

    Phi_va = Phi_val.toarray().astype(np.float64)
    norm_va = np.linalg.norm(Phi_va, axis=1, keepdims=True)
    norm_va[norm_va == 0] = 1.0
    Phi_va_n = Phi_va / norm_va
    K_val    = Phi_va_n @ Phi_tr_n.T   # (n_val, n_train)

    return K_train, K_val

print("WL engine components loaded (verbatim from NB12).")

# ============================================================
# SECTION 4: 5-Fold CV – WL Kernel + SVM
#   Strict inductive protocol (no validation leakage).
# ============================================================
print("\n" + "=" * 65)
print("SECTION 4: WL Kernel + SVM — 5-Fold CV (20 configurations)")
print("=" * 65)
print("Fold protocol (per fold):")
print("  1. Build graph topology on TRAINING data only")
print("  2. Fit WL label mapping on TRAINING graphs only")
print("  3. Construct training kernel (n_train × n_train)")
print("  4. Construct validation kernel (n_val × n_train)")
print("  5. Train SVM on training kernel")
print("  6. Predict validation fold")
print("  7. Compute metrics")
print()

fold_results_wl = []      # one row per (graph_type, height, fold)
oof_preds_wl    = []      # one row per sample × (graph_type, height)

start_total = time.time()

for gt_key in GRAPH_BUILDERS:
    gt_name = GRAPH_DISPLAY_NAMES[gt_key]
    for h in WL_HEIGHTS:
        fold_oof_rows = []     # accumulate per-sample predictions for this config
        t0 = time.time()

        for fold_id in range(N_FOLDS):
            tr_idx = df_accepted[df_accepted['fold'] != fold_id].index
            va_idx = df_accepted[df_accepted['fold'] == fold_id].index

            X_tr = df_accepted.loc[tr_idx, feature_cols]
            X_va = df_accepted.loc[va_idx, feature_cols]
            y_tr = df_accepted.loc[tr_idx, 'binary_label'].values
            y_va = df_accepted.loc[va_idx, 'binary_label'].values
            img_va  = df_accepted.loc[va_idx, 'image_path'].values
            crop_va = df_accepted.loc[va_idx, 'crop'].values
            cond_va = df_accepted.loc[va_idx, 'condition'].values

            # -- Step 1: Graph topology from training data only ----
            constructor = GRAPH_BUILDERS[gt_key](feature_cols)
            constructor.fit(X_tr)

            # -- Step 2: Discretize using training data only -------
            disc        = fit_discretizer(X_tr)
            df_tr_lbl   = transform_discretizer(X_tr, disc)
            df_va_lbl   = transform_discretizer(X_va, disc)

            # -- Step 3: Build graphs (topology fixed, labels vary) -
            graphs_train = constructor.transform(df_tr_lbl)
            graphs_val   = constructor.transform(df_va_lbl)

            # -- Step 4: WL subtree features (training-only mapper) -
            Phi_train, Phi_val, fi, mapper = wl_subtree_features(
                graphs_train, graphs_val, max_height=h)

            # -- Step 5: Build kernel matrices ---------------------
            K_train, K_val = build_kernel_matrix(Phi_train, Phi_val)

            # -- Step 6: Train SVM on precomputed kernel -----------
            svm = SVC(kernel='precomputed', C=SVM_C,
                      class_weight=SVM_CLASS_WEIGHT,
                      probability=True, random_state=RANDOM_STATE)
            svm.fit(K_train, y_tr)

            # -- Step 7: Predict ------------------------------------
            y_pred = svm.predict(K_val)
            y_prob = svm.predict_proba(K_val)[:, 1]

            # -- Step 8: Metrics ------------------------------------
            acc  = accuracy_score(y_va, y_pred)
            bacc = balanced_accuracy_score(y_va, y_pred)
            mf1  = f1_score(y_va, y_pred, average='macro', zero_division=0)
            wf1  = f1_score(y_va, y_pred, average='weighted', zero_division=0)
            prec = precision_score(y_va, y_pred, zero_division=0)   # binary positive=STRESSED
            rec  = recall_score(y_va, y_pred, zero_division=0)
            try:
                auc    = roc_auc_score(y_va, y_prob)
                pr_auc = average_precision_score(y_va, y_prob)
            except ValueError:
                auc, pr_auc = 0.0, 0.0

            oov_total = sum(mapper.oov_count(i) for i in range(h))

            fold_results_wl.append({
                'graph_type':   gt_name,
                'gt_key':       gt_key,
                'wl_height':    h,
                'fold':         fold_id,
                'n_train':      len(y_tr),
                'n_val':        len(y_va),
                'n_wl_features': Phi_train.shape[1],
                'oov_count':    oov_total,
                'accuracy':     acc,
                'balanced_accuracy': bacc,
                'macro_f1':     mf1,
                'weighted_f1':  wf1,
                'precision':    prec,
                'recall':       rec,
                'roc_auc':      auc,
                'pr_auc':       pr_auc,
            })

            for i in range(len(y_va)):
                fold_oof_rows.append({
                    'image_path':       img_va[i],
                    'crop':             crop_va[i],
                    'original_condition': cond_va[i],
                    'true_binary':      y_va[i],
                    'pred_binary':      y_pred[i],
                    'pred_prob':        y_prob[i],
                    'graph_type':       gt_name,
                    'wl_height':        h,
                    'fold':             fold_id,
                })

        elapsed = time.time() - t0
        best_fold_mf1 = max(r['macro_f1'] for r in fold_results_wl
                            if r['gt_key'] == gt_key and r['wl_height'] == h)
        mean_mf1 = np.mean([r['macro_f1'] for r in fold_results_wl
                            if r['gt_key'] == gt_key and r['wl_height'] == h])
        print(f"[{gt_name:<25}] h={h}  mean_MacroF1={mean_mf1:.4f}  "
              f"best_fold={best_fold_mf1:.4f}  time={elapsed:.1f}s")

        oof_preds_wl.extend(fold_oof_rows)

elapsed_total = time.time() - start_total
print(f"\nAll 20 configurations complete. Total time: {elapsed_total:.1f}s")

# ============================================================
# SECTION 5: Flat Baselines (comparison only)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 5: Flat Baselines (RBF SVM + Random Forest)")
print("=" * 65)

class FoldSafeScaler:
    def __init__(self):
        self._s = StandardScaler()
        self.is_fitted = False
    def fit_transform(self, X):
        self.is_fitted = True
        return self._s.fit_transform(X)
    def transform(self, X):
        assert self.is_fitted
        return self._s.transform(X)

FLAT_MODELS = {
    'Flat RBF SVM': SVC(kernel='rbf', C=1.0, class_weight='balanced',
                        probability=True, random_state=RANDOM_STATE),
    'Flat RF':      RandomForestClassifier(n_estimators=300,
                                           class_weight='balanced_subsample',
                                           random_state=RANDOM_STATE, n_jobs=-1),
}

fold_results_flat = []
oof_preds_flat    = {}

for model_name, model in FLAT_MODELS.items():
    oof_rows_flat = []
    for fold_id in range(N_FOLDS):
        tr_idx = df_accepted[df_accepted['fold'] != fold_id].index
        va_idx = df_accepted[df_accepted['fold'] == fold_id].index

        X_tr = df_accepted.loc[tr_idx, feature_cols]
        X_va = df_accepted.loc[va_idx, feature_cols]
        y_tr = df_accepted.loc[tr_idx, 'binary_label'].values
        y_va = df_accepted.loc[va_idx, 'binary_label'].values
        img_va  = df_accepted.loc[va_idx, 'image_path'].values
        crop_va = df_accepted.loc[va_idx, 'crop'].values
        cond_va = df_accepted.loc[va_idx, 'condition'].values

        sc      = FoldSafeScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_va_sc = sc.transform(X_va)

        model.fit(X_tr_sc, y_tr)
        y_pred = model.predict(X_va_sc)
        y_prob = model.predict_proba(X_va_sc)[:, 1]

        acc  = accuracy_score(y_va, y_pred)
        bacc = balanced_accuracy_score(y_va, y_pred)
        mf1  = f1_score(y_va, y_pred, average='macro', zero_division=0)
        wf1  = f1_score(y_va, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_va, y_pred, zero_division=0)
        rec  = recall_score(y_va, y_pred, zero_division=0)
        try:
            auc    = roc_auc_score(y_va, y_prob)
            pr_auc = average_precision_score(y_va, y_prob)
        except ValueError:
            auc, pr_auc = 0.0, 0.0

        fold_results_flat.append({
            'model':            model_name,
            'fold':             fold_id,
            'accuracy':         acc,
            'balanced_accuracy': bacc,
            'macro_f1':         mf1,
            'weighted_f1':      wf1,
            'precision':        prec,
            'recall':           rec,
            'roc_auc':          auc,
            'pr_auc':           pr_auc,
        })
        for i in range(len(y_va)):
            oof_rows_flat.append({
                'image_path':         img_va[i],
                'crop':               crop_va[i],
                'original_condition': cond_va[i],
                'true_binary':        y_va[i],
                'pred_binary':        y_pred[i],
                'pred_prob':          y_prob[i],
                'model':              model_name,
                'fold':               fold_id,
            })
        print(f"  [{model_name}] fold={fold_id}  acc={acc:.4f}  macro_F1={mf1:.4f}")
    oof_preds_flat[model_name] = oof_rows_flat

# ============================================================
# SECTION 6: Aggregate Results & Summary Tables
# ============================================================
print("\n" + "=" * 65)
print("SECTION 6: Aggregated Results")
print("=" * 65)

df_fold_wl   = pd.DataFrame(fold_results_wl)
df_fold_flat = pd.DataFrame(fold_results_flat)

# -- Per-configuration summary (20 WL + 2 flat) ---------------
summary_rows = []
METRIC_COLS  = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1',
                'precision', 'recall', 'roc_auc', 'pr_auc']

# WL configurations
for (gt_name, h), grp in df_fold_wl.groupby(['graph_type', 'wl_height']):
    row = {
        'Method':     'WL + SVM',
        'Graph Type': gt_name,
        'Height':     h,
    }
    for m in METRIC_COLS:
        row[m] = grp[m].mean()
    summary_rows.append(row)

# Flat baselines
for model_name, grp in df_fold_flat.groupby('model'):
    row = {
        'Method':     model_name,
        'Graph Type': 'None',
        'Height':     -1,
    }
    for m in METRIC_COLS:
        row[m] = grp[m].mean()
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows).sort_values('macro_f1', ascending=False).reset_index(drop=True)

# -- Per-graph-type best WL configuration ---------------------
best_per_graph = []
wl_only = df_summary[df_summary['Method'] == 'WL + SVM']
for gt, grp in wl_only.groupby('Graph Type'):
    best_row = grp.loc[grp['macro_f1'].idxmax()]
    best_per_graph.append({
        'Graph Type':    gt,
        'Best WL Height': int(best_row['Height']),
        'Best Macro-F1': best_row['macro_f1'],
        'Accuracy':      best_row['accuracy'],
        'Balanced Acc':  best_row['balanced_accuracy'],
        'ROC-AUC':       best_row['roc_auc'],
    })

df_graph_comparison = pd.DataFrame(best_per_graph).sort_values('Best Macro-F1', ascending=False).reset_index(drop=True)

print("\nTABLE 1: Full WL + Flat Summary (sorted by Macro-F1)")
print(df_summary[['Method', 'Graph Type', 'Height',
                   'accuracy', 'macro_f1', 'weighted_f1',
                   'balanced_accuracy', 'roc_auc']].to_string(index=False))

print("\nTABLE 2: Best WL Height per Graph Type")
print(df_graph_comparison.to_string(index=False))

# -- Identify best WL config globally -------------------------
best_wl_row   = wl_only.loc[wl_only['macro_f1'].idxmax()]
best_wl_gt    = best_wl_row['Graph Type']
best_wl_h     = int(best_wl_row['Height'])
best_wl_mf1   = best_wl_row['macro_f1']

flat_rf_mf1   = df_summary[df_summary['Method'] == 'Flat RF']['macro_f1'].values[0]
flat_svm_mf1  = df_summary[df_summary['Method'] == 'Flat RBF SVM']['macro_f1'].values[0]

print(f"\nBest WL config : {best_wl_gt}  h={best_wl_h}  Macro-F1={best_wl_mf1:.4f}")
print(f"Flat RF        : Macro-F1={flat_rf_mf1:.4f}")
print(f"Flat RBF SVM   : Macro-F1={flat_svm_mf1:.4f}")

# ============================================================
# SECTION 7: Save CSV Outputs
# ============================================================
df_fold_wl['Method'] = 'WL + SVM'
df_fold_flat_out = df_fold_flat.rename(columns={'model': 'Method'})
df_fold_flat_out['graph_type'] = 'None'
df_fold_flat_out['wl_height']  = -1
df_fold_flat_out['gt_key']     = 'flat'

df_all_folds = pd.concat([df_fold_wl, df_fold_flat_out], ignore_index=True)
df_all_folds.to_csv(f'{RESULTS_DIR}/level1_wl_fold_results.csv', index=False)
print(f"\nSaved: level1_wl_fold_results.csv")

df_summary_save = df_summary.copy()
df_summary_save.columns = [c.replace(' ', '_').lower() for c in df_summary_save.columns]
df_summary_save.to_csv(f'{RESULTS_DIR}/level1_wl_summary.csv', index=False)
print(f"Saved: level1_wl_summary.csv")

df_graph_comparison.to_csv(f'{RESULTS_DIR}/level1_graph_comparison.csv', index=False)
print(f"Saved: level1_graph_comparison.csv")

# -- OOF predictions: best WL config --------------------------
oof_best_wl = [row for row in oof_preds_wl
               if row['graph_type'] == best_wl_gt and row['wl_height'] == best_wl_h]
df_oof_wl   = pd.DataFrame(oof_best_wl)
df_oof_wl.to_csv(f'{RESULTS_DIR}/level1_oof_predictions.csv', index=False)
print(f"Saved: level1_oof_predictions.csv  ({len(df_oof_wl)} rows, config: {best_wl_gt} h={best_wl_h})")

# ============================================================
# SECTION 8: Crop Subgroup Analysis (Best WL Config OOF)
# ============================================================
print("\n" + "=" * 65)
print(f"SECTION 8: Crop Subgroup Analysis")
print(f"  Using OOF of BEST WL config: {best_wl_gt} h={best_wl_h}")
print("=" * 65)

crop_subgroup_rows = []
for crop, grp in df_oof_wl.groupby('crop'):
    y_t  = grp['true_binary'].values
    y_p  = grp['pred_binary'].values
    prec = precision_score(y_t, y_p, zero_division=0)
    rec  = recall_score(y_t, y_p, zero_division=0)
    f1   = f1_score(y_t, y_p, average='macro', zero_division=0)
    acc  = accuracy_score(y_t, y_p)
    stressed_n   = int((y_t == 1).sum())
    unstressed_n = int((y_t == 0).sum())
    crop_subgroup_rows.append({
        'Crop':           crop,
        'Samples':        len(grp),
        'Stressed':       stressed_n,
        'Non-Stressed':   unstressed_n,
        'Precision':      prec,
        'Recall':         rec,
        'Macro-F1':       f1,
        'Accuracy':       acc,
    })

df_crop = pd.DataFrame(crop_subgroup_rows).sort_values('Macro-F1', ascending=False)
df_crop.to_csv(f'{RESULTS_DIR}/level1_crop_subgroup_results.csv', index=False)
print("\nTABLE 3: Crop Subgroup Performance (Best WL Config OOF)")
print(df_crop.to_string(index=False))
print(f"\nSaved: level1_crop_subgroup_results.csv")

# ============================================================
# SECTION 9: Tomato Case Study
#   Using GLOBALLY selected best WL config (no cherry-picking)
# ============================================================
print("\n" + "=" * 65)
print(f"SECTION 9: Tomato Case Study")
print(f"  Global best config: {best_wl_gt} h={best_wl_h}  (selected on ALL crops)")
print("=" * 65)

# A. Best WL on Tomato (subset of global OOF)
df_tom_wl = df_oof_wl[df_oof_wl['crop'].str.lower() == 'tomato']
print(f"Tomato samples: {len(df_tom_wl)}")
print(df_tom_wl['original_condition'].value_counts().to_string())

def compute_binary_metrics_dict(y_t, y_p, y_prob, label):
    prec = precision_score(y_t, y_p, zero_division=0)
    rec  = recall_score(y_t, y_p, zero_division=0)
    mf1  = f1_score(y_t, y_p, average='macro', zero_division=0)
    acc  = accuracy_score(y_t, y_p)
    bacc = balanced_accuracy_score(y_t, y_p)
    try:
        auc = roc_auc_score(y_t, y_prob)
    except ValueError:
        auc = float('nan')
    return {'Model': label, 'Samples': len(y_t),
            'Precision': prec, 'Recall': rec,
            'Macro-F1': mf1, 'Accuracy': acc,
            'Balanced_Acc': bacc, 'ROC-AUC': auc}

tomato_res = []
if len(df_tom_wl) > 0:
    row_wl = compute_binary_metrics_dict(
        df_tom_wl['true_binary'].values,
        df_tom_wl['pred_binary'].values,
        df_tom_wl['pred_prob'].values,
        f'WL+SVM ({best_wl_gt} h={best_wl_h}) — Global config on Tomato'
    )
    tomato_res.append(row_wl)

# B. Flat RF on Tomato (from global OOF)
df_oof_rf = pd.DataFrame(oof_preds_flat['Flat RF'])
df_tom_rf  = df_oof_rf[df_oof_rf['crop'].str.lower() == 'tomato']
if len(df_tom_rf) > 0:
    row_rf = compute_binary_metrics_dict(
        df_tom_rf['true_binary'].values,
        df_tom_rf['pred_binary'].values,
        df_tom_rf['pred_prob'].values,
        'Flat RF — Global config on Tomato'
    )
    tomato_res.append(row_rf)

df_tomato = pd.DataFrame(tomato_res)
df_tomato.to_csv(f'{RESULTS_DIR}/level1_tomato_results.csv', index=False)
print("\nTABLE 4: Tomato Case Study")
print(df_tomato.to_string(index=False))
print(f"Saved: level1_tomato_results.csv")

# ============================================================
# SECTION 10: Confusion Matrix Analysis (Best WL + Best Flat)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 10: Confusion Matrix Analysis")
print("=" * 65)

def print_cm_analysis(y_t, y_p, label):
    cm_vals = confusion_matrix(y_t, y_p)
    tn, fp, fn, tp = cm_vals.ravel()
    print(f"\n{label}")
    print(f"  TP (Stressed correctly detected)     : {tp}")
    print(f"  TN (Healthy correctly detected)      : {tn}")
    print(f"  FP (Healthy labelled as Stressed)    : {fp}")
    print(f"  FN (Stressed labelled as Healthy)    : {fn}  <- missed diagnoses")
    stressed_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    healthy_recall  = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  Stressed Recall (sensitivity)        : {stressed_recall:.4f}")
    print(f"  Healthy Recall  (specificity)        : {healthy_recall:.4f}")
    return cm_vals

cm_wl = print_cm_analysis(
    df_oof_wl['true_binary'].values,
    df_oof_wl['pred_binary'].values,
    f"Best WL Config: {best_wl_gt} h={best_wl_h}")

# Best flat overall (by macro_f1)
best_flat_name = df_summary[df_summary['Method'].isin(['Flat RF', 'Flat RBF SVM'])].sort_values('macro_f1', ascending=False).iloc[0]['Method']
df_oof_bestflat = pd.DataFrame(oof_preds_flat[best_flat_name])
cm_flat = print_cm_analysis(
    df_oof_bestflat['true_binary'].values,
    df_oof_bestflat['pred_binary'].values,
    f"Best Flat Baseline: {best_flat_name}")

# -- FN analysis: which conditions are most often missed ------
fn_df = df_oof_wl[(df_oof_wl['true_binary'] == 1) & (df_oof_wl['pred_binary'] == 0)]
print(f"\nFalse Negatives (WL — missed stressed leaves): {len(fn_df)}")
print("  By condition:")
print(fn_df['original_condition'].value_counts().head(10).to_string())
print("  By crop:")
print(fn_df['crop'].value_counts().to_string())

# ============================================================
# SECTION 11: Comparison with 57-Class Results (NB14 reference)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 11: Comparison – Binary vs 57-Class")
print("=" * 65)

# Reference values from NB14 (wl_svm_summary.csv)
NB14_WL_BEST_MF1    = 0.3911   # knn h=2 or h=3 was top WL in 57-class
NB14_WL_BEST_CONFIG = 'kNN h=2'
NB14_FLAT_SVM_MF1   = 0.4307
NB14_FLAT_RF_MF1    = None      # not present in NB14 for WL; NB18 value is 0.550

task_comp = pd.DataFrame([
    {
        'Task':           '57-class joint classification',
        'Pipeline':       'WL + SVM (best config)',
        'Config':         NB14_WL_BEST_CONFIG,
        'Macro-F1':       NB14_WL_BEST_MF1,
        'vs_flat_delta':  NB14_WL_BEST_MF1 - NB14_FLAT_SVM_MF1,
    },
    {
        'Task':           'Level 1: Binary stress detection',
        'Pipeline':       'WL + SVM (best config)',
        'Config':         f'{best_wl_gt} h={best_wl_h}',
        'Macro-F1':       best_wl_mf1,
        'vs_flat_delta':  best_wl_mf1 - flat_rf_mf1,
    },
    {
        'Task':           '57-class joint classification',
        'Pipeline':       'Flat RBF SVM',
        'Config':         'N/A',
        'Macro-F1':       NB14_FLAT_SVM_MF1,
        'vs_flat_delta':  0.0,
    },
    {
        'Task':           'Level 1: Binary stress detection',
        'Pipeline':       'Flat RF',
        'Config':         'N/A',
        'Macro-F1':       flat_rf_mf1,
        'vs_flat_delta':  0.0,
    },
])

print("\nTABLE 5: Binary vs 57-Class Comparison")
print("NOTE: These are conceptual comparisons across different label spaces.")
print(task_comp.to_string(index=False))

wl_improvement = best_wl_mf1 - NB14_WL_BEST_MF1
print(f"\nWL Macro-F1 improvement: {NB14_WL_BEST_MF1:.4f} -> {best_wl_mf1:.4f}  (Delta = +{wl_improvement:.4f})")
flat_improvement = flat_rf_mf1 - NB14_FLAT_SVM_MF1
print(f"Flat improvement        : {NB14_FLAT_SVM_MF1:.4f} -> {flat_rf_mf1:.4f}  (Delta = +{flat_improvement:.4f})")
gap_57class    = NB14_FLAT_SVM_MF1 - NB14_WL_BEST_MF1
gap_binary     = flat_rf_mf1 - best_wl_mf1
print(f"\nGap (flat − WL) in 57-class : {gap_57class:.4f}")
print(f"Gap (flat − WL) in binary   : {gap_binary:.4f}")
if gap_binary < gap_57class:
    print("-> WL gap vs. flat NARROWED under task simplification.")
else:
    print("-> WL gap vs. flat DID NOT narrow under task simplification.")

# ============================================================
# SECTION 12: Plots
# ============================================================
print("\n" + "=" * 65)
print("SECTION 12: Generating Plots")
print("=" * 65)

sns.set_style('whitegrid')
PALETTE = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
FLAT_COLOR = '#607D8B'

# -- Plot 1: Macro-F1 by graph type (bar chart) ---------------
fig, ax = plt.subplots(figsize=(9, 5))
wl_summary_only = df_summary[df_summary['Method'] == 'WL + SVM']
df_per_graph = wl_summary_only.groupby('Graph Type')['macro_f1'].max().sort_values(ascending=False)
colors = PALETTE[:len(df_per_graph)]
bars = ax.bar(df_per_graph.index, df_per_graph.values, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(flat_rf_mf1,  color='#F44336', linestyle='--', linewidth=1.5, label=f'Flat RF ({flat_rf_mf1:.3f})')
ax.axhline(flat_svm_mf1, color='#9E9E9E', linestyle=':', linewidth=1.5, label=f'Flat RBF SVM ({flat_svm_mf1:.3f})')
for bar, val in zip(bars, df_per_graph.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('Graph Construction', fontsize=11)
ax.set_ylabel('Best Macro-F1 across WL Heights', fontsize=11)
ax.set_title('Level 1: WL+SVM Macro-F1 by Graph Type\n(Binary Stress Detection)', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, min(1.0, df_per_graph.max() + 0.12))
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/macro_f1_by_graph.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: macro_f1_by_graph.png")

# -- Plot 2: Macro-F1 by WL height (line chart per graph type) -
fig, ax = plt.subplots(figsize=(9, 5))
for idx, (gt_name, grp) in enumerate(wl_summary_only.groupby('Graph Type')):
    grp_sorted = grp.sort_values('Height')
    ax.plot(grp_sorted['Height'], grp_sorted['macro_f1'],
            marker='o', linewidth=2, label=gt_name, color=PALETTE[idx % len(PALETTE)])
ax.axhline(flat_rf_mf1,  color='#F44336', linestyle='--', linewidth=1.5, label=f'Flat RF ({flat_rf_mf1:.3f})')
ax.axhline(flat_svm_mf1, color='#9E9E9E', linestyle=':', linewidth=1.5, label=f'Flat RBF SVM ({flat_svm_mf1:.3f})')
ax.set_xlabel('WL Height (h)', fontsize=11)
ax.set_ylabel('Macro-F1', fontsize=11)
ax.set_title('Level 1: WL+SVM Macro-F1 by Height\n(Binary Stress Detection)', fontsize=12)
ax.set_xticks(WL_HEIGHTS)
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim(0.45, min(1.0, wl_summary_only['macro_f1'].max() + 0.1))
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/macro_f1_by_height.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: macro_f1_by_height.png")

# -- Plot 3: Graph × Height heatmap ----------------------------
heat_data = wl_summary_only.pivot(index='Graph Type', columns='Height', values='macro_f1')
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5,
            ax=ax, vmin=heat_data.values.min() - 0.02, vmax=heat_data.values.max() + 0.02)
ax.set_title('Level 1: WL Macro-F1 Heatmap\n(Graph Type × WL Height, Binary Stress)', fontsize=12)
ax.set_xlabel('WL Height')
ax.set_ylabel('Graph Construction')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/graph_height_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: graph_height_heatmap.png")

# -- Plot 4: Binary confusion matrix – best WL -----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax_i, (cm_arr, title) in enumerate([
    (cm_wl,   f'WL+SVM\n{best_wl_gt} h={best_wl_h}'),
    (cm_flat, f'Flat Baseline\n{best_flat_name}')
]):
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
                xticklabels=['Healthy\n(Predicted)', 'Stressed\n(Predicted)'],
                yticklabels=['Healthy\n(True)', 'Stressed\n(True)'],
                ax=axes[ax_i])
    axes[ax_i].set_title(title, fontsize=11)
    tn, fp, fn, tp = cm_arr.ravel()
    axes[ax_i].set_xlabel(f'TP={tp} TN={tn} FP={fp} FN={fn}', fontsize=9)
plt.suptitle('Level 1: Confusion Matrices (Aggregated OOF)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/confusion_matrix_wl.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_wl.png")

# -- Plot 5: ROC Curve ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for (gt_name, h), label, color in [
    ((best_wl_gt, best_wl_h), f'WL+SVM {best_wl_gt} h={best_wl_h}', '#2196F3'),
    (('flat_rf', -1),          f'Flat RF',                             '#F44336'),
]:
    if gt_name == 'flat_rf':
        y_t   = np.array([r['true_binary'] for r in oof_preds_flat['Flat RF']])
        y_pr  = np.array([r['pred_prob']   for r in oof_preds_flat['Flat RF']])
    else:
        y_t   = df_oof_wl['true_binary'].values
        y_pr  = df_oof_wl['pred_prob'].values
    fpr, tpr, _ = roc_curve(y_t, y_pr)
    auc_val = roc_auc_score(y_t, y_pr)
    ax.plot(fpr, tpr, linewidth=2, label=f'{label}  (AUC={auc_val:.3f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Level 1: ROC Curves\n(Binary Stress Detection)', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curve.png")

# -- Plot 6: PR Curve -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for (gt_name, h), label, color in [
    ((best_wl_gt, best_wl_h), f'WL+SVM {best_wl_gt} h={best_wl_h}', '#2196F3'),
    (('flat_rf', -1),          f'Flat RF',                             '#F44336'),
]:
    if gt_name == 'flat_rf':
        y_t   = np.array([r['true_binary'] for r in oof_preds_flat['Flat RF']])
        y_pr  = np.array([r['pred_prob']   for r in oof_preds_flat['Flat RF']])
    else:
        y_t   = df_oof_wl['true_binary'].values
        y_pr  = df_oof_wl['pred_prob'].values
    prec_c, rec_c, _ = precision_recall_curve(y_t, y_pr)
    pr_auc_val = average_precision_score(y_t, y_pr)
    ax.plot(rec_c, prec_c, linewidth=2, label=f'{label}  (PR-AUC={pr_auc_val:.3f})', color=color)

baseline_prev = (df_accepted['binary_label'] == 1).mean()
ax.axhline(baseline_prev, color='k', linestyle='--', linewidth=1, alpha=0.5,
           label=f'Baseline prevalence ({baseline_prev:.2f})')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Level 1: Precision-Recall Curves\n(Binary Stress Detection)', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/pr_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pr_curve.png")

# -- Plot 7: Crop subgroup F1 -----------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
df_crop_plot = df_crop.sort_values('Macro-F1', ascending=True)
colors_crop = ['#4CAF50' if f >= flat_rf_mf1 else '#FF5722' for f in df_crop_plot['Macro-F1']]
bars = ax.barh(df_crop_plot['Crop'], df_crop_plot['Macro-F1'],
               color=colors_crop, edgecolor='white')
ax.axvline(best_wl_mf1, color='#2196F3', linestyle='--', linewidth=1.5,
           label=f'Global WL+SVM ({best_wl_mf1:.3f})')
ax.axvline(flat_rf_mf1, color='#F44336', linestyle=':', linewidth=1.5,
           label=f'Flat RF ({flat_rf_mf1:.3f})')
for bar, val in zip(bars, df_crop_plot['Macro-F1']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel('Macro-F1', fontsize=11)
ax.set_title(f'Level 1: Crop Subgroup Macro-F1\n({best_wl_gt} h={best_wl_h}, Global Config)', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/crop_subgroup_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: crop_subgroup_f1.png")

# -- Plot 8: Tomato comparison (WL vs flat RF) -----------------
if len(df_tomato) >= 1:
    fig, ax = plt.subplots(figsize=(7, 4))
    models_tom = df_tomato['Model'].tolist()
    mf1_tom    = df_tomato['Macro-F1'].tolist()
    col_tom    = ['#2196F3', '#F44336'][:len(models_tom)]
    bars_tom   = ax.bar(range(len(models_tom)), mf1_tom, color=col_tom, edgecolor='white')
    ax.set_xticks(range(len(models_tom)))
    ax.set_xticklabels([m.split('—')[0].strip() for m in models_tom], rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars_tom, mf1_tom):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Macro-F1', fontsize=11)
    ax.set_title('Level 1: Tomato Subgroup – WL vs Flat RF\n(Global Model, Not Tomato-Specific)', fontsize=12)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/tomato_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: tomato_comparison.png")

# ============================================================
# SECTION 13: Feature / Graph Interpretation (qualitative)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 13: Feature & Graph Interpretation")
print("=" * 65)

# Ranking of graph constructions at 57-class vs Level 1
wl_57class_ranking = {
    'knn':         0.3911,   # best in NB14
    'correlation': 0.3795,
    'family':      0.3435,
    'hybrid':      0.3449,
    'spatial':     0.2601,
}

print("\n[A] Graph construction ranking comparison")
print(f"{'Graph':<22} {'57-class MF1':>13} {'Binary MF1':>11} {'Rank change':>12}")
print("-" * 62)
binary_mf1_lookup = {}
for gt_key_inner in GRAPH_BUILDERS:
    gt_name_inner = GRAPH_DISPLAY_NAMES[gt_key_inner]
    best_row = wl_only[wl_only['Graph Type'] == gt_name_inner].sort_values('macro_f1', ascending=False).iloc[0]
    binary_mf1_lookup[gt_key_inner] = best_row['macro_f1']

rank_57   = sorted(wl_57class_ranking, key=wl_57class_ranking.get, reverse=True)
rank_bin  = sorted(binary_mf1_lookup,  key=binary_mf1_lookup.get, reverse=True)
for gt in rank_57:
    r57 = rank_57.index(gt) + 1
    rb  = rank_bin.index(gt) + 1
    delta = r57 - rb
    symbol = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
    print(f"{GRAPH_DISPLAY_NAMES[gt]:<22} {wl_57class_ranking[gt]:>13.4f} "
          f"{binary_mf1_lookup[gt]:>11.4f} "
          f"{'rank ' + str(r57) + ' -> ' + str(rb) + ' ' + symbol:>12}")

print("\n[B] Connectivity and WL neighborhood richness")
for gt_key_inner in GRAPH_BUILDERS:
    constructor_diag = GRAPH_BUILDERS[gt_key_inner](feature_cols)
    # Use full dataset for edge count (topology is dataset-level, not fold-specific)
    constructor_diag.fit(df_accepted[feature_cols])
    n_edges = constructor_diag.base_graph.number_of_edges()
    density = nx.density(constructor_diag.base_graph)
    print(f"  {GRAPH_DISPLAY_NAMES[gt_key_inner]:<25}: {n_edges} edges, density={density:.3f}")

print("\n[C] Implication of WL neighborhood structure for binary task")
print("  WL at h=0 counts only discretized raw feature labels (Low/Medium/High).")
print("  These are independent of graph topology — so h=0 is always identical")
print("  across all graph types (same label bags, same feature vector).")
print("  At h>=1, neighborhood aggregation introduces graph-structure information.")
print("  If h=1 or h=2 improves over h=0, topology is contributing signal.")
h0_mf1  = wl_only[wl_only['Height'] == 0]['macro_f1'].mean()
h1_mf1  = wl_only[wl_only['Height'] == 1]['macro_f1'].mean()
h2_mf1  = wl_only[wl_only['Height'] == 2]['macro_f1'].mean()
h3_mf1  = wl_only[wl_only['Height'] == 3]['macro_f1'].mean()
print(f"  Mean Macro-F1 across all graphs:")
print(f"    h=0: {h0_mf1:.4f}  h=1: {h1_mf1:.4f}  h=2: {h2_mf1:.4f}  h=3: {h3_mf1:.4f}")
if h1_mf1 > h0_mf1:
    print("  -> Graph topology is contributing discriminative signal (h>0 improves).")
else:
    print("  -> Graph topology does not help; h=0 node labels already capture the signal.")

# ============================================================
# SECTION 14: Final Research Conclusion
# ============================================================
print("\n" + "=" * 65)
print("SECTION 14: FINAL RESEARCH CONCLUSION")
print("=" * 65)

best_wl_vs_flat_diff = best_wl_mf1 - flat_rf_mf1
wl_relative_improvement = (best_wl_mf1 - NB14_WL_BEST_MF1) / max(NB14_WL_BEST_MF1, 1e-6)

print(f"""
Research Question:
  "Does simplifying the OLID-I prediction task from 57-class joint
   crop-condition classification to binary stress detection improve the
   effectiveness of Weisfeiler–Lehman graph kernels, and does graph
   construction still influence performance?"

Key Findings:
  +-------------------------------------------------------------+
  |  Metric                  |  57-class   |  Level 1 (Binary)  |
  |--------------------------|-------------|--------------------|
  |  Best WL Macro-F1        |  {NB14_WL_BEST_MF1:.4f}      |  {best_wl_mf1:.4f}              |
  |  Best Flat Macro-F1      |  {NB14_FLAT_SVM_MF1:.4f}      |  {flat_rf_mf1:.4f}              |
  |  WL vs Flat gap          |  {gap_57class:.4f}      |  {gap_binary:.4f}              |
  |  Best WL config          |  {NB14_WL_BEST_CONFIG:<11}  |  {best_wl_gt} h={best_wl_h}       |
  +-------------------------------------------------------------+

  1. Does WL performance improve substantially under binary stress detection?
     WL Macro-F1: {NB14_WL_BEST_MF1:.4f} -> {best_wl_mf1:.4f}  (Delta = +{wl_improvement:.4f}, {wl_relative_improvement*100:.1f}% relative)
     {"YES — substantial improvement." if wl_improvement > 0.15 else "YES — meaningful improvement." if wl_improvement > 0.05 else "MODERATE — some improvement but not dramatic."}

  2. Does any WL graph construction outperform the flat baseline?
     Best WL ({best_wl_gt} h={best_wl_h}): {best_wl_mf1:.4f}
     Best Flat ({best_flat_name}):     {flat_rf_mf1:.4f}
     Gap:  {best_wl_vs_flat_diff:+.4f}
     {"WL+SVM MATCHES or EXCEEDS flat models — graph structure provides competitive representations." if best_wl_vs_flat_diff >= 0 else f"WL+SVM remains below flat models by {abs(best_wl_vs_flat_diff):.4f} Macro-F1."}

  3. Which graph construction performs best?
     {best_wl_gt} (Best Macro-F1 = {best_wl_mf1:.4f})

  4. Which WL height performs best overall?
     Across all graph types, mean F1 peaks at h={'  / '.join([str(h) for h,v in sorted(zip(WL_HEIGHTS, [h0_mf1, h1_mf1, h2_mf1, h3_mf1]), key=lambda x: x[1], reverse=True)[:2]])}.

  5. Is the ranking of graph constructions similar to the original 57-class experiment?
     (See Section 13 Table A for rank changes.)
     {"Rankings are largely preserved — suggesting inherent graph structural properties dominate." if len(set(rank_57[:3]) & set(rank_bin[:3])) >= 2 else "Rankings changed substantially — binary simplification changes which topology is most useful."}

  6. Does simplification of the target make graph structure more useful?
     WL gap vs. flat: 57-class={gap_57class:.4f}, Binary={gap_binary:.4f}
     {"YES — WL's disadvantage vs. flat features narrowed." if gap_binary < gap_57class else "NO — the gap did not narrow; flat models still hold a similar advantage."}

  7. If WL still underperforms flat features, what does that imply?
     The WL representation encodes graph-structural patterns as subtree
     fingerprints. If flat features outperform even on a binary task,
     it implies that the current graph topology (edges between the 38
     feature nodes) does not encode more discriminative structure than
     the raw feature magnitudes do for stress/non-stress separation.
     This suggests the bottleneck is graph topology design, not task complexity.
""")

# -- Outcome classification ------------------------------------
if wl_improvement > 0.10 and best_wl_vs_flat_diff >= -0.03:
    outcome = "A"
    outcome_desc = ("WL improves strongly and becomes competitive with flat models. "
                    "This supports the hypothesis that task complexity affects the "
                    "usefulness of WL graph representations.")
elif wl_improvement > 0.02:
    outcome = "B"
    outcome_desc = ("WL improves but remains below flat models. The graph representation "
                    "benefits from task simplification but still does not outperform "
                    "conventional feature-based learning with the current topology.")
else:
    outcome = "C"
    outcome_desc = ("WL does not improve meaningfully. Task decomposition alone does not "
                    "resolve the limitation of the current graph/WL representation, "
                    "implying a fundamental limitation in how these features are connected.")

print(f"Research Outcome: [{outcome}] {outcome_desc}")

print("\nApplication implication:")
print("  Binary stress detection (healthy vs. diseased/deficient) is substantially")
print("  more tractable than the full 57-class joint problem. Even if WL+SVM does")
print("  not outperform Random Forest, a practical two-stage screening system")
print("  (binary detector -> specialist classifier) remains highly viable.")
print()
print(f"All artifacts saved in: {RESULTS_DIR}")
print("=" * 65)
print("NB21 COMPLETE")
print("=" * 65)
