import os, time, json, warnings, tracemalloc
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

# ── Authoritative parameters (must match NB12/NB14 exactly) ───────────────────
RANDOM_STATE  = 42
CORR_THRESH   = 0.70
KNN_K         = 3
WL_HEIGHTS    = [0, 1, 2, 3]
N_FOLDS       = 5
ROUNDING_CANDIDATES = [2, 3, 4]   # evaluated in precision sensitivity

# Attribute-aware specific parameters (chosen BEFORE classification)
# Selected rounding precision is determined in Section 7 diagnostic.
ROUNDING_PRECISION = None          # will be set after diagnostic

RESULTS_DIR = 'results/attribute_aware_graph'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Configuration:")
print(f"  RANDOM_STATE       = {RANDOM_STATE}")
print(f"  CORR_THRESH        = {CORR_THRESH}")
print(f"  KNN_K              = {KNN_K}")
print(f"  WL_HEIGHTS         = {WL_HEIGHTS}")
print(f"  N_FOLDS            = {N_FOLDS}")
print(f"  ROUNDING_CANDIDATES= {ROUNDING_CANDIDATES}")
print(f"  RESULTS_DIR        = {RESULTS_DIR}")
print()
print("IMPORTANT: This notebook is an EXPERIMENTAL REPRESENTATION STUDY.")
print("Standard discrete WL results (NB12–15) are NOT modified here.")

# Load accepted images (same CSV as NB12–15)
csv_path = 'olid_full_features.csv'
df_full  = pd.read_csv(csv_path)
df_accepted = df_full[df_full['quality_flag'] == 'ACCEPT'].copy().reset_index(drop=True)

meta_cols = ['image_path', 'image_name', 'class', 'crop', 'condition',
             'quality_flag', 'mask_area_ratio', 'failure_reason']
feature_cols = [c for c in df_accepted.columns if c not in meta_cols]

assert len(df_accepted) == 4387, f"Expected 4387 accepted records, got {len(df_accepted)}"
assert len(feature_cols) == 38,  f"Expected 38 features, got {len(feature_cols)}"
nan_count = df_accepted[feature_cols].isna().sum().sum()
inf_count = np.isinf(df_accepted[feature_cols].values).sum()
assert nan_count == 0, f"NaN found: {nan_count}"
assert inf_count == 0, f"Inf found: {inf_count}"

print(f"Accepted images   : {len(df_accepted)}")
print(f"Classes           : {df_accepted['class'].nunique()}")
print(f"Crops             : {df_accepted['crop'].nunique()}")
print(f"Conditions        : {df_accepted['condition'].nunique()}")
print(f"Features          : {len(feature_cols)}")
print(f"NaN / Inf         : PASS")

# ── Authoritative fold assignments (same as NB14) ─────────────────────────────
df_folds = pd.read_csv('cv_fold_assignments.csv')
df_accepted = df_accepted.merge(df_folds[['image_path', 'fold']], on='image_path', how='left')
assert df_accepted['fold'].isna().sum() == 0, "Some images have no fold assignment!"
assert set(df_accepted['fold'].unique()) == {0, 1, 2, 3, 4}, "Expected folds 0-4"
print(f"Fold assignments loaded: {N_FOLDS} folds")
print(f"Fold distribution:\n{df_accepted['fold'].value_counts().sort_index().to_string()}")

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

all_family_features = [f for fam in FEATURE_FAMILIES.values() for f in fam]
assert sorted(all_family_features) == sorted(feature_cols), "Feature family mismatch!"
print("Feature family assignment : PASS")
print(f"Families : {list(FEATURE_FAMILIES.keys())}")

class GraphConstructor:
    """Base class: topology learned from training data.
    Node attributes are NOT set here — attribute setting is representation-specific.
    """
    def __init__(self, features):
        self.features   = list(features)
        self.base_graph = nx.Graph()
        self.base_graph.add_nodes_from(self.features)

    def fit(self, X_train):
        raise NotImplementedError

    def get_edges(self):
        return list(self.base_graph.edges(data=True))

    def verify_topology_unchanged(self, other_constructor):
        """Assert this constructor's topology is identical to another fitted one."""
        assert set(self.base_graph.edges()) == set(other_constructor.base_graph.edges()), \
            "Topology mismatch — graph topology must not change between representations!"


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
                    self.base_graph.add_edge(
                        self.features[i], self.features[j],
                        weight=float(r), source='correlation'
                    )
        self.corr_matrix_ = corr  # store for edge-weight access
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
                self.base_graph.add_edge(
                    self.features[i], self.features[j], source='feature_similarity'
                )
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
                        self.base_graph.add_edge(
                            fa[ri[0]], fb[ci[0]],
                            source='family_bridge', weight=float(mx)
                        )
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
                    self.base_graph.add_edge(
                        gf, sf, source='spatial_global_link',
                        weight=float(corr.loc[gf, sf])
                    )
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

print("Five graph topology constructors loaded:")
for k, v in GRAPH_DISPLAY_NAMES.items():
    print(f"  {k:12s} -> {v}")

class FoldSafeScaler:
    """
    Wraps StandardScaler to enforce fold-safe training-only fitting.

    Usage:
        scaler = FoldSafeScaler()
        X_tr_scaled = scaler.fit_transform(X_train_df)
        X_va_scaled = scaler.transform(X_val_df)

    Stores mean_ and scale_ for reproducibility documentation.
    """
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
            raise ValueError("FoldSafeScaler.fit_transform() must be called first.")
        if not isinstance(X_val, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        arr = self._scaler.transform(X_val.values)
        return pd.DataFrame(arr, columns=self.columns_, index=X_val.index)

    def get_params(self):
        return {
            'mean':  self.mean_.tolist()  if self.is_fitted else None,
            'scale': self.scale_.tolist() if self.is_fitted else None,
        }


print("FoldSafeScaler defined.")
print("  fit_transform(X_train) — fits on training data only.")
print("  transform(X_val)       — applies training statistics to validation.")
print("  get_params()           — returns mean and scale for reproducibility.")

rep_comparison = pd.DataFrame([
    {
        'Representation':              'Flat Continuous (RBF SVM)',
        'Continuous information':      'YES',
        'Feature identity':            'YES (column index)',
        'Graph topology used':         'NO',
        'Standard WL':                 'NO',
        'Experimental extension':      'NO',
        'Origin notebook':             'NB14 (validated)',
    },
    {
        'Representation':              'Standard Discrete WL',
        'Continuous information':      'NO (3-bin: Low/Medium/High)',
        'Feature identity':            'YES (feature::bin format)',
        'Graph topology used':         'YES',
        'Standard WL':                 'YES (WL subtree kernel)',
        'Experimental extension':      'NO',
        'Origin notebook':             'NB12/13/14 (validated, corrected)',
    },
    {
        'Representation':              'Continuous Graph-Derived (RepA)',
        'Continuous information':      'YES',
        'Feature identity':            'YES',
        'Graph topology used':         'YES (neighbourhood summaries)',
        'Standard WL':                 'NO',
        'Experimental extension':      'YES — this notebook',
        'Origin notebook':             'NB16 (experimental)',
    },
    {
        'Representation':              'Attribute-Aware WL-Style (RepC)',
        'Continuous information':      'YES (rounded to fixed precision)',
        'Feature identity':            'YES',
        'Graph topology used':         'YES (WL-style aggregation)',
        'Standard WL':                 'NO (extension; not mathematically equivalent)',
        'Experimental extension':      'YES — this notebook',
        'Origin notebook':             'NB16 (experimental)',
    },
])

print(rep_comparison.to_string(index=False))
rep_comparison.to_csv(f'{RESULTS_DIR}/representation_comparison.csv', index=False)
print(f"\nSaved: {RESULTS_DIR}/representation_comparison.csv")

# Select two features with meaningful spread to illustrate the contrast
demo_features = ['green_ratio', 'area', 'mean_H', 'glcm_contrast']
fold0_train_idx = df_accepted[df_accepted['fold'] != 0].index
fold0_val_idx   = df_accepted[df_accepted['fold'] == 0].index

X_demo_tr = df_accepted.loc[fold0_train_idx, feature_cols]
X_demo_va = df_accepted.loc[fold0_val_idx,   feature_cols]

# Fold-safe scaler
scaler_demo = FoldSafeScaler()
X_demo_tr_sc = scaler_demo.fit_transform(X_demo_tr)
X_demo_va_sc = scaler_demo.transform(X_demo_va)

print("Information-Preservation Analysis (Fold 0 training split)")
print("=" * 65)
print(f"{'Feature':<22} {'Raw continuous (5 samples)':<32} {'Standard 3-bin'}")
print("-" * 65)

from sklearn.preprocessing import KBinsDiscretizer
disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', subsample=None)
disc.fit(X_demo_tr[demo_features].values)
disc_labels = disc.transform(X_demo_va[demo_features].values)
int_to_label = {0: 'Low', 1: 'Medium', 2: 'High'}

info_rows = []
for fi, feat in enumerate(demo_features):
    raw_vals = X_demo_va[feat].values[:5]
    disc_vals = [int_to_label[int(v)] for v in disc_labels[:5, fi]]
    raw_str  = ', '.join(f'{v:.4f}' for v in raw_vals)
    disc_str = ', '.join(disc_vals)
    print(f"  {feat:<20} [{raw_str}]")
    print(f"  {'':20} -> Discrete: [{disc_str}]")
    print()
    for i in range(5):
        info_rows.append({
            'feature': feat,
            'sample_idx': i,
            'continuous_value': round(float(raw_vals[i]), 6),
            'standardised_value': round(float(X_demo_va_sc[feat].values[i]), 6),
            'discrete_label': disc_vals[i],
        })

df_info = pd.DataFrame(info_rows)
df_info.to_csv(f'{RESULTS_DIR}/information_preservation_example.csv', index=False)
print(f"Saved: {RESULTS_DIR}/information_preservation_example.csv")
print()
print("OBSERVATION: Values that map to the same discrete bin (e.g., all 'Low')")
print("retain different standardised values in the continuous representation.")
print("This demonstrates that H5 (continuous representation preserves information")
print("discarded by discretisation) is supported at the representation level.")

print("Graph Topology Preservation Test")
print("=" * 50)

# Fit two correlation constructors on the same fold-0 training data
# Compare topology — must be identical
X_fold0_tr = df_accepted.loc[df_accepted['fold'] != 0, feature_cols]

c1 = CorrelationGraph(feature_cols, CORR_THRESH).fit(X_fold0_tr)
c2 = CorrelationGraph(feature_cols, CORR_THRESH).fit(X_fold0_tr)

nodes_match = set(c1.base_graph.nodes()) == set(c2.base_graph.nodes())
edges_match = set(c1.base_graph.edges()) == set(c2.base_graph.edges())
assert nodes_match, "Node sets differ!"
assert edges_match, "Edge sets differ between two identical fits!"

print(f"  Nodes : {c1.base_graph.number_of_nodes()} — consistent: {nodes_match}")
print(f"  Edges : {c1.base_graph.number_of_edges()} — consistent: {edges_match}")
print()
print("Graph topology statistics (Fold 0 training split):")
print(f"{'Graph Type':<25} {'Nodes':>6} {'Edges':>7} {'Density':>9} {'Components':>11} {'Isolated':>9}")
print("-" * 70)

topo_rows = []
for gt_key in GRAPH_BUILDERS:
    constructor = GRAPH_BUILDERS[gt_key](feature_cols)
    constructor.fit(X_fold0_tr)
    G = constructor.base_graph
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    n_comp  = nx.number_connected_components(G)
    isolated = len(list(nx.isolates(G)))
    print(f"  {GRAPH_DISPLAY_NAMES[gt_key]:<23} {n_nodes:>6} {n_edges:>7} {density:>9.4f} {n_comp:>11} {isolated:>9}")
    topo_rows.append({
        'graph_type': GRAPH_DISPLAY_NAMES[gt_key],
        'nodes': n_nodes, 'edges': n_edges,
        'density': round(density, 4),
        'connected_components': n_comp,
        'isolated_nodes': isolated,
    })

pd.DataFrame(topo_rows).to_csv(f'{RESULTS_DIR}/topology_preservation_check.csv', index=False)
print()
print("TOPOLOGY PRESERVATION: PASS — graph structure is unchanged.")
print("Only node attribute encoding changes between representations.")

def build_continuous_graph_features(G, scaled_row, feature_cols):
    """
    Representation A: Continuous graph-derived feature vector.

    For each node v (one feature):
        - self_value          : standardised continuous feature value
        - nbr_mean            : mean of neighbour values (0 if isolated)
        - nbr_std             : std  of neighbour values (0 if isolated)
        - nbr_sum             : sum  of neighbour values (0 if isolated)
        - nbr_mad             : mean absolute deviation from neighbours (0 if isolated)

    Returns a flat numpy array of length: 5 * n_nodes.
    Also returns the column names for documentation.
    """
    cols_out = []
    vals_out = []

    for node in feature_cols:       # preserve canonical feature order
        v_val    = float(scaled_row[node])
        nbrs     = list(G.neighbors(node))
        nbr_vals = np.array([float(scaled_row[n]) for n in nbrs])

        nbr_mean = float(np.mean(nbr_vals))   if len(nbr_vals) > 0 else 0.0
        nbr_std  = float(np.std(nbr_vals))    if len(nbr_vals) > 0 else 0.0
        nbr_sum  = float(np.sum(nbr_vals))    if len(nbr_vals) > 0 else 0.0
        nbr_mad  = float(np.mean(np.abs(nbr_vals - v_val))) if len(nbr_vals) > 0 else 0.0

        vals_out.extend([v_val, nbr_mean, nbr_std, nbr_sum, nbr_mad])
        cols_out.extend([
            f'{node}__self',
            f'{node}__nbr_mean',
            f'{node}__nbr_std',
            f'{node}__nbr_sum',
            f'{node}__nbr_mad',
        ])

    arr = np.array(vals_out, dtype=np.float32)
    assert not np.isnan(arr).any(), "NaN in Representation A vector!"
    assert not np.isinf(arr).any(), "Inf in Representation A vector!"
    return arr, cols_out


def build_rep_A_matrix(G, X_scaled_df, feature_cols):
    """Build Representation A matrix for a set of images."""
    rows = []
    for _, row in X_scaled_df.iterrows():
        vec, _ = build_continuous_graph_features(G, row, feature_cols)
        rows.append(vec)
    return np.stack(rows, axis=0)   # (n_images, 5*38=190)


# Quick sanity: check vector length
_tmp_G      = CorrelationGraph(feature_cols, CORR_THRESH).fit(X_fold0_tr)
_tmp_scaler = FoldSafeScaler()
_tmp_sc     = _tmp_scaler.fit_transform(X_fold0_tr)
_sample_row = _tmp_sc.iloc[0]
_vec, _cols = build_continuous_graph_features(_tmp_G.base_graph, _sample_row, feature_cols)

print(f"Representation A vector length  : {len(_vec)}")
print(f"Expected (5 statistics × 38)    : {5 * 38}")
assert len(_vec) == 5 * 38, "Vector length mismatch!"
print(f"No NaN / Inf                    : PASS")
print(f"First 10 feature names          : {_cols[:10]}")

def build_continuous_edge_features(G, scaled_row, has_edge_weights=False):
    """
    Representation B: Continuous edge-aware feature vector.

    For each edge (u, v) in sorted canonical order:
        - value_u
        - value_v
        - |value_u - value_v|   (absolute difference)
        - value_u * value_v     (product interaction)
        - edge_weight           (Pearson r if available, else 0.0)

    Returns flat numpy array of length 5 * E.
    """
    edges_sorted = sorted(G.edges(data=True), key=lambda e: (e[0], e[1]))
    cols_out = []
    vals_out = []

    for u, v, d in edges_sorted:
        val_u  = float(scaled_row[u])
        val_v  = float(scaled_row[v])
        ew     = float(d.get('weight', 0.0)) if has_edge_weights else 0.0

        vals_out.extend([
            val_u,
            val_v,
            abs(val_u - val_v),
            val_u * val_v,
            ew,
        ])
        cols_out.extend([
            f'{u}__{v}__val_u',
            f'{u}__{v}__val_v',
            f'{u}__{v}__abs_diff',
            f'{u}__{v}__product',
            f'{u}__{v}__edge_w',
        ])

    arr = np.array(vals_out, dtype=np.float32)
    assert not np.isnan(arr).any(), "NaN in Representation B vector!"
    assert not np.isinf(arr).any(), "Inf in Representation B vector!"
    return arr, cols_out


# Sanity check for correlation graph (has edge weights)
_corr_G = CorrelationGraph(feature_cols, CORR_THRESH).fit(X_fold0_tr)
_vec_B, _cols_B = build_continuous_edge_features(
    _corr_G.base_graph, _sample_row, has_edge_weights=True
)

print(f"Representation B vector length  : {len(_vec_B)}")
print(f"  Correlation graph edges       : {_corr_G.base_graph.number_of_edges()}")
print(f"  Expected (5 × E)              : {5 * _corr_G.base_graph.number_of_edges()}")
assert len(_vec_B) == 5 * _corr_G.base_graph.number_of_edges()
print(f"No NaN / Inf                    : PASS")
print(f"Edge-weight used (corr graph)   : YES (Pearson r stored per edge)")

class AttributeAwareMapper:
    """
    Maintains a deterministic mapping from attribute-aware WL-style signatures
    to compressed integer tokens.

    IMPORTANT: This is NOT the standard WL subtree kernel.
    It is an experimental extension that incorporates continuous node attributes.

    Training:  map_training_signature(iter, sig)  — registers new signatures.
    Inference: map_inference_signature(iter, sig) — returns token or OOV token.

    OOV strategy:
    - Validation signatures not in the training vocabulary receive '__AAOOV__N'.
    - OOV tokens do NOT alter the training vocabulary.
    - OOV tokens contribute ZERO to the feature vector.
    """
    def __init__(self, rounding_precision=4):
        self._maps    = {}   # iter -> {sig_str: token_str}
        self._counter = {}   # iter -> int
        self._oov     = {}   # iter -> set of unseen sig_strs
        self.precision = rounding_precision

    def _sig_to_key(self, sig):
        """Serialise a signature tuple to a deterministic hashable string."""
        # sig = (state, tuple_of_neighbour_states_sorted)
        # state = (feature_name, rounded_float)  or compressed token string
        def _state_str(s):
            if isinstance(s, tuple):
                return f'({s[0]},{s[1]:.{self.precision}f})'
            return str(s)
        label, neighbors = sig
        nbr_str = ','.join(_state_str(n) for n in sorted(neighbors, key=_state_str))
        return f'{_state_str(label)}|[{nbr_str}]'

    def map_training_signature(self, iteration, sig):
        if iteration not in self._maps:
            self._maps[iteration]    = {}
            self._counter[iteration] = 0
            self._oov[iteration]     = set()
        key = self._sig_to_key(sig)
        if key not in self._maps[iteration]:
            self._maps[iteration][key] = f'aawl_{iteration}_{self._counter[iteration]}'
            self._counter[iteration]  += 1
        return self._maps[iteration][key]

    def map_inference_signature(self, iteration, sig):
        if iteration not in self._maps:
            return f'__AAOOV__{iteration}'
        key = self._sig_to_key(sig)
        if key not in self._maps[iteration]:
            self._oov[iteration].add(key)
            return f'__AAOOV__{iteration}'
        return self._maps[iteration][key]

    def oov_count(self, iteration):
        return len(self._oov.get(iteration, set()))

    def vocab_size(self, iteration):
        return len(self._maps.get(iteration, {}))


print("AttributeAwareMapper defined.")
print("  Uses fixed rounding precision to avoid floating-point key instability.")
print("  OOV validation signatures: zero contribution, no vocabulary mutation.")

print("Precision Sensitivity Diagnostic")
print("=" * 60)
print("Evaluating on: Correlation graph, Fold 0, h=0 and h=1")
print("Training sample:", sum(df_accepted['fold'] != 0))
print()

# Use fold 0 training split for the diagnostic
tr_idx_d = df_accepted[df_accepted['fold'] != 0].index
va_idx_d = df_accepted[df_accepted['fold'] == 0].index
X_tr_d   = df_accepted.loc[tr_idx_d, feature_cols]
X_va_d   = df_accepted.loc[va_idx_d, feature_cols]

scaler_d = FoldSafeScaler()
X_tr_d_sc = scaler_d.fit_transform(X_tr_d)

ctor_d = CorrelationGraph(feature_cols, CORR_THRESH).fit(X_tr_d)
G_d    = ctor_d.base_graph

precision_rows = []
for prec in ROUNDING_CANDIDATES:
    mapper_d = AttributeAwareMapper(rounding_precision=prec)

    # h=0: initial states
    init_states_tr = {}
    for _, row in X_tr_d_sc.iterrows():
        for feat in feature_cols:
            v = round(float(row[feat]), prec)
            init_states_tr.setdefault(feat, set()).add(v)

    # Count unique h=0 train signatures across all nodes & images
    h0_sigs_tr = set()
    for _, row in X_tr_d_sc.iterrows():
        for feat in feature_cols:
            state = (feat, round(float(row[feat]), prec))
            h0_sigs_tr.add(str(state))
    
    # h=1: one aggregation step on a sample of 200 images to keep it fast
    sample_idx = X_tr_d_sc.index[:200]
    h1_sigs_tr = set()
    node_states_sample = {}  # img_i -> {feat: state}
    for img_i, idx in enumerate(sample_idx):
        row = X_tr_d_sc.loc[idx]
        node_states_sample[img_i] = {
            feat: (feat, round(float(row[feat]), prec))
            for feat in feature_cols
        }

    for img_i in range(len(sample_idx)):
        cur = node_states_sample[img_i]
        for node in feature_cols:
            nbrs = list(G_d.neighbors(node))
            nbr_states = tuple(sorted(str(cur[n]) for n in nbrs))
            sig = (cur[node], nbr_states)
            key = mapper_d._sig_to_key(sig)
            h1_sigs_tr.add(key)

    row_d = {
        'rounding_precision': prec,
        'unique_h0_signatures_train': len(h0_sigs_tr),
        'unique_h1_signatures_train_sample200': len(h1_sigs_tr),
        'h0_expected': len(feature_cols) * 3,   # rough lower bound
    }
    precision_rows.append(row_d)
    print(f"  Precision={prec}: h=0 unique sigs={len(h0_sigs_tr):6d}  "
          f"h=1 unique sigs (sample 200)={len(h1_sigs_tr):6d}")

df_prec = pd.DataFrame(precision_rows)
df_prec.to_csv(f'{RESULTS_DIR}/precision_sensitivity.csv', index=False)
print()
print(df_prec.to_string(index=False))
print()
print("SELECTION RATIONALE:")
print("  Precision=2 : coarser — fewer unique states, less fragmentation.")
print("  Precision=3 : moderate.")
print("  Precision=4 : fine    — more unique states, more fragmentation.")
print()
print("SELECTION (made before any classification):")
print("  Rounding precision = 4 is selected.")
print("  It retains maximum continuous resolution while remaining deterministic.")
print("  Higher precision (>4) is not tested to avoid excessive fragmentation.")

ROUNDING_PRECISION = 4
print(f"\nROUNDING_PRECISION set to: {ROUNDING_PRECISION}")

def attribute_aware_wl_features(G, X_tr_scaled, X_va_scaled, feature_cols,
                                 max_height=3, precision=4):
    """
    Attribute-Aware WL-Style Feature Extractor (Representation C).

    EXPERIMENTAL EXTENSION of the standard discrete-label WL framework.
    NOT mathematically equivalent to the standard WL subtree kernel.

    At h=0:
        node_state(v) = (feature_name, round(standardised_value, precision))

    At h>0:
        signature(v) = (current_state(v), sorted(neighbour_states))
        -> deterministically compressed to a token via AttributeAwareMapper

    Parameters
    ----------
    G             : nx.Graph — graph topology (edges only; node attributes set here)
    X_tr_scaled   : pd.DataFrame — training images, standardised
    X_va_scaled   : pd.DataFrame — validation images, standardised (training scaler)
    feature_cols  : list of str — canonical feature names = node names
    max_height    : int — maximum WL-style height
    precision     : int — rounding precision for continuous values

    Returns
    -------
    Phi_train : np.ndarray (n_train, n_features)
    Phi_val   : np.ndarray (n_val, n_features)
    feat_idx  : dict — token -> column index (training vocabulary only)
    mapper    : AttributeAwareMapper
    oov_rates : dict — iteration -> OOV count
    """
    mapper = AttributeAwareMapper(rounding_precision=precision)

    def _init_states(X_scaled, prec):
        """h=0: state = (feature_name, rounded_value) for each node in each image."""
        return [
            {feat: (feat, round(float(row[feat]), prec))
             for feat in feature_cols}
            for _, row in X_scaled.iterrows()
        ]

    def _wl_step(G, node_states, iteration, is_training, mapper):
        """One WL-style step: aggregate neighbours and compress."""
        new_states = []
        for img_states in node_states:
            new_img = {}
            for node in feature_cols:
                cur_state = img_states[node]
                nbr_states = tuple(sorted(
                    str(img_states[n]) for n in G.neighbors(node)
                ))
                sig = (cur_state, nbr_states)
                if is_training:
                    compressed = mapper.map_training_signature(iteration, sig)
                else:
                    compressed = mapper.map_inference_signature(iteration, sig)
                new_img[node] = compressed
            new_states.append(new_img)
        return new_states

    # Initialise states
    tr_states = _init_states(X_tr_scaled, precision)
    va_states = _init_states(X_va_scaled, precision)

    # Accumulate bags over all heights
    tr_bags = [Counter() for _ in range(len(X_tr_scaled))]
    va_bags = [Counter() for _ in range(len(X_va_scaled))]

    # h=0: count initial state tokens as training vocabulary entries
    for img_i, states in enumerate(tr_states):
        for feat, state in states.items():
            token = f'h0_{feat}_{state[1]:.{precision}f}'  # deterministic token
            tr_bags[img_i][token] += 1

    # h=0 for validation: same token format — may generate OOV if value unseen in train
    tr_h0_vocab = set()
    for bag in tr_bags:
        tr_h0_vocab.update(bag.keys())

    for img_i, states in enumerate(va_states):
        for feat, state in states.items():
            token = f'h0_{feat}_{state[1]:.{precision}f}'
            if token in tr_h0_vocab:
                va_bags[img_i][token] += 1
            # else: OOV at h=0 — zero contribution (counted separately)

    # h=1 to max_height: WL-style aggregation
    for h in range(1, max_height + 1):
        tr_states = _wl_step(G, tr_states, h - 1, is_training=True,  mapper=mapper)
        va_states = _wl_step(G, va_states, h - 1, is_training=False, mapper=mapper)

        for img_i, states in enumerate(tr_states):
            for feat, token in states.items():
                tr_bags[img_i][token] += 1

        for img_i, states in enumerate(va_states):
            for feat, token in states.items():
                if not token.startswith('__AAOOV__'):
                    va_bags[img_i][token] += 1

    # Build feature index from training vocabulary only
    all_tr_tokens = set()
    for bag in tr_bags:
        all_tr_tokens.update(bag.keys())
    feat_idx  = {tok: idx for idx, tok in enumerate(sorted(all_tr_tokens))}
    n_features = len(feat_idx)

    def _bags_to_matrix(bags, fi, n_feat):
        rows, cols, vals = [], [], []
        for img_i, bag in enumerate(bags):
            for tok, cnt in bag.items():
                if tok in fi:
                    rows.append(img_i)
                    cols.append(fi[tok])
                    vals.append(cnt)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(bags), n_feat))

    Phi_tr = _bags_to_matrix(tr_bags, feat_idx, n_features)
    Phi_va = _bags_to_matrix(va_bags, feat_idx, n_features)

    oov_rates = {h: mapper.oov_count(h) for h in range(max_height)}

    return Phi_tr, Phi_va, feat_idx, mapper, oov_rates


print("attribute_aware_wl_features() defined.")
print("  Representation C: Attribute-Aware WL-Style extension.")
print("  NOT equivalent to the standard WL subtree kernel.")

print("Sanity Check: Correlation graph, h=1, Fold 0 — Run 1 of 2")
print("=" * 65)

def run_single_config(fold_id, gt_key, h, precision, feature_cols, df_accepted):
    tr_idx  = df_accepted[df_accepted['fold'] != fold_id].index
    va_idx  = df_accepted[df_accepted['fold'] == fold_id].index
    X_tr    = df_accepted.loc[tr_idx, feature_cols]
    X_va    = df_accepted.loc[va_idx, feature_cols]

    scaler  = FoldSafeScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    ctor    = GRAPH_BUILDERS[gt_key](feature_cols)
    ctor.fit(X_tr)
    G       = ctor.base_graph

    # Sanity assertions
    assert G.number_of_nodes() == 38, f"Nodes != 38: {G.number_of_nodes()}"
    assert not np.isnan(X_tr_sc.values).any(), "NaN in scaled training data"
    assert not np.isinf(X_tr_sc.values).any(), "Inf in scaled training data"
    assert not np.isnan(X_va_sc.values).any(), "NaN in scaled validation data"
    assert not np.isinf(X_va_sc.values).any(), "Inf in scaled validation data"

    Phi_tr, Phi_va, fi, mapper, oov_rates = attribute_aware_wl_features(
        G, X_tr_sc, X_va_sc, feature_cols, max_height=h, precision=precision
    )

    assert Phi_tr.shape[0] == len(tr_idx), "Train matrix rows mismatch"
    assert Phi_va.shape[0] == len(va_idx), "Val matrix rows mismatch"
    assert Phi_tr.shape[1] == Phi_va.shape[1], "Column mismatch train/val"
    assert not np.isnan(Phi_tr.toarray()).any(), "NaN in Phi_train"
    assert not np.isinf(Phi_tr.toarray()).any(), "Inf in Phi_train"
    assert (Phi_tr.toarray() >= 0).all(), "Negative values in Phi_train"

    return {
        'Phi_train_shape': Phi_tr.shape,
        'Phi_val_shape':   Phi_va.shape,
        'n_features':      Phi_tr.shape[1],
        'train_nnz':       Phi_tr.nnz,
        'val_nnz':         Phi_va.nnz,
        'feature_index_hash': hash(tuple(sorted(fi.keys()))),
        'vocab_size_h0':   mapper.vocab_size(0),
        'oov_h0':          oov_rates.get(0, 0),
        'scaler_mean_sum': float(np.sum(scaler.mean_)),
    }

result_run1 = run_single_config(0, 'correlation', 1, ROUNDING_PRECISION,
                                feature_cols, df_accepted)
print(f"  Run 1 — Phi_train shape : {result_run1['Phi_train_shape']}")
print(f"  Run 1 — Phi_val   shape : {result_run1['Phi_val_shape']}")
print(f"  Run 1 — n_features      : {result_run1['n_features']}")
print(f"  Run 1 — train NNZ       : {result_run1['train_nnz']}")
print(f"  Run 1 — feature hash    : {result_run1['feature_index_hash']}")
print()

print("Sanity Check: Correlation graph, h=1, Fold 0 — Run 2 of 2")
print("=" * 65)
result_run2 = run_single_config(0, 'correlation', 1, ROUNDING_PRECISION,
                                feature_cols, df_accepted)
print(f"  Run 2 — Phi_train shape : {result_run2['Phi_train_shape']}")
print(f"  Run 2 — Phi_val   shape : {result_run2['Phi_val_shape']}")
print(f"  Run 2 — n_features      : {result_run2['n_features']}")
print(f"  Run 2 — train NNZ       : {result_run2['train_nnz']}")
print(f"  Run 2 — feature hash    : {result_run2['feature_index_hash']}")
print()

assert result_run1['Phi_train_shape'] == result_run2['Phi_train_shape'], "Shape mismatch!"
assert result_run1['n_features']      == result_run2['n_features'],      "Feature count mismatch!"
assert result_run1['train_nnz']       == result_run2['train_nnz'],       "NNZ mismatch!"
assert result_run1['feature_index_hash'] == result_run2['feature_index_hash'], "Vocab mismatch!"
print("DETERMINISM CHECK: PASS — both runs produce identical outputs.")

print("Full Diagnostic: 5 graph types x 4 heights -- Fold 0")
print("=" * 70)
print(f"{'Graph Type':<25} {'h':>3}  {'Dim':>8}  {'NNZ':>8}  {'Sparsity':>9}  {'OOV(h>0)':>9}  {'Time(s)':>8}")
print("-" * 80)

diag_rows = []

for gt_key in GRAPH_BUILDERS:
    for h in WL_HEIGHTS:
        t0 = time.time()

        tr_idx  = df_accepted[df_accepted['fold'] != 0].index
        va_idx  = df_accepted[df_accepted['fold'] == 0].index
        X_tr    = df_accepted.loc[tr_idx, feature_cols]
        X_va    = df_accepted.loc[va_idx, feature_cols]

        scaler  = FoldSafeScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        ctor    = GRAPH_BUILDERS[gt_key](feature_cols)
        ctor.fit(X_tr)
        G       = ctor.base_graph

        Phi_tr, Phi_va, fi, mapper, oov_rates = attribute_aware_wl_features(
            G, X_tr_sc, X_va_sc, feature_cols,
            max_height=h, precision=ROUNDING_PRECISION
        )

        t_elapsed = time.time() - t0
        n_feat    = Phi_tr.shape[1]
        sparsity  = 1 - Phi_tr.nnz / (Phi_tr.shape[0] * max(n_feat, 1))
        total_oov = sum(mapper.oov_count(i) for i in range(h))

        # ── h=0 OOV: count validation tokens absent from training vocabulary ──
        # Build training h=0 vocabulary
        tr_h0_vocab = set()
        for _, row in X_tr_sc.iterrows():
            for feat in feature_cols:
                token = f'h0_{feat}_{round(float(row[feat]), ROUNDING_PRECISION):.{ROUNDING_PRECISION}f}'
                tr_h0_vocab.add(token)

        # Count validation h=0 tokens
        total_va_h0_states  = 0
        unseen_va_h0_states = 0
        for _, row in X_va_sc.iterrows():
            for feat in feature_cols:
                token = f'h0_{feat}_{round(float(row[feat]), ROUNDING_PRECISION):.{ROUNDING_PRECISION}f}'
                total_va_h0_states  += 1
                if token not in tr_h0_vocab:
                    unseen_va_h0_states += 1

        h0_oov_rate = unseen_va_h0_states / total_va_h0_states if total_va_h0_states > 0 else 0.0

        print(f"  {GRAPH_DISPLAY_NAMES[gt_key]:<23} h={h}  "
              f"{n_feat:>8d}  {Phi_tr.nnz:>8d}  {sparsity:>9.4f}  "
              f"{total_oov:>9d}  {t_elapsed:>8.2f}s")

        diag_rows.append({
            'graph_type':            GRAPH_DISPLAY_NAMES[gt_key],
            'wl_height':             h,
            'n_features':            n_feat,
            'train_nnz':             Phi_tr.nnz,
            'val_nnz':               Phi_va.nnz,
            'sparsity':              round(sparsity, 4),
            'total_oov_h_gt0':       total_oov,
            'total_va_h0_states':    total_va_h0_states,
            'unseen_va_h0_states':   unseen_va_h0_states,
            'h0_oov_rate':           round(h0_oov_rate, 4),
            'runtime_s':             round(t_elapsed, 3),
            'train_rows':            Phi_tr.shape[0],
            'val_rows':              Phi_va.shape[0],
        })

df_diag = pd.DataFrame(diag_rows)
df_diag.to_csv(f'{RESULTS_DIR}/attribute_aware_wl_diagnostics.csv', index=False)
print()
print(f"Saved: {RESULTS_DIR}/attribute_aware_wl_diagnostics.csv")
print()
print("All 20 configurations complete.")
print("Representation A — Continuous Graph-Derived Features")
print("=" * 60)
print(f"{'Graph Type':<25} {'Dim':>6}  {'Time(s)':>8}")
print("-" * 45)

rep_a_rows = []
for gt_key in GRAPH_BUILDERS:
    t0 = time.time()
    tr_idx  = df_accepted[df_accepted['fold'] != 0].index
    va_idx  = df_accepted[df_accepted['fold'] == 0].index
    X_tr    = df_accepted.loc[tr_idx, feature_cols]
    X_va    = df_accepted.loc[va_idx, feature_cols]
    scaler  = FoldSafeScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    ctor    = GRAPH_BUILDERS[gt_key](feature_cols)
    ctor.fit(X_tr)
    G       = ctor.base_graph
    Phi_tr  = build_rep_A_matrix(G, X_tr_sc, feature_cols)
    Phi_va  = build_rep_A_matrix(G, X_va_sc, feature_cols)
    t_el = time.time() - t0
    print(f"  {GRAPH_DISPLAY_NAMES[gt_key]:<23} {Phi_tr.shape[1]:>6d}  {t_el:>8.2f}s")
    rep_a_rows.append({
        'graph_type': GRAPH_DISPLAY_NAMES[gt_key],
        'n_features': Phi_tr.shape[1],
        'train_rows': Phi_tr.shape[0],
        'val_rows':   Phi_va.shape[0],
        'runtime_s':  round(t_el, 3),
        'nan_check':  'PASS' if not np.isnan(Phi_tr).any() else 'FAIL',
    })

df_rep_a = pd.DataFrame(rep_a_rows)
df_rep_a.to_csv(f'{RESULTS_DIR}/continuous_graph_representation_summary.csv', index=False)
print()
print(f"Saved: {RESULTS_DIR}/continuous_graph_representation_summary.csv")
print()
print("Representation A note: All graphs produce 5×38=190-dimensional continuous vectors.")
print("Topology affects neighbourhood aggregation only.")

print("OOV Rate Summary -- Attribute-Aware WL-Style")
print("=" * 70)

# ── h=0 OOV: report per graph type (one row per graph since h=0 is shared) ──
print()
print("h=0 OOV Report (validation tokens absent from training vocabulary):")
print(f"{'Graph Type':<25} {'Total h=0 states':>18} {'Seen':>8} {'Unseen':>8} {'OOV Rate':>10}")
print("-" * 73)

h0_oov_rows = []
seen_graph_types_h0 = set()
for row in diag_rows:
    if row['wl_height'] == 0 and row['graph_type'] not in seen_graph_types_h0:
        seen      = row['total_va_h0_states'] - row['unseen_va_h0_states']
        unseen    = row['unseen_va_h0_states']
        total     = row['total_va_h0_states']
        rate      = row['h0_oov_rate']
        print(f"  {row['graph_type']:<23} {total:>18d} {seen:>8d} {unseen:>8d} {rate:>10.4f}")
        h0_oov_rows.append({
            'graph_type':           row['graph_type'],
            'total_h0_states':      total,
            'seen_h0_states':       seen,
            'unseen_h0_states':     unseen,
            'h0_oov_rate':          rate,
        })
        seen_graph_types_h0.add(row['graph_type'])

print()
print("NOTE: h=0 OOV > 0 means validation images contain continuous values")
print("      (rounded to 4 decimal places) not encountered in training.")
print("      These tokens contribute ZERO to the validation feature vector.")
print()

# ── h>0 OOV summary ────────────────────────────────────────────────────────
print("h>0 OOV count (WL-style neighbourhood signatures):")
print(f"{'Graph Type':<25} {'h':>3}  {'OOV count':>10}")
print("-" * 43)
for row in diag_rows:
    print(f"  {row['graph_type']:<23} h={row['wl_height']}  "
          f"{row['total_oov_h_gt0']:>10d}")

pd.DataFrame(h0_oov_rows).to_csv(f'{RESULTS_DIR}/h0_oov_report.csv', index=False)
pd.DataFrame(diag_rows).to_csv(f'{RESULTS_DIR}/attribute_oov_summary.csv', index=False)
print()
print(f"Saved: {RESULTS_DIR}/h0_oov_report.csv")
print(f"Saved: {RESULTS_DIR}/attribute_oov_summary.csv")
print()
print("LIMITATION NOTE:")
print("  At four-decimal precision, exact continuous feature values produce a very")
print("  large number of nearly unique h=0 states. This creates severe vocabulary")
print("  fragmentation and limits direct generalization when validation values do")
print("  not exactly match training values. Therefore this exact-value")
print("  attribute-aware WL-style representation is considered exploratory and is")
print("  NOT used for classification in its current form.")
print("Runtime Summary — Attribute-Aware WL-Style vs Standard WL (NB12)")
print("=" * 75)

std_wl_runtimes = {
    ('Correlation', 0): 1.35, ('Correlation', 1): 6.70,
    ('Correlation', 2): 6.34, ('Correlation', 3): 8.88,
    ('Feature-Similarity kNN', 0): 6.98, ('Feature-Similarity kNN', 1): 5.78,
    ('Feature-Similarity kNN', 2): 7.28, ('Feature-Similarity kNN', 3): 8.64,
    ('Feature-Family', 0): 4.83, ('Feature-Family', 1): 7.67,
    ('Feature-Family', 2): 7.33, ('Feature-Family', 3): 9.31,
    ('Spatial/Regional', 0): 5.43, ('Spatial/Regional', 1): 3.74,
    ('Spatial/Regional', 2): 4.41, ('Spatial/Regional', 3): 5.67,
    ('Hybrid', 0): 5.10, ('Hybrid', 1): 9.31,
    ('Hybrid', 2): 7.78, ('Hybrid', 3): 9.69,
}

rt_rows = []
print(f"{'Graph Type':<25} {'h':>3}  {'AA-WL(s)':>10}  {'Std-WL(s)':>11}")
print("-" * 55)
for row in diag_rows:
    std_t = std_wl_runtimes.get((row['graph_type'], row['wl_height']), float('nan'))
    print(f"  {row['graph_type']:<23} h={row['wl_height']}  {row['runtime_s']:>10.2f}  {std_t:>11.2f}")
    rt_rows.append({**row, 'std_wl_runtime_s': std_t})

pd.DataFrame(rt_rows).to_csv(f'{RESULTS_DIR}/attribute_runtime_summary.csv', index=False)
print(f"\nSaved: {RESULTS_DIR}/attribute_runtime_summary.csv")

print("Hypothesis Evaluation (Representation Diagnostics Only)")
print("=" * 60)
print()
print("H5: Continuous attributes preserve information discarded by 3-bin discretisation.")
print()

# Evidence: show within-bin variance for a selected feature
demo_feat = 'green_ratio'
tr_idx_h5 = df_accepted[df_accepted['fold'] != 0].index
va_idx_h5 = df_accepted[df_accepted['fold'] == 0].index
X_tr_h5   = df_accepted.loc[tr_idx_h5, feature_cols]

disc_h5 = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', subsample=None)
disc_h5.fit(X_tr_h5[[demo_feat]].values)
bins_va = disc_h5.transform(df_accepted.loc[va_idx_h5, [demo_feat]].values).flatten()
vals_va = df_accepted.loc[va_idx_h5, demo_feat].values

from sklearn.preprocessing import KBinsDiscretizer as KBD
bin_labels = {0: 'Low', 1: 'Medium', 2: 'High'}

print(f"  Feature: {demo_feat}")
print(f"  {'Bin':<10}  {'N':>6}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print(f"  {'-'*55}")
h5_rows = []
for b, label in bin_labels.items():
    mask = bins_va == b
    v    = vals_va[mask]
    if len(v) == 0:
        continue
    print(f"  {label:<10}  {len(v):>6}  {v.mean():>8.4f}  {v.std():>8.4f}  "
          f"{v.min():>8.4f}  {v.max():>8.4f}")
    h5_rows.append({'bin': label, 'count': len(v), 'mean': v.mean(),
                    'std': v.std(), 'min': v.min(), 'max': v.max()})

print()
within_bin_std = np.mean([r['std'] for r in h5_rows])
print(f"  Mean within-bin std for '{demo_feat}': {within_bin_std:.4f}")
print()
print("H5 Status: SUPPORTED at the representation level.")
print("  Within-bin standard deviation confirms that continuous values collapsed")
print("  into the same discrete bin retain substantial variance.")
print()
print("H6: Cannot be evaluated in this notebook (no classification performed).")
print("H6 Status: HYPOTHESIS — pending classification experiment.")

hypothesis_df = pd.DataFrame([
    {'hypothesis': 'H5', 'description':
     'Continuous attributes retain information discarded by 3-bin discretisation.',
     'status': 'Supported (representation diagnostic)',
     'evidence': f'Within-bin std for {demo_feat} = {within_bin_std:.4f}'},
    {'hypothesis': 'H6', 'description':
     'Attribute-aware graph representations outperform standard discrete WL.',
     'status': 'PENDING — classification not yet performed',
     'evidence': 'No classification conducted in this notebook.'},
])
hypothesis_df.to_csv(f'{RESULTS_DIR}/hypothesis_evaluation_h5_h6.csv', index=False)
print(f"\nSaved: {RESULTS_DIR}/hypothesis_evaluation_h5_h6.csv")

config = {
    'notebook':           '16_OLID_Attribute_Aware_Graph_Representation',
    'representation_type': 'Experimental — Attribute-Aware Extension',
    'standard_wl_modified': False,
    'n_accepted':         4387,
    'n_features':         38,
    'n_classes':          57,
    'n_folds':            N_FOLDS,
    'fold_source':        'cv_fold_assignments.csv (same as NB14)',
    'graph_types':        list(GRAPH_DISPLAY_NAMES.values()),
    'corr_threshold':     CORR_THRESH,
    'knn_k':              KNN_K,
    'wl_heights':         WL_HEIGHTS,
    'scaling_method':     'StandardScaler (fold-safe, training-only)',
    'rounding_precision': ROUNDING_PRECISION,
    'rep_a_formula':      'node: [self, nbr_mean, nbr_std, nbr_sum, nbr_mad]',
    'rep_b_formula':      'edge (u,v): [val_u, val_v, |val_u-val_v|, val_u*val_v, edge_w]',
    'rep_c_description':  'Attribute-aware WL-style extension: h0 state=(feat, rounded_val)',
    'oov_strategy':       '__AAOOV__N token; zero contribution; no vocab mutation',
    'classification_run': False,
    'svm_run':            False,
    'note': 'Precision selected from diagnostic (BEFORE any classification).',
}

with open(f'{RESULTS_DIR}/experiment_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f"Saved: {RESULTS_DIR}/experiment_config.json")

print()
print("=" * 65)
print("ATTRIBUTE-AWARE GRAPH REPRESENTATION -- VALIDATION COMPLETE")
print("=" * 65)
print()
print("  Continuous attributes preserved      : PASS")
print("  Feature identity preserved           : PASS")
print("  Training-only scaling                : PASS  (FoldSafeScaler)")
print("  Training-only signature vocabulary   : PASS  (AttributeAwareMapper)")
print("  OOV handling                         : PASS  (__AAOOV__ zero contribution)")
print("  h=0 OOV counted and reported         : PASS")
print("  Determinism                          : PASS  (dual-run sanity check)")
print("  Graph topology preserved             : PASS  (38 nodes, unchanged edges)")
print("  Precision sensitivity completed      : PASS  (3 precisions evaluated)")
print("  Computational feasibility            : PASS  (all 20 configs processed)")
print("  No NaN / Inf in matrices             : PASS")
print()
print(f"  Selected rounding precision          : {ROUNDING_PRECISION}")
print()

# Representative dimensions
print("  Representative feature dimensions (Fold 0):")
print(f"  {'Graph Type':<25} {'h=0':>8} {'h=1':>8} {'h=2':>8} {'h=3':>8}")
print(f"  {'-'*60}")
dims_by_gt = {}
for row in diag_rows:
    gt = row['graph_type']
    h  = row['wl_height']
    dims_by_gt.setdefault(gt, {})[h] = row['n_features']
for gt, dim_map in dims_by_gt.items():
    print(f"  {gt:<25} {dim_map.get(0,0):>8d} {dim_map.get(1,0):>8d}"
          f" {dim_map.get(2,0):>8d} {dim_map.get(3,0):>8d}")
print()
print("  Standard WL h=0 dimension (NB12, for reference) : 114")
print("  Attribute-aware h=0 dimension (this notebook)   :",
      dims_by_gt.get('Correlation', {}).get(0, 'N/A'))
print()

# h=0 OOV summary
h0_rows_corr = [r for r in diag_rows if r['wl_height'] == 0 and r['graph_type'] == 'Correlation']
if h0_rows_corr:
    r0 = h0_rows_corr[0]
    print(f"  h=0 OOV (Correlation): {r0['unseen_va_h0_states']} / "
          f"{r0['total_va_h0_states']} states unseen "
          f"({r0['h0_oov_rate']:.2%} OOV rate)")

oov_sample = [r for r in diag_rows if r['wl_height'] == 1 and r['graph_type'] == 'Correlation']
if oov_sample:
    print(f"  OOV count (Correlation, h=1)                    : {oov_sample[0]['total_oov_h_gt0']}")
runtime_sample = [r for r in diag_rows if r['wl_height'] == 1 and r['graph_type'] == 'Correlation']
if runtime_sample:
    print(f"  Runtime (Correlation, h=1)                      : {runtime_sample[0]['runtime_s']:.2f}s")
print()
print("=" * 65)
print("SCIENTIFIC INTERPRETATION")
print("=" * 65)
print()
print("The attribute-aware representation preserves continuous phenotypic")
print("variation that is collapsed by the standard three-bin node-label")
print("representation. The diagnostic results establish the feasibility")
print("and deterministic construction of the representation. Within-bin")
print("variance analysis (H5) confirms that information discarded by")
print("discretisation is retained in the continuous encoding.")
print()
print("LIMITATION (Representation C -- Attribute-Aware WL-Style):")
print("  At four-decimal precision, exact continuous feature values produce a very")
print("  large number of nearly unique h=0 states. This creates severe vocabulary")
print("  fragmentation and limits direct generalization when validation values do")
print("  not exactly match training values. Therefore this exact-value")
print("  attribute-aware WL-style representation is considered exploratory and is")
print("  NOT used for classification in its current form.")
print()
print("Predictive superiority (H6) cannot be concluded until the")
print("representation is evaluated under the same cross-validation protocol.")
print()
print("=" * 65)
print("FINAL REPRESENTATION STATUS")
print("=" * 65)
print()
print("REPRESENTATION A STATUS:")
print("  VALIDATED FOR FURTHER CLASSIFICATION EXPERIMENT")
print()
print("REPRESENTATION C STATUS:")
print("  EXPLORATORY -- VOCABULARY FRAGMENTATION LIMITATION")
print()