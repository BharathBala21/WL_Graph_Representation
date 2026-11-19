"""WL graph-kernel classifier for leaf measurement graphs.

Same method as the research pipeline (NB11-NB14, NB22): every leaf is a graph whose
38 nodes are the measurements; node labels are training-fold quantile bins
(Low / Medium / High); edges come from one of five topologies learned on the
training data; the WL subtree kernel feeds an SVM.

This module packages that method as a fit / predict estimator so a trained model can
be saved and applied to new photos. Graph builders follow the definitions in
Task_Direction_Audit/Tomato_Nested_CV/run_tomato_nested_cv.py, and the WL relabelling
is equivalent to its networkx implementation (verified by tests/test_wl_equivalence.py).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer, QuantileTransformer, StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_FAMILIES, FEATURES, SPATIAL

N = len(FEATURES)
IDX = {f: i for i, f in enumerate(FEATURES)}


class LeafMeasurementKernel:
    """Laplacian similarity on training percentiles of measurements and within-leaf contrasts."""

    @staticmethod
    def measurements(X):
        X = np.asarray(X, float)
        def col(name):
            return X[:, IDX[name]]
        rgb = X[:, [IDX['mean_R'], IDX['mean_G'], IDX['mean_B']]]
        chromaticity = rgb / np.maximum(rgb.sum(axis=1, keepdims=True), 1e-9)
        variation = [col('std_' + channel) / np.maximum(col('mean_' + channel), 1e-9)
                     for channel in ('R', 'G', 'B', 'S', 'V', 'L')]
        return np.column_stack([X, chromaticity, *variation, col('core_mean_H') - col('edge_mean_H'),
                                col('core_yellow_ratio') - col('edge_yellow_ratio')])

    def fit(self, X):
        self.ranks_ = QuantileTransformer(n_quantiles=min(100, len(X)), random_state=42)
        self.train_ = self.ranks_.fit_transform(self.measurements(X))
        return self

    def kernel(self, X=None):
        ranks = self.train_ if X is None else self.ranks_.transform(self.measurements(X))
        return np.exp((-4 / ranks.shape[1]) * cdist(ranks, self.train_, 'cityblock'))


# ---------------------------------------------------------------- topologies
def _abs_corr(X):
    C = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
    np.fill_diagonal(C, 0.0)
    return C


def adj_correlation(X, threshold=0.70):
    return _abs_corr(X) >= threshold


def adj_knn(X, k=3):
    Z = StandardScaler().fit_transform(X).T
    _, ind = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    A = np.zeros((N, N), bool)
    for i, nb in enumerate(ind):
        A[i, nb[1:]] = True
        A[nb[1:], i] = True
    return A


def adj_family(X):
    C = _abs_corr(X)
    A = np.zeros((N, N), bool)
    fams = [[IDX[f] for f in fl] for fl in FEATURE_FAMILIES.values()]
    for ii in fams:
        for a in ii:
            for b in ii:
                if a != b:
                    A[a, b] = True
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            sub = C[np.ix_(fams[i], fams[j])]
            r, c = np.unravel_index(np.argmax(sub), sub.shape)
            A[fams[i][r], fams[j][c]] = A[fams[j][c], fams[i][r]] = True
    return A


def adj_spatial(X, threshold=0.60):
    C = _abs_corr(X)
    A = np.zeros((N, N), bool)
    core = [IDX[f] for f in SPATIAL if f.startswith('core')]
    edge = [IDX[f] for f in SPATIAL if f.startswith('edge')]
    for grp in (core, edge):
        for a in grp:
            for b in grp:
                if a != b:
                    A[a, b] = True
    for c, e in zip(core, edge):
        A[c, e] = A[e, c] = True
    spatial_idx = core + edge
    for g in range(N):
        if g in spatial_idx:
            continue
        for s in spatial_idx:
            if C[g, s] >= threshold:
                A[g, s] = A[s, g] = True
    return A


def adj_hybrid(X):
    return adj_correlation(X, 0.80) | adj_family(X) | adj_spatial(X, 0.70)


GRAPHS = {'correlation': adj_correlation, 'knn': adj_knn, 'family': adj_family,
          'spatial': adj_spatial, 'hybrid': adj_hybrid}
GRAPH_NAMES = {'correlation': 'Correlation', 'knn': 'Feature-Similarity kNN', 'family': 'Feature-Family',
               'spatial': 'Spatial', 'hybrid': 'Hybrid'}


# ---------------------------------------------------------------- WL features
class WLFeaturizer:
    """Fit on training leaves; transform any leaves into WL subtree-count vectors."""

    def __init__(self, graph='hybrid', h=2, n_bins=3):
        self.graph, self.h, self.n_bins = graph, h, n_bins

    def fit(self, X, *, discretizer=None):
        X = np.asarray(X, float)
        self.disc_ = discretizer if discretizer is not None else KBinsDiscretizer(
            n_bins=self.n_bins, encode='ordinal', strategy='quantile', subsample=None).fit(X)
        self.A_ = GRAPHS[self.graph](X)
        self.nbrs_ = [np.where(self.A_[v])[0] for v in range(N)]
        L = self.disc_.transform(X).astype(np.int64)
        sizes = [[self.n_bins] * N]
        self.luts_ = []
        for _ in range(self.h):
            off = self._offsets(sizes[-1])
            newL = np.empty_like(L)
            luts, sz = [], []
            for v, nb in enumerate(self.nbrs_):
                sig = self._sig(L, off, v, nb)
                uniq, inv = np.unique(sig, axis=0, return_inverse=True)
                newL[:, v] = inv.ravel()
                luts.append({tuple(r): i for i, r in enumerate(uniq.tolist())})
                sz.append(len(uniq))
            self.luts_.append(luts)
            sizes.append(sz)
            L = newL
        self.sizes_ = sizes
        return self

    @staticmethod
    def _offsets(sizes):
        return np.concatenate([[0], np.cumsum(sizes)[:-1]])

    @staticmethod
    def _sig(L, off, v, nb):
        # own label + sorted multiset of neighbour labels; neighbour labels carry node identity
        g = np.where(L[:, nb] >= 0, L[:, nb] + off[nb], -1)
        return np.concatenate([L[:, [v]], np.sort(g, axis=1)], axis=1)

    def labels(self, X):
        """Node labels of every leaf at WL iterations 0..h (list of h+1 arrays, -1 = pattern unseen in training)."""
        X = np.asarray(X, float)
        L = self.disc_.transform(X).astype(np.int64)
        levels = [L]
        for it, luts in enumerate(self.luts_):
            off = self._offsets(self.sizes_[it])
            newL = np.full_like(L, -1)
            for v, nb in enumerate(self.nbrs_):
                sig = self._sig(L, off, v, nb)
                newL[:, v] = [luts[v].get(tuple(r), -1) for r in sig.tolist()]  # -1 = unseen pattern
            levels.append(newL)
            L = newL
        return levels

    def transform(self, X):
        return self.encode_levels(self.labels(X))

    def encode_levels(self, levels):
        """Assemble one CSR matrix, keeping the original (iteration, node, label) column order."""
        columns = []
        offset = 0
        for Lk, sz in zip(levels, self.sizes_):
            columns.append(np.where(Lk >= 0, offset + self._offsets(sz) + Lk, -1))
            offset += sum(sz)
        ids = np.concatenate(columns, axis=1)
        valid = ids >= 0
        indices = ids[valid]
        indptr = np.r_[0, np.cumsum(valid.sum(axis=1))]
        return sp.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(ids), offset))

    def node_levels(self, X):
        """Low / Medium / High level (0, 1, 2) of every measurement."""
        return self.disc_.transform(np.asarray(X, float)).astype(int)


def cosine_kernel(P, Q=None):
    def _norm(M):
        n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
        n[n == 0] = 1.0
        return n
    Q = P if Q is None else Q
    return (Q @ P.T).toarray() / np.outer(_norm(Q), _norm(P))


class WLKernelSVC:
    """WL subtree kernel + SVM (C = 1, class-balanced), as in the research notebooks."""

    def __init__(self, graph='hybrid', h=2, C=1.0, probability=True, random_state=42,
                 add_measurements=False):
        self.graph, self.h, self.C = graph, h, C
        self.probability, self.random_state = probability, random_state
        self.add_measurements = add_measurements

    def fit(self, X, y):
        self.feat_ = WLFeaturizer(self.graph, self.h).fit(X)
        self.Ptr_ = self.feat_.transform(X)
        self.measurements_ = LeafMeasurementKernel().fit(X) if self.add_measurements else None
        self.svc_ = SVC(kernel='precomputed', C=self.C, class_weight='balanced',
                        probability=self.probability, random_state=self.random_state)
        self.svc_.fit(self.combine_kernel(cosine_kernel(self.Ptr_)), y)
        self.classes_ = self.svc_.classes_
        return self

    def combine_kernel(self, wl_kernel, X=None):
        # Models saved before measurement kernels were introduced contain WL state only.
        measurements = getattr(self, 'measurements_', None)
        return wl_kernel if measurements is None else (wl_kernel + measurements.kernel(X)) * 0.5

    def _k(self, X):
        return self.combine_kernel(cosine_kernel(self.Ptr_, self.feat_.transform(X)), X)

    def predict(self, X):
        return self.svc_.predict(self._k(X))

    def predict_proba(self, X):
        return self.svc_.predict_proba(self._k(X))


def stress_classifier(probability=True):
    """Fixed binary classifier used for every crop in the revised routed pipeline."""
    return WLKernelSVC('correlation', h=1, C=4, probability=probability, add_measurements=True)
