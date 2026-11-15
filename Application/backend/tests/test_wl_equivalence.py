"""The backend WL engine must give the same kernels as the research implementation.

Reference: Task_Direction_Audit/Tomato_Nested_CV/run_tomato_nested_cv.py (networkx WL).
Run from the repository root:  python -m pytest Application/backend/tests -q
"""
import importlib.machinery
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'Application', 'backend'))
from app.features import FEATURES  # noqa: E402
from app.wl import WLFeaturizer, cosine_kernel  # noqa: E402

REF_DIR = os.path.join(ROOT, 'Task_Direction_Audit', 'Tomato_Nested_CV')
REF = os.path.join(REF_DIR, 'run_tomato_nested_cv.py')
REF_PYC = os.path.join(REF_DIR, '__pycache__', 'run_tomato_nested_cv.cpython-311.pyc')
DATA = os.path.join(ROOT, 'full_pipeline', 'olid_full_features.csv')


def _load_reference():
    """Load the research WL implementation from source, or from its compiled cache."""
    if os.path.exists(REF):
        loader = importlib.machinery.SourceFileLoader('ref', REF)
    else:
        loader = importlib.machinery.SourcelessFileLoader('ref', REF_PYC)
    spec = importlib.util.spec_from_loader('ref', loader)
    ref = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(ROOT)  # the reference script creates its output folder relative to the cwd
    try:
        loader.exec_module(ref)
    finally:
        os.chdir(cwd)
    return ref


@pytest.mark.skipif(not (os.path.exists(REF) or os.path.exists(REF_PYC)), reason='reference implementation not present')
@pytest.mark.parametrize('graph', ['correlation', 'knn', 'family', 'spatial', 'hybrid'])
def test_matches_reference(graph):
    ref = _load_reference()

    df = pd.read_csv(DATA)
    df = df[(df.quality_flag == 'ACCEPT') & (df.crop == 'tomato')].reset_index(drop=True)
    X = df[FEATURES]
    tr, te = X.iloc[:450].reset_index(drop=True), X.iloc[450:].reset_index(drop=True)

    builders = {'correlation': lambda c: c.fit(tr, threshold=0.7), 'knn': lambda c: c.fit(tr, k=3),
                'family': lambda c: c.fit(tr, ref.FEATURE_FAMILIES),
                'spatial': lambda c: c.fit(tr, ref.SPATIAL_FEATURES), 'hybrid': lambda c: c.fit(tr)}
    for h in [0, 1, 2, 3]:
        c = ref.GRAPH_BUILDERS[graph](FEATURES)
        builders[graph](c)
        disc = ref.fit_discretizer(tr)
        Pt, Pv, _, _ = ref.wl_subtree_features(c.transform(ref.transform_discretizer(tr, disc)),
                                               c.transform(ref.transform_discretizer(te, disc)), max_height=h)
        K_tr, K_te = ref.build_kernel_matrix(Pt, Pv)

        f = WLFeaturizer(graph, h).fit(tr.values)
        P, Q = f.transform(tr.values), f.transform(te.values)
        assert np.allclose(cosine_kernel(P), K_tr), (graph, h, 'train')
        assert np.allclose(cosine_kernel(P, Q), K_te), (graph, h, 'test')
