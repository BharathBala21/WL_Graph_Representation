"""Revision experiments (round four): SVM cost tuning, graph null models and joint-matching baselines.

Run from the repository root:  python frontires_draft/revision_experiments.py
Writes supplement S20-S24. Uses the archived pooled folds (S1) and the benchmark's known-crop splits
(StratifiedKFold(5, shuffle, seed 42) within each crop), so results pair with the earlier tables.

Questions addressed
  1. Does the depth-zero to depth-three gain survive tuning the SVM cost C?  (S20, S21)
  2. Does the gain come from shared neighbourhood patterns, or from deep patterns that are unique to each
     training image (which changes the kernel's scale between training and test)?  "Private-pattern"
     control: depth-zero matches plus private training-only coordinates of the same size.  (S20, S21)
  3. Do joint descriptor matches without any graph do as well?  Polynomial kernels on the one-hot
     quantile states count matching pairs (degree 2) and triples (degree 3) of descriptor states.  (S20, S21)
  4. How often do held-out labels match the training vocabulary at each depth?  (S23)
  5. Is the hybrid graph better than random graphs: label permutations, degree-preserving rewiring and
     random graphs with the same edge count?  (S22)
  6. Nested selection over graph x depth x C for pooled, known-crop and routed tasks.  (S21, S24)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Application/backend"))
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from app.features import FEATURES  # noqa: E402
from app.wl import GRAPHS, WLFeaturizer, adj_hybrid  # noqa: E402

N = len(FEATURES)
SEED = 42
CS = [0.01, 0.1, 1.0, 10.0, 100.0]
GRAPH_ORDER = ["correlation", "knn", "family", "spatial", "hybrid"]  # benchmark grid order
DEPTHS = [0, 1, 2, 3]
OUT = HERE / "supplement"
N_PERM, N_REWIRE, N_RANDOM = 200, 100, 100


# ---------------------------------------------------------------- data
def load():
    feat = pd.read_csv(ROOT / "full_pipeline/olid_full_features.csv").query("quality_flag == 'ACCEPT'")
    data = feat.merge(pd.read_csv(HERE / "supplement/S1_image_folds.csv"),
                      on=["image_name", "crop", "condition"], validate="one_to_one").reset_index(drop=True)
    assert len(data) == 4387
    return data, data[FEATURES].to_numpy(float), data.condition.ne("healthy").to_numpy(int)


def metrics(y, pred):
    return dict(macro_f1=f1_score(y, pred, average="macro"), balanced_accuracy=balanced_accuracy_score(y, pred),
                healthy_recall=recall_score(y, pred, pos_label=0, zero_division=0),
                stressed_recall=recall_score(y, pred, pos_label=1, zero_division=0))


# ---------------------------------------------------------------- kernels
def level_blocks(engine, X):
    """Sparse count matrices, one per WL level (0..h); unseen patterns have no column."""
    P = engine.transform(X)
    widths = [sum(s) for s in engine.sizes_]
    edges = np.concatenate([[0], np.cumsum(widths)])
    return [P[:, edges[i]:edges[i + 1]] for i in range(len(widths))]


def depth_kernels(engine, Xtr, Xte, depths=DEPTHS):
    """Yield (depth, K_train, K_test) for the cosine-normalised WL kernel truncated at each depth."""
    Btr, Bte = level_blocks(engine, Xtr), level_blocks(engine, Xte)
    G = np.zeros((Xtr.shape[0], Xtr.shape[0]))
    T = np.zeros((Xte.shape[0], Xtr.shape[0]))
    ntr, nte = np.zeros(Xtr.shape[0]), np.zeros(Xte.shape[0])
    for d, (a, b) in enumerate(zip(Btr, Bte)):
        G += (a @ a.T).toarray()
        T += (b @ a.T).toarray()
        ntr += np.asarray(a.multiply(a).sum(1)).ravel()
        nte += np.asarray(b.multiply(b).sum(1)).ravel()
        if d in depths:
            str_, ste = np.sqrt(np.maximum(ntr, 1e-12)), np.sqrt(np.maximum(nte, 1e-12))
            yield d, G / np.outer(str_, str_), T / np.outer(ste, str_)


def match_counts(Ltr, Lte):
    """Number of matching descriptor states between images (one-hot dot product of depth-zero bins)."""
    mtr = sum((Ltr[:, [v]] == Ltr[:, v]).astype(np.float64) for v in range(N))
    mte = sum((Lte[:, [v]] == Ltr[:, v]).astype(np.float64) for v in range(N))
    return mtr, mte


def baseline_kernels(engine, Xtr, Xte):
    """Graph-free joint-matching baselines and the private-pattern control."""
    Ltr, Lte = engine.node_levels(Xtr), engine.node_levels(Xte)
    mtr, mte = match_counts(Ltr, Lte)
    out = {}
    for deg in (2, 3):
        out[f"poly{deg}"] = ((mtr / N) ** deg, (mte / N) ** deg)
    # private-pattern control: each training image gets 3N coordinates of its own (deep patterns never shared),
    # held-out images get none, exactly as if no deep pattern recurred; norms follow the WL depth-three case.
    extra = 3 * N
    Ktr = mtr / (N + extra) + np.eye(len(mtr)) * extra / (N + extra)
    Kte = mte / np.sqrt(N * (N + extra))
    out["private"] = (Ktr, Kte)
    return out


def fit_scores(Ktr, ytr, Kte, yte, Cs=CS):
    res = {}
    for C in Cs:
        pred = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, ytr).predict(Kte)
        res[C] = (pred, metrics(yte, pred))
    return res


def all_configs(Xtr, ytr, Xte, yte, graphs=GRAPH_ORDER, baselines=True):
    """Scores for every (model, graph, depth, C) on one train/test split, plus match statistics."""
    rows, preds, match = [], {}, []
    for g in graphs:
        engine = WLFeaturizer(g, 3).fit(Xtr)
        for d, Ktr, Kte in depth_kernels(engine, Xtr, Xte):
            if d == 0 and g != graphs[0]:
                continue  # depth zero is identical for every graph
            key_g = "any" if d == 0 else g
            for C, (pred, m) in fit_scores(Ktr, ytr, Kte, yte).items():
                rows.append(dict(model="WL", graph=key_g, depth=d, C=C, **m))
                preds[("WL", key_g, d, C)] = pred
        if baselines and g == graphs[0]:
            for name, (Ktr, Kte) in baseline_kernels(engine, Xtr, Xte).items():
                for C, (pred, m) in fit_scores(Ktr, ytr, Kte, yte).items():
                    rows.append(dict(model=name, graph="none", depth=-1, C=C, **m))
                    preds[(name, "none", -1, C)] = pred
        Ltr, Lte = engine.labels(Xtr), engine.labels(Xte)
        for d in DEPTHS:
            tr_labels = Ltr[d]
            uniq = np.mean([np.mean(np.unique(tr_labels[:, v], return_counts=True)[1][tr_labels[:, v]] == 1)
                            for v in range(N)])
            match.append(dict(graph=g, depth=d, test_matched=float(np.mean(Lte[d] >= 0)),
                              train_singleton=float(uniq), vocabulary=int(sum(engine.sizes_[d]))))
    return pd.DataFrame(rows), preds, pd.DataFrame(match)


def select(inner: pd.DataFrame, models, graphs, depths, Cs):
    """Best configuration by mean inner macro-F1; ties go to the first in grid order (as GridSearchCV)."""
    sub = inner[inner.model.isin(models) & inner.C.isin(Cs)
                & (((inner.model == "WL") & inner.graph.isin(graphs + ["any"]) & inner.depth.isin(depths))
                   | (inner.model != "WL"))]
    mean = sub.groupby(["model", "graph", "depth", "C"], sort=False).macro_f1.mean().reset_index()
    order = {g: i for i, g in enumerate(["any"] + GRAPH_ORDER + ["none"])}
    mean["o"] = [(order[g], d, list(CS).index(c)) for g, d, c in zip(mean.graph, mean.depth, mean.C)]
    mean = mean.sort_values("o", kind="stable")
    best = mean.loc[mean.macro_f1.idxmax()] if not mean.empty else None
    return best


SELECTIONS = {
    # name: (models, graphs, depths, Cs)
    "WL nested (graph x depth 1-3, C=1; original benchmark grid)": (["WL"], GRAPH_ORDER, [1, 2, 3], [1.0]),
    "WL nested (graph x depth 0-3 x C)": (["WL"], GRAPH_ORDER, DEPTHS, CS),
    "WL depth 0, tuned C": (["WL"], GRAPH_ORDER, [0], CS),
    "WL hybrid depth 3, tuned C": (["WL"], ["hybrid"], [3], CS),
    "WL hybrid depth 3, C=1": (["WL"], ["hybrid"], [3], [1.0]),
    "Pairwise matches (poly2), tuned C": (["poly2"], [], [], CS),
    "Triple matches (poly3), tuned C": (["poly3"], [], [], CS),
    "Private-pattern control, tuned C": (["private"], [], [], CS),
    "Private-pattern control, C=1": (["private"], [], [], [1.0]),
}


def nested_split(Xtr, ytr, Xte, yte, graphs=GRAPH_ORDER):
    """Inner 3-fold CV on the training split, then outer evaluation of every selection rule."""
    inner_rows = []
    for k, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xtr, ytr)):
        r, _, _ = all_configs(Xtr[a], ytr[a], Xtr[b], ytr[b], graphs)
        inner_rows.append(r.assign(inner=k))
    inner = pd.concat(inner_rows)
    outer, preds, match = all_configs(Xtr, ytr, Xte, yte, graphs)
    chosen = []
    for name, (models, gs, ds, cs) in SELECTIONS.items():
        best = select(inner, models, gs, ds, cs)
        key = (best.model, best.graph, int(best.depth), float(best.C))
        o = outer[(outer.model == key[0]) & (outer.graph == key[1]) & (outer.depth == key[2]) & (outer.C == key[3])]
        chosen.append(dict(rule=name, model=key[0], graph=key[1], depth=key[2], C=key[3],
                           inner_macro_f1=best.macro_f1, **o.iloc[0][list(metrics(yte, yte))].to_dict()))
        chosen[-1]["_pred"] = preds[key]
    return outer, pd.DataFrame(chosen), match


# ---------------------------------------------------------------- tasks
def pooled(data, X, y):
    outs, sels, matches = [], [], []
    for fold in sorted(data.fold.unique()):
        t0 = time.time()
        tr, te = data.fold.ne(fold).to_numpy(), data.fold.eq(fold).to_numpy()
        o, s, m = nested_split(X[tr], y[tr], X[te], y[te])
        outs.append(o.assign(task="pooled", crop="all", fold=fold))
        sels.append(s.drop(columns="_pred").assign(task="pooled", crop="all", fold=fold))
        matches.append(m.assign(task="pooled", crop="all", fold=fold))
        print(f"pooled fold {fold}: {time.time() - t0:.0f}s", flush=True)
    return outs, sels, matches


def known_crop(data, X, y):
    outs, sels = [], []
    for crop in sorted(data.crop.unique()):
        rows = np.where(data.crop == crop)[0]
        yc = y[rows]
        n = min(5, int(np.bincount(yc).min()))
        for f, (a, b) in enumerate(StratifiedKFold(n, shuffle=True, random_state=SEED).split(rows, yc)):
            o, s, _ = nested_split(X[rows[a]], yc[a], X[rows[b]], yc[b])
            outs.append(o.assign(task="known_crop", crop=crop, fold=f))
            sels.append(s.drop(columns="_pred").assign(task="known_crop", crop=crop, fold=f))
        print(f"known crop {crop} done", flush=True)
    return outs, sels


def routed(data, X, y):
    """Descriptor router (300-tree random forest) then a crop model chosen by nested CV on that crop's training images."""
    sels = []
    for fold in sorted(data.fold.unique()):
        tr, te = np.where(data.fold.ne(fold))[0], np.where(data.fold.eq(fold))[0]
        router = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1).fit(X[tr], data.crop.values[tr])
        routed_crop = router.predict(X[te])
        preds = {"predicted": {}, "true": {}}
        for crop in sorted(data.crop.unique()):
            ctr = tr[data.crop.values[tr] == crop]
            for route, mask in (("predicted", routed_crop == crop), ("true", data.crop.values[te] == crop)):
                cte = te[mask]
                if len(cte) == 0:
                    continue
                _, s, _ = nested_split(X[ctr], y[ctr], X[cte], y[cte])
                for _, r in s.iterrows():
                    preds[route].setdefault(r.rule, {}).update(dict(zip(cte, r._pred)))
        for route, by_rule in preds.items():
            for rule, mapping in by_rule.items():
                idx = np.array(sorted(mapping))
                pred = np.array([mapping[i] for i in idx])
                sels.append(dict(task=f"routed_{route}", fold=fold, rule=rule, n=len(idx),
                                 router_accuracy=float(np.mean(routed_crop == data.crop.values[te])),
                                 **metrics(y[idx], pred)))
        print(f"routed fold {fold} done", flush=True)
    return sels


# ---------------------------------------------------------------- graph null models
def rewire(A, rng, swaps_per_edge=10):
    """Degree-preserving double-edge swaps (no self loops, no multi-edges)."""
    A = A.copy()
    edges = np.array(np.where(np.triu(A, 1))).T.tolist()
    m = len(edges)
    target, done = swaps_per_edge * m, 0
    for _ in range(50 * target):
        if done >= target:
            break
        i, j = rng.choice(m, 2, replace=False)
        (a, b), (c, d) = edges[i], edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or A[a, d] or A[c, b]:
            continue
        A[a, b] = A[b, a] = A[c, d] = A[d, c] = False
        A[a, d] = A[d, a] = A[c, b] = A[b, c] = True
        edges[i], edges[j] = [a, d], [c, b]
        done += 1
    return A


def random_same_edges(A, rng):
    m = int(np.triu(A, 1).sum())
    iu = np.array(np.triu_indices(N, 1)).T
    pick = iu[rng.choice(len(iu), m, replace=False)]
    B = np.zeros_like(A)
    B[pick[:, 0], pick[:, 1]] = B[pick[:, 1], pick[:, 0]] = True
    return B


def nulls(data, X, y, path=None, depths=(1, 2, 3), Cs=(0.1, 1.0, 10.0)):
    """Hybrid graph versus random graphs; appends to `path` after every graph so an interrupted run resumes."""
    rows = []
    controls = ([("permuted", s) for s in range(N_PERM)] + [("rewired", s) for s in range(N_REWIRE)]
                + [("random", s) for s in range(N_RANDOM)])
    folds = sorted(data.fold.unique())
    splits = [(data.fold.ne(f).to_numpy(), data.fold.eq(f).to_numpy()) for f in folds]
    done = set()
    if path is not None and Path(path).exists():
        prev = pd.read_csv(path)
        done = set(zip(prev.control, prev.seed))
    t0 = time.time()
    for kind, seed in [("original", 0)] + controls:
        if (kind, seed) in done:
            continue
        rows = []
        if kind == "permuted":
            order = np.random.default_rng(seed).permutation(N)
            GRAPHS["null"] = lambda x, o=order: adj_hybrid(x)[np.ix_(o, o)]
        elif kind == "rewired":
            GRAPHS["null"] = lambda x, s=seed: rewire(adj_hybrid(x), np.random.default_rng(1000 + s))
        elif kind == "random":
            GRAPHS["null"] = lambda x, s=seed: random_same_edges(adj_hybrid(x), np.random.default_rng(2000 + s))
        else:
            GRAPHS["null"] = adj_hybrid
        for f, (tr, te) in zip(folds, splits):
            engine = WLFeaturizer("null", 3).fit(X[tr])
            for d, Ktr, Kte in depth_kernels(engine, X[tr], X[te], depths=list(depths)):
                for C in Cs:
                    pred = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, y[tr]).predict(Kte)
                    rows.append(dict(control=kind, seed=seed, fold=f, depth=d, C=C,
                                     edges=int(engine.A_.sum() // 2), vocabulary=sum(sum(s) for s in engine.sizes_[:d + 1]),
                                     **metrics(y[te], pred)))
        if path is not None:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not Path(path).exists(), index=False)
        if seed % 10 == 0:
            print(f"nulls: {kind} {seed} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def main():
    data, X, y = load()
    t0 = time.time()
    outs, sels, matches = pooled(data, X, y)
    pd.concat(outs).to_csv(OUT / "S20_c_and_baseline_grid_pooled.csv", index=False)
    pd.concat(matches).to_csv(OUT / "S23_vocabulary_matching.csv", index=False)
    kouts, ksels = known_crop(data, X, y)
    pd.concat(kouts).to_csv(OUT / "S20b_c_and_baseline_grid_known_crop.csv", index=False)
    pd.concat(sels + ksels).to_csv(OUT / "S21_nested_selection.csv", index=False)
    pd.DataFrame(routed(data, X, y)).to_csv(OUT / "S24_routed_nested.csv", index=False)
    print(f"tuning done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(nulls(data, X, y)).to_csv(OUT / "S22_graph_nulls.csv", index=False)
    print(f"all done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "nulls":
        # resumable rerun used for S22: depth three, C = 1 (the value nested tuning chose for hybrid depth three)
        N_PERM, N_REWIRE, N_RANDOM = 100, 50, 50
        data, X, y = load()
        nulls(data, X, y, path=OUT / "S22_graph_nulls.csv", depths=(3,), Cs=(1.0,))
        print("nulls done", flush=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke":
        N_PERM = N_REWIRE = N_RANDOM = 1
        data, X, y = load()
        tr, te = data.fold.ne(1).to_numpy(), data.fold.eq(1).to_numpy()
        t0 = time.time()
        o, s, m = all_configs(X[tr], y[tr], X[te], y[te])
        print(f"{time.time() - t0:.0f}s")
        print(o[(o.C == 1.0)].pivot_table(index=["model", "graph"], columns="depth", values="macro_f1").round(3))
        print(o[o.graph.isin(["any", "hybrid", "none"])].pivot_table(index=["model", "graph", "depth"], columns="C",
                                                                     values="macro_f1").round(3))
        print(m.round(3).to_string())
        A = adj_hybrid(X[tr])
        B = rewire(A, np.random.default_rng(0))
        print("rewire degrees equal:", (A.sum(0) == B.sum(0)).all(), "changed edges:", int((A & ~B).sum() // 2),
              "of", int(A.sum() // 2))
    else:
        main()
