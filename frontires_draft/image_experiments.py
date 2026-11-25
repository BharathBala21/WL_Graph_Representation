"""Revision image experiments (round four). Needs the OLID-I photos in dataset/OLID-I/images (git-ignored).

Run from the repository root, in order:
  python frontires_draft/image_experiments.py index           # find every photo of the feature table
  python frontires_draft/image_experiments.py cache           # 224 px copies of all 4,749 photos
  python frontires_draft/image_experiments.py embed           # frozen MobileNetV3-Large / EfficientNet-B0 embeddings
  python frontires_draft/image_experiments.py coverage        # S25, S26: CNN on every submitted photo
  python frontires_draft/image_experiments.py robust-extract  # 240 photos x 8 versions in every pooled test fold
  python frontires_draft/image_experiments.py robust-embed
  python frontires_draft/image_experiments.py robust-eval     # S27, S28: five-fold robustness

Coverage question: the descriptor pipeline cannot classify the 362 photos its mask filter rejects, but a frozen
image encoder needs no mask. The 4,387 accepted photos keep their archived folds (S1). Each of the 362 rejected
photos is assigned a fold by shuffling within its crop-condition class (seed 42) and dealing folds in turn.
Model A trains on the accepted training photos only (exactly the published training sets) and scores every photo
of the held-out fold, so only coverage changes. Model B also trains on the rejected training photos.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "New_Models_Baselines/03_Benchmark_Suite"))
sys.path.insert(0, str(ROOT / "New_Models_Baselines/04_CNN_Frozen_Features"))
import bench  # noqa: E402,F401  (torch / LightGBM before scikit-learn)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = ROOT / "dataset/OLID-I"   # derived caches and embeddings (git-ignored)
IMAGES = Path(os.environ.get("OLID_IMAGES", DATA / "images"))  # unpacked OLID-I photo folders
CACHE, EMB = DATA / "cache224", DATA / "embeddings"
RCACHE = DATA / "robust224"
INDEX = DATA / "image_index.csv"
OUT = HERE / "supplement"
ROBUST_FEATS = DATA / "robustness_features_5fold.csv"
SIDE, SEED, N_ROBUST = 224, 42, 240
PERTURBATIONS = ["clean", "darker_25", "brighter_25", "warm_cast", "blur", "jpeg_q20", "noise", "half_resolution"]
BACKBONES = {"MobileNetV3-Large": "mobilenet_v3_large", "EfficientNet-B0": "efficientnet_b0"}
CONDITION_NAMES = {"N_K": "Nitrogen + potassium", "N": "Nitrogen", "K": "Potassium", "MIT": "Mite",
                   "LM": "Leaf miner", "PM": "Powdery mildew", "PLEI": "Pumpkin leaf-eating insect"}


def table():
    """All 4,749 photos in feature-table order with local paths and folds."""
    df = pd.read_csv(ROOT / "full_pipeline/olid_full_features.csv")
    idx = pd.read_csv(INDEX)
    df = df.merge(idx, on="image_name", how="left", validate="one_to_one")
    s1 = pd.read_csv(OUT / "S1_image_folds.csv")[["image_name", "fold"]]
    df = df.merge(s1, on="image_name", how="left", validate="one_to_one")
    rej = df.fold.isna()
    rng = np.random.default_rng(SEED)
    start = 0  # deal folds in turn, continuing across classes so fold sizes stay balanced
    for _, g in df[rej].groupby(["crop", "condition"]):
        order = rng.permutation(g.index.to_numpy())
        df.loc[order, "fold"] = (start + np.arange(len(order))) % 5
        start = (start + len(order)) % 5
    df["fold"] = df.fold.astype(int)
    df["accepted"] = df.quality_flag.eq("ACCEPT")
    df["stressed"] = df.condition.ne("healthy").astype(int)
    df["cache"] = [str(CACHE / (Path(n).stem.replace(" ", "_") + ".jpg")) for n in df.image_name]
    return df


# ---------------------------------------------------------------- 1. photos, cache, embeddings
def index():
    found = {}
    for dirpath, _, files in os.walk(IMAGES):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                found.setdefault(f, os.path.join(dirpath, f))
    df = pd.read_csv(ROOT / "full_pipeline/olid_full_features.csv")
    missing = [n for n in df.image_name if n not in found]
    print(f"{len(found)} photo files; {len(df) - len(missing)} of {len(df)} table images found")
    if missing:
        print("missing examples:", missing[:5])
    pd.DataFrame({"image_name": df.image_name, "local_path": [found.get(n) for n in df.image_name]}).to_csv(INDEX, index=False)


def _cache_one(args):
    import cv2
    src, out = args
    if not os.path.exists(out):
        img = cv2.imread(src, cv2.IMREAD_REDUCED_COLOR_4)  # as cnn_pipeline.py: 1/4-size decode, then resize
        cv2.imwrite(out, cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out


def cache(workers=4):
    CACHE.mkdir(parents=True, exist_ok=True)
    df = table()
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_cache_one, zip(df.local_path, df.cache), chunksize=16))
    print("cached", len(df))


def embed():
    import torch
    from cnn_pipeline import embed_paths, load_backbone
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    EMB.mkdir(parents=True, exist_ok=True)
    df = table()
    for name, arch in BACKBONES.items():
        t0 = time.time()
        E = embed_paths(load_backbone(name), list(df.cache))
        np.save(EMB / f"{arch}.npy", E.astype(np.float32))
        print(f"{name}: {E.shape} in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 2. coverage
def coverage():
    from cnn_pipeline import head
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
    df = table()
    y = df.stressed.to_numpy()
    fold_rows, calls = [], []
    for name, arch in BACKBONES.items():
        E = np.load(EMB / f"{arch}.npy")
        for f in range(5):
            te = np.where(df.fold == f)[0]
            for variant, train_mask in (("A: accepted training photos", df.accepted),
                                        ("B: all training photos", pd.Series(True, index=df.index))):
                tr = np.where((df.fold != f) & train_mask)[0]
                pred = head().fit(E[tr], y[tr]).predict(E[te])
                acc = df.accepted.to_numpy()[te]
                for cohort, m in (("accepted", acc), ("all submitted", np.ones_like(acc)), ("rejected", ~acc)):
                    if m.sum() == 0:
                        continue
                    yy, pp = y[te][m], pred[m]
                    fold_rows.append(dict(model=name, variant=variant, fold=f, cohort=cohort, n=int(m.sum()),
                                          n_healthy=int((yy == 0).sum()),
                                          macro_f1=f1_score(yy, pp, average="macro") if len(set(yy)) > 1 else np.nan,
                                          balanced_accuracy=balanced_accuracy_score(yy, pp) if len(set(yy)) > 1 else np.nan,
                                          healthy_recall=recall_score(yy, pp, pos_label=0, zero_division=np.nan),
                                          stressed_recall=recall_score(yy, pp, pos_label=1, zero_division=np.nan)))
                calls.append(pd.DataFrame(dict(model=name, variant=variant, image_name=df.image_name.values[te],
                                               prediction=pred)))
        print(name, "done", flush=True)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "S25_cnn_all_photos_folds.csv", index=False)
    calls = pd.concat(calls)
    # per-condition screens out of every submitted photo; WL calls come from S11 (accepted photos only)
    wl = pd.read_csv(OUT / "S11_pooled_predictions.csv")[["image_name", "prediction"]]
    rows = []
    for cond in list(CONDITION_NAMES) + ["healthy"]:
        sub = df[df.condition == cond]
        n, a = len(sub), int(sub.accepted.sum())
        wl_pos = int(wl[wl.image_name.isin(sub.image_name)].prediction.sum())
        r = dict(condition=CONDITION_NAMES.get(cond, "Healthy"), original=n, accepted=a, wl_positive=wl_pos)
        for name in BACKBONES:
            for v in ("A", "B"):
                c = calls[(calls.model == name) & calls.variant.str.startswith(v) & calls.image_name.isin(sub.image_name)]
                r[f"{name} {v} positive"] = int(c.prediction.sum())
                acc_names = set(sub[sub.accepted].image_name)
                r[f"{name} {v} positive among accepted"] = int(c[c.image_name.isin(acc_names)].prediction.sum())
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "S26_condition_coverage_cnn.csv", index=False)
    summary = folds.groupby(["model", "variant", "cohort"])[["n", "macro_f1", "balanced_accuracy", "healthy_recall",
                                                             "stressed_recall"]].mean().round(3)
    print(summary.to_string())
    print(pd.DataFrame(rows).to_string())


# ---------------------------------------------------------------- 3. five-fold robustness
def robust_sample(df_acc, fold):
    """Same stratified rule as run_robustness.extract, applied to every pooled test fold."""
    test = df_acc[df_acc.fold == fold]
    return test.groupby(["crop", "stressed"], group_keys=False).apply(
        lambda g: g.sample(max(2, round(N_ROBUST * len(g) / len(test))), random_state=SEED) if len(g) > 2 else g)


def _robust_one(args):
    import cv2
    from run_robustness import perturb
    from app.features import FEATURES, measure_leaf
    path, name, fold = args
    img = cv2.imread(path)
    rows = []
    for kind in PERTURBATIONS:
        p = perturb(img, kind)
        out = RCACHE / kind / (Path(name).stem.replace(" ", "_") + ".jpg")
        if not out.exists():
            cv2.imwrite(str(out), cv2.resize(p, (SIDE, SIDE), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        m = measure_leaf(p)
        row = dict(image_name=name, fold=fold, perturbation=kind, quality_flag=m["quality_flag"])
        if m["features"]:
            row.update(dict(zip(FEATURES, m["features"])))
        rows.append(row)
        del p, m
    return rows


def robust_extract(workers=2):
    for kind in PERTURBATIONS:
        (RCACHE / kind).mkdir(parents=True, exist_ok=True)
    df = table()
    acc = df[df.accepted]
    sample = pd.concat([robust_sample(acc, f) for f in range(5)])
    old = pd.read_csv(ROOT / "New_Models_Baselines/03_Benchmark_Suite/results/robustness_features.csv")
    old_names = {Path(p.replace("\\", "/")).name for p in old.image_path}
    s0 = set(sample[sample.fold == 0].image_name)
    print(f"fold-0 sample reproduces the published 240 photos: {s0 == old_names} ({len(s0 & old_names)} shared)")
    done = set(pd.read_csv(ROBUST_FEATS).image_name) if ROBUST_FEATS.exists() else set()
    todo = [(p, n, f) for p, n, f in zip(sample.local_path, sample.image_name, sample.fold) if n not in done]
    print(f"{len(sample)} sampled photos, {len(todo)} to process", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, rows in enumerate(ex.map(_robust_one, todo, chunksize=1)):
            pd.DataFrame(rows).to_csv(ROBUST_FEATS, mode="a", header=not ROBUST_FEATS.exists(), index=False)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)} ({(time.time() - t0) / (i + 1):.1f}s per photo)", flush=True)


def robust_embed():
    import torch
    from cnn_pipeline import embed_paths, load_backbone
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    names = sorted(pd.read_csv(ROBUST_FEATS).image_name.unique())
    for name, arch in BACKBONES.items():
        model = load_backbone(name)
        R = {k: embed_paths(model, [str(RCACHE / k / (Path(n).stem.replace(" ", "_") + ".jpg")) for n in names])
             for k in PERTURBATIONS}
        np.savez(EMB / f"{arch}_robust5.npz", names=np.array(names), **R)
        print(name, "embedded", flush=True)


def robust_models():
    """The benchmark's settings (bench.models / bench.tuned_models), built here because bench.models also builds
    AdaBoost with an argument that newer scikit-learn versions no longer accept."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    S = bench.SEED
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    t = bench.tuned_models("level1")
    return {
        "WL + SVM (hybrid, h=3)": bench.WLSVM("hybrid", 3),
        "Logistic Regression": scaled(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=S)),
        "RBF SVM": scaled(SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=S)),
        "k-NN (k=5)": scaled(KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=S, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=S, n_jobs=1),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(class_weight="balanced", random_state=S),
        "XGBoost": bench.Encoded(XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                                               n_jobs=1, random_state=S, verbosity=0), balance=True),
        "LightGBM": bench.Encoded(LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", n_jobs=1,
                                                 random_state=S, verbose=-1)),
        "MLP (128-64)": scaled(MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=500, random_state=S)),
        "Extra Trees (tuned)": t["Extra Trees (tuned)"],
        "XGBoost (tuned)": t["XGBoost (tuned)"],
    }


def robust_eval():
    import warnings
    from cnn_pipeline import head
    from sklearn.metrics import balanced_accuracy_score, f1_score
    df = table()
    acc = df[df.accepted].reset_index(drop=True)
    feats = pd.read_csv(ROBUST_FEATS).merge(df[["image_name", "stressed"]], on="image_name")
    rows, preds = [], []
    for f in range(5):
        train = acc[acc.fold != f]
        Xtr, ytr = train[bench.FEATURES].to_numpy(float), train.stressed.to_numpy()
        ff = feats[feats.fold == f]
        for name, model in robust_models().items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xtr, ytr)
            for kind in PERTURBATIONS:
                d = ff[ff.perturbation == kind]
                ok = d[d.quality_flag == "ACCEPT"]
                p = model.predict(ok[bench.FEATURES].to_numpy(float))
                preds.append(pd.DataFrame(dict(fold=f, model=name, perturbation=kind, image_name=ok.image_name.values,
                                               y_true=ok.stressed.values, y_pred=p)))
                rows.append(dict(fold=f, model=name, perturbation=kind, n=len(d), rejected=int(len(d) - len(ok)),
                                 macro_f1=f1_score(ok.stressed, p, average="macro"),
                                 balanced_accuracy=balanced_accuracy_score(ok.stressed, p)))
        for name, arch in BACKBONES.items():
            E = np.load(EMB / f"{arch}.npy")
            tr = np.where(df.accepted & (df.fold != f))[0]
            clf = head().fit(E[tr], df.stressed.values[tr])
            R = np.load(EMB / f"{arch}_robust5.npz")
            pos = {n: i for i, n in enumerate(R["names"])}
            names = sorted(ff.image_name.unique())
            yy = df.set_index("image_name").loc[names, "stressed"].to_numpy()
            for kind in PERTURBATIONS:
                p = clf.predict(R[kind][[pos[n] for n in names]])
                preds.append(pd.DataFrame(dict(fold=f, model=name, perturbation=kind, image_name=names,
                                               y_true=yy, y_pred=p)))
                rows.append(dict(fold=f, model=name, perturbation=kind, n=len(names), rejected=0,
                                 macro_f1=f1_score(yy, p, average="macro"), balanced_accuracy=balanced_accuracy_score(yy, p)))
        print(f"robustness fold {f} done", flush=True)
    rows, preds = pd.DataFrame(rows), pd.concat(preds)
    clean = rows[rows.perturbation == "clean"].set_index(["fold", "model"]).macro_f1
    rows["change"] = rows.macro_f1.to_numpy() - clean.loc[list(zip(rows.fold, rows.model))].to_numpy()
    rows.to_csv(OUT / "S27_robustness_5fold.csv", index=False)
    # common-image cohort per fold: photos with a prediction from every model under every change
    common = []
    for f in range(5):
        p = preds[preds.fold == f]
        k = p.groupby("image_name").size()
        keep = set(k[k == p.groupby(["model", "perturbation"]).ngroups].index)
        for (m, kind), g in p[p.image_name.isin(keep)].groupby(["model", "perturbation"]):
            common.append(dict(fold=f, model=m, perturbation=kind, n=len(g),
                               macro_f1=f1_score(g.y_true, g.y_pred, average="macro")))
    common = pd.DataFrame(common)
    c0 = common[common.perturbation == "clean"].set_index(["fold", "model"]).macro_f1
    common["change"] = common.macro_f1.to_numpy() - c0.loc[list(zip(common.fold, common.model))].to_numpy()
    common.to_csv(OUT / "S28_robustness_5fold_common.csv", index=False)
    preds.to_csv(DATA / "robustness_5fold_predictions.csv", index=False)
    # summary: mean change over the 7 changes, then mean and SD over the 5 folds
    for name, t in (("all retained", rows), ("common cohort", common)):
        per_fold = t[t.perturbation != "clean"].groupby(["model", "fold"]).change.mean()
        s = per_fold.groupby("model").agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
        print(f"\n{name}\n", s.round(3).to_string())


if __name__ == "__main__":
    {"index": index, "cache": cache, "embed": embed, "coverage": coverage, "robust-extract": robust_extract,
     "robust-embed": robust_embed, "robust-eval": robust_eval}[sys.argv[1]]()
