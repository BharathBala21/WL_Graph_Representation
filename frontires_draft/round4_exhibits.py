"""Round-four exhibits: mechanism controls, graph nulls, segmentation coverage with image encoders, five-fold robustness.

Called at the end of refresh_exhibits.py. Inputs are the data sheets written by revision_experiments.py (S20-S24)
and image_experiments.py (S25-S28).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RULES = [  # (rule in S21/S24, label in the manuscript)
    ("WL depth 0, tuned C", "Descriptor states only (WL depth 0)"),
    ("Private-pattern control, C=1", "Depth 0 + private training patterns"),
    ("Pairwise matches (poly2), tuned C", "Pairs of descriptor states, no graph"),
    ("Triple matches (poly3), tuned C", "Triples of descriptor states, no graph"),
    ("WL hybrid depth 3, C=1", "WL, hybrid graph, depth 3"),
    ("WL nested (graph x depth 1-3, C=1; original benchmark grid)", "WL, nested graph and depth"),
    ("WL nested (graph x depth 0-3 x C)", "WL, nested graph, depth and $C$"),
]
NULL_LABELS = {"permuted": "Permuted descriptor positions", "rewired": "Degree-preserving rewiring",
               "random": "Random graph, same edge count"}
MODEL_NAMES = {"WL + SVM (hybrid, h=3)": "Selected WL", "Extra Trees (tuned)": "Tuned Extra Trees",
               "XGBoost (tuned)": "Tuned XGBoost", "EfficientNet-B0": "EfficientNet-B0",
               "MobileNetV3-Large": "MobileNetV3", "k-NN (k=5)": "Nearest neighbors", "RBF SVM": "RBF SVM",
               "Random Forest": "Random forest", "Extra Trees": "Extra Trees", "Gradient Boosting (HistGB)":
               "Histogram boosting", "XGBoost": "XGBoost", "LightGBM": "LightGBM", "Logistic Regression":
               "Logistic regression", "MLP (128-64)": "MLP"}
CHANGES = ["darker_25", "brighter_25", "warm_cast", "blur", "jpeg_q20", "noise", "half_resolution"]
CHANGE_LABELS = ["Darker", "Brighter", "Warm cast", "Blur", "JPEG", "Noise", "Half size"]


def update_round4(result, here, save):
    sup = here / "supplement"

    def emit(name, rows):
        (here / (name + "_rows.tex")).write_text("\n".join(" & ".join(r) + r" \\" for r in rows) + "\n",
                                                 encoding="utf-8")

    sel = pd.read_csv(sup / "S21_nested_selection.csv")
    routed = pd.read_csv(sup / "S24_routed_nested.csv")
    nulls = pd.read_csv(sup / "S22_graph_nulls.csv")
    pooled = sel[sel.task == "pooled"]
    known = sel[sel.task == "known_crop"].groupby(["rule", "crop"]).macro_f1.mean().groupby("rule").mean()
    rpred = routed[routed.task == "routed_predicted"].groupby("rule").macro_f1.mean()

    # ---- mechanism table
    rows = []
    for rule, label in RULES:
        p = pooled[pooled.rule == rule]
        rows.append([label, f"{p.macro_f1.mean():.3f} $\\pm$ {p.macro_f1.std():.3f}",
                     f"{p.balanced_accuracy.mean():.3f}", f"{known[rule]:.3f}", f"{rpred[rule]:.3f}"])
    per_graph = nulls.groupby(["control", "seed"]).macro_f1.mean()
    original = float(per_graph.loc["original"].iloc[0])
    for kind, label in NULL_LABELS.items():
        s = per_graph.loc[kind]
        bal = nulls[nulls.control == kind].groupby("seed").balanced_accuracy.mean().mean()
        rows.append([f"{label} ({len(s)} graphs)", f"{s.mean():.3f} [{s.min():.3f}, {s.max():.3f}]",
                     f"{bal:.3f}", "---", "---"])
    emit("mechanism", rows)

    null_summary = []
    for kind in NULL_LABELS:
        s = per_graph.loc[kind]
        null_summary.append(dict(control=kind, graphs=len(s), mean=s.mean(), min=s.min(), max=s.max(),
                                 original=original, share_at_or_above_original=float((s >= original).mean()),
                                 empirical_p_one_sided=(1 + int((s >= original).sum())) / (1 + len(s))))
    pd.DataFrame(null_summary).to_csv(sup / "S22b_graph_null_summary.csv", index=False)

    # ---- Figure 4: (A) topology/depth sweep, (B) hybrid graph against random-graph nulls
    depth0 = pooled[pooled.rule == "WL depth 0, tuned C"].macro_f1.mean()
    triple = pooled[pooled.rule == "Triple matches (poly3), tuned C"].macro_f1.mean()
    private = pooled[pooled.rule == "Private-pattern control, C=1"].macro_f1.mean()
    abl = pd.read_csv(sup / "S8_topology_depth.csv")
    fig, (a, ax) = plt.subplots(1, 2, figsize=(7.4, 3.3), layout="constrained", width_ratios=[1, 1.15])
    for (name, g), marker in zip(abl.groupby("graph"), ["o", "s", "^", "D", "x"]):
        g = g.sort_values("depth")
        a.plot(g.depth, g.macro_f1, marker=marker, label=name, markersize=4)
    a.axhline(triple, color="grey", ls="--", lw=.8)
    a.text(0, triple + .002, "triples, no graph", fontsize=6.5, color="grey")
    a.set(xticks=[0, 1, 2, 3], xlabel="WL refinement depth", ylabel="Mean pooled macro-F1", ylim=(.59, .71),
          title="A   Graph and depth")
    a.legend(frameon=False, loc="lower right", fontsize=6.5, ncol=2)
    a.grid(axis="y", alpha=.18)
    lo = min(per_graph.min(), depth0) - .005
    bins = np.linspace(lo, max(per_graph.max(), original) + .005, 36)
    short = {"permuted": "Permuted positions", "rewired": "Rewired, same degrees", "random": "Random, same edge count"}
    for kind, color in zip(NULL_LABELS, ["#4C72B0", "#DD8452", "#55A868"]):
        ax.hist(per_graph.loc[kind], bins=bins, alpha=.55, label=short[kind], color=color)
    ax.yaxis.get_major_locator().set_params(integer=True)
    top = ax.get_ylim()[1]
    for x, text, style in [(original, "hybrid graph", "-"), (triple, "triples, no graph", "--"),
                           (private, "private patterns", "-."), (depth0, "depth 0", ":")]:
        ax.axvline(x, color="black", ls=style, lw=.9)
        ax.text(x, top * .98, text + " ", rotation=90, va="top", ha="right", fontsize=6.5)
    ax.set(xlabel="Pooled macro-F1, depth 3", ylabel="Number of graphs", title="B   Random graphs")
    ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(.27, .45))
    save(fig, "Figure4.jpg")

    # ---- comparison table with fold SDs, MLP and the graph-free reference
    folds = [pd.read_csv(result / f"benchmark_level{i}_folds.csv") for i in (1, 2, 3)]
    names = {"Gradient Boosting (HistGB)": "Histogram boosting", "LightGBM": "LightGBM",
             "Extra Trees (tuned)": "Tuned Extra Trees", "XGBoost": "XGBoost", "XGBoost (tuned)": "Tuned XGBoost",
             "MLP (128-64)": "MLP", "CNN MobileNetV3-Large (frozen) + LR": "MobileNetV3",
             "CNN EfficientNet-B0 (frozen) + LR": "EfficientNet-B0", "WL + SVM": "Selected WL",
             "WL + SVM (tuned)": "Nested WL", "GIN": "GIN", "GIN (tuned)": "Nested GIN", "GCN": "GCN", "GAT": "GAT",
             "Random Forest": "Random forest"}

    def ms(d, model, col="macro_f1"):
        g = d[d.model_family == model][col]
        return f"{g.mean():.3f} $\\pm$ {g.std():.3f}" if len(g) else "---"

    def crop_mean(model):
        g = folds[1][folds[1].model_family == model]
        return f"{g.groupby('crop').macro_f1.mean().mean():.3f}" if len(g) else "---"

    rows = []
    for model, label in names.items():
        pool = ms(folds[0], model)
        if model == "WL + SVM (tuned)":
            p = pooled[pooled.rule == RULES[5][0]].macro_f1
            pool = f"{p.mean():.3f} $\\pm$ {p.std():.3f}"
        oracle = folds[2][folds[2].model_family == model].oracle_macro_f1
        rows.append([label, pool, crop_mean(model), ms(folds[2], model), f"{oracle.mean():.3f}"])
    rule = "Triple matches (poly3), tuned C"
    p = pooled[pooled.rule == rule].macro_f1
    r = routed[(routed.task == "routed_predicted") & (routed.rule == rule)].macro_f1
    t = routed[(routed.task == "routed_true") & (routed.rule == rule)].macro_f1
    rows.append(["Triples, no graph", f"{p.mean():.3f} $\\pm$ {p.std():.3f}", f"{known[rule]:.3f}",
                 f"{r.mean():.3f} $\\pm$ {r.std():.3f}", f"{t.mean():.3f}"])
    emit("comparison", rows)

    # ---- extra paired known-crop comparisons for nested WL (same 40 crop-fold splits)
    from scipy.stats import wilcoxon
    kc = sel[sel.task == "known_crop"].pivot_table(index=["crop", "fold"], columns="rule", values="macro_f1")
    bench2 = folds[1].pivot_table(index=["crop", "fold"], columns="model_family", values="macro_f1")
    nested = kc[RULES[5][0]]
    gap = (nested - bench2["WL + SVM (tuned)"]).abs().max()
    assert gap < 1e-9, f"rerun nested WL differs from the archived benchmark by {gap}"
    extra, paired_rows = [], []
    for label, other in [("Triples, no graph", kc["Triple matches (poly3), tuned C"]),
                         ("Tuned Extra Trees", bench2["Extra Trees (tuned)"]),
                         ("EfficientNet-B0", bench2["CNN EfficientNet-B0 (frozen) + LR"])]:
        delta = nested - other
        crop_delta = delta.groupby("crop").mean()
        fp, cp = wilcoxon(delta).pvalue, wilcoxon(crop_delta, method="exact").pvalue
        extra.append(dict(wl="Nested WL", comparator=label, mean_difference=delta.mean(),
                          wl_wins_of_40=int(delta.gt(0).sum()), fold_pair_p_unadjusted=fp, crop_pair_p_unadjusted=cp))
        paired_rows.append(["Nested WL", label, f"{delta.mean():+.3f}", str(int(delta.gt(0).sum())),
                            "$<0.0001$" if fp < .0001 else f"{fp:.3f}", f"{cp:.3f}"])
    s17 = pd.read_csv(sup / "S17_paired_comparisons.csv")
    pd.concat([s17[~((s17.wl == "Nested WL") & s17.comparator.isin([r["comparator"] for r in extra]))],
               pd.DataFrame(extra)]).to_csv(sup / "S17_paired_comparisons.csv", index=False)
    with open(here / "paired_rows.tex", "a", encoding="utf-8") as fh:
        fh.write("\n".join(" & ".join(r) + r" \\" for r in paired_rows) + "\n")

    # ---- coverage with image encoders (S26)
    cov = pd.read_csv(sup / "S26_condition_coverage_cnn.csv")
    rows = []
    for _, c in cov[cov.condition != "Healthy"].iterrows():
        eff = c["EfficientNet-B0 A positive"]
        rows.append([c.condition, str(c.original), str(c.accepted), str(c.wl_positive),
                     f"{100 * c.wl_positive / c.original:.1f}", str(eff), f"{100 * eff / c.original:.1f}"])
    emit("error", rows)

    # ---- five-fold robustness (S27, S28)
    rob = pd.read_csv(sup / "S27_robustness_5fold.csv")
    com = pd.read_csv(sup / "S28_robustness_5fold_common.csv")
    summary = []
    for name, t in (("retained", rob), ("common", com)):
        pf = t[t.perturbation != "clean"].groupby(["model", "fold"]).change.mean()
        s = pf.groupby("model").agg(["mean", "std", "min", "max"])
        s["cohort"] = name
        summary.append(s.reset_index())
    summary = pd.concat(summary)
    rej = rob[rob.perturbation != "clean"].groupby(["model", "perturbation"]).rejected.sum().reset_index()
    summary.to_csv(sup / "S27b_robustness_5fold_summary.csv", index=False)
    rej.to_csv(sup / "S27c_robustness_rejections.csv", index=False)
    clean = rob[rob.perturbation == "clean"].groupby("model").macro_f1.agg(["mean", "std"])
    ret = summary[summary.cohort == "retained"].set_index("model")
    cmn = summary[summary.cohort == "common"].set_index("model")
    order = ret.sort_values("mean", ascending=False).index
    emit("robust", [[MODEL_NAMES.get(m, m), f"{clean.loc[m, 'mean']:.3f}",
                     f"${ret.loc[m, 'mean']:+.3f} \\pm {ret.loc[m, 'std']:.3f}$",
                     f"${cmn.loc[m, 'mean']:+.3f} \\pm {cmn.loc[m, 'std']:.3f}$"] for m in order])

    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    show = ["WL + SVM (hybrid, h=3)", "Extra Trees (tuned)", "XGBoost (tuned)", "EfficientNet-B0"]
    per = rob[rob.perturbation != "clean"].groupby(["model", "perturbation"]).change.agg(["mean", "std"])
    for j, m in enumerate(show):
        g = per.loc[m].loc[CHANGES]
        ax.bar([i + (j - 1.5) * .2 for i in range(7)], g["mean"], yerr=g["std"], width=.19, capsize=1.5,
               error_kw={"lw": .6}, label=MODEL_NAMES[m])
    ax.axhline(0, c="black", lw=.7)
    ax.set_xticks(range(7), CHANGE_LABELS)
    ax.set_ylabel("Macro-F1 difference from clean images")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=8)
    save(fig, "Figure6.jpg")
