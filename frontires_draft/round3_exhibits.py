"""Build manuscript tables, plots and data sheets from the round-three fold records."""
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

def update_round3(result, here, save):
    supplement = here / "supplement"

    def read(stem):
        return pd.read_csv(result / (stem + ".csv"))

    def emit(name, rows):
        (here / (name + "_rows.tex")).write_text(
            "\n".join(" & ".join(row) + r" \\" for row in rows) + "\n", encoding="utf-8"
        )

    folds = [read(f"benchmark_level{i}_folds") for i in (1, 2, 3)]
    overall = [d.groupby("model_family").macro_f1.mean() for d in folds]
    per_crop = folds[1].pivot(index=["crop", "fold"], columns="model_family", values="macro_f1")
    assert len(per_crop) == 40 and per_crop.notna().all().all()
    oracle = folds[2].groupby("model_family").oracle_macro_f1.mean()
    names = {
        "Gradient Boosting (HistGB)": "Histogram boosting", "LightGBM": "LightGBM",
        "Extra Trees (tuned)": "Tuned Extra Trees", "XGBoost": "XGBoost",
        "XGBoost (tuned)": "Tuned XGBoost", "CNN MobileNetV3-Large (frozen) + LR": "MobileNetV3",
        "CNN EfficientNet-B0 (frozen) + LR": "EfficientNet-B0", "WL + SVM": "Selected WL",
        "WL + SVM (tuned)": "Nested WL", "GIN": "GIN", "GIN (tuned)": "Nested GIN",
        "GCN": "GCN", "GAT": "GAT", "Random Forest": "Random forest",
    }
    def value(series, model):
        return f"{series[model]:.3f}" if model in series else "---"

    emit("comparison", [[label, *[value(s, model) for s in overall], value(oracle, model)]
                        for model, label in names.items()])
    folds[0].to_csv(supplement / "S3_pooled.csv", index=False)
    per_crop.groupby("crop").mean().to_csv(supplement / "S4_known_crop.csv")
    folds[1].to_csv(supplement / "S15_known_crop_folds.csv", index=False)

    graph = read("graph_study_level1_folds")
    assert len(graph) == 115 and graph.groupby(["graph", "model"]).fold.nunique().eq(5).all()
    graph_models = ["WL + SVM", "GIN", "GCN", "GAT"]
    graph_scores = graph.pivot_table(index="graph", columns="model", values="macro_f1")
    emit("graphs", [[label, *[f"{graph_scores.loc[key, m]:.3f}" if pd.notna(graph_scores.loc[key, m]) else "---"
                             for m in graph_models]]
                    for key, label in [("hybrid", "Hybrid"), ("family", "Family"), ("knn", "kNN"),
                                       ("correlation", "Correlation"), ("spatial", "Regional"), ("none", "No inter-node edges")]])
    graph.to_csv(supplement / "S16_graph_study.csv", index=False)

    comparisons = []
    for model in ["GIN", "GCN", "GAT", "GIN (tuned)", "XGBoost", "XGBoost (tuned)"]:
        wl = "WL + SVM (tuned)" if model == "GIN (tuned)" else "WL + SVM"
        delta = per_crop[wl] - per_crop[model]
        crop_delta = delta.groupby("crop").mean()
        comparisons.append((names[wl], names[model], delta.mean(), int(delta.gt(0).sum()),
                            wilcoxon(delta).pvalue, wilcoxon(crop_delta, method="exact").pvalue))
    stats = pd.DataFrame(comparisons, columns=["wl", "comparator", "mean_difference", "wl_wins_of_40",
                                              "fold_pair_p_unadjusted", "crop_pair_p_unadjusted"])
    stats.to_csv(supplement / "S17_paired_comparisons.csv", index=False)
    emit("paired", [[w, m, f"{d:+.3f}", str(n), "$<0.0001$" if f < .0001 else f"{f:.3f}", f"{c:.3f}"]
                    for w, m, d, n, f, c in comparisons])

    curves = pd.concat([read(f"learning_curve_level{i}") for i in (1, 2)])
    curves.to_csv(supplement / "S6_learning_curves.csv", index=False)
    rob = read("robustness_level1")
    rob.to_csv(supplement / "S5_perturbations.csv", index=False)
    clean = rob.query("perturbation == 'clean'").set_index("model_family").macro_f1
    change = rob.query("perturbation != 'clean'").groupby("model_family").macro_f1.mean() - clean
    shap = read("xgboost_shap_level1")
    shap.to_csv(supplement / "S18_xgboost_shap.csv", index=False)
    read("xgboost_shap_family_level1").to_csv(supplement / "S19_xgboost_shap_families.csv", index=False)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), layout="constrained")
    wanted = graph_models + ["XGBoost"]
    means = (curves.query("task == 'level2'").groupby(["model_family", "fraction", "crop"]).macro_f1.mean()
             .groupby(level=[0, 1]).mean())
    for model, marker in zip(wanted, ["o", "s", "^", "D", "x"]):
        series = means.loc[model]
        a.plot(series.index * 100, series, label=names[model], marker=marker, markersize=4)
    a.set(title="A   Training data", xlabel="Training images used (%)", ylabel="Mean crop macro-F1")
    a.legend(frameon=False, fontsize=7, ncol=2)
    bmodels = ["CNN EfficientNet-B0 (frozen) + LR"] + graph_models + ["XGBoost (tuned)"]
    b.barh([names[m] for m in bmodels], change.loc[bmodels], color="#306873")
    b.invert_yaxis()
    b.set(title="B   Simulated acquisition changes", xlabel="Mean macro-F1 change from clean")
    save(fig, "Figure8.jpg")
