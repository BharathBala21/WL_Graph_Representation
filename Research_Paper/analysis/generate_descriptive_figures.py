from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "analysis" / "input" / "olid_full_features.csv"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
df = pd.read_csv(CSV)
accepted = df.loc[df["quality_flag"].eq("ACCEPT")].copy()

# Selected descriptors are shown as distributions across all accepted full-table records.
features = ["area", "circularity", "eccentricity", "mean_G", "mean_H", "mean_L",
            "green_ratio", "yellow_ratio", "glcm_contrast", "glcm_homogeneity",
            "glcm_energy", "core_yellow_ratio", "edge_yellow_ratio"]
features = [f for f in features if f in accepted.columns]
fig, axes = plt.subplots(4, 4, figsize=(12, 8.5))
for ax, feature in zip(axes.flat, features):
    x = pd.to_numeric(accepted[feature], errors="coerce").dropna()
    ax.hist(x, bins=35, color="#6D91C5", edgecolor="white", linewidth=0.35)
    ax.set_title(feature, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=.18)
for ax in axes.flat[len(features):]:
    ax.axis("off")
fig.suptitle("Distributions of selected descriptors across accepted OLID-I feature records", fontsize=12)
fig.tight_layout(rect=[0,0,1,.97])
fig.savefig(FIG/"fig3_feature_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Correlation heat map across the 38 descriptors on the accepted full-table cohort.
feature_cols = [c for c in accepted.columns if c not in {"image_path", "image_name", "class", "crop", "condition", "quality_flag", "mask_area_ratio", "failure_reason"}]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(accepted[c])]
C = accepted[feature_cols].corr(method="pearson")
fig, ax = plt.subplots(figsize=(10.5, 8.8))
im = ax.imshow(C.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
ax.set_xticks(range(len(feature_cols)), feature_cols, rotation=90, fontsize=5.2)
ax.set_yticks(range(len(feature_cols)), feature_cols, fontsize=5.2)
ax.set_title("Pearson correlation among 38 handcrafted descriptors (accepted records)", fontsize=11)
fig.colorbar(im, ax=ax, fraction=.025, pad=.02, label="Pearson r")
fig.tight_layout()
fig.savefig(FIG/"fig4_feature_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Show stored mask-area-ratio distribution across the complete 4,749-row table.
ratio = pd.to_numeric(df["mask_area_ratio"], errors="coerce").dropna()
fig, ax = plt.subplots(figsize=(7.0, 3.3))
ax.hist(ratio, bins=60, color="#D95F5F", edgecolor="white", linewidth=.3, alpha=.72)
ax.axvline(.01, color="#333333", linestyle="--", linewidth=1.1, label="Lower acceptance threshold (0.01)")
ax.axvline(.98, color="#4C72B0", linestyle=":", linewidth=1.2, label="Upper suspicious-mask threshold (0.98)")
ax.set_xlabel("Stored mask area / image area")
ax.set_ylabel("Number of records")
ax.set_title("Mask-area ratio distribution for the full derived-feature table (n=4,749)")
ax.legend(frameon=False, fontsize=7)
ax.grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(FIG/"fig8_mask_ratio_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Generated descriptive figures from the full feature table:", ", ".join(["fig3_feature_distributions.png", "fig4_feature_correlation_heatmap.png", "fig8_mask_ratio_distribution.png"]))
