"""Regenerate the four manuscript figures from the bundled analysis CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PAPER = Path(__file__).resolve().parent.parent
AN = PAPER / "analysis"
FIG = PAPER / "figures"
SUP = PAPER / "supplementary"
FIG.mkdir(exist_ok=True)

wl = pd.read_csv(AN / "wl_fold_results.csv")
base = pd.read_csv(AN / "baseline_fold_results.csv")
ws = pd.read_csv(AN / "wl_summary_by_topology_and_height.csv")
bs = pd.read_csv(AN / "baseline_summary.csv")
crop = pd.read_csv(AN / "crop_acceptance_analysis.csv")
cond = pd.read_csv(AN / "condition_acceptance_analysis.csv")
edges = pd.read_csv(SUP / "graph_topology_edges_first_fold.csv")

# Set compact, print-sized figure dimensions: 6.5 inches wide to avoid
# shrinking labels when figures are placed into a one-column journal manuscript.

# Figure 1: study workflow, arranged as a readable two-row sequence.
fig, ax = plt.subplots(figsize=(6.5, 3.35))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis("off")
box_w, box_h = 3.55, 1.28
xs = [0.4, 4.72, 9.04]
top_y, bot_y = 4.25, 1.78
items = [
    (xs[0], top_y, "1. OLID-I records", "4,749 images\n8 crops, 57 classes"),
    (xs[1], top_y, "2. Segmentation + QC", "ExG/Otsu + morphology\n4,387 accepted; 362 excluded"),
    (xs[2], top_y, "3. Handcrafted traits", "Shape, colour, texture\nand regional descriptors"),
    (xs[2], bot_y, "4. Feature graphs", "38 nodes per image\n5 alternative topologies"),
    (xs[1], bot_y, "5. WL subtree kernel", "H = 0, 1, 2, 3\nnormalized subtree counts"),
    (xs[0], bot_y, "6. Classification", "Class-balanced SVM\nvs. flat-feature baselines"),
]
for x,y,title,sub in items:
    box = FancyBboxPatch((x,y),box_w,box_h,boxstyle="round,pad=0.06,rounding_size=0.08",
                         linewidth=1.0,edgecolor="#365C7D",facecolor="#EEF4F8")
    ax.add_patch(box)
    ax.text(x+box_w/2,y+0.85,title,ha="center",va="center",fontsize=7.5,fontweight="bold",color="#20394E")
    ax.text(x+box_w/2,y+0.37,sub,ha="center",va="center",fontsize=6.4,color="#22313F")
# Top row flows left-to-right.
for i in [0,1]:
    ax.add_patch(FancyArrowPatch((xs[i]+box_w+0.04,top_y+box_h/2),(xs[i+1]-0.07,top_y+box_h/2),
                                 arrowstyle="-|>",mutation_scale=10,lw=1.1,color="#5D6D7E"))
# Transition to graph construction, then bottom row flows right-to-left.
ax.add_patch(FancyArrowPatch((xs[2]+box_w/2,top_y-0.04),(xs[2]+box_w/2,bot_y+box_h+0.06),
                             arrowstyle="-|>",mutation_scale=10,lw=1.1,color="#5D6D7E"))
ax.add_patch(FancyArrowPatch((xs[2]-0.08,bot_y+box_h/2),(xs[1]+box_w+0.08,bot_y+box_h/2),
                             arrowstyle="-|>",mutation_scale=10,lw=1.1,color="#5D6D7E"))
ax.add_patch(FancyArrowPatch((xs[1]-0.08,bot_y+box_h/2),(xs[0]+box_w+0.08,bot_y+box_h/2),
                             arrowstyle="-|>",mutation_scale=10,lw=1.1,color="#5D6D7E"))
# Cross-validation note.
cv_box = FancyBboxPatch((0.75,0.45),11.5,0.58,boxstyle="round,pad=0.04,rounding_size=0.06",
                        linewidth=0.85,edgecolor="#6C7A89",facecolor="#F6F7F8")
ax.add_patch(cv_box)
ax.text(6.5,0.74,"Stratified 5-fold CV (seed 42); fit scaling, discretization and topology on training folds only",
        ha="center",va="center",fontsize=7.7,color="#2F3B46")
ax.text(6.5,6.15,"Study workflow",ha="center",va="center",fontsize=10.5,fontweight="bold")
fig.savefig(FIG / "fig2_workflow.pdf",bbox_inches="tight",pad_inches=0.05)
fig.savefig(FIG / "fig2_workflow.png",dpi=300,bbox_inches="tight",pad_inches=0.05)
plt.close(fig)

# Figure 2: five graph adjacency matrices. Stored edge lists are the first CV fold.
topologies = ["correlation","knn","family","spatial","hybrid"]
labels = {
    "correlation":"Correlation\n|r| >= 0.70", "knn":"Feature kNN\nk = 3",
    "family":"Feature-family\ncliques + bridges", "spatial":"Spatial/regional\nlinks",
    "hybrid":"Hybrid\nunion"
}
fig, axes = plt.subplots(1,5,figsize=(6.5,2.55))
for ax,kind in zip(axes,topologies):
    sub=edges[edges.topology.eq(kind)]
    A=np.zeros((38,38),dtype=np.uint8)
    for r in sub.itertuples(index=False): A[int(r.i),int(r.j)]=A[int(r.j),int(r.i)]=1
    ax.imshow(A,interpolation="nearest",cmap="Blues",vmin=0,vmax=1,origin="lower",aspect="equal")
    ax.set_title(labels[kind],fontsize=7.2,fontweight="bold",pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{len(sub)} edges",fontsize=6.9,labelpad=2)
fig.suptitle("Feature-graph topologies from the first training fold",fontsize=9.5,fontweight="bold",y=1.02)
fig.text(.5,-.025,"All graphs contain the same 38 descriptor nodes; off-diagonal marks indicate undirected edges.",ha="center",fontsize=6.6)
fig.tight_layout(pad=.45,w_pad=.35)
fig.savefig(FIG / "fig5_graph_topologies.pdf",bbox_inches="tight",pad_inches=0.06)
fig.savefig(FIG / "fig5_graph_topologies.png",dpi=300,bbox_inches="tight",pad_inches=0.06)
plt.close(fig)

# Figure 3: graph-kernel performance by WL height against flat-model references.
graph_styles={"correlation":("Correlation","#0072B2","o"),"knn":("Feature-similarity kNN","#D55E00","s"),
              "family":("Feature-family","#009E73","^"),"spatial":("Spatial/regional","#CC79A7","D"),
              "hybrid":("Hybrid","#E69F00","P")}
base_styles={"Random Forest":("#222222","--"),"Linear SVM":("#666666",":"),
             "Flat RBF SVM":("#8C564B","-."),"KNN (k=5)":("#4C78A8",(0,(3,1,1,1)))}
fig,axes=plt.subplots(1,2,figsize=(6.5,3.45))
for ax,metric,title in zip(axes,["macro_f1","accuracy"],["Macro-F1","Accuracy"]):
    for kind,(label,color,marker) in graph_styles.items():
        sub=ws[ws.graph_type.eq(kind)].sort_values("wl_height")
        x=sub.wl_height.to_numpy(); m=sub[f"mean_{metric}"].to_numpy(); sd=sub[f"sd_{metric}"].to_numpy()
        ax.plot(x,m,marker=marker,lw=1.3,ms=3.4,label=label,color=color)
        ax.fill_between(x,m-sd,m+sd,color=color,alpha=.12,linewidth=0)
    for method,(color,style) in base_styles.items():
        row=bs[bs.method.eq(method)].iloc[0]
        ax.axhline(row[f"mean_{metric}"],color=color,ls=style,lw=0.9,label=f"{method} (flat)")
    ax.set_xticks([0,1,2,3]); ax.set_xlabel("WL height",fontsize=7.8); ax.set_ylabel(title,fontsize=7.8)
    ax.set_title(title,fontsize=8.5,fontweight="bold",pad=3)
    ax.tick_params(axis="both",labelsize=6.7,pad=2); ax.grid(axis="y",alpha=.22)
    ax.set_ylim(0.30,0.64 if metric=="accuracy" else 0.60)
handles,leglabels=axes[0].get_legend_handles_labels()
fig.legend(handles,leglabels,loc="lower center",ncol=4,frameon=False,fontsize=6.2,
           bbox_to_anchor=(.5,-.015),columnspacing=1.0,handlelength=2.0)
fig.suptitle("Five-fold performance: graph kernels versus flat features",fontsize=9.5,fontweight="bold",y=.99)
fig.subplots_adjust(left=.09,right=.99,bottom=.20,top=.83,wspace=.27)
fig.savefig(FIG / "fig8_cv_performance.pdf",bbox_inches="tight",pad_inches=0.05)
fig.savefig(FIG / "fig8_cv_performance.png",dpi=300,bbox_inches="tight",pad_inches=0.05)
plt.close(fig)

# Figure 4: mask acceptance by crop and eight lowest-retention conditions.
crop=crop.sort_values("acceptance_rate")
cond=cond.sort_values("acceptance_rate").head(8)
fig,axes=plt.subplots(1,2,figsize=(6.5,3.3))
axes[0].barh(crop.crop.str.replace("_"," "),crop.acceptance_rate*100,color="#4C78A8")
axes[0].set_xlabel("Accepted records (%)",fontsize=7.7); axes[0].set_title("By crop",fontsize=8.6,fontweight="bold")
for i,v in enumerate(crop.acceptance_rate*100): axes[0].text(v+0.6,i,f"{v:.1f}%",va="center",fontsize=6.7)
axes[0].set_xlim(0,112); axes[0].grid(axis="x",alpha=.22)
axes[1].barh(cond.condition,cond.acceptance_rate*100,color="#E39D45")
axes[1].set_xlabel("Accepted records (%)",fontsize=7.7); axes[1].set_title("Lowest-acceptance conditions",fontsize=8.6,fontweight="bold")
for i,v in enumerate(cond.acceptance_rate*100): axes[1].text(v+0.6,i,f"{v:.1f}%",va="center",fontsize=6.7)
axes[1].set_xlim(0,112); axes[1].grid(axis="x",alpha=.22)
for ax in axes:
    ax.invert_yaxis(); ax.tick_params(axis="y",labelsize=6.5,pad=1); ax.tick_params(axis="x",labelsize=6.5,pad=2)
fig.suptitle("Retention varies across crops and stress conditions",fontsize=9.5,fontweight="bold",y=.99)
fig.subplots_adjust(left=.16,right=.98,bottom=.14,top=.85,wspace=.55)
fig.savefig(FIG / "fig6_acceptance_audit.pdf",bbox_inches="tight",pad_inches=.05)
fig.savefig(FIG / "fig6_acceptance_audit.png",dpi=300,bbox_inches="tight",pad_inches=.05)
plt.close(fig)
print("Rebuilt manuscript figures in", FIG)
