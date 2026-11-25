"""Refresh manuscript exhibits from tracked experiments. Run with Python, pandas, matplotlib and Pillow."""
from pathlib import Path
import json
import re
import shutil
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT = ROOT / "New_Models_Baselines/03_Benchmark_Suite/results"
SUP = HERE / "supplement"
FIG = HERE / "figures"
for directory in (SUP, FIG):
    directory.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name):
    fig.savefig(FIG / name, dpi=350, facecolor="white", pil_kwargs={"quality": 97})
    plt.close(fig)

features = pd.read_csv(ROOT / "full_pipeline/olid_full_features.csv")
accepted = features.query("quality_flag == 'ACCEPT'").copy()
accepted["number"] = accepted.image_name.str.extract(r"\((\d+)\)")[0].astype(int)
log = json.loads((ROOT / "Application/public/plantcare/oof_level3.json").read_text())
folds = pd.DataFrame([(log["crops"][r[0]], log["conditions"][r[1]], r[2], r[6]) for r in log["rows"]],
                     columns=["crop","condition","number","fold"])
manifest = accepted.merge(folds, how="outer", validate="one_to_one", indicator=True)
assert manifest._merge.eq("both").all() and manifest.image_name.is_unique
manifest[["image_name","crop","condition","fold"]].to_csv(SUP / "S1_image_folds.csv", index=False)

if "--folds-only" in sys.argv:
    print(f"Validated and wrote fold assignments for {len(manifest)} images.")
    raise SystemExit(0)

byfold = pd.read_csv(RESULT / "benchmark_level3_folds.csv")
summary = pd.read_csv(RESULT / "summary_level3.csv").set_index("Model")
for model, g in byfold.groupby("model_family"):
    assert g.fold.nunique() == len(g) == 5
    assert g.sort_values("fold").n_test.tolist() == manifest.groupby("fold").size().tolist()
    for col, published in [("macro_f1","Macro-F1"),("balanced_accuracy","Balanced acc."),("specificity_healthy","Specificity (healthy found)")]:
        if model in summary.index:
            assert abs(g[col].mean() - summary.loc[model,published]) < .000051
for src,dst in [("benchmark_level3_folds","S2_routed_folds"),("summary_level1","S3_pooled"),("summary_level2_per_crop","S4_known_crop"),("summary_robustness_level1","S5_perturbations"),("summary_learning_curve_level2","S6_learning_curves")]:
    shutil.copyfile(RESULT / (src+".csv"), SUP / (dst+".csv"))
shutil.copyfile(ROOT/"Research_Paper/data/feature_dictionary.csv",SUP/"S7_descriptors.csv")
nb = json.loads(next((ROOT/"full_pipeline/decomposition").glob("21*FIXED*.ipynb")).read_text(encoding="utf8"))
output = next("".join(o["text"]) for c in nb["cells"] for o in c.get("outputs",[]) if "TABLE 5: Comprehensive" in "".join(o.get("text",[])))
scores = []
for line in output.splitlines():
    match = re.match(r"Level-1 \(Binary\)\s+WL \+ SVM\s+(.+?)\s+(\d)\s+([\d.]+)\s+([\d.]+)",line)
    if match:
        name, h, acc, f1 = match.groups()
        scores.append((name,int(h),float(acc),float(f1)))
abl = pd.DataFrame(scores,columns=["graph","depth","accuracy","macro_f1"])
assert len(abl)==20
abl.to_csv(SUP/"S8_topology_depth.csv",index=False)

fig,ax=plt.subplots(figsize=(6.8,3.35),layout="constrained")
for (name,g),marker in zip(abl.groupby("graph"),["o","s","^","D","x"]):
    g=g.sort_values("depth")
    ax.plot(g.depth,g.macro_f1,marker=marker,label=name)
ax.set(xticks=[0,1,2,3],xlabel="WL refinement depth",ylabel="Mean macro-F1",ylim=(.59,.71))
ax.legend(frameon=False,loc="upper left",fontsize=8,ncol=2)
ax.grid(axis="y",alpha=.18)
save(fig,"Figure4.jpg")

names={"WL + SVM (tuned)":"WL, nested","WL + SVM":"WL, selected","Extra Trees (tuned)":"Extra Trees, tuned","CNN EfficientNet-B0 (frozen) + LR":"EfficientNet-B0","CNN MobileNetV3-Large (frozen) + LR":"MobileNetV3","Random Forest":"Random forest"}
per=pd.read_csv(RESULT/"summary_level2_per_crop.csv").set_index("crop")
fig,ax=plt.subplots(figsize=(7,3.5),layout="constrained")
for i,(model,short) in enumerate(list(names.items())[:4]):
    ax.scatter(range(len(per)),per[model],label=short,s=36,marker=["o","x","^","s"][i])
ax.set_xticks(range(len(per)),[c.replace("_","\n") for c in per.index])
ax.set_ylabel("Mean macro-F1 within crop")
ax.legend(frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(.5,1.0))
ax.grid(axis="y",alpha=.2)
save(fig,"Figure5.jpg")
rob=pd.read_csv(RESULT/"robustness_level1.csv")
fig,ax=plt.subplots(figsize=(7,3.8),layout="constrained")
kinds=["darker_25","brighter_25","warm_cast","blur","jpeg_q20","noise","half_resolution"]
for j,m in enumerate(["WL + SVM","Extra Trees (tuned)","CNN EfficientNet-B0 (frozen) + LR"]):
    g=rob[rob.model_family==m].set_index("perturbation")
    ax.bar([i+(j-1)*.24 for i in range(7)],g.loc[kinds,"macro_f1"]-g.loc["clean","macro_f1"],width=.23,label=names[m])
ax.axhline(0,c="black",lw=.7)
ax.set_xticks(range(7),["Darker","Brighter","Warm cast","Blur","JPEG","Noise","Half size"])
ax.set_ylabel("Macro-F1 difference from clean images")
ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,1.0))
save(fig,"Figure6.jpg")

lines=[]
for model in ["CNN EfficientNet-B0 (frozen) + LR","CNN MobileNetV3-Large (frozen) + LR","Extra Trees (tuned)","WL + SVM","WL + SVM (tuned)","Random Forest"]:
    row=summary.loc[model]
    cells=[names[model],f"{row['Macro-F1']:.3f} $\\pm$ {row['Macro-F1 SD']:.3f}"]
    cells.extend(f"{row[k]:.3f}" for k in ["Balanced acc.","Specificity (healthy found)","Sensitivity (stressed found)","Accuracy"])
    lines.append(" & ".join(cells)+r" \\")
prev=accepted.condition.ne("healthy").mean()
lines.append(f"Always stressed & {prev/(1+prev):.3f} & 0.500 & 0.000 & 1.000 & {prev:.3f}"+r" \\")
(HERE/"routed_rows.tex").write_text("\n".join(lines)+"\n",encoding="utf8")
print("Matched",len(manifest),"image identifiers; checked",len(byfold),"routed fold records.")

# Hold the evaluated leaves constant across all changes and model families.
from sklearn.metrics import f1_score
pred = pd.read_csv(RESULT/"robustness_level1_predictions.csv")
assert not pred.duplicated(["model_family","perturbation","image_path"]).any()
counts = pred.groupby("image_path").size()
shared = counts[counts == pred.model_family.nunique()*pred.perturbation.nunique()].index
balanced = pred[pred.image_path.isin(shared)]
assert balanced.groupby("image_path").y_true.nunique().eq(1).all()
scores = []
for (family,kind),g in balanced.groupby(["model_family","perturbation"]):
    scores.append((family,kind,len(g),int(g.y_true.sum()),f1_score(g.y_true,g.y_pred,average="macro")))
paired = pd.DataFrame(scores,columns=["model","change","images","stressed","macro_f1"])
paired["difference"] = paired.macro_f1 - paired.model.map(paired.query("change=='clean'").set_index("model").macro_f1)
paired.to_csv(SUP/"S9_common_images.csv",index=False)
print("Common-image comparison:",len(shared))
print(paired[paired.change.ne("clean")].groupby("model").difference.mean().sort_values(ascending=False).head(8))

coverage = []
for crop,g in features.groupby("crop"):
    kept=g[g.quality_flag.eq("ACCEPT")]
    healthy=int(kept.condition.eq("healthy").sum())
    coverage.append(f"{crop.replace('_',' ').capitalize()} & {len(g)} & {len(kept)} & {100*len(kept)/len(g):.1f} & {healthy} & {len(kept)-healthy}"+r" \\")
(HERE/"coverage_rows.tex").write_text("\n".join(coverage)+"\n",encoding="utf8")

from analysis_figures import run
run(save, manifest, RESULT)

# Manuscript tables derived from archived folds and the pooled control predictions.
ctrl=pd.read_csv(SUP/"S10_topology_controls.csv")
pp=pd.read_csv(SUP/"S11_pooled_predictions.csv")
assert len(pp)==len(manifest) and not pp.image_name.duplicated().any()
assert (pp.merge(manifest,on=["image_name","crop","condition","fold"],validate="one_to_one").shape[0]==len(manifest))
matrix=ctrl.pivot_table(index="fold",columns=["graph","depth"],values="macro_f1")
assert ctrl.groupby(["fold","graph","depth"]).size().value_counts().to_dict()=={1:10,5:5}
control_rows=[]
for f,g in matrix.iterrows():
    native=g[("hybrid",3)]
    avg=g[("permuted",3)]
    control_rows.append(f"{f+1} & {g[('hybrid',0)]:.3f} & {native:.3f} & {avg:.3f} & {native-avg:+.3f}"+r" \\")
def rowsfile(name,lines):
    (HERE/name).write_text("\n".join(lines)+"\n",encoding="utf8")
rowsfile("control_rows.tex",control_rows)
errors=[]
for label,name in [("N_K","Nitrogen + potassium"),("N","Nitrogen"),("K","Potassium"),("MIT","Mite"),("LM","Leaf miner"),("PM","Powdery mildew"),("PLEI","Pumpkin leaf-eating insect")]:
    g=pp[pp.condition.eq(label)]
    total=int(features.condition.eq(label).sum())
    found=int(g.prediction.sum())
    errors.append(f"{name} & {total} & {len(g)} & {found} & {100*found/len(g):.1f} & {100*found/total:.1f}"+r" \\")
rowsfile("error_rows.tex",errors)
pooled=pd.read_csv(RESULT/"summary_level1.csv").set_index("Model")
known=pd.read_csv(RESULT/"summary_level2.csv").set_index("Model")
comparison=[]
for model,label in [("Gradient Boosting (HistGB)","Histogram boosting"),("LightGBM","LightGBM"),("Extra Trees (tuned)","Tuned Extra Trees"),("CNN MobileNetV3-Large (frozen) + LR","MobileNetV3"),("CNN EfficientNet-B0 (frozen) + LR","EfficientNet-B0"),("WL + SVM","Selected WL"),("WL + SVM (tuned)","Nested WL"),("Random Forest","Random forest")]:
    pool=f"{pooled.loc[model,'Macro-F1']:.3f}" if model in pooled.index else "---"
    comparison.append(f"{label} & {pool} & {known.loc[model,'Macro-F1 (mean of crops)']:.3f} & {summary.loc[model,'Macro-F1']:.3f} & {summary.loc[model,'Macro-F1 if crop were known']:.3f}"+r" \\")
rowsfile("comparison_rows.tex",comparison)
assert abs(ctrl.query("graph=='hybrid' and depth==3").macro_f1.mean()-pooled.loc["WL + SVM","Macro-F1"])<.00005

# Secondary endpoint from the separately archived 57-class reanalysis.
paper_results=ROOT/"Research_Paper/analysis"
multi=pd.read_csv(paper_results/"best_wl_by_topology.csv").rename(columns={"graph_type":"model"})
flat=pd.read_csv(paper_results/"baseline_summary.csv").rename(columns={"method":"model"})
multi["source"]="best_wl_by_topology.csv"
flat["source"]="baseline_summary.csv"
pd.concat([multi,flat],ignore_index=True)[["source","model","selected_height_by_macro_f1","mean_accuracy","mean_macro_f1","sd_macro_f1"]].to_csv(SUP/"S13_multiclass_context.csv",index=False)

from round3_exhibits import update_round3
update_round3(RESULT, HERE, save)

from round4_exhibits import update_round4
update_round4(RESULT, HERE, save)
