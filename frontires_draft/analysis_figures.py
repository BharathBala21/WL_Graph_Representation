"""Additional figures for the Frontiers manuscript. Called by refresh_exhibits.py."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUP = HERE/"supplement"
sys.path.insert(0,str(ROOT/"Application/backend"))
from app.features import FEATURES, segment_leaf
from app.wl import GRAPHS

def run(save, manifest, result):
    photo = ROOT/"Research_Paper/data/illustrative_images"
    fig, axes = plt.subplots(2,3,figsize=(7.0,4.45),layout="constrained")
    for i,(prefix,title) in enumerate([("01_","A   Healthy"),("08_","B   Nitrogen deficiency"),("15_","C   Powdery mildew")]):
        image = cv2.imread(str(next(photo.glob(prefix+"*.JPG"))))
        _,mask,frac,flag = segment_leaf(image)
        axes[0,i].imshow(cv2.cvtColor(cv2.resize(image,(600,600)),cv2.COLOR_BGR2RGB))
        axes[0,i].set_title(title,fontsize=10,loc="left")
        axes[1,i].imshow(cv2.resize(mask,(600,600),interpolation=cv2.INTER_NEAREST),cmap="gray",vmin=0,vmax=255)
        axes[1,i].set_title(f"{100*frac:.1f}% of frame",fontsize=9)
    for ax in axes.flat: ax.axis("off")
    save(fig,"Figure1.jpg")

    train=manifest.loc[manifest.fold.ne(0),FEATURES].to_numpy()
    fig,axs=plt.subplots(2,3,figsize=(7.0,4.55),layout="constrained")
    ax=axs.flat[0]
    corr=np.corrcoef(train,rowvar=False)
    ax.imshow(abs(corr),cmap="Blues",vmin=0,vmax=1,interpolation="nearest")
    ax.set_title("A   Absolute correlation",loc="left",fontsize=9)
    for ax,(key,builder),letter in zip(list(axs.flat)[1:],GRAPHS.items(),"BCDEF"):
        adj=builder(train)
        ax.imshow(adj,cmap="Greys",interpolation="nearest")
        ax.set_title(f"{letter}   {key.capitalize()} ({np.triu(adj,1).sum()})",loc="left",fontsize=9)
    for ax in axs.flat:
        ax.set_xticks([0,9,19,29,37],[1,10,20,30,38])
        ax.set_yticks([0,9,19,29,37],[1,10,20,30,38])
        ax.tick_params(labelsize=7,length=2)
        for edge in [9.5,15.5,21.5,27.5,29.5,33.5]:
            ax.axvline(edge,color="#c27348",lw=.35)
            ax.axhline(edge,color="#c27348",lw=.35)
    save(fig,"Figure2.jpg")
    # A rank comparison within each crop, descriptive only.
    records=[]
    for crop,g in manifest.groupby("crop"):
        y=g.condition.ne("healthy")
        for trait in FEATURES:
            records.append((crop,trait,int(y.sum()),int((~y).sum()),2*roc_auc_score(y,g[trait])-1))
    effects=pd.DataFrame(records,columns=["crop","descriptor","stressed","healthy","rank_difference"])
    effects.to_csv(SUP/"S12_descriptor_contrasts.csv",index=False)
    traits=["area","circularity","mean_H","green_ratio","glcm_contrast","edge_yellow_ratio"]
    pivot=effects.pivot(index="crop",columns="descriptor",values="rank_difference")[traits]
    fig,ax=plt.subplots(figsize=(7.0,3.8),layout="constrained")
    ax.imshow(pivot,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    ax.set_yticks(range(len(pivot)),[s.replace("_"," ").capitalize() for s in pivot.index])
    ax.set_xticks(range(6),["Area","Circularity","Mean hue","Green ratio","Texture\ncontrast","Edge yellow\nratio"])
    for (y,x),val in np.ndenumerate(pivot.to_numpy()):
        ax.text(x,y,f"{val:+.2f}",ha="center",va="center",color="white" if abs(val)>.55 else "black",fontsize=9)
    save(fig,"Figure3.jpg")
    print(pivot.round(2).to_string())

    fig,(a,b)=plt.subplots(1,2,figsize=(7,3.0),layout="constrained",gridspec_kw={"width_ratios":[1.25,1]})
    curve=pd.read_csv(result/"summary_learning_curve_level2.csv").set_index("Model")
    fracs=[10,20,35,50,75,100]
    for m,label,mark in [("WL + SVM","Selected WL","o"),("RBF SVM","RBF SVM","s"),("CNN MobileNetV3-Large (frozen) + LR","MobileNetV3","^")]:
        a.plot(fracs,curve.loc[m,[f"{p}% data" for p in fracs]],marker=mark,label=label)
    a.legend(fontsize=8,frameon=False)
    a.set(xlabel="Training images used (%)",ylabel="Mean crop macro-F1",title="A   Learning curves")
    timing=pd.read_csv(result/"summary_level3.csv").set_index("Model")
    models=["WL + SVM","Extra Trees (tuned)","CNN EfficientNet-B0 (frozen) + LR","CNN MobileNetV3-Large (frozen) + LR"]
    times=timing.loc[models,"End-to-end ms/photo"].values
    b.barh(range(4),times,color=["#28666e","#75a7a5","#8e542d","#c48255"])
    b.set_yticks(range(4),["WL","Extra Trees","EfficientNet","MobileNet"])
    b.invert_yaxis()
    for i,t in enumerate(times): b.text(t+15,i,f"{t:.0f}",va="center",fontsize=8)
    b.set(xlabel="Estimated time (ms/image)",xlim=(0,1360),title="B   Routed inference")
    save(fig,"Figure7.jpg")
