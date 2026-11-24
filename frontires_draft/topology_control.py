"""Paired topology controls on the archived pooled folds; run from any directory."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Application/backend"))
from app.features import FEATURES
from app.wl import WLFeaturizer, GRAPHS, adj_hybrid, cosine_kernel

def permuted_graph(seed):
    order = np.random.default_rng(seed).permutation(len(FEATURES))
    def build(x):
        graph = adj_hybrid(x)
        return graph[np.ix_(order, order)]
    return build

def run():
    feat = pd.read_csv(ROOT/"full_pipeline/olid_full_features.csv").query("quality_flag == 'ACCEPT'")
    data = feat.merge(pd.read_csv(HERE/"supplement/S1_image_folds.csv"), on=["image_name","crop","condition"], validate="one_to_one")
    assert len(data) == len(feat) == 4387
    x = data[FEATURES].to_numpy()
    y = data.condition.ne("healthy").to_numpy(dtype=int)
    scores, predictions = [], []
    for fold in sorted(data.fold.unique()):
        train = data.fold.ne(fold).to_numpy()
        for setting, depth, seed in [("hybrid",0,None),("hybrid",3,None)] + [("permuted",3,i) for i in range(5)]:
            if seed is not None:
                GRAPHS["permuted"] = permuted_graph(seed)
            engine = WLFeaturizer(setting, depth).fit(x[train])
            p = engine.transform(x[train])
            clf = SVC(kernel="precomputed", C=1, class_weight="balanced").fit(cosine_kernel(p),y[train])
            test = engine.transform(x[~train])
            pred = clf.predict(cosine_kernel(p,test))
            tn,fp,fn,tp = confusion_matrix(y[~train],pred,labels=[0,1]).ravel()
            scores.append((fold,setting,depth,seed,engine.A_.sum()//2,p.shape[1],f1_score(y[~train],pred,average="macro"),balanced_accuracy_score(y[~train],pred),tn,fp,fn,tp))
            if setting=="hybrid" and depth==3:
                records=data.loc[~train,["image_name","crop","condition","fold"]].copy()
                records["prediction"]=pred
                predictions.append(records)
        print("Finished fold",fold,flush=True)
    frame=pd.DataFrame(scores,columns="fold graph depth seed edges dimensions macro_f1 balanced_accuracy tn fp fn tp".split())
    frame.to_csv(HERE/"supplement/S10_topology_controls.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(HERE/"supplement/S11_pooled_predictions.csv",index=False)
    print(frame.groupby(["graph","depth"]).macro_f1.mean())
if __name__ == "__main__":
    run()
