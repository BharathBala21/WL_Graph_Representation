"""Trace a predetermined held-out image through the pooled hybrid model."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"Application/backend"))
from app.features import FEATURES
from app.wl import WLFeaturizer, cosine_kernel

def trace():
    df=pd.read_csv(ROOT/"full_pipeline/olid_full_features.csv").merge(
        pd.read_csv(HERE/"supplement/S1_image_folds.csv"),on=["image_name","crop","condition"],validate="one_to_one")
    tr=df[df.fold!=0]
    te=df.query("fold==0 and crop=='cucumber' and condition=='healthy'").sort_values("image_name").iloc[:1]
    x=tr[FEATURES].to_numpy()
    sample=te[FEATURES].to_numpy()
    e=WLFeaturizer("hybrid",3).fit(x)
    phi=e.transform(x)
    clf=SVC(kernel="precomputed",C=1,class_weight="balanced").fit(cosine_kernel(phi),tr.condition.ne("healthy").astype(int))
    k=cosine_kernel(phi,e.transform(sample))
    margin=float(clf.decision_function(k)[0])
    vote=clf.dual_coef_[0]*k[0,clf.support_]
    assert abs(margin-float(clf.intercept_[0])-vote.sum())<1e-7
    yhat=int(clf.predict(k)[0])
    ident=te.image_name.iloc[0]
    saved=pd.read_csv(HERE/"supplement/S11_pooled_predictions.csv").set_index("image_name")
    assert yhat==saved.loc[ident,"prediction"]
    labels=e.labels(sample)
    rows=[]
    for index,(feature,value,b) in enumerate(zip(FEATURES,sample[0],e.node_levels(sample)[0])):
        rows.append((ident,feature,value,int(b),yhat,margin,*[int(level[0,index]>=0) for level in labels]))
    pd.DataFrame(rows,columns=["image_name","descriptor","value","bin_0_low_2_high","prediction","svm_margin","known_h0","known_h1","known_h2","known_h3"]).to_csv(HERE/"supplement/S14_held_out_trace.csv",index=False)
    print(ident,"margin",margin,"prediction",yhat,"known labels",[(level[0]>=0).sum() for level in labels])
    print([row[1:4] for row in rows if row[1] in ("area","mean_H","glcm_contrast","green_ratio")])
if __name__=="__main__":
    trace()
