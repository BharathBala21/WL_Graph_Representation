import os, json, time, warnings
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
warnings.filterwarnings('ignore')

ROOT=os.path.join(os.path.dirname(__file__),'input')
OUT=os.path.dirname(__file__)
os.makedirs(OUT,exist_ok=True)
df0=pd.read_csv(os.path.join(ROOT,'olid_full_features.csv'))
df=df0[df0['quality_flag'].eq('ACCEPT')].copy().reset_index(drop=True)
meta={'image_path','image_name','class','crop','condition','quality_flag','mask_area_ratio','failure_reason'}
feature_cols=[c for c in df.columns if c not in meta]
assert len(df0)==4749, len(df0)
assert len(df)==4387, len(df)
assert len(feature_cols)==38, (len(feature_cols),feature_cols)
assert df[feature_cols].isna().sum().sum()==0
X=df[feature_cols].astype(float).reset_index(drop=True)
y=df['class'].astype(str).to_numpy()

families = {
'Shape':['area','perimeter','aspect_ratio','circularity','extent','solidity','convex_area','major_axis_length','minor_axis_length','eccentricity'],
'RGB':['mean_R','std_R','mean_G','std_G','mean_B','std_B'],
'HSV':['mean_H','std_H','mean_S','std_S','mean_V','std_V'],
'Lab':['mean_L','std_L','mean_a','std_a','mean_b','std_b'],
'DerivedColor':['green_ratio','yellow_ratio'],
'Texture':['glcm_contrast','glcm_homogeneity','glcm_energy','glcm_correlation'],
'Spatial':['core_mean_H','core_yellow_ratio','edge_mean_H','edge_yellow_ratio']}
assert sorted(sum(families.values(),[]))==sorted(feature_cols)
idx={c:i for i,c in enumerate(feature_cols)}
spatial=[c for c in feature_cols if c.startswith('core_') or c.startswith('edge_')]

def edge_set_add(E,i,j):
    if i!=j: E.add(tuple(sorted((int(i),int(j)))))

def graph_edges(kind, Xtr):
    # Topology construction follows current source in notebook 12_source.py / notebook 12.
    n=len(feature_cols); E=set()
    if kind=='correlation':
        C=Xtr.corr(method='pearson').abs().values
        for i in range(n):
            for j in range(i+1,n):
                if C[i,j]>=0.70: edge_set_add(E,i,j)
    elif kind=='knn':
        z=StandardScaler().fit_transform(Xtr[feature_cols].values)
        nn=NearestNeighbors(n_neighbors=4,metric='euclidean').fit(z.T)
        inds=nn.kneighbors(z.T,return_distance=False)
        for i,row in enumerate(inds):
            for j in row[1:]: edge_set_add(E,i,j)
    elif kind=='family':
        C=Xtr[feature_cols].corr().abs()
        for fvals in families.values():
            ids=[idx[c] for c in fvals]
            for p in range(len(ids)):
                for q in range(p+1,len(ids)): edge_set_add(E,ids[p],ids[q])
        fkeys=list(families.keys())
        for a in range(len(fkeys)):
            for b in range(a+1,len(fkeys)):
                fa=families[fkeys[a]]; fb=families[fkeys[b]]
                sub=C.loc[fa,fb]
                mx=sub.max().max()
                if pd.notna(mx):
                    # Preserve first max in family feature order, matching np.where(...)[0].
                    loc=np.argwhere(sub.values==mx)[0]
                    edge_set_add(E,idx[fa[int(loc[0])]],idx[fb[int(loc[1])]])
    elif kind=='spatial':
        C=Xtr[feature_cols].corr().abs()
        core=[c for c in spatial if c.startswith('core_')]
        edge=[c for c in spatial if c.startswith('edge_')]
        glob=[c for c in feature_cols if c not in spatial]
        for group in (core,edge):
            for p in range(len(group)):
                for q in range(p+1,len(group)): edge_set_add(E,idx[group[p]],idx[group[q]])
        for cf in core:
            ef='edge_'+cf.replace('core_','')
            if ef in edge: edge_set_add(E,idx[cf],idx[ef])
        for gf in glob:
            for sf in spatial:
                if C.loc[gf,sf]>=0.60: edge_set_add(E,idx[gf],idx[sf])
    elif kind=='hybrid':
        # Union of corr>=.80, family, and spatial with threshold .70
        # Hybrid's correlation threshold is defined explicitly below.
        C=Xtr.corr(method='pearson').abs().values
        for i in range(n):
            for j in range(i+1,n):
                if C[i,j]>=0.80: edge_set_add(E,i,j)
        E |= graph_edges('family',Xtr)
        # spatial with cutoff .70
        Cdf=Xtr[feature_cols].corr().abs()
        core=[c for c in spatial if c.startswith('core_')]
        edge=[c for c in spatial if c.startswith('edge_')]
        glob=[c for c in feature_cols if c not in spatial]
        for group in (core,edge):
            for p in range(len(group)):
                for q in range(p+1,len(group)): edge_set_add(E,idx[group[p]],idx[group[q]])
        for cf in core:
            ef='edge_'+cf.replace('core_','')
            if ef in edge: edge_set_add(E,idx[cf],idx[ef])
        for gf in glob:
            for sf in spatial:
                if Cdf.loc[gf,sf]>=0.70: edge_set_add(E,idx[gf],idx[sf])
    else: raise ValueError(kind)
    return E

# special branch needed to make type threshold 0.80 explicit in the graph constructor above
def neighbor_lists(E):
    N=[[] for _ in feature_cols]
    for i,j in E:
        N[i].append(j); N[j].append(i)
    return [tuple(sorted(a)) for a in N]

def make_initial_labels(Xtr,Xva):
    disc=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='quantile',subsample=None)
    disc.fit(Xtr[feature_cols].values)
    def trans(Xmat):
        A=disc.transform(Xmat[feature_cols].values).astype(int)
        lbls=[]
        cats=['Low','Medium','High']
        for row in A:
            lbls.append([f'{feature_cols[j]}::{cats[row[j]]}' for j in range(len(feature_cols))])
        return lbls
    return trans(Xtr),trans(Xva)

def wl_bags(train_labels,val_labels,neighbors,max_h=3):
    # Separate count bags for each iteration. Label compression maps are fit on training rows only.
    tr_cur=[list(r) for r in train_labels]
    va_cur=[list(r) for r in val_labels]
    tr_bags=[[] for _ in range(max_h+1)]
    va_bags=[[] for _ in range(max_h+1)]
    tr_bags[0]=[Counter(row) for row in tr_cur]
    va_bags[0]=[Counter(row) for row in va_cur]
    for h in range(1,max_h+1):
        mapper={}; counter=0
        # Training signatures register in deterministic row and node order.
        new_tr=[]
        for row in tr_cur:
            nr=[]
            for i,label in enumerate(row):
                sig=(label,tuple(sorted(row[j] for j in neighbors[i])))
                if sig not in mapper:
                    mapper[sig]=f'wl_{h-1}_{counter}'; counter+=1
                nr.append(mapper[sig])
            new_tr.append(nr)
        new_va=[]
        for row in va_cur:
            nr=[]
            for i,label in enumerate(row):
                sig=(label,tuple(sorted(row[j] for j in neighbors[i])))
                nr.append(mapper.get(sig,f'__OOV__{h-1}'))
            new_va.append(nr)
        tr_cur,va_cur=new_tr,new_va
        tr_bags[h]=[Counter(row) for row in tr_cur]
        va_bags[h]=[Counter(x for x in row if not x.startswith('__OOV__')) for row in va_cur]
    return tr_bags,va_bags

def to_sparse(bags_tr,bags_va):
    vocab=sorted(set().union(*(b.keys() for b in bags_tr)))
    fidx={x:i for i,x in enumerate(vocab)}
    def build(bags):
        rr=[]; cc=[]; vv=[]
        for r,b in enumerate(bags):
            for token,count in b.items():
                col=fidx.get(token)
                if col is not None:
                    rr.append(r);cc.append(col);vv.append(count)
        return csr_matrix((vv,(rr,cc)),shape=(len(bags),len(vocab)),dtype=np.float64)
    return build(bags_tr),build(bags_va)

def normalized_kernel(Phi_tr,Phi_va):
    Ktr=(Phi_tr @ Phi_tr.T).toarray()
    Kva=(Phi_va @ Phi_tr.T).toarray()
    dtr=np.sqrt(np.maximum(np.diag(Ktr),1e-12))
    dva=np.sqrt(np.maximum(np.asarray(Phi_va.multiply(Phi_va).sum(axis=1)).ravel(),1e-12))
    Ktr=Ktr/(dtr[:,None]*dtr[None,:])
    Kva=Kva/(dva[:,None]*dtr[None,:])
    return Ktr,Kva

def metrics(ytrue,ypred):
    return {'accuracy':accuracy_score(ytrue,ypred),
            'macro_f1':f1_score(ytrue,ypred,average='macro',zero_division=0),
            'weighted_f1':f1_score(ytrue,ypred,average='weighted',zero_division=0),
            'macro_precision':precision_score(ytrue,ypred,average='macro',zero_division=0),
            'macro_recall':recall_score(ytrue,ypred,average='macro',zero_division=0),
            'balanced_accuracy':balanced_accuracy_score(ytrue,ypred)}

cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
folds=list(cv.split(X,y))
fold_assignment=np.full(len(y),-1,dtype=int)
for f,(_,va) in enumerate(folds): fold_assignment[va]=f
pd.DataFrame({'image_path':df['image_path'],'fold':fold_assignment,'class':y}).to_csv(os.path.join(OUT,'cv_fold_assignments_recomputed.csv'),index=False)

# Flat baselines on the same newly generated fold assignment.
baseline_rows=[]
for model_name in ['Flat RBF SVM','Linear SVM','Random Forest','KNN (k=5)']:
    for f,(tr,va) in enumerate(folds):
        scaler=StandardScaler()
        Xtr=scaler.fit_transform(X.iloc[tr].values)
        Xva=scaler.transform(X.iloc[va].values)
        if model_name=='Flat RBF SVM': model=SVC(kernel='rbf',C=1.0,class_weight='balanced',random_state=42)
        elif model_name=='Linear SVM': model=SVC(kernel='linear',C=1.0,class_weight='balanced',random_state=42)
        elif model_name=='Random Forest': model=RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',random_state=42,n_jobs=-1)
        else:
            from sklearn.neighbors import KNeighborsClassifier
            model=KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
        t=time.time(); model.fit(Xtr,y[tr]); fit_s=time.time()-t
        yp=model.predict(Xva)
        baseline_rows.append({'method':model_name,'fold':f,**metrics(y[va],yp),'fit_s':fit_s})
        print(f'BASE {model_name:16s} fold {f}: acc={baseline_rows[-1]["accuracy"]:.4f}, macroF1={baseline_rows[-1]["macro_f1"]:.4f}',flush=True)
base_df=pd.DataFrame(baseline_rows)
base_df.to_csv(os.path.join(OUT,'baseline_fold_results.csv'),index=False)
base_summary=base_df.groupby('method').agg(**{f'mean_{c}':(c,'mean') for c in ['accuracy','macro_f1','weighted_f1','macro_precision','macro_recall','balanced_accuracy']},**{f'sd_{c}':(c,'std') for c in ['accuracy','macro_f1','weighted_f1','macro_precision','macro_recall','balanced_accuracy']}).reset_index()
base_summary.to_csv(os.path.join(OUT,'baseline_summary.csv'),index=False)

# Main WL graph-kernel comparison, using the current source-level feature identity in h=0 labels.
wl_rows=[]
all_fold_metrics=[]
for f,(tr,va) in enumerate(folds):
    Xtr=X.iloc[tr].reset_index(drop=True); Xva=X.iloc[va].reset_index(drop=True)
    tr_init,va_init=make_initial_labels(Xtr,Xva)
    for graph_type in ['correlation','knn','family','spatial','hybrid']:
        tic=time.time()
        E=graph_edges(graph_type,Xtr)
        neighbors=neighbor_lists(E)
        bags_tr,bags_va=wl_bags(tr_init,va_init,neighbors,3)
        # WL subtree kernel at height h accumulates label counts from levels 0..h.
        for h in range(4):
            cum_tr=[Counter() for _ in bags_tr[0]]
            cum_va=[Counter() for _ in bags_va[0]]
            for lev in range(h+1):
                for i,b in enumerate(bags_tr[lev]): cum_tr[i].update(b)
                for i,b in enumerate(bags_va[lev]): cum_va[i].update(b)
            Phi_tr,Phi_va=to_sparse(cum_tr,cum_va)
            Ktr,Kva=normalized_kernel(Phi_tr,Phi_va)
            clf=SVC(kernel='precomputed',C=1.0,class_weight='balanced',random_state=42)
            t=time.time(); clf.fit(Ktr,y[tr]); fit_s=time.time()-t
            yp=clf.predict(Kva)
            m=metrics(y[va],yp)
            row={'task':'57-class','method':'WL + SVM','graph_type':graph_type,'wl_height':h,'fold':f,'n_nodes':38,'n_edges':len(E),'n_train':len(tr),'n_val':len(va),'feature_dim':Phi_tr.shape[1],'fit_s':fit_s,**m}
            wl_rows.append(row)
            print(f'WL fold={f} graph={graph_type:11s} h={h} dim={Phi_tr.shape[1]:6d} E={len(E):3d} acc={m["accuracy"]:.4f} macroF1={m["macro_f1"]:.4f} SVM={fit_s:.2f}s',flush=True)
            # Checkpoint every evaluated configuration so a long run never loses prior folds.
            pd.DataFrame(wl_rows).to_csv(os.path.join(OUT,'wl_fold_results_checkpoint.csv'),index=False)
        all_fold_metrics.append({'fold':f,'graph_type':graph_type,'n_edges':len(E),'graph_wl_time_s':time.time()-tic})
        del bags_tr,bags_va

wl_df=pd.DataFrame(wl_rows)
wl_df.to_csv(os.path.join(OUT,'wl_fold_results.csv'),index=False)
wl_summary=wl_df.groupby(['graph_type','wl_height']).agg(**{f'mean_{c}':(c,'mean') for c in ['accuracy','macro_f1','weighted_f1','macro_precision','macro_recall','balanced_accuracy']},**{f'sd_{c}':(c,'std') for c in ['accuracy','macro_f1','weighted_f1','macro_precision','macro_recall','balanced_accuracy']},mean_edges=('n_edges','mean'),min_edges=('n_edges','min'),max_edges=('n_edges','max'),mean_dim=('feature_dim','mean')).reset_index()
wl_summary=wl_summary.sort_values('mean_macro_f1',ascending=False)
wl_summary.to_csv(os.path.join(OUT,'wl_summary.csv'),index=False)
wl_summary.sort_values(['graph_type','wl_height']).to_csv(os.path.join(OUT,'wl_summary_by_topology_and_height.csv'),index=False)
pd.DataFrame(all_fold_metrics).to_csv(os.path.join(OUT,'graph_fold_stats.csv'),index=False)
best_rows=[]
for graph_name, group in wl_summary.groupby('graph_type'):
    selected=group.sort_values(['mean_macro_f1','wl_height'],ascending=[False,True]).iloc[0]
    best_rows.append({'graph_type':graph_name,'selected_height_by_macro_f1':int(selected.wl_height),
                      'mean_edges':selected.mean_edges,'mean_accuracy':selected.mean_accuracy,'sd_accuracy':selected.sd_accuracy,
                      'mean_macro_f1':selected.mean_macro_f1,'sd_macro_f1':selected.sd_macro_f1,
                      'mean_weighted_f1':selected.mean_weighted_f1,'sd_weighted_f1':selected.sd_weighted_f1,
                      'mean_feature_dim':selected.mean_dim})
pd.DataFrame(best_rows).sort_values('mean_macro_f1',ascending=False).to_csv(os.path.join(OUT,'best_wl_by_topology.csv'),index=False)
print('\n=== BASELINE SUMMARY ===')
print(base_summary.to_string(index=False))
print('\n=== WL SUMMARY (sorted macro F1) ===')
print(wl_summary.to_string(index=False))
print('\n=== BEST WL VS BASELINES ===')
print('Best WL:',wl_summary.iloc[0].to_dict())
print('Baselines ranked:',base_summary.sort_values('mean_macro_f1',ascending=False)[['method','mean_accuracy','mean_macro_f1','sd_macro_f1']].to_dict('records'))
with open(os.path.join(OUT,'run_metadata.json'),'w') as f:
    json.dump({'n_original':len(df0),'n_accepted':len(df),'n_excluded':len(df0)-len(df),'n_features':len(feature_cols),'n_classes':len(np.unique(y)),'n_crops':int(df['crop'].nunique()),'n_conditions':int(df['condition'].nunique()),'cv':'StratifiedKFold(n_splits=5, shuffle=True, random_state=42)','feature_identity_preserved_in_h0':True,'source_basis':'corrected current graph/WL source logic; recomputed because embedded notebook outputs were stale relative to patched source','software_versions':{'numpy':np.__version__,'pandas':pd.__version__}},f,indent=2)
