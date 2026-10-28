import os,time
p=os.path.join(os.path.dirname(__file__),'recompute_models.py')
s=open(p).read(); s=s[:s.index('\ncv=StratifiedKFold')]
exec(s,globals())
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
folds=list(cv.split(X,y))
outfile=os.path.join(OUT,'wl_fold_results_checkpoint.csv')
finalfile=os.path.join(OUT,'wl_fold_results.csv')
if os.path.exists(outfile):
    rows=pd.read_csv(outfile).to_dict('records')
elif os.path.exists(finalfile):
    rows=pd.read_csv(finalfile).to_dict('records')
    pd.DataFrame(rows).to_csv(outfile,index=False)
else: rows=[]
completed={(int(r['fold']),r['graph_type'],int(r['wl_height'])) for r in rows}
print('RESUME starting; completed:',len(completed), sorted(completed)[:3],flush=True)
for f,(tr,va) in enumerate(folds):
    Xtr=X.iloc[tr].reset_index(drop=True); Xva=X.iloc[va].reset_index(drop=True)
    pending=[g for g in ['correlation','knn','family','spatial','hybrid'] if any((f,g,h) not in completed for h in range(4))]
    if not pending: continue
    tr_init,va_init=make_initial_labels(Xtr,Xva)
    for graph_type in pending:
        if all((f,graph_type,h) in completed for h in range(4)): continue
        print(f'GRAPH START fold={f} graph={graph_type}',flush=True); tic=time.time()
        E=graph_edges(graph_type,Xtr)
        neighbors=neighbor_lists(E)
        bags_tr,bags_va=wl_bags(tr_init,va_init,neighbors,3)
        print(f'GRAPH BAGS fold={f} graph={graph_type} edges={len(E)} time={time.time()-tic:.2f}s',flush=True)
        for h in range(4):
            if (f,graph_type,h) in completed: continue
            cum_tr=[Counter() for _ in bags_tr[0]]; cum_va=[Counter() for _ in bags_va[0]]
            for lev in range(h+1):
                for i,b in enumerate(bags_tr[lev]): cum_tr[i].update(b)
                for i,b in enumerate(bags_va[lev]): cum_va[i].update(b)
            Phi_tr,Phi_va=to_sparse(cum_tr,cum_va)
            Ktr,Kva=normalized_kernel(Phi_tr,Phi_va)
            clf=SVC(kernel='precomputed',C=1.0,class_weight='balanced',random_state=42)
            t=time.time(); clf.fit(Ktr,y[tr]); fit_s=time.time()-t
            yp=clf.predict(Kva); m=metrics(y[va],yp)
            row={'task':'57-class','method':'WL + SVM','graph_type':graph_type,'wl_height':h,'fold':f,'n_nodes':38,'n_edges':len(E),'n_train':len(tr),'n_val':len(va),'feature_dim':Phi_tr.shape[1],'fit_s':fit_s,**m}
            rows.append(row); completed.add((f,graph_type,h))
            pd.DataFrame(rows).sort_values(['fold','graph_type','wl_height']).to_csv(outfile,index=False)
            print(f'WL fold={f} graph={graph_type:11s} h={h} dim={Phi_tr.shape[1]:6d} E={len(E):3d} acc={m["accuracy"]:.4f} macroF1={m["macro_f1"]:.4f} SVM={fit_s:.2f}s total={time.time()-tic:.1f}s',flush=True)
        del bags_tr,bags_va
# Finalize the full fold-level outputs and summaries for manuscript figure regeneration.
wl_df=pd.DataFrame(rows).sort_values(['fold','graph_type','wl_height'])
wl_df.to_csv(os.path.join(OUT,'wl_fold_results.csv'),index=False)
metrics_cols=['accuracy','macro_f1','weighted_f1','macro_precision','macro_recall','balanced_accuracy']
agg={}
for c in metrics_cols:
    agg[f'mean_{c}']=(c,'mean'); agg[f'sd_{c}']=(c,'std')
wl_summary=wl_df.groupby(['graph_type','wl_height']).agg(**agg,mean_edges=('n_edges','mean'),min_edges=('n_edges','min'),max_edges=('n_edges','max'),mean_dim=('feature_dim','mean')).reset_index()
wl_summary.to_csv(os.path.join(OUT,'wl_summary_by_topology_and_height.csv'),index=False)
wl_summary.to_csv(os.path.join(OUT,'wl_summary.csv'),index=False)
best_rows=[]
for graph_name, group in wl_summary.groupby('graph_type'):
    selected=group.sort_values(['mean_macro_f1','wl_height'],ascending=[False,True]).iloc[0]
    best_rows.append({'graph_type':graph_name,'selected_height_by_macro_f1':int(selected.wl_height),
                      'mean_edges':selected.mean_edges,'mean_accuracy':selected.mean_accuracy,'sd_accuracy':selected.sd_accuracy,
                      'mean_macro_f1':selected.mean_macro_f1,'sd_macro_f1':selected.sd_macro_f1,
                      'mean_weighted_f1':selected.mean_weighted_f1,'sd_weighted_f1':selected.sd_weighted_f1,
                      'mean_feature_dim':selected.mean_dim})
pd.DataFrame(best_rows).sort_values('mean_macro_f1',ascending=False).to_csv(os.path.join(OUT,'best_wl_by_topology.csv'),index=False)
print('ALL DONE',len(rows),flush=True)
