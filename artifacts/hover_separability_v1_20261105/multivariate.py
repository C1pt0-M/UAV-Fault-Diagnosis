"""Simple-feature classifier on late-hover windows: train on train, score val.
Gives a rough check whether 26-channel window statistics jointly separate
normal from each fault class where M0 fails. No neural training."""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
DATA='D:/UAV-Data/causal_input_v1/'; OUT='D:/UAV-Fault-Diagnosis/artifacts/hover_separability_v1_20261105/'
def pop(split):
    L=pd.read_csv(DATA+f'ledgers/{split}_included_windows.csv',low_memory=False)
    p=L.case_uid.str.split('/');L['flight'],L['cls']=p.str[1],p.str[2]
    L=L[(L.flight=='hover')&(L.window_end_row*0.02>30)]
    n=L[(L.qualified_y==0)&(L.normal_origin=='normal_flight')].assign(cls='normal')
    return pd.concat([n,L[L.qualified_y==1]]).reset_index(drop=True)
def feats(P):
    out=[];cache={}
    for r in P.itertuples():
        k=r.case_uid.replace('/','__')
        if k not in cache:
            z=np.load(DATA+'windows/'+k+'.npz');cache={k:dict(zip(z['original_cache_index'].tolist(),z['x']))}
        x=cache[k][r.original_cache_index].astype(np.float64)
        d=np.abs(np.diff(x,axis=1))
        out.append(np.concatenate([x.mean(1),x.std(1),x.min(1),x.max(1),d.mean(1)]))
    return np.array(out)
T,V=pop('train'),pop('val');FT,FV=feats(T),feats(V)
res={}
for cls in sorted(set(T.cls)-{'normal'}):
    mt=T.cls.isin(['normal',cls]).to_numpy();mv=V.cls.isin(['normal',cls]).to_numpy()
    yt=(T.cls[mt]==cls).astype(int);yv=(V.cls[mv]==cls).astype(int)
    w=np.where(yt==1,0.5/yt.mean(),0.5/(1-yt.mean()))
    m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,random_state=0).fit(FT[mt],yt,sample_weight=w)
    s=m.predict_proba(FV[mv])[:,1]
    # normal FPR at the threshold that keeps 95% fault recall on val
    thr=np.quantile(s[yv.values==1],0.05);fpr=float((s[yv.values==0]>=thr).mean())
    res[cls]={'val_auc':round(float(roc_auc_score(yv,s)),3),'normal_fpr_at_95pct_recall':round(fpr,3)}
json.dump(res,open(OUT+'multivariate.json','w'),indent=2)
print(pd.DataFrame(res).T)
