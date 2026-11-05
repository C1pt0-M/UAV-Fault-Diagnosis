"""ens_groupcv_seed43_v1: rule a-d (m0s_stats_branch_v1) on seed-43 M0 vs seed-43 ENS."""
import json, numpy as np, pandas as pd
DATA='D:/UAV-Data/causal_input_v1/'; OUT='D:/UAV-Fault-Diagnosis/artifacts/ens_groupcv_seed43_v1_20261105/'
L=pd.read_csv(DATA+'ledgers/train_included_windows.csv',low_memory=False)
g=np.load(OUT+'gbt_heldout.npz'); assert (L.original_cache_index.values==g['original_indices']).all()
L['fold']=g['fold']; L['GBT']=g['prob']; pos={v:i for i,v in enumerate(L.original_cache_index)}
m0=np.full(len(L),np.nan)
for f in range(3):
    z=np.load(OUT+f'fold{f}/heldout_predictions.npz'); ii=[pos[v] for v in z['original_indices']]
    assert (L.fold.values[ii]==f).all(); m0[ii]=z['probabilities'][:,1]
assert np.isfinite(m0).all()
L['M0']=m0>0.5; L['GBTb']=L.GBT>=0.5; L['ENS']=((m0+L.GBT)/2)>=0.5
s=L.case_uid.str.split('/'); L['flight'],L['cls']=s.str[1],s.str[2]; L['late']=(L.flight=='hover')&(L.window_end_row*0.02>30)
def st(d,a):
    n=d.qualified_y==0; f=~n; ind=n&(d.normal_origin=='normal_flight')
    return dict(nfp=int(d[a][n].sum()),tp=int(d[a][f].sum()),fn=int(f.sum()),late=int(d[a][ind&d.late].sum()),
        late_dom={k:int(d[a][ind&d.late&(d.domain==k)].sum()) for k in ['SIL','HIL']},
        nonhover=int(d[a][ind&(d.flight!='hover')].sum()),
        out={c:int(d[a][f&~d.late&(d.cls==c)].sum()) for c in sorted(d.cls[f].unique())},
        outn={c:int((f&~d.late&(d.cls==c)).sum()) for c in sorted(d.cls[f].unique())},
        lt={c:(int(d[a][f&d.late&(d.cls==c)].sum()),int((f&d.late&(d.cls==c)).sum())) for c in ['GPS','barometer']})
res={}
for arm in ['GBTb','ENS']:
    B={f:st(L[L.fold==f],'M0') for f in range(3)}; P={f:st(L[L.fold==f],arm) for f in range(3)}; Bp,Pp=st(L,'M0'),st(L,arm)
    a=all(P[f]['nfp']<B[f]['nfp'] for f in range(3)); b=all(P[f]['tp']>=B[f]['tp']-0.005*B[f]['fn'] for f in range(3))
    c=Pp['late']<Bp['late']; drop={k:round(100*(Bp['out'][k]-Pp['out'][k])/Bp['outn'][k],3) for k in Bp['out']}; d=max(drop.values())<=1.0
    res[arm]={'a':a,'b':b,'c':c,'d':d,'PASS':bool(a and b and c and d),'class_drop_outside_late_pp':drop,
        'per_fold':{f:{'M0':(B[f]['nfp'],B[f]['tp']),arm:(P[f]['nfp'],P[f]['tp']),'fault_n':B[f]['fn']} for f in range(3)},
        'pooled':{'M0':{k:Bp[k] for k in ['nfp','tp','late','late_dom','nonhover']},arm:{k:Pp[k] for k in ['nfp','tp','late','late_dom','nonhover']}},
        'e_late_GPS_baro':{'M0':Bp['lt'],arm:Pp['lt']}}
json.dump(res,open(OUT+'evaluation.json','w'),indent=2)
print(json.dumps(res,indent=1))
