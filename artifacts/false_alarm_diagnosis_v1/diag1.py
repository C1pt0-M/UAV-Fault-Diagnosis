import numpy as np, pandas as pd, json
R='/root/autodl-tmp/UAV-Fault-Diagnosis/artifacts/t_selection_mstca_v1/cv/'
L=pd.read_csv('/root/autodl-tmp/causal_source_v1_20261031/unpacked/data/ledgers/train_included_windows.csv')
L['flight']=L.case_uid.str.split('/').str[1]; L['t_end']=L.window_end_row*.02
L['kind']=np.where(L.qualified_y==1,'fault',np.where(L.normal_origin=='normal_flight','indep_normal','prefault_normal'))
fp=np.zeros(len(L))
for seed in [42,43,44,45,46]:
    for f in range(3):
        d=np.load(R+f'seed{seed}_fold{f}/heldout_predictions.npz'); e=json.load(open(R+f'seed{seed}_fold{f}/selections.json'))['T0']['epoch']
        p=pd.Series(d['probabilities'][e-1].astype(float),index=d['original_cache_index'])
        L.loc[L.original_cache_index.isin(p.index),'p%d'%seed]=p.reindex(L.original_cache_index).dropna().values if False else None
    # simpler: build full vector
for seed in [42,43,44,45,46]:
    s=pd.Series(dtype=float)
    for f in range(3):
        d=np.load(R+f'seed{seed}_fold{f}/heldout_predictions.npz'); e=json.load(open(R+f'seed{seed}_fold{f}/selections.json'))['T0']['epoch']
        s=pd.concat([s,pd.Series(d['probabilities'][e-1].astype(float),index=d['original_cache_index'])])
    p=s.reindex(L.original_cache_index).to_numpy(); y=L.qualified_y.to_numpy()
    fs=np.sort(p[y==1])[::-1]; t=fs[int(np.ceil(.97*len(fs)))-1]; fp+=((y==0)&(p>=t))
L['fp']=fp/5
L['tbin']=pd.cut(L.t_end,[0,10,20,30,45,60,90,1e9])
print('== D2 window counts by flight x time bin x kind (train) ==')
print(L.pivot_table(index=['flight','tbin'],columns='kind',values='fp',aggfunc='size',observed=True).fillna(0).astype(int).to_string())
print('\n== D1 indep-normal FP rate by flight x time bin ==')
I=L[L.kind=='indep_normal']
t=I.groupby(['flight','tbin'],observed=True).agg(win=('fp','size'),fp=('fp','sum')); t['rate']=(t.fp/t.win).round(3); print(t.to_string())
print('\n== FP concentration over independent normal flights ==')
c=I.groupby('case_uid').fp.sum().sort_values(ascending=False); tot=c.sum()
print('flights',len(c),'with any FP(mean over seeds>0)',int((c>0).sum()),'top10% flights share',round(c.head(len(c)//10).sum()/tot,3),'top25%',round(c.head(len(c)//4).sum()/tot,3))
print('\n== onset time of fault cases by flight (s) ==')
F=L[L.fault_onset_row.notna()].groupby('case_uid').agg(flight=('flight','first'),onset=('fault_onset_row','first'),end=('t_end','max'))
F['onset']*=.02; print(F.groupby('flight')[['onset','end']].describe(percentiles=[.1,.5,.9]).round(1).to_string())
I2=I.groupby('case_uid').agg(flight=('flight','first'),end=('t_end','max')); print('\nindep normal flight length', I2.groupby('flight').end.describe(percentiles=[.1,.5,.9]).round(1).to_string())
