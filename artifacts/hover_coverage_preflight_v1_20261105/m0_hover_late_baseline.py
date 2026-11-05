import numpy as np, pandas as pd
L=pd.read_csv('D:/UAV-Data/causal_input_v1/ledgers/val_all_windows.csv',low_memory=False).set_index('original_cache_index')
A='artifacts/causal_pure_mstca_m0_v1_20261101/results/runs/'
for arm in ['uniform_replacement','balanced_source_normal']:
    z=np.load(A+arm+'/source_validation_predictions.npz')
    d=L.loc[z['original_indices']].copy()
    assert (d.qualified_y.values==z['labels']).all() and d.supervised_valid.all()
    d['pred']=(z['probabilities'][:,1]>z['probabilities'][:,0]).astype(int)
    d['t']=d.window_end_row*0.02
    d['flight']=d.case_uid.str.split('/').str[1]
    print('==',arm, 'n',len(d))
    ind=d[(d.normal_origin=='normal_flight')]
    print(' all normal FPR %d/%d'%(d[d.qualified_y==0].pred.sum(),(d.qualified_y==0).sum()),
          ' indep normal %d/%d'%(ind.pred.sum(),len(ind)),' fault recall %d/%d'%(d[d.qualified_y==1].pred.sum(),(d.qualified_y==1).sum()))
    h=ind[ind.flight=='hover']
    for name,s in [('hover all',h),('hover<=30',h[h.t<=30]),('hover>30',h[h.t>30]),('hover30-40',h[(h.t>30)&(h.t<=40)]),('hover>40',h[h.t>40])]:
        print('  %-11s FP %3d/%3d = %.3f%%'%(name,s.pred.sum(),len(s),100*s.pred.mean()))
    late=h[h.t>30]
    pc=late.groupby('case_uid').pred.mean(); pg=late.groupby('association_group').pred.mean()
    print('  hover>30 per-case FPR: mean %.3f median %.3f cases_with_FP %d/%d | group-equal mean %.3f'%(pc.mean(),pc.median(),(pc>0).sum(),len(pc),pg.mean()))
    print('  per-case:',pc.round(2).to_dict())
    hf=d[(d.flight=='hover')&(d.qualified_y==1)]
    print('  hover fault recall >30: %d/%d'%(hf[hf.t>30].pred.sum(),(hf.t>30).sum()))
