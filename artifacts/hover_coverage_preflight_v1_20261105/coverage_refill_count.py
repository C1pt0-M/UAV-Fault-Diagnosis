import pandas as pd, numpy as np
for sp in ['train','val']:
    d=pd.read_csv(f'ledgers/{sp}_all_windows.csv',low_memory=False)
    d['flight']=d.case_uid.str.split('/').str[1]
    cases=d[(d.flight=='hover')&(d.case_uid.str.contains('/no_fault/'))].case_uid.dropna().unique()
    rows=[]
    for c in cases:
        z=np.load('cases/'+c.replace('/','__')+'.npz')
        iv=z['row_input_valid']; lv=z['row_label_valid']; st=z['reference_state']
        n=len(iv); bad=np.concatenate([[0],np.cumsum(~iv)])
        lbad=np.concatenate([[0],np.cumsum(~lv | (st!=0))])
        starts=np.arange(0,n-150+1,5); ends=starts+150
        ok=(bad[ends]-bad[starts]==0)&lv[ends-1]&(st[ends-1]==0)
        okw=ok&(lbad[ends]-lbad[starts]==0)
        late=ends*0.02>30
        cap=d[d.case_uid==c]
        rows.append(dict(case=c,dom=c.split('/')[0],n_s=n*0.02,all_ok=ok.sum(),late_ok=(ok&late).sum(),late_okw=(okw&late).sum(),
          cap_late=int(((cap.window_end_row*0.02>30)&cap.supervised_valid).sum()),stmax=int(st.max()),wend=cap.window_end_row.max()))
    r=pd.DataFrame(rows)
    print(sp,len(r),'cases; duration s: median %.1f min %.1f max %.1f'%(r.n_s.median(),r.n_s.min(),r.n_s.max()))
    print(' stride5 valid all=%d late=%d late_wholewin=%d | cap64 late=%d'%(r.all_ok.sum(),r.late_ok.sum(),r.late_okw.sum(),r.cap_late.sum()))
    print(' max state:',r.stmax.max(),' max cap end row vs n check:',(r.wend<=r.n_s/0.02).all())
    print(r.groupby('dom')[['all_ok','late_ok','cap_late']].sum())
    print(r.n_s.describe().round(1).to_dict())
