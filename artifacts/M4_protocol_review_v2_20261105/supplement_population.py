"""Additional state/domain physical summaries from published v1 CSV only."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'M4_design_preflight_v1_20261105'
FEATURES=['vz_median_mps','speed_median_mps','speed_std_mps','gyro_p90_radps']


def weights(f):
    counts=f[['case_uid','association_group']].drop_duplicates().groupby('association_group').size()
    return 1/(f.association_group.nunique()*f.association_group.map(counts).to_numpy()*f.groupby('case_uid').original_cache_index.transform('size').to_numpy())


def quant(x,w):
    order=np.argsort(x,kind='stable');cs=np.cumsum(w[order]);cs/=cs[-1]
    return x[order][np.minimum(np.searchsorted(cs,[.1,.25,.5,.75,.9]),len(x)-1)]


def main():
    f=pd.read_csv(SRC/'window_state_ledger.csv.gz',dtype={'association_group':str},keep_default_na=False)
    nested=pd.read_csv(SRC/'nested_group_assignments_3.csv',dtype={'association_group':str})
    masks={'D1':f.known_normal,'D2':f.known_normal&f.normal_origin.eq('normal_flight'),'D3':f.known_normal&f.normal_origin.eq('normal_part_of_fault_case')}
    rows=[]
    for fold in range(3):
        role=f.association_group.map(nested[nested.outer_evaluation_fold.eq(fold)].set_index('association_group').role)
        for pop,mask in masks.items():
            for (r,s,d),p in f[mask].assign(role=role[mask]).groupby(['role','state','domain']):
                w=weights(p)
                for name in FEATURES:
                    x=p[name].to_numpy()
                    rows.append({'outer_fold':fold,'population':pop,'role':r,'state':s,'domain':d,'feature':name,'windows':len(p),'cases':p.case_uid.nunique(),'groups':p.association_group.nunique(),'weighted_mean':w@x,**dict(zip(['p10','p25','p50','p75','p90'],quant(x,w)))})
    pd.DataFrame(rows).to_csv(OUT/'normal_state_domain_distributions.csv',index=False,mode='x')
    p=pd.read_csv(SRC/'case_pre_post_state_pairs.csv',dtype={'association_group':str})
    rows=[]
    for (fault,domain),g in p.groupby(['fault_directory','domain']):
        for feature in FEATURES:
            x=g[feature+'_delta'].to_numpy()
            rows.append({'fault_directory':fault,'domain':domain,'feature':feature,'cases':len(g),'groups':g.association_group.nunique(),'state_changed_cases':int(g.state_changed.sum()),'median_gap_s':g.endpoint_gap_s.median(),**dict(zip(['p10','p25','p50','p75','p90'],np.quantile(x,[.1,.25,.5,.75,.9])))})
    pd.DataFrame(rows).to_csv(OUT/'pre_post_delta_by_fault_domain.csv',index=False,mode='x')
    print('SUPPLEMENT_SOURCE_V1_ONLY_COMPLETE')


if __name__=='__main__':main()
