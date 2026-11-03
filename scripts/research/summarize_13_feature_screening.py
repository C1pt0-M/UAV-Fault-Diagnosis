"""Summarize finite13-feature diagnostics, retaining conditional redundancy."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,json
import numpy as np
import pandas as pd
from analyze_13_feature_screening import IDS,corr,distribution


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New summary directory required')
    f=pd.read_csv(a.results/'window_features.csv');responses=pd.read_csv(a.results/'feature_fault_response.csv');a.output.mkdir(parents=True)
    pair_rows=[];case_rows=[]
    for (split,uid,label),part in f[f.label_group.isin(['active','normal'])].groupby(['split','case_uid','label_group'],sort=True):
        metadata=part.iloc[0];c=corr(part)
        for i,x in enumerate(IDS):
            for y in IDS[i+1:]:case_rows.append({'split':split,'case_uid':uid,'domain':metadata.domain,'kind':metadata.kind,'subtype':metadata.subtype,'label_group':label,'feature_a':x,'feature_b':y,'windows':len(part),'spearman':c.loc[x,y]})
    case_frame=pd.DataFrame(case_rows)
    for split in ['train','val']:
        for subtype in ['motor','accelerometer','gyroscope','barometer','magnetometer','GPS']:
            part=case_frame[(case_frame.split==split)&(case_frame.subtype==subtype)&(case_frame.label_group=='active')]
            for (x,y),g in part.groupby(['feature_a','feature_b'],sort=True):
                valid=g.spearman.dropna();pair_rows.append({'split':split,'subtype':subtype,'feature_a':x,'feature_b':y,'cases':len(valid),
                                                          'median_within_case_spearman':valid.median() if len(valid) else None,
                                                          'min':valid.min() if len(valid) else None,'max':valid.max() if len(valid) else None,
                                                          'SIL':g[g.domain=='SIL'].spearman.iloc[0] if len(g[g.domain=='SIL']) else None,
                                                          'HIL':g[g.domain=='HIL'].spearman.iloc[0] if len(g[g.domain=='HIL']) else None})
    case_frame.to_csv(a.output/'correlation_by_case_label.csv',index=False);pd.DataFrame(pair_rows).to_csv(a.output/'correlation_by_fault_within_case.csv',index=False)
    summary=[]
    for (name,split,subtype),part in responses.groupby(['feature_id','split','subtype'],sort=True):
        a1=part.auc_active_vs_same_case_pre_fixed_direction.dropna();a2=part.auc_vs_same_observed_state_independent_normals.dropna()
        summary.append({'feature_id':name,'split':split,'subtype':subtype,'cases':part.case_uid.nunique(),
                        'same_case_auc_case_mean':a1.mean() if len(a1) else None,'same_case_auc_min':a1.min() if len(a1) else None,'same_case_auc_max':a1.max() if len(a1) else None,
                        'state_matched_auc_case_mean':a2.mean() if len(a2) else None,'state_matched_auc_min':a2.min() if len(a2) else None,'state_matched_auc_max':a2.max() if len(a2) else None,
                        'matched_active_windows':int(part.matched_active_windows.sum()),'unmatched_active_windows':int(part.unmatched_active_windows.sum()),
                        'same_case_signed_effect_median':part.shift_in_train_normal_IQR.median(),
                        'same_case_auc_domain_direction_agrees':bool((a1.gt(.5).all() or a1.lt(.5).all())) if len(a1)==2 else None,
                        'state_matched_auc_domain_direction_agrees':bool((a2.gt(.5).all() or a2.lt(.5).all())) if len(a2)==2 else None})
    pd.DataFrame(summary).to_csv(a.output/'fault_response_case_aggregates.csv',index=False)
    # Distribution overlap uses case-state medians, equal case contributions.
    # Histogram edges come solely from the limited Train-normal diagnostic
    # reference; tails outside that support have explicit overflow bins.
    reference=f[(f.split=='train')&(f.kind=='normal')];overlap=[]
    for name in IDS:
        edges=np.unique(np.quantile(reference[name],[0,.25,.5,.75,1]));edges=np.r_[-np.inf,edges,np.inf]
        for (split,subtype),part in f[(f.kind=='fault')&(f.label_group=='active')].groupby(['split','subtype']):
            positives=part.groupby('case_uid')[name].median().to_numpy()
            normals=f[(f.split==split)&(f.kind=='normal')&(f.case_uid.str.split('/').str[1]=='hover')].groupby('case_uid')[name].median().to_numpy()
            if not len(positives) or not len(normals):continue
            p=np.histogram(positives,bins=edges)[0]/len(positives);q=np.histogram(normals,bins=edges)[0]/len(normals)
            overlap.append({'feature_id':name,'split':split,'subtype':subtype,'positive_case_count':len(positives),'normal_case_count':len(normals),'histogram_overlap':float(np.minimum(p,q).sum()),
                            'definition':'Train-normal quartile edges plus tails; one median per case. Coarse descriptive sensitivity check, n_fault2; not density truth or incremental feature value.'})
    pd.DataFrame(overlap).to_csv(a.output/'case_distribution_overlap.csv',index=False)
    print('COMPLETE_CONDITIONAL_CASE_LABEL_CORRELATION_AND_FAULT_SUMMARIES')


if __name__=='__main__':main()
