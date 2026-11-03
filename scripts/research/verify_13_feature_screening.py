"""Independent checks on finite13-feature diagnostics; no models or raw logs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from analyze_13_feature_screening import IDS,sha,read


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['results','data','features','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve independent verification')
    s=read(a.results/'analysis_completion.json')
    for path,item in s['inputs'].items():assert sha(path)==item['sha256'],path
    f=pd.read_csv(a.results/'window_features.csv');ledger=pd.read_csv(a.results/'window_qualification.csv');assert len(f)==2644 and len(ledger)==2944
    assert not f.duplicated(['split','original_cache_index']).any() and set(f.split)=={'train','val'} and set(f.domain)=={'SIL','HIL'}
    cv=f.groupby(['split','case_uid'],sort=True)[IDS].median().reset_index();train=cv[cv.split=='train'];ranks=np.column_stack([rankdata(train[name]) for name in IDS]);corr=np.corrcoef(ranks,rowvar=False)
    actual=pd.read_csv(a.results/'feature_correlation.csv',index_col=0).loc[IDS,IDS].to_numpy();assert np.max(np.abs(corr-actual))<1e-14;assert np.allclose(actual,actual.T,rtol=0,atol=1e-15)
    response=pd.read_csv(a.results/'feature_fault_response.csv');max_auc=0.
    for _,row in response.iterrows():
        case=f[(f.split==row.split)&(f.case_uid==row.case_uid)];pre=case[case.label_group=='normal'][row.feature_id].to_numpy();active=case[case.label_group=='active'][row.feature_id].to_numpy()
        expected=float(np.mean((active[:,None]>pre[None,:])+.5*(active[:,None]==pre[None,:])))
        error=abs(expected-row.auc_active_vs_same_case_pre_fixed_direction);assert error<1e-14;max_auc=max(max_auc,error)
        controls=f[(f.split==row.split)&(f.domain==row.domain)&(f.kind=='normal')&(f.case_uid.str.split('/').str[1]=='hover')]
        states=case[case.label_group=='active'];weighted=[]
        for state,part in states.groupby('observed_state'):
            negative=controls[controls.observed_state==state];scores=[]
            for _,normal in negative.groupby('case_uid'):
                positive=part[row.feature_id].to_numpy();negative_values=normal[row.feature_id].to_numpy()
                scores.append(float(np.mean((positive[:,None]>negative_values[None,:])+.5*(positive[:,None]==negative_values[None,:]))))
            if scores:weighted.append((len(part),np.mean(scores)))
        n=sum(x for x,_ in weighted);expected=sum(n*v for n,v in weighted)/n if n else np.nan
        if n:assert abs(expected-row.auc_vs_same_observed_state_independent_normals)<1e-14
        assert n==row.matched_active_windows
    feature_errors=[];checked=0
    norm=read(a.data/'norm_stats.json');mean=np.asarray(norm['mean'],dtype=np.float32)[:,None];std=np.asarray(norm['std'],dtype=np.float32)[:,None];columns=norm['columns']
    for uid,case in f.groupby('case_uid'):
        indices=case.sort_values('window_start_row').iloc[np.unique(np.linspace(0,len(case)-1,min(3,len(case))).round().astype(int))]
        key=uid.replace('/','__')+'.npz'
        with np.load(a.data/'cases'/key,allow_pickle=False) as z:b={k:z[k] for k in ['canonical_rows','original_cache_index','window_start_row','window_end_row']}
        with np.load(a.features/'cases'/key,allow_pickle=False) as z:residual=z['residual'];valid=z['valid']
        for _,row in indices.iterrows():
            st,en=int(row.window_start_row),int(row.window_end_row);position=np.flatnonzero(b['original_cache_index']==row.original_cache_index);assert len(position)==1
            assert valid[st:en].all() and b['window_start_row'][position[0]]==st and b['window_end_row'][position[0]]==en
            x=b['canonical_rows'][st:en].copy();ai=[columns.index('accel_'+a) for a in 'xyz'];x[:,columns.index('acc_norm')]=np.linalg.norm(x[:,ai],axis=1);x[:,columns.index('baro_alt_relative')]-=x[0,columns.index('baro_alt_relative')];z=(x.astype(np.float32).T-mean)/std
            u=z[[columns.index('actuator_ctrl_'+str(i)) for i in range(1,5)]];o=z[[columns.index('actuator_out_'+str(i)) for i in range(1,5)]];gap=np.abs(u-o);acc=z[columns.index('acc_norm')];gyro=z[[columns.index('gyro_'+a) for a in 'xyz']];quat=z[[columns.index('att_q_'+a) for a in 'xyz']];baro=z[columns.index('baro_alt_relative')];vz=z[columns.index('vel_z')]
            old=np.stack([o.std(0),np.ptp(o,axis=0),gap.mean(0),gap.std(0),np.concatenate([u,o]).mean(0)/np.maximum(acc,.1),np.abs(acc),np.linalg.norm(gyro,axis=0),np.abs(np.diff(baro,prepend=baro[0])-vz),np.linalg.norm(quat,axis=0)])
            for i,name in enumerate(IDS):
                expected=float(np.median(old[i].astype(np.float64))) if i<9 else float(np.median(residual[st:en,i-9]));error=abs(expected-row[name]);assert error<3e-6 if i<9 else error<1e-12;feature_errors.append(error)
            checked+=1
    stability=pd.read_csv(a.results/'feature_normal_stability.csv');paired=pd.read_csv(a.results/'normal_state_paired_shifts.csv');ref=read(a.results/'diagnostic_reference.json')
    for _,row in paired[paired.paired].iterrows():
        part=stability[(stability.case_uid==row.case_uid)&(stability.feature_id==row.feature_id)].set_index('observed_state');bstate,astate=row.contrast.split(' minus ');delta=part.loc[bstate,'median']-part.loc[astate,'median'];assert abs(delta-row.paired_shift)<1e-12
        assert abs(delta/ref[row.feature_id]['iqr']-row.paired_shift_in_train_normal_IQR)<1e-10
    record={'status':'PASS_INDEPENDENT13_FEATURE_SCREENING_CHECK','checked_at':datetime.now(timezone.utc).isoformat(),'case_count':46,'paired_common150_windows':len(f),'all_qualification_IDs':len(ledger),'independently_rebuilt_representative_windows':checked,
            '13x13_train_Spearman_max_error':float(np.max(np.abs(corr-actual))),'all312_same_case_signed_AUCs_checked':len(response),'max_auc_error':max_auc,'state_matched_AUCs_also_recomputed':True,
            '13_feature_reconstruction_max_error':float(max(feature_errors)),'normal_signal_state_pairs_recomputed':int(paired.paired.sum()),'all_inputs_unchanged':True,'training_server_GPU_test_Real_access':False,'verifier_sha256':sha(Path(__file__))}
    a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
