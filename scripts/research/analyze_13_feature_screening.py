"""Finite source-only13 candidate screening diagnostics, no classifier or training.

The case list is frozen to the union of two pre-existing diagnostic selections.
The old deterministic function is reused; N1..N4 come from saved continuous M1
sidecars. All statistics retain case/split/domain, missingness and original IDs.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
from check_kinematic_feature_response import legacy_features, fixed_auc
from uav_fault.causal_input import canonical_window

IDS = [f'O{i}' for i in range(1,10)] + [f'N{i}' for i in range(1,5)]
KEYS = ['old_output_std','old_output_range','old_ctrl_output_gap_mean','old_ctrl_output_gap_std',
        'old_effort','old_acc_deviation','old_gyro_norm','old_vertical_consistency','old_attitude_energy',
        'rh_signed_m','rR_x_rad','rR_y_rad','rR_z_rad']
STATES = ['climb_proxy','descent_proxy','low_motion_proxy','maneuver_proxy','translation_proxy','other_motion']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def distribution(values):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    if not len(values):return {'n':0,'min':None,'q05':None,'q25':None,'median':None,'q75':None,'q95':None,'max':None}
    result=dict(zip(['min','q05','q25','median','q75','q95','max'],map(float,np.quantile(values,[0,.05,.25,.5,.75,.95,1]))))
    return dict(n=len(values),**result)


def observed_motion(rows,columns):
    rows=np.asarray(rows,dtype=float)
    if rows.shape!=(150,26) or not np.isfinite(rows).all():raise ValueError('Finite past150 physical base rows required')
    v=rows[:,[columns.index('vel_'+a) for a in 'xyz']]
    g=rows[:,[columns.index('gyro_'+a) for a in 'xyz']]
    speed=np.linalg.norm(v,axis=1);rotation=np.linalg.norm(g,axis=1);vz=v[:,2]
    if np.quantile(rotation,.9)>=.3 or np.std(speed)>=.5:state='maneuver_proxy'
    elif np.mean(vz<=-.5)>=.8:state='climb_proxy'
    elif np.mean(vz>=.5)>=.8:state='descent_proxy'
    elif np.mean(speed<=.3)>=.8 and np.mean(rotation<=.15)>=.8:state='low_motion_proxy'
    elif np.mean(speed>.3)>=.8:state='translation_proxy'
    else:state='other_motion'
    return {'observed_state':state,'vel_z_median_mps':float(np.median(vz)),
            'speed_median_mps':float(np.median(speed)),'speed_std_mps':float(np.std(speed)),
            'gyro_p90_radps':float(np.quantile(rotation,.9))}


def corr(frame,method='spearman'):
    return frame[IDS].corr(method=method,min_periods=4)


def balanced_auc(frame,feature,positive,control):
    """Average every normal case's AUC equally; preserve fixed signed direction."""
    positives=frame.loc[positive,feature].to_numpy(dtype=float)
    negatives=frame.loc[control]
    scores=[]
    for _,case in negatives.groupby('case_uid',sort=True):
        score=fixed_auc(case[feature].to_numpy(dtype=float),positives)
        if score is not None:scores.append(score)
    return float(np.mean(scores)) if scores else None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','data','features','historical','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous screening results')
    plan=read(a.plan);selected=plan['cases']
    if len(selected)!=46 or len({c['case_uid'] for c in selected})!=46 or any(c['split'] not in ['train','val'] or c['domain'] not in ['SIL','HIL'] for c in selected):raise ValueError('Only frozen46 source cases allowed')
    feature_manifest=read(a.features/'build_manifest.json');norm=read(a.data/'norm_stats.json');protocol=read(ROOT/'configs/experiments/causal_m1_continuous150_v1.json')
    if sha(a.features/'build_manifest.json')!=protocol['feature_manifest_sha256']:raise ValueError('M1 feature identity changed')
    old=read(ROOT/protocol['M0_protocol_path']);inputs={};torch.set_num_threads(2)
    def identity(path,expected=None):
        actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Input identity changed: '+str(path))
        inputs[str(path)]={'sha256':actual,'bytes':path.stat().st_size}
    for relative in ['norm_stats.json','input_contract.json','ledgers/train_included_windows.csv','ledgers/val_included_windows.csv']:
        identity(a.data/relative,old['data_identities'][relative])
    for path in [a.plan,a.features/'build_manifest.json',Path(__file__),ROOT/'scripts/research/check_kinematic_feature_response.py',ROOT/'packages/uav_fault/uav_fault/models/cnn/lstm.py']:
        identity(path)
    c=read(a.historical/'results/completion.json')
    for name,item in c['output_identities'].items():identity(a.historical/'results'/name,item['sha256'])
    for name,h in c['input_sha256'].items():identity(ROOT/name,h)
    metadata={split:pd.read_csv(a.data/'ledgers'/(split+'_included_windows.csv'),keep_default_na=False,low_memory=False) for split in ['train','val']}
    windows=[];ledgers=[];coverage=[];parameters=[];frame_tails={name:[] for name in IDS};finite_counts={name:0 for name in IDS};independent_window_counts={name:0 for name in IDS}
    normal_frame_tails={name:[] for name in IDS};normal_independent_counts={name:0 for name in IDS};o5_clipped=0;o5_frames=0
    for case in selected:
        uid=case['case_uid'];key=uid.replace('/','__');receipt_path=a.data/'case_records'/(key+'.json');record=read(receipt_path)
        if record['split']!=case['split'] or record['role']!=case['split'] or record['domain']!=case['domain']:raise ValueError('Case source role mismatch')
        bp=a.data/'cases'/(key+'.npz');sp=a.features/'cases'/(key+'.npz')
        expected=next(h for name,h in record['output_identities'].items() if Path(name).name==bp.name)
        identity(receipt_path);identity(bp,expected);identity(sp,feature_manifest['files']['cases/'+sp.name]['sha256'])
        with np.load(bp,allow_pickle=False) as z:b={k:z[k] for k in z.files}
        with np.load(sp,allow_pickle=False) as z:s={k:z[k] for k in z.files}
        for name in ['timestamp_us','window_start_row','window_end_row','original_cache_index']:
            if not np.array_equal(b[name],s[name]):raise ValueError('Sidecar/base alignment failed: '+name)
        if not np.array_equal(b['reference_state'],s['original_reference_state']):raise ValueError('Historical labels changed')
        row_meta=metadata[case['split']][metadata[case['split']].case_uid==uid].set_index('original_cache_index')
        association=case['case_uid'].split('/')[1]+'/'+case['subtype']+'/'+record['group']
        stats=dict(case,association_group=association,original_windows=len(b['original_cache_index']),base_admitted=int(b['supervised_valid'].sum()),N1_admitted=0,attitude_admitted=0,joint_admitted=0)
        if case['kind']=='fault':
            # Bounded metadata only: preserve the original literal parameter
            # arrays, without guessing which index means bias/noise/severity.
            tipath=Path(record['raw_ulog']).parent.parent/'TestInfo.csv'
            if tipath.is_file():
                identity(tipath)
                with tipath.open(encoding='utf-8-sig',newline='') as stream:ti={r[0]:r[1] for r in csv.reader(stream) if len(r)>=2}
                parameters.append(dict(case,association_group=association,fault_id_literal=ti.get('Fault ID',''),fault_parameter_literal=ti.get('Fault Parameter',''),control_sequence_literal=ti.get('Control Sequence',''),declared_injection_time_literal=ti.get('Fault injection time',''),severity_semantics='NOT_DECODED_NO_CALIBRATED_INTENSITY_ORDER'))
            else:parameters.append(dict(case,association_group=association,fault_id_literal='',fault_parameter_literal='',control_sequence_literal='',declared_injection_time_literal='',severity_semantics='MISSING_TESTINFO'))
        for i,(start,end,oid,base_ok) in enumerate(zip(b['window_start_row'],b['window_end_row'],b['original_cache_index'],b['supervised_valid'])):
            start,end,oid=int(start),int(end),int(oid)
            n1=bool(s['valid'][start:end,0].all());att=bool(s['valid'][start:end,1:].all());common=bool(base_ok and n1 and att)
            if common!=bool(s['eligible'][i]):raise ValueError('Full150 qualification differs from frozen M1 sidecar')
            stats['N1_admitted']+=bool(base_ok and n1);stats['attitude_admitted']+=bool(base_ok and att);stats['joint_admitted']+=common
            ledger=dict(case,association_group=association,original_cache_index=oid,window_start_row=start,window_end_row=end,base_admitted=bool(base_ok),N1_full150=n1,attitude_full150=att,common_eligible=common,vertical_reason_bits=int(np.bitwise_or.reduce(s['reason'][start:end,0])),attitude_reason_bits=int(np.bitwise_or.reduce(s['reason'][start:end,1])))
            ledgers.append(ledger)
            if not base_ok:continue
            if oid not in row_meta.index:raise ValueError('Original admitted ID missing from source-only ledger')
            linked=row_meta.loc[oid]
            if linked.window_start_row!=start or linked.window_end_row!=end or linked.qualified_y!=b['reference_y'][i]:raise ValueError('Window ID/time/label differs')
            physical=b['canonical_rows'][start:end]
            x=canonical_window(physical,norm['columns'],b['row_input_valid'][start:end],mean=norm['mean'],std=norm['std'])
            old_values=legacy_features(x,norm['columns'])
            o5_clipped+=int((x[norm['columns'].index('acc_norm')]<=.1).sum());o5_frames+=150
            residual=s['residual'][start:end].T
            for j,name in enumerate(IDS):
                values=old_values[j] if j<9 else residual[j-9]
                valid=True if j<9 else n1 if j==9 else att
                if valid:
                    if not np.isfinite(values).all():raise ValueError('Nonfinite admitted candidate: '+name)
                    independent_window_counts[name]+=1;finite_counts[name]+=len(values);frame_tails[name].append(values.copy())
                    if case['kind']=='normal':normal_independent_counts[name]+=1;normal_frame_tails[name].append(values.copy())
            if not common:continue
            states=b['reference_state'][start:end];known=b['row_label_valid'][start:end]
            group='normal' if known.all() and (states==0).all() else 'active' if known.all() and (states==1).all() else 'mixed_or_unknown'
            motion=observed_motion(physical,norm['columns'])
            row=dict(case,association_group=association,original_cache_index=oid,window_start_row=start,window_end_row=end,
                     first_timestamp_s=float(b['timestamp_us'][start]/1e6),last_timestamp_s=float(b['timestamp_us'][end-1]/1e6),
                     label_group=group,phase=str(linked.phase),fault_fraction=float(linked.fault_fraction),**motion)
            all_values=np.vstack([old_values,residual])
            for j,name in enumerate(IDS):
                values=all_values[j];row[name]=float(np.median(values));row[name+'_p95']=float(np.quantile(values,.95));row[name+'_std']=float(np.std(values))
            windows.append(row)
        coverage.append(stats);print('SCREEN_CASE',case['split'],uid,'base',stats['base_admitted'],'common',stats['joint_admitted'],flush=True)
    f=pd.DataFrame(windows)
    if f.duplicated(['split','original_cache_index']).any():raise ValueError('Duplicate source original IDs')
    normal=f[(f.kind=='normal')&(f.label_group=='normal')];train_normal=normal[normal.split=='train']
    reference={}
    for name in IDS:
        d=distribution(train_normal[name]);reference[name]={'median':d['median'],'iqr':d['q75']-d['q25'],'windows':len(train_normal),'cases':train_normal.case_uid.nunique(),'diagnostic_only':True,'model_normalization_changed':False}
    case_vectors=f.groupby(['split','case_uid','domain','kind','subtype','association_group'],sort=True)[IDS].median().reset_index()
    primary=case_vectors[case_vectors.split=='train'];primary_corr=corr(primary)
    matrices=[];matrix_metadata=[]
    def matrix(label,part,aggregation):
        for method in ['spearman','pearson']:
            c=corr(part,method)
            matrices.append((label,method,c))
            matrix_metadata.append({'scope':label,'method':method,'rows':len(part),'cases':int(part.case_uid.nunique()) if 'case_uid' in part else None,'aggregation':aggregation,'constant_features':[n for n in IDS if part[n].nunique()<2]})
    matrix('train_case_medians',primary,'One median13-vector per case, same150 population; primary exploratory matrix')
    matrix('val_case_medians',case_vectors[case_vectors.split=='val'],'One median vector per exposed-Val case, descriptive confirmation')
    matrix('all_case_medians',case_vectors,'One median vector per case; mixes splits and classes, supplementary only')
    pair_vectors=case_vectors.groupby(['split','association_group'],sort=True)[IDS].median().reset_index()
    pair_vectors['case_uid']=pair_vectors.association_group
    matrix('train_pair_group_medians',pair_vectors[pair_vectors.split=='train'],'Collapse SIL/HIL pair group before correlation; no independent-window confidence')
    for split in ['train','val']:
        for domain in ['SIL','HIL']:
            matrix(split+'_'+domain+'_case_medians',case_vectors[(case_vectors.split==split)&(case_vectors.domain==domain)],'One median vector per case')
        for kind in ['normal','fault']:
            matrix(split+'_'+kind+'_case_medians',case_vectors[(case_vectors.split==split)&(case_vectors.kind==kind)],'One median vector per case')
        for subtype in ['motor','accelerometer','gyroscope','barometer','magnetometer','GPS']:
            part=f[(f.split==split)&(f.subtype==subtype)&(f.label_group=='active')].groupby('case_uid')[IDS].median().reset_index()
            matrix(split+'_'+subtype+'_active_case_medians',part,'Only2 SIL/HIL cases: correlation intentionally undefined with min_periods4')
        for state in STATES:
            part=normal[(normal.split==split)&(normal.observed_state==state)].groupby('case_uid')[IDS].median().reset_index()
            matrix(split+'_normal_'+state,part,'One case-state median; signal proxy not online truth')
    within=[]
    for (split,uid),part in f.groupby(['split','case_uid'],sort=True):
        c=corr(part)
        for j,name in enumerate(IDS):
            for other in IDS[j+1:]:within.append({'split':split,'case_uid':uid,'feature_a':name,'feature_b':other,'spearman':c.loc[name,other],'windows':len(part)})
    within_frame=pd.DataFrame(within);redundancy=[]
    for j,name in enumerate(IDS):
        for other in IDS[j+1:]:
            w=within_frame[(within_frame.feature_a==name)&(within_frame.feature_b==other)&(within_frame.split=='train')].spearman.dropna()
            redundancy.append({'feature_a':name,'feature_b':other,'train_case_spearman':primary_corr.loc[name,other],
                               'train_case_pearson':corr(primary,'pearson').loc[name,other],
                               'val_case_spearman':corr(case_vectors[case_vectors.split=='val']).loc[name,other],
                               'train_pair_group_spearman':corr(pair_vectors[pair_vectors.split=='train']).loc[name,other],
                               'within_train_cases':len(w),'within_train_spearman_median':float(w.median()) if len(w) else np.nan,
                               'within_train_spearman_q10':float(w.quantile(.1)) if len(w) else np.nan,'within_train_spearman_q90':float(w.quantile(.9)) if len(w) else np.nan})
    response=[]
    for (split,uid),case in f[f.kind=='fault'].groupby(['split','case_uid'],sort=True):
        meta=case.iloc[0];active=case[case.label_group=='active'];pre=case[case.label_group=='normal']
        controls=normal[(normal.split==split)&(normal.domain==meta.domain)&(normal.case_uid.str.split('/').str[1]=='hover')]
        for name in IDS:
            dpre,dact=distribution(pre[name]),distribution(active[name]);scale=reference[name]['iqr'];delta=dact['median']-dpre['median'] if len(pre) and len(active) else None
            state_details=[]
            for state,part in active.groupby('observed_state'):
                matched=controls[controls.observed_state==state]
                auc=balanced_auc(f,name,part.index,matched.index) if len(matched) else None
                state_details.append({'state':state,'active_windows':len(part),'normal_windows':len(matched),'normal_cases':int(matched.case_uid.nunique()),'auc':auc})
            matched=[x for x in state_details if x['auc'] is not None];n=sum(x['active_windows'] for x in matched)
            response.append({'feature_id':name,'split':split,'case_uid':uid,'domain':meta.domain,'subtype':meta.subtype,'association_group':meta.association_group,
                             'pre_windows':len(pre),'active_windows':len(active),'pre_median':dpre['median'],'active_median':dact['median'],
                             'active_q05':dact['q05'],'active_q95':dact['q95'],'active_min':dact['min'],'active_max':dact['max'],
                             'active_minus_pre':delta,'shift_in_train_normal_IQR':delta/scale if delta is not None and scale>1e-12 else None,
                             'auc_active_vs_same_case_pre_fixed_direction':fixed_auc(pre[name],active[name]),
                             'auc_vs_same_observed_state_independent_normals':sum(x['active_windows']*x['auc'] for x in matched)/n if n else None,
                             'matched_active_windows':n,'unmatched_active_windows':len(active)-n,'state_details':json.dumps(state_details,ensure_ascii=False)})
    stability=[];stability_pairs=[]
    for (split,uid,state),part in normal.groupby(['split','case_uid','observed_state'],sort=True):
        meta=part.iloc[0]
        for name in IDS:
            d=distribution(part[name]);scale=reference[name]['iqr']
            stability.append({'feature_id':name,'split':split,'case_uid':uid,'domain':meta.domain,'association_group':meta.association_group,
                              'directory':uid.split('/')[1],'observed_state':state,'windows':len(part),**{k:v for k,v in d.items() if k!='n'},
                              'median_offset_in_train_normal_IQR':(d['median']-reference[name]['median'])/scale if scale>1e-12 else None})
    stability_frame=pd.DataFrame(stability)
    for (split,uid),part in stability_frame.groupby(['split','case_uid'],sort=True):
        for name in IDS:
            temp=part[part.feature_id==name].set_index('observed_state');scale=reference[name]['iqr']
            for astate,bstate in [('climb_proxy','low_motion_proxy'),('maneuver_proxy','low_motion_proxy'),('translation_proxy','low_motion_proxy')]:
                available=astate in temp.index and bstate in temp.index
                delta=float(temp.loc[bstate,'median']-temp.loc[astate,'median']) if available else None
                stability_pairs.append({'feature_id':name,'split':split,'case_uid':uid,'contrast':bstate+' minus '+astate,'paired':available,
                                        'state_a_windows':int(temp.loc[astate,'windows']) if astate in temp.index else 0,
                                        'state_b_windows':int(temp.loc[bstate,'windows']) if bstate in temp.index else 0,
                                        'paired_shift':delta,'paired_shift_in_train_normal_IQR':delta/scale if delta is not None and scale>1e-12 else None})
    # Pair-matched SIL/HIL normal-state shifts, not pooled independent-window tests.
    domain_shifts=[]
    for (split,pair,state,name),part in stability_frame.groupby(['split','association_group','observed_state','feature_id'],sort=True):
        if set(part.domain)=={'SIL','HIL'}:
            temp=part.set_index('domain');delta=float(temp.loc['HIL','median']-temp.loc['SIL','median']);scale=reference[name]['iqr']
            domain_shifts.append({'feature_id':name,'split':split,'association_group':pair,'observed_state':state,
                                  'HIL_minus_SIL_median':delta,'shift_in_train_normal_IQR':delta/scale if scale>1e-12 else None})
    validity=[]
    for name in IDS:
        v=np.concatenate(frame_tails[name]);nv=np.concatenate(normal_frame_tails[name]);d=distribution(v);nd=distribution(nv)
        validity.append({'feature_id':name,'base_admitted_windows':sum(x['base_admitted'] for x in coverage),'independent_valid_windows':independent_window_counts[name],
                         'own_full150_valid_fraction':independent_window_counts[name]/sum(x['base_admitted'] for x in coverage),
                         'observed_frame_contributions':finite_counts[name],'all_admitted_values_finite':True,'normal_independent_valid_windows':normal_independent_counts[name],
                         **{'raw_'+k:v for k,v in d.items()},**{'normal_raw_'+k:v for k,v in nd.items()}})
    distributions=[]
    for (split,kind,label),part in f.groupby(['split','kind','label_group'],sort=True):
        for name in IDS:
            d=distribution(part[name]);distributions.append({'feature_id':name,'split':split,'kind':kind,'label_group':label,'cases':part.case_uid.nunique(),**d})
    a.output.mkdir(parents=True,exist_ok=False)
    outputs={'window_features.csv':f,'window_qualification.csv':pd.DataFrame(ledgers),'case_coverage.csv':pd.DataFrame(coverage),
             'case_feature_medians.csv':case_vectors,'feature_correlation.csv':primary_corr,'feature_correlation_pearson.csv':corr(primary,'pearson'),
             'feature_correlation_by_group.csv':pd.DataFrame([{'scope':label,'method':method,'feature_a':x,'feature_b':y,'correlation':c.loc[x,y]} for label,method,c in matrices for x in IDS for y in IDS]),
             'within_case_correlations.csv':within_frame,'redundancy_pairs.csv':pd.DataFrame(redundancy),
             'feature_fault_response.csv':pd.DataFrame(response),'feature_normal_stability.csv':stability_frame,
             'normal_state_paired_shifts.csv':pd.DataFrame(stability_pairs),'normal_domain_paired_shifts.csv':pd.DataFrame(domain_shifts),
             'feature_validity.csv':pd.DataFrame(validity),'feature_distributions.csv':pd.DataFrame(distributions),'fault_parameter_metadata.csv':pd.DataFrame(parameters)}
    for name,df in outputs.items():df.to_csv(a.output/name,index=name in ['feature_correlation.csv','feature_correlation_pearson.csv'])
    write(a.output/'diagnostic_reference.json',reference);write(a.output/'correlation_scopes.json',matrix_metadata)
    for path,item in inputs.items():
        if sha(path)!=item['sha256']:raise ValueError('Input changed during analysis: '+path)
    result={'status':'COMPLETE_BOUNDED46_CASE13_FEATURE_CPU_ANALYSIS','completed_at':datetime.now(timezone.utc).isoformat(),
            'plan_sha256':sha(a.plan),'inputs':inputs,'features':dict(zip(IDS,KEYS)),'cases':len(selected),'train_cases':sum(x['split']=='train' for x in selected),
            'windows':len(f),'qualification_windows':len(ledgers),'base_admitted_windows':sum(x['base_admitted'] for x in coverage),
            'own_N1_windows':sum(x['N1_admitted'] for x in coverage),'own_attitude_windows':sum(x['attitude_admitted'] for x in coverage),
            'normal_windows':len(normal),'train_reference_windows':len(train_normal),'train_reference_cases':int(train_normal.case_uid.nunique()),
            'window_labels':f.groupby(['split','kind','label_group']).size().reset_index(name='windows').to_dict(orient='records'),
            'normal_states':normal.groupby(['split','observed_state']).agg(windows=('case_uid','size'),cases=('case_uid','nunique')).reset_index().to_dict(orient='records'),
            'O5_acc_z_denominator_clipped_frame_contributions':o5_clipped,'O5_total_old_admitted_frame_contributions':o5_frames,
            'no_training_classifier_inference_GPU_or_server':True,'raw_ULog_read':False,'final_test_Real_read':False,
            'no_old_assets_or_normalization_changed':True,'source_script_sha256':sha(Path(__file__))}
    write(a.output/'analysis_completion.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
