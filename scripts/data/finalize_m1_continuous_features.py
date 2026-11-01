"""Finalize already generated source-only continuous M1 sidecars, no model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/research'))
from validate_kinematic_residuals import sha


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=read(args.plan);out=Path(plan['output_root']);source=Path(plan['source_root'])
    if (out/'build_manifest.json').exists():raise FileExistsError('Preserve finalized M1 dataset')
    if read(out/'generation_status.json')['status']!='COMPLETE_CASE_FEATURES_AWAITING_DATASET_FINALIZATION':raise ValueError('All source cases must finish first')
    for name,expected in plan['source_code_sha256'].items():
        if sha(name)!=expected:raise ValueError('Pinned source changed:'+name)
    receipts={}; normal_count=0;normal_windows=0;mu=np.zeros(4);m2=np.zeros(4);cases_with_fit=set();rows_by_split={};case_ids=[];sidecars={}
    for task in plan['tasks']:
        uid=task['case_uid'];key=uid.replace('/','__');rp=out/'case_records'/(key+'.json');r=read(rp);cp=out/'cases'/r['sidecar_file']
        assert r['source_case_sha256']==task['case_sha256'] and sha(cp)==r['sidecar_sha256']
        receipts[uid]=r;sidecars[cp.relative_to(out).as_posix()]=r['sidecar_sha256']
        sidecars[rp.relative_to(out).as_posix()]=sha(rp)
        if task['split']!='train':continue
        with np.load(cp,allow_pickle=False) as d:
            selected=d['eligible']&d['whole_normal'];values=d['residual'];starts=d['window_start_row']
            for s in starts[selected]:
                x=values[s:s+150];assert x.shape==(150,4) and np.isfinite(x).all()
                n=len(x);mean=x.mean(0);local=((x-mean)**2).sum(0);delta=mean-mu
                m2+=local+delta*delta*normal_count*n/(normal_count+n);mu+=delta*n/(normal_count+n);normal_count+=n;normal_windows+=1;cases_with_fit.add(uid)
    if normal_count==0:raise ValueError('No valid train-normal residuals for normalization')
    std=np.sqrt(m2/normal_count)
    if not np.isfinite(mu).all() or not np.isfinite(std).all() or (std<=1e-12).any():raise ValueError('Residual training scale degenerate; no fallback')
    stats={'version':plan['version'],'columns':plan['columns'],'mean':mu.tolist(),'std':std.tolist(),'fit_split':'source_train_normal_only','fit_windows':normal_windows,'fit_frame_contributions':normal_count,'fit_cases':len(cases_with_fit),'qualification':'Original admitted; N1..N4 valid at all150 positions; all150 reference states known and zero','weighting':'All150 frames from every qualifying original-cap64 source train normal window, overlap retained; population variance','source_base_norm_sha256':plan['source_data_identities']['norm_stats.json'],'generation_plan_sha256':sha(args.plan)}
    save(out/'residual_norm_stats.json',stats)
    coverage={};export={}
    for split in ['train','val']:
        all_path=source/'ledgers'/(split+'_all_windows.csv');assert sha(all_path)==plan['all_ledger_identities'][split]
        included_path=source/'ledgers'/(split+'_included_windows.csv');assert sha(included_path)==plan['source_data_identities']['ledgers/'+split+'_included_windows.csv']
        all_rows=pd.read_csv(all_path,keep_default_na=False,low_memory=False,dtype={'case_uid':str,'association_group':str})
        frame=pd.read_csv(included_path,keep_default_na=False,low_memory=False,dtype={'case_uid':str,'association_group':str})
        frame['flight_status']=frame.case_uid.str.split('/').str[1];frame['fault_directory']=frame.case_uid.str.split('/').str[2]
        frame['base_compact_index']=np.arange(len(frame));eligible=np.zeros(len(frame),dtype=bool);reasons={};window_records={}
        side_values=[];base_indices=[];original_ids=[];labels=[];case_list=[]
        for uid,g in frame.groupby('case_uid',sort=False):
            r=receipts[uid]
            with np.load(out/'cases'/r['sidecar_file'],allow_pickle=False) as d:
                by_id={int(v):i for i,v in enumerate(d['original_cache_index'])}
                for row in g.itertuples():
                    j=by_id[int(row.original_cache_index)];s=int(d['window_start_row'][j]);assert s==int(row.window_start_row) and int(d['original_reference_y'][j])==int(row.qualified_y)
                    valid=bool(d['eligible'][j]);reason=int(np.bitwise_or.reduce(d['reason'][s:s+150].reshape(-1)))
                    count=int(d['window_residual_valid'][j].sum());eligible[int(row.base_compact_index)]=valid
                    window_records[int(row.original_cache_index)]={'m1_valid':valid,'residual_valid_positions':count,'residual_reason_bits':reason,'residual_case_failure':r['whole_case_failure'] or ''}
                    if valid:
                        x=d['residual'][s:s+150].T.astype(np.float32);assert x.shape==(4,150) and np.isfinite(x).all()
                        side_values.append(x);base_indices.append(int(row.base_compact_index));original_ids.append(int(row.original_cache_index));labels.append(int(row.qualified_y));case_list.append(uid)
                    else:reasons[str(reason)]=reasons.get(str(reason),0)+1
        # Restore exact compact ledger order rather than group iteration order.
        order=np.argsort(base_indices);indices=np.array(base_indices,dtype=np.int64)[order];ids=np.array(original_ids,dtype=np.int64)[order];ys=np.array(labels,dtype=np.int64)[order]
        x=np.stack(side_values)[order];assert np.array_equal(np.flatnonzero(eligible),indices)
        payload={'version':plan['version'],'split':split,'columns':plan['columns'],'normalization_applied':False,'residual':torch.from_numpy(x),'full_window_valid':torch.ones((len(x),150),dtype=torch.bool),'base_compact_index':torch.from_numpy(indices),'original_cache_index':torch.from_numpy(ids),'y':torch.from_numpy(ys),'residual_norm_sha256':sha(out/'residual_norm_stats.json'),'source_cache_sha256':plan['source_data_identities']['cache/'+split+'.pt']}
        cp=out/'cache'/(split+'.pt');torch.save(payload,cp)
        all_rows['m1_valid']=[bool(window_records.get(int(i),{}).get('m1_valid',False)) for i in all_rows.original_cache_index]
        all_rows['residual_valid_positions']=[window_records.get(int(i),{}).get('residual_valid_positions',0) for i in all_rows.original_cache_index]
        all_rows['residual_reason_bits']=[window_records.get(int(i),{}).get('residual_reason_bits',0) for i in all_rows.original_cache_index]
        all_rows['residual_case_failure']=[window_records.get(int(i),{}).get('residual_case_failure','base_not_admitted_no_rescue') for i in all_rows.original_cache_index]
        all_rows['m1_compact_index']=-1;lookup={int(i):n for n,i in enumerate(ids)};all_rows['m1_compact_index']=[lookup.get(int(i),-1) for i in all_rows.original_cache_index]
        for name,f in [('all',all_rows),('included',all_rows[all_rows.m1_valid]),('excluded',all_rows[~all_rows.m1_valid])]:
            f.to_csv(out/'ledgers'/(split+'_'+name+'_windows.csv'),index=False)
        def group_counts(cols):
            f=frame.assign(m1_valid=eligible);result=[]
            for key,g in f.groupby(cols,dropna=False):
                key=key if isinstance(key,tuple) else (key,);key=tuple(v.item() if isinstance(v,np.generic) else v for v in key);result.append(dict(zip(cols,key),original_admitted=len(g),M1_eligible=int(g.m1_valid.sum()),excluded=int((~g.m1_valid).sum())))
            return result
        coverage[split]={'original_window_denominator':len(all_rows),'original_base_admitted':len(frame),'M1_common_eligible':len(indices),'newly_excluded':int((~eligible).sum()),'new_exclusion_fraction':float((~eligible).mean()),'all_admitted_IDs_preserved':bool(eligible.all()),'reason_bit_combinations':reasons,'by_domain_family':group_counts(['domain','case_level2_label']),'by_phase':group_counts(['domain','phase']),'by_log_scenario':group_counts(['domain','flight_status']),'by_fault_directory':group_counts(['domain','fault_directory']),'by_case':group_counts(['case_uid']),'by_label':group_counts(['qualified_y'])}
        export[split]={'file':'cache/'+split+'.pt','sha256':sha(cp),'windows':len(indices),'base_population':len(frame),'normalization_applied':False}
    save(out/'coverage.json',coverage)
    entries={p.relative_to(out).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in [out/'generation_plan.json',out/'residual_norm_stats.json',out/'coverage.json',*list((out/'ledgers').glob('*.csv')),*list((out/'cache').glob('*.pt'))]}
    entries.update({name:{'sha256':h,'bytes':(out/name).stat().st_size} for name,h in sidecars.items()})
    manifest={'status':'COMPLETE_SOURCE_M1_CONTINUOUS150_FEATURES_PENDING_TRAINING_GATE','version':plan['version'],'completed_at':datetime.now(timezone.utc).isoformat(),'cases':len(receipts),'columns':plan['columns'],'horizon_us':200000,'window_frames':150,'source_data_identities':plan['source_data_identities'],'coverage':coverage,'residual_norm_sha256':sha(out/'residual_norm_stats.json'),'exports':export,'files':entries,'old_assets_modified':False,'test_or_Real_read':False,'formal_training_enabled':False,'M0_matched_required':not all(coverage[s]['all_admitted_IDs_preserved'] for s in ['train','val'])}
    save(out/'build_manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'coverage':coverage,'normalization':stats,'M0_matched_required':manifest['M0_matched_required']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
