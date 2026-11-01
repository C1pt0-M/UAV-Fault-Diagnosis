"""Independent M1 export checks against original source IDs and saved evidence.

No training/model inference, no final-test/Real. Does not loosen validity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/uav_fault'))
sys.path.insert(0,str(ROOT/'scripts/research'))
from validate_kinematic_residuals import sha
from uav_fault.data.m1_continuous import M1ContinuousDataset
from replay_m1_continuous import stream_check


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve M1 verification')
    plan=read(a.plan);root=Path(plan['output_root']);base=Path(plan['source_root']);manifest=read(root/'build_manifest.json')
    for name,e in manifest['files'].items():
        f=root/name;assert sha(f)==e['sha256'] and f.stat().st_size==e['bytes'],name
    source_stats=base/'norm_stats.json';assert sha(source_stats)==plan['source_data_identities']['norm_stats.json']
    stats=read(root/'residual_norm_stats.json');count=0;total=np.zeros(4,dtype=np.longdouble);sumsq=np.zeros(4,dtype=np.longdouble);normal_windows=0;old_overlap=0;max_error=0.;stream_cases=0
    by_split={'train':{},'val':{}}
    for task in plan['tasks']:
        uid=task['case_uid'];key=uid.replace('/','__');receipt=read(root/'case_records'/(key+'.json'));assert receipt['source_case_sha256']==task['case_sha256']
        with np.load(root/'cases'/(key+'.npz'),allow_pickle=False) as side,np.load(task['case_file'],allow_pickle=False) as src:
            residual_values=side['residual'];source_starts=src['window_start_row'];source_states=src['reference_state'];source_labels_valid=src['row_label_valid'];source_ids=src['original_cache_index'];source_y=src['reference_y']
            assert np.array_equal(side['timestamp_us'],src['timestamp_us'])
            assert np.array_equal(side['window_start_row'],src['window_start_row'])
            assert np.array_equal(side['original_cache_index'],src['original_cache_index'])
            assert np.array_equal(side['original_supervised_valid'],src['supervised_valid'])
            assert side['residual'].shape==(len(src['timestamp_us']),4) and side['valid'].shape==side['residual'].shape
            reasons=side['reason'];valid=np.column_stack([reasons[:,0]==0,np.repeat((reasons[:,1]==0)[:,None],3,axis=1)])
            assert np.array_equal(valid,side['valid']) and np.isfinite(side['residual'][valid]).all()
            masks=valid.all(1)[src['window_start_row'][:,None]+np.arange(150)]
            expected=src['supervised_valid']&masks.all(1)
            assert np.array_equal(side['window_residual_valid'],masks) and np.array_equal(side['eligible'],expected)
            whole=np.array([source_labels_valid[s:s+150].all() and (source_states[s:s+150]==0).all() for s in source_starts])
            assert np.array_equal(whole,side['whole_normal'])
            for j in np.flatnonzero(expected):
                original=int(source_ids[j]);s=int(source_starts[j]);by_split[task['split']][original]=(uid,s,int(source_y[j]))
            if task['split']=='train':
                for s in source_starts[expected&whole]:
                    x=residual_values[s:s+150].astype(np.longdouble);assert np.isfinite(x).all();total+=x.sum(0);sumsq+=(x*x).sum(0);count+=150;normal_windows+=1
            # Reuse every available previously saved case, never select by score.
            old=ROOT/'artifacts/kinematic_residual_discrimination_v1_20261101/results'/(key+'_residuals.npz')
            if not old.exists():old=ROOT/'artifacts/kinematic_residual_validation_v1_20261101/results_final'/(key+'.npz')
            if old.exists():
                with np.load(old,allow_pickle=False) as prior:
                    common=prior['joint_valid']&side['valid'].all(1);common[:10]=False
                    if common.any():
                        other=np.column_stack([prior['vertical_m'],prior['attitude_rad']]);err=float(np.max(np.abs(other[common]-side['residual'][common])));assert err<1e-11;max_error=max(max_error,err);old_overlap+=int(common.sum())
                if 'prehistory_timestamp_us' in side and stream_cases<4:
                    stream_check({k:side[k] for k in side.files},read(base/'input_contract.json')['max_age_us']);stream_cases+=1
    mean=np.array(total/count,dtype=float);std=np.sqrt(np.array(sumsq/count-(total/count)**2,dtype=float))
    assert count==stats['fit_frame_contributions'] and normal_windows==stats['fit_windows']
    np.testing.assert_allclose(mean,stats['mean'],rtol=0,atol=1e-11);np.testing.assert_allclose(std,stats['std'],rtol=0,atol=1e-10)
    sample_checks=0
    for split in ['train','val']:
        ds=M1ContinuousDataset(base,root,split=split);lookup=by_split[split]
        assert ds.original_cache_indices.tolist()==sorted(lookup)
        labels=np.array(ds.labels);assert all(lookup[int(i)][2]==int(y) for i,y in zip(ds.original_cache_indices,labels))
        for i in [0,len(ds)//2,len(ds)-1]:
            x,r,y=ds[i];base_x,base_y=ds.base[int(ds.base_indices[i])];assert torch.equal(x,base_x) and y==base_y
            uid,s,label=lookup[int(ds.original_cache_indices[i])]
            with np.load(root/'cases'/(uid.replace('/','__')+'.npz'),allow_pickle=False) as side:
                raw=side['residual'][s:s+150].astype(np.float32).T
                expected=(raw-np.array(stats['mean'],dtype=np.float32)[:,None])/np.array(stats['std'],dtype=np.float32)[:,None]
                np.testing.assert_array_equal(r.numpy(),expected);sample_checks+=1
    for name,h in plan['source_data_identities'].items():assert sha(base/name)==h,name
    result={'status':'PASS_INDEPENDENT_FULL_CONTINUOUS_M1_FEATURES','feature_manifest_sha256':sha(root/'build_manifest.json'),'cases':len(plan['tasks']),'all_window_IDs_masks_and_labels_verified':True,'normal_fit_frame_contributions':count,'normal_fit_windows':normal_windows,'independent_mean_max_error':float(np.max(np.abs(mean-np.array(stats['mean'])))),'independent_std_max_error':float(np.max(np.abs(std-np.array(stats['std'])))),'old_frozen_overlap_grid_points':old_overlap,'old_formula_max_error':max_error,'bounded_native_stream_parity_cases':stream_cases,'dataset_first_middle_last_checks':sample_checks,'source_data_and_base_statistics_unchanged':True,'formal_training_or_test_Real':False,'coverage':manifest['coverage']}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
