"""M3 source-Train-only scaling and Train/Val identity/finite preflight.

No training, model/checkpoint loading, GPU or final-test/Real access. Computes
old9 deterministically in bounded CPU batches; only stores new scaling/evidence.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse,json,sys,hashlib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
from uav_fault.data.m3_features import CANDIDATE_IDS,old_candidate_features
from uav_fault.data.causal_cache import sha
from smoke_normal_source_sampling import sampling_weights,probability_table


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['data','features','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('New M3 preparation directory required')
    p=read(ROOT/'configs/experiments/causal_m1_continuous150_v1.json');old=read(ROOT/p['M0_protocol_path'])
    if sha(a.features/'build_manifest.json')!=p['feature_manifest_sha256']:raise ValueError('Frozen continuous manifest changed')
    manifest=read(a.features/'build_manifest.json');norm=read(a.data/'norm_stats.json');rn=read(a.features/'residual_norm_stats.json')
    if rn['fit_split']!='source_train_normal_only' or rn['fit_windows']!=74700 or sha(a.features/'residual_norm_stats.json')!=manifest['residual_norm_sha256']:
        raise ValueError('Only frozen Train-normal residual statistics permitted')
    inputs={}
    def pin(path,expected=None):
        actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Historical input changed: '+str(path))
        inputs[str(path)]={'sha256':actual,'bytes':path.stat().st_size}
    for rel in ['norm_stats.json','input_contract.json','ledgers/train_included_windows.csv','ledgers/val_included_windows.csv','cache/train.pt','cache/val.pt']:
        pin(a.data/rel,old['data_identities'][rel])
    for rel,h in p['code_identities'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Historical frozen code changed: '+rel)
    pin(a.features/'build_manifest.json');pin(a.features/'residual_norm_stats.json');pin(Path(__file__));pin(ROOT/'packages/uav_fault/uav_fault/data/m3_features.py')
    frames={};datasets={};identity={};splits={};mu=torch.tensor(norm['mean'],dtype=torch.float32).view(1,26,1);std=torch.tensor(norm['std'],dtype=torch.float32).view(1,26,1)
    for split in ['train','val']:
        entry=manifest['exports'][split];pin(a.features/entry['file'],entry['sha256'])
        base=torch.load(a.data/'cache'/(split+'.pt'),map_location='cpu',weights_only=True,mmap=True)
        residual=torch.load(a.features/entry['file'],map_location='cpu',weights_only=True,mmap=True)
        f=pd.read_csv(a.data/'ledgers'/(split+'_included_windows.csv'),keep_default_na=False,low_memory=False,dtype={'association_group':str,'case_uid':str})
        index=residual['base_compact_index'];selected=f.iloc[index.numpy()].copy().reset_index(drop=True)
        if base['split']!=split or residual['split']!=split or base['normalization_applied'] is not False or residual['normalization_applied'] is not False:raise ValueError('Wrong split/scale contract')
        if len(index)!=p['data_counts'][split] or not torch.all(index[1:]>index[:-1]) or not residual['full_window_valid'].all():raise ValueError('Common150 population/validity changed')
        if not torch.equal(base['original_cache_index'][index],residual['original_cache_index']) or not np.array_equal(selected.original_cache_index,residual['original_cache_index']):raise ValueError('Original ID alignment changed')
        if not torch.equal(base['y'][index],residual['y']) or not np.array_equal(selected.qualified_y,residual['y']):raise ValueError('Original labels changed')
        if not selected.split.eq(split).all() or not selected.domain.isin(['SIL','HIL']).all():raise ValueError('Non-source data')
        if not selected.window_end_row.sub(selected.window_start_row).eq(150).all():raise ValueError('Full150 range required')
        frames[split]=selected;datasets[split]=(base,residual)
        identity[split]={'windows':len(index),'original_ids_sha256':tensor_digest(residual['original_cache_index']),'labels_sha256':tensor_digest(residual['y']),
                         'residual_cache_sha256':entry['sha256'],'base_cache_sha256':old['data_identities']['cache/'+split+'.pt'],
                         'base_compact_index_sha256':tensor_digest(index),'metadata_case_count':selected.case_uid.nunique(),
                         'window_id_time_rows_sha256':hashlib.sha256(selected[['original_cache_index','window_start_row','window_end_row']].to_csv(index=False).encode()).hexdigest()}
    if set(frames['train'].case_uid)&set(frames['val'].case_uid) or set(frames['train'].association_group)&set(frames['val'].association_group):raise ValueError('Case/pair leakage')
    trainbase,trainres=datasets['train'];t=frames['train']
    fit=(trainres['y'].numpy()==0)&t.whole_window_label_valid.to_numpy(dtype=bool)&t.fault_fraction.eq(0).to_numpy()
    if fit.sum()!=rn['fit_windows']:raise ValueError('Exact frozen74700 Train-normal fit population required')
    allsum=np.zeros(13);sumsq=np.zeros(13);count=0;central_mean=np.zeros(13);central_M2=np.zeros(13)
    torch.set_num_threads(4)
    for split in ['train','val']:
        base,res=datasets[split];total=len(res['y']);minv=np.full(13,np.inf);maxv=np.full(13,-np.inf);finite=0
        for start in range(0,total,512):
            end=min(start+512,total);x=(base['x'][res['base_compact_index'][start:end]]-mu)/std
            old_values=old_candidate_features(x);raw=torch.cat([old_values,res['residual'][start:end]],dim=1).numpy().astype(np.float64)
            if raw.shape!=(end-start,13,150) or not np.isfinite(raw).all():raise ValueError('Nonfinite candidate before fitting/selector; stop entire preflight')
            minv=np.minimum(minv,raw.min(axis=(0,2)));maxv=np.maximum(maxv,raw.max(axis=(0,2)));finite+=len(raw)
            if split=='train':
                normal=raw[fit[start:end]]
                if len(normal):
                    n=len(normal)*150;sum1=normal.sum(axis=(0,2));s2=np.square(normal).sum(axis=(0,2));mean1=sum1/n
                    m2=np.square(normal-mean1[None,:,None]).sum(axis=(0,2));delta=mean1-central_mean
                    central_M2+=m2+delta*delta*(count*n/(count+n));central_mean+=delta*n/(count+n)
                    allsum+=sum1;sumsq+=s2;count+=n
        splits[split]={'windows_checked_all13':finite,'raw_min':minv.tolist(),'raw_max':maxv.tolist(),'finite_before_BN':True,
                       'original_normal_windows':int(((frames[split].case_level2_label=='normal')&(frames[split].qualified_y==0)).sum()),
                       'fault_windows':int((frames[split].qualified_y==1).sum()),'normal_windows':int((frames[split].qualified_y==0).sum())}
        print('M3_SCALING_SOURCE_SPLIT_CHECKED',split,total,flush=True)
    mean=allsum/count;variance=sumsq/count-mean*mean;scale=np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale>0).all():raise ValueError('Degenerate feature std: stop, no fallback')
    if np.max(np.abs(central_mean-mean))>1e-10 or np.max(np.abs(np.sqrt(central_M2/count)-scale))>1e-9:raise ValueError('Independent central-moment scaling mismatch')
    normal_ids=trainres['original_cache_index'][torch.from_numpy(fit)]
    # Retain the original N statistics fitted in float64 on continuous physical
    # case values. Exported residual cache float32 is checked, never re-fitted
    # as a new N scaler or used to overwrite historical statistics.
    n_mean_error=np.abs(mean[9:]-np.asarray(rn['mean']));n_std_error=np.abs(scale[9:]-np.asarray(rn['std']))
    if n_mean_error.max()>1e-7 or n_std_error.max()>1e-7:raise ValueError('Residual Train-reference check differs from frozen physical fit')
    merged_mean=mean[:9].tolist()+rn['mean'];merged_std=scale[:9].tolist()+rn['std']
    origins=np.where(t.qualified_y==1,'fault_endpoint',np.where(t.case_level2_label=='normal','original_normal_case','normal_part_of_fault_case'))
    weights=sampling_weights(trainres['y'].numpy(),origins,True);sampling=probability_table(weights,trainres['y'].numpy(),origins)
    metadata=pd.concat([frames['train'].assign(source_split='train'),frames['val'].assign(source_split='val')])
    fault_counts=[]
    for (split,domain,subtype),part in metadata[metadata.qualified_y==1].assign(subtype=lambda q:q.case_uid.str.split('/').str[2]).groupby(['source_split','domain','subtype']):
        fault_counts.append({'split':split,'domain':domain,'subtype':subtype,'fault_windows':len(part),'cases':part.case_uid.nunique()})
    stats={'status':'FROZEN_SOURCE_TRAIN_NORMAL_M3_SCALING','version':'m3_old9_secondary_z_and_frozen_N4_v1','candidate_ids':list(CANDIDATE_IDS),
           'fit_split':'source_train_known_normal_only','fit_windows':int(fit.sum()),'fit_frame_contributions':count,'fit_cases':t[fit].case_uid.nunique(),
           'fit_original_ID_sha256':tensor_digest(normal_ids),'weighting':'All150 frames per qualifying original cap64 window, overlap retained, population variance. No Val fitting.',
           'base_norm_sha256':sha(a.data/'norm_stats.json'),'residual_stats_sha256':sha(a.features/'residual_norm_stats.json'),
           'old9':{'formula':'unchanged OnlineKGMSTCA._relation_features on frozen normalized base26','mean':mean[:9].tolist(),'std':scale[:9].tolist(),
                   'reason':'Old9 derived outputs have different offsets/scales and ratios; one common Train-normal secondary affine scale improves optimization comparability without changing definitions. New M3 preprocessing, not historical M2.'},
           'new4':{'mean':rn['mean'],'std':rn['std'],'source':'Exact frozen M1 Train-normal74700 window physical fit','float32_export_recompute_mean_max_error':float(n_mean_error.max()),'float32_export_recompute_std_max_error':float(n_std_error.max())},
           'mean':merged_mean,'std':merged_std,'source_split_identities':identity,'variance_policy':'population; reject zero/nonfinite scale, no silent epsilon floor',
           'finite_outliers':'retain; no clipping/winsorization/log/abs/sign change/new residual horizon','missing_policy':'Reject whole sample before selector/BN, preserve original-ID exclusion. New mismatch/invalid in this frozen population stops preflight; do not silently shrink or refill.',
           'independent_central_mean_max_error':float(np.max(np.abs(central_mean-mean))),'independent_central_std_max_error':float(np.max(np.abs(np.sqrt(central_M2/count)-scale)))}
    a.output.mkdir(parents=True,exist_ok=False);save(a.output/'normalization.json',stats)
    rows=[]
    for j,name in enumerate(CANDIDATE_IDS):
        rows.append({'id':name,'unit':'z-space statistical proxy' if j<9 else 'm' if j==9 else 'rad','mean':merged_mean[j],'std':merged_std[j],
                     'transform':'(raw_candidate-mean)/std; O raw=historical formula on base-z, N raw=physical signed residual',
                     'fit_split':'source_train_known_normal_only','fit_windows':int(fit.sum()),'fit_frames':count,'clip':'none','invalid':'reject entire window before selector and BatchNorm'})
    pd.DataFrame(rows).to_csv(a.output/'normalization_table.csv',index=False)
    result={'status':'PASS_M3_SOURCE_NORMALIZATION_AND_ALL13_FINITE_POPULATION','checked_at':datetime.now(timezone.utc).isoformat(),'inputs':inputs,
            'normalization_sha256':sha(a.output/'normalization.json'),'splits':splits,'common_case_and_pair_leakage':0,'source_split_identities':identity,
            'balanced_sampling':sampling,'sampler_weights_sha256':tensor_digest(weights),'sampler_generator_seed':53,'loader_generator_seed':143,
            'draws_per_epoch':len(trainres['y']),'gradient_steps_per_epoch':int(np.ceil(len(trainres['y'])/512)),
            'fault_directory_domain_counts':fault_counts,'fit_windows':int(fit.sum()),'fit_cases':t[fit].case_uid.nunique(),
            'N_statistics_unchanged':True,'fit_Val_Test_Real_used':False,'all13_data_coverage_shrunk':False,'training_or_GPU_server_used':False,
            'feature_cache_policy':'old9 computed on demand from base-z to avoid duplicate giant tensors; strict dataset code provided, full new old9 sidecar not required.'}
    save(a.output/'data_preflight.json',result)
    print('FROZEN_M3_SCALING',json.dumps({'mean':merged_mean,'std':merged_std,'windows':int(fit.sum()),'frames':count},ensure_ascii=False))


if __name__=='__main__':main()
