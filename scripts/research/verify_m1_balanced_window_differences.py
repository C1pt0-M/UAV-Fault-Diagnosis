"""Independently check paired Balanced diagnostics against saved source assets."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['analysis','features','results','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve independent analysis verification')
    manifest=read(a.analysis/'manifest.json')
    for name,item in manifest['files'].items():
        assert sha(a.analysis/name)==item['sha256']
    summary=read(a.analysis/'summary.json')
    for name,item in summary['inputs'].items():
        path=Path(item['path']) if name=='protocol' else Path(name)
        assert sha(path)==item['sha256'],name
    paired=pd.read_csv(a.analysis/'all_paired_predictions.csv',keep_default_na=False)
    changed=pd.read_csv(a.analysis/'changed_windows.csv',keep_default_na=False)
    assert len(paired)==32381 and paired.original_cache_index.is_unique
    table=paired.set_index('original_cache_index');source=[]
    for variant in ['M0_matched','M1']:
        path=a.results/'runs'/variant/'balanced_source_normal/source_validation_predictions.npz'
        with np.load(path,allow_pickle=False) as z:
            df=pd.DataFrame({'id':z['original_indices'],'label':z['labels'],'prediction':z['probabilities'].argmax(1),'score':z['probabilities'][:,1]}).set_index('id').loc[table.index]
        assert np.array_equal(df.label,table.qualified_y)
        # pandas writes original float32 probabilities with enough decimal
        # digits to recover their float32 values, not their float64 expansion.
        np.testing.assert_array_equal(df.score.to_numpy(),table[variant.replace('_matched','')+'_fault_probability'].to_numpy(dtype=np.float32))
        source.append(df)
    names={(1,1,0):'new_miss',(1,0,1):'recovered_fault',(1,0,0):'shared_miss',(1,1,1):'shared_TP',
           (0,1,0):'resolved_FP',(0,0,1):'new_FP',(0,1,1):'shared_FP',(0,0,0):'shared_TN'}
    expected=[names[(y,p0,p1)] for y,p0,p1 in zip(source[0].label,source[0].prediction,source[1].prediction)]
    assert expected==table.group.tolist()
    counts=pd.Series(expected).value_counts().to_dict()
    assert counts=={x['group']:x['windows'] for x in summary['groups']}
    assert counts['new_miss']==192 and counts['recovered_fault']==68 and counts['resolved_FP']==166 and counts['new_FP']==29
    assert set(changed.original_cache_index)==set(table[table.group.isin(['new_miss','recovered_fault','resolved_FP','new_FP'])].index)
    # Re-read continuous per-case residuals for every changed original window,
    # not just the selected examples, and compare exported physical summaries.
    max_residual_error=0.;checked_cases=0;checked_windows=0
    feature_manifest=read(a.features/'build_manifest.json')
    for uid,g in changed.groupby('case_uid'):
        name='cases/'+uid.replace('/','__')+'.npz';path=a.features/name
        assert sha(path)==feature_manifest['files'][name]['sha256']
        with np.load(path,allow_pickle=False) as z:
            values=z['residual'];valid=z['valid'];ids=z['original_cache_index'];starts=z['window_start_row'];ends=z['window_end_row']
            for _,row in g.iterrows():
                k=np.flatnonzero(ids==row.original_cache_index);assert len(k)==1;k=k.item()
                assert starts[k]==row.window_start_row and ends[k]==row.window_end_row and valid[starts[k]:ends[k]].all()
                r=values[starts[k]:ends[k]].astype(np.float32).astype(np.float64)
                for i in range(4):
                    for suffix,value in [('signed_median',np.median(r[:,i])),('abs_median',np.median(np.abs(r[:,i]))),('abs_p95',np.quantile(np.abs(r[:,i]),.95)),('std',np.std(r[:,i]))]:
                        error=abs(row[f'N{i+1}_{suffix}']-value);max_residual_error=max(error,max_residual_error);assert error<1e-12
                assert abs(np.median(np.linalg.norm(r[:,1:],axis=1))-row.attitude_norm_median_rad)<1e-12
                checked_windows+=1
        checked_cases+=1
    # Collapse adapter projection and class-margin head into a16D vector, then
    # compute the direct contribution without the model/adapter implementation.
    state=torch.load(a.results/'runs/M1/balanced_source_normal/best.pt',map_location='cpu',weights_only=True)
    export=torch.load(a.features/'cache/val.pt',map_location='cpu',weights_only=True,mmap=True)
    norm=read(a.features/'residual_norm_stats.json');mu=torch.tensor(norm['mean'],dtype=torch.float32).view(4,1);std=torch.tensor(norm['std'],dtype=torch.float32).view(4,1)
    class_margin=state['head.1.weight'][1]-state['head.1.weight'][0]
    collapsed_weight=class_margin@state['residual_adapter.6.weight'];bias=class_margin@state['residual_adapter.6.bias'];torch.set_num_threads(4)
    direct=[]
    with torch.inference_mode():
        for start in range(0,len(export['y']),512):
            r=(export['residual'][start:start+512]-mu)/std
            h=F.relu(F.conv1d(r,state['residual_adapter.0.weight'],state['residual_adapter.0.bias'],padding=1))
            h=F.relu(F.conv1d(h,state['residual_adapter.2.weight'],state['residual_adapter.2.bias'],padding=1)).mean(2)
            direct.extend((h@collapsed_weight+bias).tolist())
    direct=pd.Series(direct,index=export['original_cache_index'].numpy())
    joined=direct.loc[changed.original_cache_index].to_numpy();error=np.abs(joined-changed.M1_residual_margin.to_numpy());assert error.max()<1e-5
    assert np.array_equal(joined<0,changed.M1_residual_margin.to_numpy()<0)
    pivotal={}
    for name in ['new_miss','resolved_FP']:
        g=changed[changed.group==name];margin=g.M1_margin.to_numpy();direct_values=direct.loc[g.original_cache_index].to_numpy();common=margin-direct_values
        assert np.array_equal(common>0,g.M1_common_path_margin.to_numpy()>0)
        pivotal[name]=int(((common>0)&(margin<0)).sum())
    assert pivotal=={'new_miss':105,'resolved_FP':84}
    controls=pd.read_csv(a.analysis/'within_case_controls.csv')
    for _,pair in controls[controls.control_original_id.notna()].iterrows():
        target=table.loc[int(pair.target_original_id)];control=table.loc[int(pair.control_original_id)]
        assert target.case_uid==control.case_uid
        assert target.phase==control.phase
        assert control.group==('shared_TP' if target.group=='new_miss' else 'shared_FP')
    result={'status':'PASS_INDEPENDENT_PAIRED_BALANCED_WINDOW_DIFFERENCE_VERIFICATION','checked_at':datetime.now(timezone.utc).isoformat(),
            'paired_IDs':len(paired),'changed_windows_reconstructed_from_case_sidecar':checked_windows,'changed_case_sidecars':checked_cases,
            'physical_summary_max_error':max_residual_error,'collapsed16D_vs_original_adapter_max_changed_margin_error':float(error.max()),
            'direct_branch_pivotal_counts':pivotal,'group_counts':counts,'all_historical_input_and_output_SHA_verified':True,
            'adapter_output_constant_bias_margin':float(bias),
            'constant_bias_interpretation':'Contribution includes learned adapter output bias; split into bias and input-dependent16D term is algebraic parameterization, not independent causal importance.',
            'analysis_script_sha256':summary['analysis_source_sha256'],'verifier_script_sha256':sha(Path(__file__)),
            'training_threshold_or_model_asset_changes':False,'final_test_or_Real_read':False}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
