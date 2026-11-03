"""Sensitivity to proposed recall tolerances from saved source-Val scores.

No classifier inference, new thresholds or acceptance choice. Every proposed
margin stays a labelled scenario; actual M3 acceptance parameters remain null.
"""
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_FLOOR
import argparse,json,sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
from uav_fault.data.causal_cache import sha

SUBTYPES=['GPS','accelerometer','barometer','gyroscope','magnetometer','motor']


def allowed_extra_misses(windows,delta_pp):
    if windows<=0 or delta_pp is None or not np.isfinite(delta_pp) or delta_pp<0:raise ValueError('Explicit nonnegative percentage-point margin and positive population required')
    return int((Decimal(str(delta_pp))*Decimal(int(windows))/Decimal(100)).to_integral_value(rounding=ROUND_FLOOR))


def recall_gate(losses_pp,overall_delta_pp,per_fault_delta_pp):
    if overall_delta_pp is None or set(per_fault_delta_pp)!=set(SUBTYPES) or any(v is None for v in per_fault_delta_pp.values()):
        return {'status':'PENDING_USER_RECALL_LIMITS','pass':None}
    if set(losses_pp)!=set(['overall']+SUBTYPES) or any(not np.isfinite(v) for v in losses_pp.values()):
        raise ValueError('All overall/six-subtype recall differences required')
    if any(not np.isfinite(v) or v<0 for v in [overall_delta_pp]+list(per_fault_delta_pp.values())):
        raise ValueError('Recall losses need finite nonnegative margins in percentage points')
    failed=[]
    if losses_pp['overall']>overall_delta_pp:failed.append('overall')
    failed.extend(k for k in SUBTYPES if losses_pp[k]>per_fault_delta_pp[k])
    return {'status':'FAIL' if failed else 'PASS','pass':not failed,'failed_recall_scopes':failed}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['data','features','results','original-m0','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve recall sensitivity evidence')
    source=pd.read_csv(a.data/'ledgers/val_included_windows.csv',keep_default_na=False,low_memory=False)
    export=torch.load(a.features/'cache/val.pt',map_location='cpu',weights_only=True,mmap=True);f=source.iloc[export['base_compact_index'].numpy()].reset_index(drop=True)
    assert len(f)==32381 and np.array_equal(f.original_cache_index,export['original_cache_index'])
    labels=f.qualified_y.to_numpy(dtype=np.int64);directory=f.case_uid.str.split('/').str[2].to_numpy();domains=f.domain.to_numpy()
    inputs={};predictions={};full_predictions={};ties={}
    manifest=json.loads((a.results/'manifest.json').read_text(encoding='utf-8'))
    original_manifest=json.loads((a.original_m0/'verified_m0_results_manifest.json').read_text(encoding='utf-8'))
    for arm in ['balanced_source_normal','uniform_replacement']:
        suffix='' if arm=='balanced_source_normal' else '_Uniform'
        for variant in ['M0_matched','M1']:
            rel='runs/'+variant+'/'+arm+'/source_validation_predictions.npz';path=a.results/rel
            if sha(path)!=manifest['files'][rel]['sha256']:raise ValueError('Saved formal prediction identity changed')
            inputs[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as z:
                assert np.array_equal(z['original_indices'],f.original_cache_index) and np.array_equal(z['labels'],labels)
                predictions[variant+suffix]=z['probabilities'].argmax(1)
                ties[variant+suffix]=int((z['probabilities'][:,0]==z['probabilities'][:,1]).sum())
        relative='runs/'+arm+'/source_validation_predictions.npz';original=a.original_m0/relative
        if sha(original)!=original_manifest['files'][relative]['sha256']:raise ValueError('Original saved M0 identity changed')
        inputs[str(original)]=sha(original)
        with np.load(original,allow_pickle=False) as z:
            assert np.array_equal(z['original_indices'],source.original_cache_index) and np.array_equal(z['labels'],source.qualified_y)
            predictions['original_M0_saved_on_common_IDs'+suffix]=z['probabilities'][export['base_compact_index'].numpy()].argmax(1)
            full_predictions['original_M0_full_Val'+suffix]=z['probabilities'].argmax(1)
            ties['original_M0_full_Val'+suffix]=int((z['probabilities'][:,0]==z['probabilities'][:,1]).sum())
    rows=[]
    for model,pred in dict(predictions,**full_predictions).items():
        selected=source if model in full_predictions else f
        lab=selected.qualified_y.to_numpy(dtype=np.int64);d=selected.domain.to_numpy();sub=selected.case_uid.str.split('/').str[2].to_numpy()
        for domain in ['all','SIL','HIL']:
            dm=np.ones(len(selected),dtype=bool) if domain=='all' else d==domain
            for subtype in ['overall']+SUBTYPES:
                mask=(lab==1)&dm
                if subtype!='overall':mask&=sub==subtype
                n=int(mask.sum());fn=int((pred[mask]==0).sum());rows.append({'model':model,'domain':domain,'scope':subtype,'fault_windows':n,'missed_windows':fn,'recall':(n-fn)/n})
    metrics=pd.DataFrame(rows);budget=[]
    candidates=[0.,.1,.25,.5,1.,2.,5.]
    for _,row in metrics[(metrics.model=='M0_matched')&(metrics.domain=='all')].iterrows():
        for delta in candidates:
            budget.append({'scope':row.scope,'fault_windows':row.fault_windows,'delta_percentage_points':delta,'allowed_net_additional_missed_windows':allowed_extra_misses(row.fault_windows,delta),'historical_matched_M0_recall':row.recall})
    total=metrics[(metrics.domain=='all')].pivot(index='scope',columns='model',values='recall');losses={str(k):float(100*(row.M0_matched-row.M1)) for k,row in total.iterrows()}
    scenarios=[]
    for delta in candidates:
        gate=recall_gate(losses,delta,{k:delta for k in SUBTYPES});scenarios.append({'scenario':'same_delta_all_scopes_'+str(delta)+'pp','overall_delta_pp':delta,'per_fault_delta_pp':{k:delta for k in SUBTYPES},'historical_M1_vs_matched_M0':gate,'adopted':False})
    for label,overall,per in [('strict_recall_preservation',0.,{k:0. for k in SUBTYPES}),
                             ('suggested_small_exploratory_loss_only',.25,{k:(.1 if k=='motor' else .5) for k in SUBTYPES})]:
        scenarios.append({'scenario':label,'overall_delta_pp':overall,'per_fault_delta_pp':per,'historical_M1_vs_matched_M0':recall_gate(losses,overall,per),'adopted':False})
    a.output.mkdir(parents=True,exist_ok=False);metrics.to_csv(a.output/'saved_recall_context.csv',index=False);pd.DataFrame(budget).to_csv(a.output/'recall_tolerance_window_budgets.csv',index=False)
    result={'status':'COMPLETE_RECALL_TOLERANCE_SCENARIOS_NO_LIMIT_ADOPTED','created_at':datetime.now(timezone.utc).isoformat(),'inputs':inputs,'classification_threshold':.5,'threshold_tie_rule':'Historical argmax: exact equal logits/probabilities tie to normal, no threshold search','saved_equal_probability_counts':ties,
            'historical_M1_recall_loss_pp_vs_matched_M0':losses,'scenarios':scenarios,'official_M3_recall_delta':None,'official_M3_per_fault_delta':{k:None for k in SUBTYPES},
            'relative_reference':'New M3 C0 best selected at fixedMacroF1, same common151833/32381 population, untrained shared start. Old trained matchedM0 illustrates margins only.',
            'strict_recommendation':'If research claim says maintain fault detection, start with0pp overall and each subtype. This is point-estimate preservation, not statistical noninferiority.',
            'optional_exploratory_recommendation':'Only if user explicitly accepts bounded loss: illustrative overall0.25pp, six sensors/types0.5pp except motor0.1pp. It is a research tradeoff, not safety/clinical standard and not chosen to admit M1.',
            'user_must_confirm':['Overall allowed absolute percentage-point loss','Each six subtype loss, may be0 or distinct','Whether per-domain recalls are hard gates or stratified warnings','Optional absolute per-type recall floors; current magnetometer reference already low','Recall gate as post-selection acceptance, not checkpoint reselection'],
            'no_threshold_search_or_model_training':True,'no_final_test_Real_or_server_GPU':True}
    (a.output/'recall_sensitivity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
