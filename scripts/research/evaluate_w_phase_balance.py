"""W train-CV evaluation. All-class constraints include late-hover faults."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research'))
import train_w_phase_balance as train
from train_w_phase_balance import s,t
import evaluate_t_selection as te
import evaluate_s_ablation as m
import evaluate_v_confirm as v
import numpy as np
import pandas as pd
import torch
ARMS=('W0','W1','W2')
BASELINE=ROOT/'artifacts/t_selection_mstca_v1/cv'

def adopt(decisions,ratios):
    passed=[a for a in ('W1','W2') if decisions[a]['verdict']=='PASS']
    if not passed:return 'KEEP_T0'
    if len(passed)==1:return 'ADOPT_'+passed[0]
    if ratios['W1']==ratios['W2']:
        raise ValueError('Protocol has no exact aggregate-ratio tie-break; STOP for clarification')
    return 'ADOPT_'+min(passed,key=lambda a:ratios[a])

def decide(point,uncertainty):
    result=v.criteria(list(point['per_seed_ratios'].values()),uncertainty['ratio_ci'],point['C2_delta_pp'],point['C3_drop_pp'],uncertainty['delta_ap_ci'])
    checks=dict(zip(('C1','C2','C3','C4'),result['criteria'].values()))
    return {'verdict':'PASS' if all(checks.values()) else 'FAIL','criteria':checks,
            'failed_criteria':[key for key,value in checks.items() if not value]}

def load_scores(frame,root,*,baseline_only=False):
    scores={a:{} for a in (('W0',) if baseline_only else ARMS)}
    epoch100={a:{} for a in ('W1','W2')}
    reports={};choices={};samplers={}
    for seed in train.SEEDS:
        for arm in scores:
            pooled=np.full(len(frame),np.nan);last=np.full(len(frame),np.nan)
            for fold in range(3):
                positions,_=s.split_ledger(frame,fold)
                idx=positions['heldout'];part=frame.iloc[idx]
                folder=BASELINE/f'seed{seed}_fold{fold}' if arm=='W0' else Path(root)/f'{arm}_seed{seed}_fold{fold}'
                report=s.read_json(folder/'report.json');selection=s.read_json(folder/'selections.json')
                if (report.get('status')!='COMPLETE' or report.get('phase')!='cv' or report.get('seed')!=seed or report.get('fold')!=fold
                    or report.get('epochs')!=100 or len(report.get('history',[]))!=100):
                    raise ValueError('Only complete matching 100-epoch CV run may enter W evaluation')
                t.require_formal_environment(report['environment'])
                if report['initial_digest']!=t.golden_digests()[('mstca',seed)]:raise ValueError('Run golden mismatch')
                replay=t.replay_t0([r['inner_class_balanced_ce'] for r in report['history']])
                if replay!=selection['T0']:raise ValueError('Selection differs from unweighted inner CE replay')
                if arm=='W0':
                    if report['protocol_sha256']!=t.PROTOCOL_SHA:raise ValueError('Wrong frozen T baseline')
                else:
                    if report['arm']!=arm or report['protocol_commit']!=train.FREEZE_COMMIT:raise ValueError('Wrong W arm/revision')
                    _,stats=train.phase_weights(frame.iloc[positions['fit']],arm)
                    if stats!=report['sampler']:raise ValueError('Sampler differs from FIT-only r1.1 calculation')
                    samplers[f'{arm}/fold{fold}']=stats
                    for epoch in {replay['epoch'],100}:
                        if not(folder/f'checkpoints/epoch_{epoch}.pt').is_file():raise ValueError('Missing selected/100 checkpoint')
                probabilities=te.load_prediction(folder/'heldout_predictions.npz',part.original_cache_index.to_numpy(dtype=np.int64),
                    part.qualified_y.to_numpy(dtype=np.int64),100)
                pooled[idx]=probabilities[replay['epoch']-1]
                last[idx]=probabilities[99]
                choices[f'{arm}/seed{seed}/fold{fold}']=replay
                reports[f'{arm}/seed{seed}/fold{fold}']={k:report[k] for k in ('status','epochs')}
                if arm!='W0':reports[f'{arm}/seed{seed}/fold{fold}']['wall_time_s']=report['wall_time_s']
            if not np.isfinite(pooled).all() or not np.isfinite(last).all():raise ValueError('Incomplete pooled population')
            scores[arm][seed]=pooled
            if arm!='W0':epoch100[arm][seed]=last
    return scores,epoch100,reports,choices,samplers

def w0_gate(frame,scores):
    actual={str(seed):m.fp_at_recall(frame.qualified_y.to_numpy(),scores['W0'][seed])['fp'] for seed in (42,43,44)}
    if actual!={'42':3595,'43':3588,'44':4610}:raise RuntimeError('W0 FP reproduction mismatch: '+str(actual)+' STOP')
    return {'status':'PASS','fp_r97':actual}

def prepare(frame,scores):
    y,_,classes,_,late=m.strata(frame)
    if set(scores)!=set(ARMS) or any(set(scores[a])!=set(train.SEEDS) for a in ARMS):raise ValueError('Need all 3 arms and 5 seeds')
    masks={name:(y==1)&(classes==name) for name in sorted(set(classes[y==1]))}
    return {'y':y,'scores':scores,'cache':{(a,sd):m.MetricCache(y,scores[a][sd]) for a in ARMS for sd in train.SEEDS},
        'masks':masks,'outside_masks':{name:mask&~late for name,mask in masks.items()}}

def measure(prepared,weights=None,*,outside=False):
    y=prepared['y'];w=np.ones(len(y),dtype=np.int64) if weights is None else weights
    basic={key:cache.basic(w) for key,cache in prepared['cache'].items()}
    result={}
    for arm in ('W1','W2'):
        ratios=[];deltas=[];matched={};c2={};c3={};per_c2={};per_c3={}
        for seed in train.SEEDS:
            a,b=basic[(arm,seed)],basic[('W0',seed)]
            ratios.append(m.ratio(a['fp'],b['fp']))
            deltas.append(a['ap']-b['ap'] if a['ap'] is not None and b['ap'] is not None else None)
            cache=prepared['cache'][(arm,seed)]
            matched[str(seed)]=v.matched_fp_threshold(y,cache.score,b['fp'],w,normal_order=cache.order)
        masks=prepared['outside_masks'] if outside else prepared['masks']
        for cls,mask in masks.items():
            mass=int(w[mask].sum());changes=[];drops=[];per_c2[cls]={};per_c3[cls]={}
            for seed in train.SEEDS:
                p0,p1=prepared['scores']['W0'][seed],prepared['scores'][arm][seed]
                t0,t1=basic[('W0',seed)]['threshold'],basic[(arm,seed)]['threshold'];tm=matched[str(seed)]['threshold']
                n0=int(w[mask&(p0>=t0)].sum()) if t0 is not None else None
                n1=int(w[mask&(p1>=t1)].sum()) if t1 is not None else None
                nm=int(w[mask&(p1>=tm)].sum()) if tm is not None else None
                change=nm-n0 if nm is not None and n0 is not None else None
                drop=n0-n1 if n0 is not None and n1 is not None else None
                changes.append(change);drops.append(drop)
                per_c2[cls][str(seed)]=100*change/mass if mass and change is not None else None
                per_c3[cls][str(seed)]=100*drop/mass if mass and drop is not None else None
            c2[cls]=100*sum(changes)/(5*mass) if mass and all(n is not None for n in changes) else None
            c3[cls]=100*sum(drops)/(5*mass) if mass and all(n is not None for n in drops) else None
        result[arm]={'per_seed_ratios':dict(zip(map(str,train.SEEDS),ratios)),'aggregate_ratio':m.geometric_mean(ratios),
            'per_seed_delta_ap':dict(zip(map(str,train.SEEDS),deltas)),
            'mean_delta_ap':float(np.mean(deltas)) if all(d is not None for d in deltas) else None,
            'C2_delta_pp':c2,'C3_drop_pp':c3,'C2_per_seed_delta_pp':per_c2,'C3_per_seed_drop_pp':per_c3,'matched_fp':matched}
    between=[m.ratio(basic[('W1',sd)]['fp'],basic[('W2',sd)]['fp']) for sd in train.SEEDS]
    result['W1_vs_W2']={'aggregate_ratio':m.geometric_mean(between)}
    return result

def bootstrap(prepared,groups,replicates=2000,seed=20261106):
    groups=np.asarray(groups)
    if len(groups)!=len(prepared['y']) or any(not isinstance(g,str) or not g for g in groups):raise ValueError('Invalid bootstrap group IDs')
    unique,inverse=np.unique(groups,return_inverse=True);rng=np.random.default_rng(seed)
    samples={arm:{'ratio':[],'ap':[],'C2':{c:[] for c in prepared['masks']},'C3':{c:[] for c in prepared['masks']},
        'per_seed_ratio':{str(sd):[] for sd in train.SEEDS},'per_seed_ap':{str(sd):[] for sd in train.SEEDS}} for arm in ('W1','W2')}
    between=[]
    for _ in range(replicates):
        counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        point=measure(prepared,counts[inverse])  # One weights object for all 15 caches; t* recalculated.
        between.append(point['W1_vs_W2']['aggregate_ratio'])
        for arm,store in samples.items():
            value=point[arm];store['ratio'].append(value['aggregate_ratio']);store['ap'].append(value['mean_delta_ap'])
            for c in store['C2']:
                store['C2'][c].append(value['C2_delta_pp'][c]);store['C3'][c].append(value['C3_drop_pp'][c])
            for sd in map(str,train.SEEDS):
                store['per_seed_ratio'][sd].append(value['per_seed_ratios'][sd]);store['per_seed_ap'][sd].append(value['per_seed_delta_ap'][sd])
    results={}
    for arm,store in samples.items():
        results[arm]={'ratio_ci':m.percentile(store['ratio']),'delta_ap_ci':m.percentile(store['ap']),
            'C2_ci_pp':{c:m.percentile(x) for c,x in store['C2'].items()},'C3_ci_pp':{c:m.percentile(x) for c,x in store['C3'].items()},
            'per_seed_ratio_ci':{sd:m.percentile(x) for sd,x in store['per_seed_ratio'].items()},
            'per_seed_delta_ap_ci':{sd:m.percentile(x) for sd,x in store['per_seed_ap'].items()},
            'undefined_ratio_replicates':sum(x is None for x in store['ratio']),
            'undefined_ap_replicates':sum(x is None for x in store['ap']),
            'undefined_C2':{c:sum(v is None for v in x) for c,x in store['C2'].items()},
            'undefined_C3':{c:sum(v is None for v in x) for c,x in store['C3'].items()}}
    return {'replicates':replicates,'rng_seed':seed,'groups':len(unique),'shared_all_arms_and_seeds':True,
        'comparisons':results,'W1_vs_W2_ratio_ci':m.percentile(between),'undefined_policy':'No pseudocount, drop or redraw'}

def normal_diagnostics(frame,scores,metrics):
    y,flight,_,independent,_=m.strata(frame);end=frame.window_end_row.to_numpy()*.02
    timebin=pd.cut(end,[0,10,20,30,45,60,90,np.inf],labels=['[0,10]','(10,20]','(20,30]','(30,45]','(45,60]','(60,90]','(90,inf]'],include_lowest=True)
    phase=np.empty(len(frame),dtype=object);refs={}
    for fold in range(3):
        idx,_=s.split_ledger(frame,fold);refs[str(fold)]=train.onset_references(frame.iloc[idx['fit']])
        _,phase[idx['heldout']]=train.cells(frame.iloc[idx['heldout']],refs[str(fold)])
    result={}
    for arm in ARMS:
        alarms={sd:scores[arm][sd]>=metrics[arm][str(sd)]['primary']['threshold'] for sd in train.SEEDS}
        def describe(mask):
            n=int(mask.sum());fp={str(sd):int(a[mask].sum()) for sd,a in alarms.items()}
            return {'windows':n,'per_seed_fp':fp,'mean_fp':float(np.mean(list(fp.values()))),
                    'mean_fp_rate':float(np.mean(list(fp.values())))/n if n else None}
        bins={f+'/'+str(b):describe(independent&(flight==f)&(np.asarray(timebin)==b)) for f in train.FLIGHTS for b in timebin.categories}
        cells={f+'/'+p:describe(independent&(flight==f)&(phase==p)) for f in train.FLIGHTS for p in ('pre','post')}
        counts={str(sd):int(frame.loc[independent&a,'case_uid'].nunique()) for sd,a in alarms.items()}
        result[arm]={'flight_time_bin':bins,'flight_phase':cells,
            'prefault_normal':describe(frame.normal_origin.eq('normal_part_of_fault_case').to_numpy()),
            'independent_normal_flights':int(frame.loc[independent,'case_uid'].nunique()),
            'alarmed_flights_per_seed':counts,'alarmed_flights_seed_mean':float(np.mean(list(counts.values())))}
    return {'arms':result,'phase_onset_references_by_outer_fold':refs,'time_bin_rule':'right-closed; first includes zero; empty cell rate=null',
        'phase_rule':'each held-out window uses only its outer-fold FIT onset references',
        'prefault_rule':'frozen normal_origin=normal_part_of_fault_case (includes no-onset fault cases)'}

def evaluate(args):
    output=Path(args.output)
    if output.exists():raise FileExistsError('Fresh W evaluation/gate output required')
    t.require_formal_environment();torch.set_num_threads(4)
    identities=t.verify_data(args.data,local=True)
    frame=s.read_ledger(args.data);s.fold_sizes(frame,s.read_json(s.DEFAULT_PROTOCOL))
    if len(frame)!=162069:raise ValueError('Wrong train-CV population')
    scores,last,reports,choices,samplers=load_scores(frame,args.runs,baseline_only=args.w0_gate)
    gate=w0_gate(frame,scores)
    output.mkdir(parents=True,exist_ok=False)
    if args.w0_gate:
        s.write_json(output/'w0_gate.json',gate);return gate
    try:
        prepared=prepare(frame,scores);point=measure(prepared);ci=bootstrap(prepared,frame.association_group.to_numpy())
        decisions={a:decide(point[a],ci['comparisons'][a]) for a in ('W1','W2')}
        adoption=adopt(decisions,{a:point[a]['aggregate_ratio'] for a in ('W1','W2')})
        metrics={a:{str(sd):te.score_metrics(frame,scores[a][sd]) for sd in train.SEEDS} for a in ARMS}
        all_masks=prepared['masks']
        for a in ARMS:
            for sd in train.SEEDS:
                threshold=metrics[a][str(sd)]['primary']['threshold']
                metrics[a][str(sd)]['class_recall_all_fault']={c:float((scores[a][sd][mask]>=threshold).mean()) for c,mask in all_masks.items()}
        diag=normal_diagnostics(frame,scores,metrics)
        if any(diag['arms'][a]['independent_normal_flights']!=334 for a in ARMS):raise ValueError('Independent-normal flight population mismatch')
        stability={a:{'ddof0':float(np.std(np.log([metrics[a][str(sd)]['primary']['fp'] for sd in train.SEEDS]),ddof=0)),
                      'ddof1':float(np.std(np.log([metrics[a][str(sd)]['primary']['fp'] for sd in train.SEEDS]),ddof=1))}
                   if all(metrics[a][str(sd)]['primary']['fp']>0 for sd in train.SEEDS) else {'ddof0':None,'ddof1':None} for a in ARMS}
        result={'status':'COMPLETE','protocol_commit':train.FREEZE_COMMIT,'population':len(frame),'W0_gate':gate,
            'data_identities':identities,'decisions':decisions,'adoption':adoption,'comparisons':point,'uncertainty':ci,
            'metrics':metrics,'normal_diagnostics':diag,'selected_epochs':choices,'samplers':samplers,'seed_log_fp_sd':stability,
            'V_style_excluding_late_hover_fault_descriptive':measure(prepared,outside=True),
            'epoch100_fp_r97_descriptive':{a:{str(sd):m.fp_at_recall(frame.qualified_y.to_numpy(),last[a][sd])['fp'] for sd in train.SEEDS} for a in ('W1','W2')},
            'run_reports':reports,'disclosure':s.read_json(train.PROTOCOL)['confirmation_disclosure']}
        s.write_json(output/'evaluation.json',result)
        return result
    except Exception as exc:
        s.write_json(output/'FAILED.json',{'status':'FAILED','error':repr(exc)});raise

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=train.DATA);p.add_argument('--runs',type=Path,default=train.OUTPUT/'cv')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--w0-gate',action='store_true');return p
if __name__=='__main__':
    args=parser().parse_args();train.train_only_guard(args.data);r=evaluate(args);print(r.get('adoption',r['status']))
