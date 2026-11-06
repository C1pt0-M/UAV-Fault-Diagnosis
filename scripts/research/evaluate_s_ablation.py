"""Evaluate complete S-ablation runs; incomplete/synthetic inputs never yield real claims.

Cluster bootstrap uses shared group multiplicities (equivalent to repeating
sampled clusters), reusing score sorts, recomputing t* for every arm/seed/draw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
import numpy as np
import pandas as pd
import train_s_ablation as train

ARMS=('S0','S1','S2','S3')
SEEDS=(42,43,44)
BOOTSTRAP_REPLICATES=2000
BOOTSTRAP_SEED=20261106


def finite(value):
    value=float(value)
    return value if math.isfinite(value) else None


def ratio(a,b):
    return None if a is None or b is None or b==0 else finite(a/b)


def geometric_mean(values):
    if any(v is None or v<0 for v in values):
        return None
    return 0. if any(v==0 for v in values) else finite(np.exp(np.log(values).mean()))


class MetricCache:
    def __init__(self,y,score):
        self.y=np.asarray(y,dtype=np.int64)
        self.score=np.asarray(score,dtype=np.float64)
        if self.y.shape!=self.score.shape or not np.isin(self.y,[0,1]).all() or not np.isfinite(self.score).all() or ((self.score<0)|(self.score>1)).any():
            raise ValueError('Invalid binary labels/probabilities')
        self.order=np.argsort(-self.score,kind='stable')
        self.fault_order=self.order[self.y[self.order]==1]
        ordered=self.score[self.order]
        self.ends=np.r_[np.flatnonzero(ordered[:-1]!=ordered[1:]),len(ordered)-1]

    def basic(self,weights=None,recall=.97):
        w=np.ones(len(self.y),dtype=np.int64) if weights is None else np.asarray(weights)
        if w.shape!=self.y.shape or (w<0).any():
            raise ValueError('Invalid cluster multiplicities')
        fault_mass=w[self.fault_order].sum()
        if fault_mass==0:
            return {'threshold':None,'fp':None,'ap':None}
        cumulative=np.cumsum(w[self.fault_order],dtype=np.float64)
        index=int(np.searchsorted(cumulative,recall*fault_mass,side='left'))
        threshold=float(self.score[self.fault_order[index]])
        fp=int(w[(self.y==0)&(self.score>=threshold)].sum())
        ordered_w=w[self.order]
        tp=np.cumsum(ordered_w*self.y[self.order],dtype=np.float64)[self.ends]
        total=np.cumsum(ordered_w,dtype=np.float64)[self.ends]
        precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        ap=float(np.sum(np.diff(np.r_[0.,tp/fault_mass])*precision))
        return {'threshold':threshold,'fp':fp,'ap':ap}


def fp_at_recall(y,score,recall=.97):
    if not 0<recall<=1:
        raise ValueError('Recall must be in (0,1]')
    return MetricCache(y,score).basic(recall=recall)


def strata(frame):
    parts=frame.case_uid.str.split('/')
    if not parts.map(len).ge(3).all():
        raise ValueError('Malformed case_uid')
    flight=parts.str[1].to_numpy()
    classes=parts.str[2].to_numpy()
    y=frame.qualified_y.to_numpy(dtype=np.int64)
    independent=(y==0)&frame.normal_origin.eq('normal_flight').to_numpy()
    late=(flight=='hover')&(frame.window_end_row.to_numpy()*.02>30)
    return y,flight,classes,independent,late


def score_metrics(frame,score):
    y,flight,classes,independent,late=strata(frame)
    score=np.asarray(score,dtype=np.float64)
    primary=fp_at_recall(y,score)
    r95=fp_at_recall(y,score,.95)
    fixed=score>.5
    matched=score>=primary['threshold'] if primary['threshold'] is not None else np.zeros(len(y),bool)
    def recall(mask,pred):
        return float(pred[mask].mean()) if mask.any() else None
    outside=(y==1)&~late
    fault_classes=sorted(set(classes[y==1]))
    p_y=np.where(y==1,score,1-score)
    with np.errstate(divide='ignore'):
        losses=-np.log(p_y)
    ce=finite(.5*(losses[y==0].mean()+losses[y==1].mean())) if (y==0).any() and (y==1).any() else None
    return {'primary':primary,'ap':primary['ap'],'fp_at_r95':r95,
        'fp_r97_outside_late_hover_normal':int(matched[(y==0)&~(independent&late)].sum()),
        'fixed_p_gt_half':{'all_normal_fp':int(fixed[y==0].sum()),'fault_tp':int(fixed[y==1].sum()),
            'normal_n':int((y==0).sum()),'fault_n':int((y==1).sum()),
            'late_hover_normal_fp':{d:int(fixed[independent&late&frame.domain.eq(d).to_numpy()].sum()) for d in ('SIL','HIL')},
            'independent_nonhover_fp':int(fixed[independent&(flight!='hover')].sum()),
            'class_recall_outside_late_hover_fault':{c:recall(outside&(classes==c),fixed) for c in fault_classes},
            'late_hover_fault_recall':{c:recall((y==1)&late&(classes==c),fixed) for c in ('GPS','barometer')}},
        'matched_class_recall_outside_late_hover_fault':{c:recall(outside&(classes==c),matched) for c in fault_classes},
        'class_balanced_ce':ce,'brier_binary':float(np.mean((score-y)**2))}


def percentile(values):
    if any(v is None or not math.isfinite(v) for v in values):
        return None
    return np.percentile(values,[2.5,97.5]).tolist()


def paired_bootstrap(y,scores,groups,replicates=BOOTSTRAP_REPLICATES,seed=BOOTSTRAP_SEED):
    groups=np.asarray(groups,dtype=str)
    unique,inverse=np.unique(groups,return_inverse=True)
    if len(groups)!=len(y) or not len(unique):
        raise ValueError('Bootstrap groups/population mismatch')
    cache={(a,s):MetricCache(y,scores[a][s]) for a in ARMS for s in SEEDS}
    values={a:{'ratio':{str(s):[] for s in SEEDS},'delta_ap':{str(s):[] for s in SEEDS},
               'aggregate_ratio':[],'aggregate_delta_ap':[]} for a in ARMS[1:]}
    rng=np.random.default_rng(seed)
    plan_hash=hashlib.sha256()
    for _ in range(replicates):
        counts=np.bincount(rng.integers(0,len(unique),size=len(unique)),minlength=len(unique))
        plan_hash.update(counts.astype('<i8').tobytes())
        weights=counts[inverse]  # SAME object/replicate for all 12 score vectors.
        metrics={key:item.basic(weights) for key,item in cache.items()}
        for arm in ARMS[1:]:
            ratios=[]; deltas=[]
            for s in SEEDS:
                base,current=metrics[('S0',s)],metrics[(arm,s)]
                r=ratio(current['fp'],base['fp'])
                d=finite(current['ap']-base['ap']) if current['ap'] is not None and base['ap'] is not None else None
                values[arm]['ratio'][str(s)].append(r)
                values[arm]['delta_ap'][str(s)].append(d)
                ratios.append(r);deltas.append(d)
            values[arm]['aggregate_ratio'].append(geometric_mean(ratios))
            values[arm]['aggregate_delta_ap'].append(float(np.mean(deltas)) if all(d is not None for d in deltas) else None)
    result={}
    for arm,v in values.items():
        result[arm]={'per_seed_ratio_ci':{s:percentile(x) for s,x in v['ratio'].items()},
            'per_seed_delta_ap_ci':{s:percentile(x) for s,x in v['delta_ap'].items()},
            'aggregate_ratio_ci':percentile(v['aggregate_ratio']),
            'aggregate_delta_ap_ci':percentile(v['aggregate_delta_ap']),
            'undefined_aggregate_ratio_replicates':sum(x is None for x in v['aggregate_ratio']),
            'undefined_aggregate_ap_replicates':sum(x is None for x in v['aggregate_delta_ap'])}
    return {'replicates':replicates,'rng_seed':seed,'groups':len(unique),
            'shared_resample_plan_sha256':plan_hash.hexdigest(),'paired_across_all_arms_and_seeds':True,
            'ap_seed_aggregate':'arithmetic mean of paired seed AP differences',
            'undefined_policy':'No pseudocount, redraw or dropped replicate; an undefined replicate makes its CI unavailable',
            'comparisons':result}


def decide(arm,ratios,ratio_ci,ap_ci,class_drops_pp):
    if arm not in ARMS[1:]:
        raise ValueError('Only S1/S2/S3 comparisons have verdicts')
    result={'verdict':'INCONCLUSIVE','effect_below_margin':False,'reason':None}
    if any(r is None for r in ratios) or max(sum(r<1 for r in ratios),sum(r>1 for r in ratios))<2:
        result['reason']='seed_consistency'
        return result
    if not class_drops_pp or any(v is None for v in class_drops_pp.values()):
        result['reason']='class_constraint_unestimable'
        return result
    if max(class_drops_pp.values())>1.0:
        result.update(verdict='TRADE_OFF',reason='class_constraint')
    elif ratio_ci is None:
        result['reason']='ratio_ci_unestimable'
    elif ratio_ci[0]>1:
        result.update(verdict='MSTCA_BETTER' if arm=='S3' else 'COMPONENT_HELPS',
                      effect_below_margin=ratio_ci[1]<1.10,reason='lower_bound_gt_one')
    elif ratio_ci[1]<1.10:
        result.update(verdict='NO_ADVANTAGE' if arm=='S3' else 'REMOVABLE',reason='upper_bound_lt_margin')
    else:
        result['reason']='ratio_ci_crosses_margin'
    if result['verdict']!='INCONCLUSIVE':
        if ap_ci is None:
            result.update(pre_ap_verdict=result['verdict'],verdict='INCONCLUSIVE',reason='ap_ci_unestimable')
        else:
            worse=result['verdict'] in ('COMPONENT_HELPS','MSTCA_BETTER')
            if result['verdict']=='TRADE_OFF':
                worse=geometric_mean(ratios)>1
            opposed=ap_ci[0]>0 if worse else ap_ci[1]<0
            if opposed:
                result.update(pre_ap_verdict=result['verdict'],verdict='INCONCLUSIVE',reason='AP_consistency')
    return result


def comparison_results(metrics,uncertainty):
    points={}; decisions={}
    for arm in ARMS[1:]:
        ratios=[ratio(metrics[arm][str(s)]['primary']['fp'],metrics['S0'][str(s)]['primary']['fp']) for s in SEEDS]
        deltas=[metrics[arm][str(s)]['ap']-metrics['S0'][str(s)]['ap'] for s in SEEDS]
        classes=metrics['S0']['42']['matched_class_recall_outside_late_hover_fault']
        drops={}
        for c in classes:
            b=[metrics['S0'][str(s)]['matched_class_recall_outside_late_hover_fault'][c] for s in SEEDS]
            a=[metrics[arm][str(s)]['matched_class_recall_outside_late_hover_fault'][c] for s in SEEDS]
            drops[c]=100*float(np.mean(np.subtract(b,a))) if all(x is not None for x in b+a) else None
        u=uncertainty['comparisons'][arm]
        points[arm]={'per_seed_ratios':dict(zip(map(str,SEEDS),ratios)),
            'aggregate_ratio':geometric_mean(ratios),'per_seed_delta_ap':dict(zip(map(str,SEEDS),deltas)),
            'aggregate_delta_ap':float(np.mean(deltas)),'class_recall_drop_seed_mean_pp':drops,**u}
        decisions[arm]=decide(arm,ratios,u['aggregate_ratio_ci'],u['aggregate_delta_ap_ci'],drops)
    return points,decisions


def load_runs(root,frame,phase,synthetic_demo=False):
    missing=[]; reports={}; arrays={a:{} for a in ARMS}; last={a:{} for a in ARMS}
    _,outer=train.split_ledger(frame,0)
    gold=train.golden_digests()
    implementations=set()
    for arm in ARMS:
        for seed in SEEDS:
            best_pool=np.full(len(frame),np.nan);last_pool=np.full(len(frame),np.nan)
            for fold in (range(3) if phase=='cv' else [0]):
                directory=Path(root)/f'{arm}_seed{seed}_fold{fold}'
                report_path=directory/'report.json'
                if not report_path.exists():
                    missing.append(str(directory));continue
                r=train.read_json(report_path)
                allowed=('COMPLETE','COMPLETE_CPU_SMOKE_NOT_FORMAL','COMPLETE_SYNTHETIC_DEMO') if synthetic_demo else ('COMPLETE',)
                if r.get('status') not in allowed:
                    missing.append(str(directory)+': not eligible COMPLETE');continue
                if r.get('phase')!=phase or r.get('arm')!=arm or r.get('seed')!=seed or r.get('fold')!=fold or r.get('protocol_sha256')!=train.PROTOCOL_SHA:
                    raise ValueError('Run identity/protocol mismatch: '+str(directory))
                if not synthetic_demo and (r.get('cpu_smoke') or r.get('synthetic_demo') or r.get('determinism_probe')):
                    raise ValueError('Non-formal run mixed into real evaluation')
                variant={'S0':'mstca','S1':'mstca_no_se','S2':'mstca_single_k7','S3':'cnn1d_matched'}[arm]
                if r['initial_digest']!=gold[(variant,seed)]:
                    raise ValueError('Run initialization differs from pinned golden')
                if not synthetic_demo:
                    train.require_formal_environment(r['environment'])
                    if r['implementation_files']!=train.implementation_manifest()['files']:
                        raise ValueError('Evaluation implementation differs from the phase-1 pinned files')
                    expected_data=train.read_json(train.BASELINE)['data_identities']
                    if r['data_identities']['matched_sha256']!=expected_data or r['data_identities']['skipped_sha256']:
                        raise ValueError('Formal run must verify all nine data identities')
                    implementations.add(r['implementation_manifest_sha256'])
                idx=np.flatnonzero(outer==fold) if phase=='cv' else np.arange(len(frame))
                expected_ids=frame.original_cache_index.to_numpy()[idx]
                for which,pool in [('best',best_pool),('last',last_pool)]:
                    path=directory/f'heldout_predictions_{which}.npz'
                    if not path.exists():
                        missing.append(str(path));continue
                    if train.sha(path)!=r['prediction_sha256'][which]:
                        raise ValueError('Prediction artifact hash mismatch')
                    with np.load(path,allow_pickle=False) as z:
                        ids=z['original_cache_index'];probs=z['probabilities']
                    if ids.dtype!=np.int64 or len(np.unique(ids))!=len(ids) or not np.array_equal(ids,expected_ids):
                        raise ValueError('Held-out IDs must exactly match the fold population/order')
                    if probs.shape!=(len(idx),2) or not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any() or not np.allclose(probs.sum(1),1,atol=1e-7):
                        raise ValueError('Invalid held-out probability matrix')
                    pool[idx]=probs[:,1]
                reports[str(directory)]=r
            arrays[arm][seed]=best_pool;last[arm][seed]=last_pool
    if len(implementations)>1:
        raise ValueError('Implementation manifest changed across runs')
    if missing:
        return None,None,reports,missing
    if any(not np.isfinite(x).all() for data in (arrays,last) for seeds in data.values() for x in seeds.values()):
        raise ValueError('Incomplete pooled predictions')
    return arrays,last,reports,[]


def historical_sanity(frame,scores,roots):
    result={}
    for seed,root in roots.items():
        if root is None:
            result[str(seed)]={'status':'NOT_SUPPLIED_NOT_CHECKED'};continue
        mapping={int(v):i for i,v in enumerate(frame.original_cache_index)}
        old=np.full(len(frame),np.nan)
        for fold in range(3):
            with np.load(Path(root)/f'fold{fold}'/'heldout_predictions.npz',allow_pickle=False) as z:
                ids=z['original_indices'];p=z['probabilities'][:,1]
            if any(int(i) not in mapping for i in ids):
                raise ValueError('Historical sanity population differs')
            old[[mapping[int(i)] for i in ids]]=p
        if not np.isfinite(old).all():
            raise ValueError('Incomplete historical source-train predictions')
        y=frame.qualified_y.to_numpy()
        result[str(seed)]={'status':'SANITY_ONLY_NOT_CONTROLLED',
            'historical_normal_fp':int(((old>.5)&(y==0)).sum()),
            'new_normal_fp':int(((scores['S0'][seed]>.5)&(y==0)).sum()),
            'historical_fault_tp':int(((old>.5)&(y==1)).sum()),
            'new_fault_tp':int(((scores['S0'][seed]>.5)&(y==1)).sum())}
    return result


def evaluate(args):
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh evaluation output required')
    phase1=None
    if args.phase=='val':
        phase1=train.phase1_gate(args.phase1_verdicts,args.phase1_verdicts_sha256)
    p,_=train.load_protocol(args.protocol)
    if args.synthetic_demo:
        if not args.ledger or not output.resolve().is_relative_to(train.SMOKE_ROOT.resolve()):
            raise ValueError('Synthetic demo needs an explicit artifact ledger and smoke-root output')
        frame=pd.read_csv(args.ledger,dtype={'association_group':str,'case_uid':str},keep_default_na=False)
    else:
        split='train' if args.phase=='cv' else 'val'
        expected=train.read_json(train.BASELINE)['data_identities'][f'ledgers/{split}_included_windows.csv']
        if train.sha(Path(args.data)/'ledgers'/f'{split}_included_windows.csv')!=expected:
            raise ValueError('Evaluation ledger identity mismatch')
        frame=train.read_ledger(args.data,split)
        if args.phase=='cv':
            train.fold_sizes(frame,p)
    scores,last,reports,missing=load_runs(args.runs,frame,args.phase,args.synthetic_demo)
    output.mkdir(parents=True,exist_ok=False)
    if missing:
        result={'status':'INCOMPLETE','phase':args.phase,'synthetic_demo':args.synthetic_demo,
                'protocol_sha256':train.PROTOCOL_SHA,'missing_runs_or_files':missing,
                'decisions':{},'performance_claim':False}
    else:
        metrics={a:{str(s):score_metrics(frame,scores[a][s]) for s in SEEDS} for a in ARMS}
        sensitivity={a:{str(s):{'primary':fp_at_recall(frame.qualified_y.to_numpy(),last[a][s]),
                              'ap':fp_at_recall(frame.qualified_y.to_numpy(),last[a][s])['ap']} for s in SEEDS} for a in ARMS}
        result={'status':'COMPLETE_SYNTHETIC_DEMO_NOT_RESULT' if args.synthetic_demo else 'COMPLETE',
            'phase':args.phase,'synthetic_demo':args.synthetic_demo,'performance_claim':not args.synthetic_demo,
            'protocol_sha256':train.PROTOCOL_SHA,'population':len(frame),'metrics_best':metrics,
            'last_epoch_sensitivity':sensitivity,
            'run_resources':[{k:r.get(k) for k in ('arm','seed','fold','best_epoch','last_epoch','epochs','wall_time_s','param_count','macs')} for r in reports.values()],
            'run_report_sha256':{path:train.sha(Path(path)/'report.json') for path in reports}}
        if args.phase=='cv':
            uncertainty=paired_bootstrap(frame.qualified_y.to_numpy(),scores,frame.association_group.to_numpy())
            points,decisions=comparison_results(metrics,uncertainty)
            result.update(uncertainty=uncertainty,comparisons=points,decisions=decisions,
                historical_sanity=historical_sanity(frame,scores,{42:args.historical_seed42_root,43:args.historical_seed43_root}))
        else:
            confirmations={}
            for arm in ARMS[1:]:
                ratios=[ratio(metrics[arm][str(s)]['primary']['fp'],metrics['S0'][str(s)]['primary']['fp']) for s in SEEDS]
                point=geometric_mean(ratios)
                verdict=phase1['decisions'][arm]['verdict']
                confirms=(point is not None and point>1) if verdict in ('COMPONENT_HELPS','MSTCA_BETTER') else (
                    point is not None and point<1.10) if verdict in ('REMOVABLE','NO_ADVANTAGE') else None
                confirmations[arm]={'phase1_verdict_unchanged':verdict,'val_ratio':point,
                    'confirmation':'NOT_APPLICABLE' if confirms is None else ('CONFIRMED' if confirms else 'NOT_CONFIRMED')}
            result.update(confirmations=confirmations,decisions=phase1['decisions'],phase1_verdicts_sha256=args.phase1_verdicts_sha256)
    train.write_json(output/'evaluation.json',result)
    lines=['# S ablation evaluation',f"Status: {result['status']}",
           '**SYNTHETIC DEMO — NOT SCIENTIFIC RESULTS**' if args.synthetic_demo else 'Source-only evaluation']
    for arm,decision in result.get('decisions',{}).items():
        lines.append(f"- {arm}: {decision['verdict']} ({decision.get('reason')})")
    if missing:
        lines.append(f'INCOMPLETE: {len(missing)} required runs/files unavailable; no conclusions.')
    with (output/'summary.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    return result


def build_demo(smoke_runs,data,output):
    """Mix four actual small smoke runs with 32 clearly labelled synthetic runs.

    Reads train metadata only, never additional train X or val contents.
    """
    import shutil
    output=Path(output)
    if output.exists() or not output.resolve().is_relative_to(train.SMOKE_ROOT.resolve()):
        raise FileExistsError('New demo directory under smoke root required')
    frame=train.read_ledger(data)
    _,folds=train.split_ledger(frame,0)
    first=train.read_json(Path(smoke_runs)/'S0_seed42_fold0'/'population.json')['heldout']
    selected=np.flatnonzero(frame.original_cache_index.isin(first).to_numpy()).tolist()
    for fold in (1,2):
        selected.extend(train.take_smoke(frame,np.flatnonzero(folds==fold),256).tolist())
    demo=frame.iloc[sorted(selected)].copy().reset_index(drop=True)
    demo['compact_index']=np.arange(len(demo))
    output.mkdir(parents=True,exist_ok=False)
    with (output/'ledger.csv').open('x',encoding='utf-8',newline='') as stream:
        demo.to_csv(stream,index=False)
    _,demo_folds=train.split_ledger(demo,0)
    gold=train.golden_digests()
    for arm in ARMS:
        for seed in SEEDS:
            for fold in range(3):
                directory=output/'runs'/f'{arm}_seed{seed}_fold{fold}'
                directory.mkdir(parents=True,exist_ok=False)
                if seed==42 and fold==0:
                    source=Path(smoke_runs)/directory.name
                    r=train.read_json(source/'report.json')
                    for which in ('best','last'):
                        shutil.copyfile(source/f'heldout_predictions_{which}.npz',directory/f'heldout_predictions_{which}.npz')
                else:
                    part=demo.iloc[np.flatnonzero(demo_folds==fold)]
                    rng=np.random.default_rng(seed*100+fold)
                    y=part.qualified_y.to_numpy()
                    score=np.clip(np.where(y==1,.45+.5*rng.random(len(y)),.2+.6*rng.random(len(y)))
                                  -int(arm[1])*.012*(y==0),.001,.999)
                    variant={'S0':'mstca','S1':'mstca_no_se','S2':'mstca_single_k7','S3':'cnn1d_matched'}[arm]
                    counts={'S0':(105938,6764928),'S1':(103810,6762880),'S2':(105106,6640128),'S3':(105929,7097374)}
                    r={'status':'COMPLETE_SYNTHETIC_DEMO','phase':'cv','arm':arm,'seed':seed,'fold':fold,
                       'synthetic_demo':True,'protocol_sha256':train.PROTOCOL_SHA,'initial_digest':gold[(variant,seed)],
                       'param_count':counts[arm][0],'macs':counts[arm][1],'epochs':2,'best_epoch':1,'last_epoch':2,
                       'wall_time_s':None,'prediction_sha256':{}}
                    for which in ('best','last'):
                        scores=score if which=='best' else np.clip(score+.005,.001,.999)
                        path=directory/f'heldout_predictions_{which}.npz'
                        with path.open('xb') as stream:
                            np.savez_compressed(stream,probabilities=np.stack([1-scores,scores],axis=1),
                                                original_cache_index=part.original_cache_index.to_numpy(dtype=np.int64))
                        r['prediction_sha256'][which]=train.sha(path)
                train.write_json(directory/'report.json',r)
    train.write_json(output/'DEMO_ONLY.json',{'scope':'SYNTHETIC_DEMO_NOT_RESULT','real_smoke_runs':4,'synthetic_runs':32,
        'train_metadata_rows':len(demo),'additional_X_windows_read':0})
    return output


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=train.DEFAULT_PROTOCOL)
    p.add_argument('--data',type=Path,default=Path('D:/UAV-Data/causal_input_v1'))
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--phase',choices=['cv','val'],default='cv')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--synthetic-demo',action='store_true')
    p.add_argument('--ledger',type=Path)
    p.add_argument('--build-demo',action='store_true',help='Build a labelled mixed synthetic fixture, not evaluate results')
    p.add_argument('--phase1-verdicts',type=Path)
    p.add_argument('--phase1-verdicts-sha256')
    p.add_argument('--historical-seed42-root',type=Path)
    p.add_argument('--historical-seed43-root',type=Path)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.build_demo:
        print(build_demo(args.runs,args.data,args.output))
    else:
        result=evaluate(args)
        print(json.dumps({'status':result['status'],'synthetic_demo':result['synthetic_demo']},indent=2))
