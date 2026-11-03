"""Collect and independently verify completed M3 source-Val artifacts.

Read-only: no model forward, optimizer, threshold search, Test/Real access or
new training. Metrics come from each saved best-validation NPZ.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
from train_causal_m3 import metric_report
from train_source_normal_sampling import state_digest
from uav_fault.models.cnn.mstca_m3 import ADAPTER_ANCHOR,load_fixed_adapter
from uav_fault.models.train_online_detector import get_warmup_cosine_lr
VARIANTS = ['C0','Ccap','CO','CO9','CV','CR','CN','CON']
FAULTS = ['GPS','accelerometer','barometer','gyroscope','magnetometer','motor']
DOMAINS = ['SIL','HIL']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def cm_metrics(y, pred):
    y = np.asarray(y, dtype=np.int64); pred = np.asarray(pred, dtype=np.int64)
    cm = np.bincount(y*2+pred, minlength=4).reshape(2,2)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    normal_recall = tn/(tn+fp) if tn+fp else None
    f1_normal = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else None
    return {'windows': int(len(y)), 'confusion_matrix': cm.tolist(), 'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp,
            'accuracy': float((tn+tp)/len(y)) if len(y) else None, 'precision':precision,
            'fault_recall':recall, 'fault_f1':f1, 'normal_fpr':float(fp/(tn+fp)) if tn+fp else None,
            'normal_recall':normal_recall, 'normal_f1':f1_normal,
            'macro_f1':float(np.mean([x for x in [f1_normal, f1] if x is not None])) if (f1_normal is not None or f1 is not None) else None}


def ap_score(y, score):
    y=np.asarray(y,dtype=np.int64);score=np.asarray(score,dtype=float)
    if len(y)==0 or int(y.sum())==0:return None
    order=np.argsort(-score,kind='stable');ys=y[order];ss=score[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:])+1,len(ss)]
    tp=np.cumsum(ys)[ends-1];recall=tp/ys.sum()
    return float(np.sum(np.diff(np.r_[0.,recall])*(tp/ends)))


def observed_states(raw, columns):
    # Same fixed proxy rules as the frozen 13-feature analysis, vectorized.
    ci={name:i for i,name in enumerate(columns)}
    v=raw[:,[ci['vel_x'],ci['vel_y'],ci['vel_z']],:].astype(np.float64)
    g=raw[:,[ci['gyro_x'],ci['gyro_y'],ci['gyro_z']],:].astype(np.float64)
    speed=np.linalg.norm(v,axis=1);rotation=np.linalg.norm(g,axis=1);vz=v[:,2]
    state=np.full(len(raw),'other_motion',dtype=object)
    maneuver=(np.quantile(rotation,.9,axis=1)>=.3)|(np.std(speed,axis=1)>=.5)
    state[maneuver]='maneuver_proxy'
    remaining=~maneuver
    climb=remaining&(np.mean(vz<=-.5,axis=1)>=.8);state[climb]='climb_proxy';remaining &= ~climb
    descent=remaining&(np.mean(vz>=.5,axis=1)>=.8);state[descent]='descent_proxy';remaining &= ~descent
    low=remaining&(np.mean(speed<=.3,axis=1)>=.8)&(np.mean(rotation<=.15,axis=1)>=.8);state[low]='low_motion_proxy';remaining &= ~low
    translation=remaining&(np.mean(speed>.3,axis=1)>=.8);state[translation]='translation_proxy'
    return state


def hover_segment_masks(frame, normal_mask):
    """Keep hover separate from other log scenarios when slicing crop time."""
    hover = np.asarray(normal_mask, dtype=bool) & frame.flight_status.eq('hover').to_numpy()
    seconds = (frame.window_end_row.to_numpy()-1)*.02
    return {'first30s': hover & (seconds<=30), 'after30s': hover & (seconds>30)}


def metrics_match(saved, recomputed, path=''):
    if isinstance(saved,dict) and isinstance(recomputed,dict):
        if saved.keys()!=recomputed.keys():return False
        return all(metrics_match(saved[k],recomputed[k],path+'/'+str(k)) for k in saved)
    if path.endswith('/average_precision') and isinstance(saved,(int,float)) and isinstance(recomputed,(int,float)):
        return math.isclose(float(saved),float(recomputed),rel_tol=0,abs_tol=5e-13)
    return saved==recomputed


def check_selection(history, early):
    selected={'best_score':-float('inf'),'best_epoch':None,'significant_reference':-float('inf'),'bad_epochs':0}
    best_scores=[];stop_epoch=None
    for index,item in enumerate(history,1):
        if item.get('epoch')!=index:raise ValueError('History epochs not contiguous')
        score=float(item['source_validation']['overall']['macro_f1'])
        if not math.isfinite(score) or not math.isfinite(float(item['train_loss'])):raise ValueError('Nonfinite history metric')
        save=score>selected['best_score'];result=dict(selected)
        if save:result['best_score']=score;result['best_epoch']=index
        if score>selected['significant_reference']+early['significant_delta']:
            result['significant_reference']=score;result['bad_epochs']=0
        elif index>=early['min_epochs']:
            result['bad_epochs']+=1
        stop=index>=early['min_epochs'] and result['bad_epochs']>=early['patience']
        if result != item['selection']:raise ValueError('Saved selection state differs at epoch '+str(index))
        selected=result;best_scores.append(score)
        if stop:
            if index!=len(history):raise ValueError('History continues after early stop')
            stop_epoch=index
    if selected['best_epoch'] != int(np.argmax(np.asarray(best_scores)))+1:
        raise ValueError('Best epoch is not earliest strict Macro-F1 maximum')
    if stop_epoch is None and len(history)!=100:
        raise ValueError('Training ended before max epochs without registered early stop')
    return selected, stop_epoch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collected',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--preflight',type=Path,required=True)
    ap.add_argument('--execution',type=Path,required=True)
    ap.add_argument('--gate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior M3 collection')
    a.output.mkdir(parents=True,exist_ok=False)
    root=a.collected
    manifest=read(root/'source_manifest.json')
    for name,item in manifest['files'].items():
        path=root/name
        if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Collected archive member SHA mismatch: '+name)
    p=read(a.preflight);e=read(a.execution);gate=read(a.gate)
    for path in [a.preflight,a.execution]:
        if sha(path)!=path.with_suffix('.json.sha256').read_text().strip():raise ValueError('Frozen protocol lock mismatch')
        copied=root/'code'/path.resolve().relative_to(ROOT)
        if sha(copied)!=sha(path):raise ValueError('Collected execution protocol differs from local protocol')
    if sha(a.preflight)!=e['parent_sha256'] or gate['execution_sha256']!=sha(a.execution) or gate['status']!='PASS_M3_FIXED_ANCHOR_AND_ENTRYPOINTS':
        raise ValueError('Parent or accepted execution identity mismatch')
    for name,expected in e['code_identities'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen execution code changed: '+name)
    if e['jobs']!=VARIANTS or e['overall_delta_pp']!=0 or any(e['per_fault_delta_pp'].values()) or e['domain_overall_delta_pp']!={'SIL':0.,'HIL':0.}:
        raise ValueError('Execution constraints changed')
    # Frozen common validation population and metadata, read-only.
    expected=p['data']['source_split_identities']['val']
    data_identities=read(ROOT/'configs/experiments/causal_pure_mstca_m0_v1.json')['data_identities']
    input_paths=[a.data/'cache/val.pt',a.features/'cache/val.pt',a.data/'ledgers/val_included_windows.csv',a.data/'norm_stats.json',ROOT/p['normalization']['path']]
    input_hashes=[expected['base_cache_sha256'],expected['residual_cache_sha256'],data_identities['ledgers/val_included_windows.csv'],data_identities['norm_stats.json'],p['normalization']['sha256']]
    for path,h in zip(input_paths,input_hashes):
        if sha(path)!=h:raise ValueError('Frozen input identity mismatch: '+str(path))
    export=torch.load(a.features/'cache/val.pt',map_location='cpu',weights_only=True,mmap=True)
    ledger=pd.read_csv(a.data/'ledgers/val_included_windows.csv',keep_default_na=False,low_memory=False)
    indices=export['base_compact_index'].numpy();frame=ledger.iloc[indices].copy().reset_index(drop=True)
    if len(frame)!=32381 or not np.array_equal(frame.original_cache_index,export['original_cache_index'].numpy()) or not np.array_equal(frame.qualified_y,export['y'].numpy()):
        raise ValueError('Frozen validation ID/label population differs')
    frame['label']=frame.qualified_y.astype(int);frame['normal_origin']=np.where(frame.label==1,'fault_endpoint',np.where(frame.case_level2_label=='normal','original_normal_case','normal_part_of_fault_case'))
    frame['flight_status']=frame.case_uid.str.split('/').str[1];frame['fault_directory']=frame.case_uid.str.split('/').str[2];frame['cache_index']=np.arange(len(frame))
    base=torch.load(a.data/'cache/val.pt',map_location='cpu',weights_only=True,mmap=True)
    raw=base['x'][indices].numpy()
    frame['observed_state']=observed_states(raw,p['data']['base_columns'])
    if sha(ADAPTER_ANCHOR)!=e['adapter_anchor_sha256']:
        raise ValueError('Fixed adapter anchor bytes changed')
    fixed_adapter=load_fixed_adapter()
    untrained_m0=torch.load(ROOT/p['initialization']['actual_mapped_untrained_file'],map_location='cpu',weights_only=True)
    common_digest=p['initialization']['shared_state_digest']
    expected_draws=read(ROOT/'artifacts/M3_preflight_v1_20261103/sampler_cpu_check.json')['reference_epoch_originalIDs_SHA256']
    normal_mask=frame.label.to_numpy()==0
    original_normal=normal_mask&(frame.normal_origin.to_numpy()=='original_normal_case')
    current=[];curves=[];category=[];domains=[];domain_faults=[];states=[];files=[];checks=[];saved_predictions={}
    for variant in VARIANTS:
        run=root/'runs'/variant;report=read(run/'report.json');history=read(run/'history.json');progress=read(run/'progress.json')
        if report.get('status')!='COMPLETE_FORMAL_M3':raise ValueError('Job incomplete: '+variant)
        if report.get('variant')!=variant or report.get('seed')!=42 or report.get('arm')!='balanced_source_normal' or report.get('Test_Real_used') is not False:raise ValueError('Job contract mismatch: '+variant)
        if len(history)!=report['epochs'] or report['epochs']>100 or report['optimizer_steps']!=sum(x['gradient_steps'] for x in history):raise ValueError('Job history/update count mismatch: '+variant)
        selected,stop=check_selection(history,p['validation']['early_stop'])
        if int(report['best_epoch'])!=int(selected['best_epoch']):raise ValueError('Best epoch mismatch: '+variant)
        if report['source_validation']!=history[report['best_epoch']-1]['source_validation']:raise ValueError('Best history/report metrics differ '+variant)
        if report['execution_sha256']!=sha(a.execution) or report['parent_sha256']!=sha(a.preflight):raise ValueError('Run protocol identities differ '+variant)
        if progress['epoch']!=len(history) or progress['optimizer_steps']!=report['optimizer_steps']:raise ValueError('Final progress disagrees with report '+variant)
        log=(root/'logs'/(variant+'.log')).read_text(encoding='utf-8')
        epoch_lines=[line for line in log.splitlines() if line.startswith('M3_EPOCH ')]
        if len(epoch_lines)!=len(history) or 'Traceback (most recent call last)' in log:raise ValueError('Missing epochs or error in training log '+variant)
        initial_state=torch.load(run/'initial.pt',map_location='cpu',weights_only=True)
        if state_digest(initial_state)!=report['initial_state_digest']:
            raise ValueError('Initial state digest mismatch '+variant)
        shared={k:v for k,v in initial_state.items() if k.startswith(('encoder.','head.'))}
        if state_digest(shared)!=common_digest:
            raise ValueError('M0 public tensor/buffer state mismatch '+variant)
        if variant=='C0':
            for k,v in untrained_m0.items():
                target='encoder.'+k[len('raw_encoder.'):] if k.startswith('raw_encoder.') else k
                if target in shared and not torch.equal(v,shared[target]):raise ValueError('C0 mapped M0 tensor mismatch '+target)
        else:
            for k,v in fixed_adapter.items():
                if not torch.equal(initial_state['adapter.'+k],v):raise ValueError('Adapter anchor tensor mismatch '+variant+' '+k)
        for i,item in enumerate(history):
            if item['sampling_original_ID_sha256']!=expected_draws[i]:raise ValueError('Sampler digest mismatch '+variant+' epoch '+str(i+1))
            if item['gradient_steps']!=297 or not math.isclose(item['lr'],get_warmup_cosine_lr(i,5,100,3e-4,1e-6),rel_tol=0,abs_tol=1e-18):raise ValueError('Training schedule/step count mismatch '+variant+' epoch '+str(i+1))
            h=item['source_validation'];o=h['overall'];curves.append({'variant':variant,'epoch':item['epoch'],'train_loss':item['train_loss'],'val_accuracy':o['accuracy'],'val_macro_f1':o['macro_f1'],'val_fault_recall':o['fault_recall'],'val_normal_fpr':o['normal_window_fpr'],'val_fault_precision':h.get('fault_precision'),'val_average_precision':h.get('average_precision'),'is_best':item['epoch']==report['best_epoch'],'significant_reference':item['selection']['significant_reference'],'bad_epochs':item['selection']['bad_epochs']})
        best=run/'best.pt';last=run/'last.pt';initial=run/'initial.pt';pred_path=run/'best_validation_predictions.npz'
        for path in [best,last,initial,pred_path]:
            if not path.is_file():raise FileNotFoundError(f'{variant}: {path.name}')
        best_state=torch.load(best,map_location='cpu',weights_only=True)
        last_state=torch.load(last,map_location='cpu',weights_only=True)
        if set(best_state)!=set(initial_state) or set(last_state)!=set(initial_state) or any(best_state[k].shape!=initial_state[k].shape for k in best_state):raise ValueError('Checkpoint state schema mismatch '+variant)
        if not all(torch.isfinite(v).all() for v in best_state.values()) or not all(torch.isfinite(v).all() for v in last_state.values()):raise ValueError('Nonfinite checkpoint weight '+variant)
        if sha(best)!=report['best_sha256']:raise ValueError('Best weight SHA mismatch '+variant)
        with np.load(pred_path,allow_pickle=False) as z:
            q=z['probabilities'];y=z['labels'];ids=z['original_indices']
        if q.shape!=(32381,2) or not np.isfinite(q).all() or not np.allclose(q.sum(1),1,rtol=0,atol=1e-6) or not np.array_equal(y,frame.label.to_numpy()) or not np.array_equal(ids,frame.original_cache_index.to_numpy()):
            raise ValueError('Saved best predictions mismatch '+variant)
        pred=q.argmax(1);score=q[:,1];overall=cm_metrics(y,pred);overall['average_precision']=ap_score(y,score);overall['macro_f1_reported']=report['source_validation']['overall']['macro_f1']
        saved_predictions[variant]=(y.copy(),pred.copy())
        reconstructed=metric_report(y,q,frame)
        if not metrics_match(reconstructed,report['source_validation']):
            raise ValueError('Saved best probability metrics do not reproduce report: '+variant)
        for key in ['overall','normal_origin','normal_by_status','original_normal_by_status','fault_by_case_class','six_faults','domains','faults_by_domain','normal_observed_states','hover_crop_segments']:
            if reconstructed[key]!=report['source_validation'][key]:
                raise ValueError('Exact confusion/count metrics mismatch in '+variant+': '+key)
        if report.get('optimizer_steps') != report['epochs']*297:
            raise ValueError('Optimizer update count does not match per-epoch batch count '+variant)
        # Same normal pool for each fault category: a conditional fault detector metric.
        for fault in FAULTS:
            pos=(y==1)&(frame.fault_directory.to_numpy()==fault);sel=pos|normal_mask;cm=cm_metrics(pos[sel].astype(np.int64),pred[sel]);cm.update(variant=variant,scope='fault_vs_same_normal_pool',fault=fault,positive_windows=int(pos.sum()),negative_normal_windows=int(normal_mask.sum()))
            category.append(cm)
        for domain in DOMAINS:
            sel=frame.domain.to_numpy()==domain; m=cm_metrics(y[sel],pred[sel]);m.update(variant=variant,scope='domain_overall',domain=domain);domains.append(m)
            for fault in FAULTS:
                positive=(y==1)&frame.fault_directory.eq(fault).to_numpy()&sel
                normal=normal_mask&sel
                population=positive|normal
                m=cm_metrics(y[population],pred[population])
                m.update(variant=variant,domain=domain,fault=fault,scope='fault_vs_same_domain_normal_pool',positive_windows=int(positive.sum()),negative_normal_windows=int(normal.sum()))
                domain_faults.append(m)
        for scope,mask in [('all_normal',normal_mask),('independent_normal',original_normal)]:
            states.append({'variant':variant,'scope':scope,'stratum_type':'population','stratum':'all_known_normal','windows':int(mask.sum()),'false_positives':int(pred[mask].sum()),'fpr':float(pred[mask].mean()) if mask.any() else None})
            for state in sorted(frame.loc[mask,'observed_state'].unique()):
                sel=mask&(frame.observed_state.to_numpy()==state);states.append({'variant':variant,'scope':scope,'stratum_type':'observed_state','stratum':state,'windows':int(sel.sum()),'false_positives':int(pred[sel].sum()),'fpr':float(pred[sel].mean()) if sel.any() else None})
            for flight in sorted(frame.loc[mask,'flight_status'].unique()):
                sel=mask&(frame.flight_status.to_numpy()==flight);states.append({'variant':variant,'scope':scope,'stratum_type':'log_flight_status','stratum':flight,'windows':int(sel.sum()),'false_positives':int(pred[sel].sum()),'fpr':float(pred[sel].mean()) if sel.any() else None})
            for segment,sel in hover_segment_masks(frame,mask).items():
                states.append({'variant':variant,'scope':scope,'stratum_type':'hover_crop_segment','stratum':segment,'windows':int(sel.sum()),'false_positives':int(pred[sel].sum()),'fpr':float(pred[sel].mean()) if sel.any() else None})
        expected_parameters=105938 if variant=='C0' else 109538
        actual_parameters=sum(v.numel() for k,v in best_state.items() if not k.endswith(('running_mean','running_var','num_batches_tracked')))
        if int(report['parameters'])!=expected_parameters or actual_parameters!=expected_parameters:raise ValueError('Frozen parameter count mismatch '+variant)
        current.append({'variant':variant,'parameters':int(report['parameters']),'epochs':report['epochs'],'best_epoch':report['best_epoch'],'optimizer_steps':report['optimizer_steps'],'best_macro_f1':overall['macro_f1'],'accuracy':overall['accuracy'],'precision':overall['precision'],'recall':overall['fault_recall'],'fault_f1':overall['fault_f1'],'normal_fpr':overall['normal_fpr'],'fault_precision':overall['precision'],'average_precision':overall['average_precision'],'confusion_matrix':overall['confusion_matrix'],'best_weight_sha256':sha(best),'best_weight_bytes':best.stat().st_size,'initial_state_digest':report['initial_state_digest'],'train_loss_final':history[-1]['train_loss'],'train_loss_min':min(float(x['train_loss']) for x in history),'val_loss_available':False,'train_metrics_available':False,'prediction_file_sha256':sha(pred_path),'history_sha256':sha(run/'history.json'),'log_sha256':sha(root/'logs'/ (variant+'.log')),'all_weights_finite':True,'best_selection_valid':True,'predictions_valid':True})
        files.extend([{'variant':variant,'role':role,'path':path.relative_to(root).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size} for role,path in [('initial_weight',initial),('best_weight',best),('last_weight',last),('history',run/'history.json'),('report',run/'report.json'),('progress',run/'progress.json'),('best_validation_predictions',pred_path),('log',root/'logs'/(variant+'.log'))]])
        checks.append({'variant':variant,'status':'PASS','history_selection':True,'sampler_digests':True,'finite_weights':True,'saved_predictions':True,'test_real_used':False})
    df=pd.DataFrame(current);base_row=df[df.variant=='C0'].iloc[0]
    # Compare integer TP counts to avoid floating-point boundary decisions.
    base_fault={r['fault']:r for r in category if r['variant']=='C0'};base_domain={r['domain']:r for r in domains if r['variant']=='C0'}
    constraint=[]
    for variant in VARIANTS:
        row=df[df.variant==variant].iloc[0];fault_rows={r['fault']:r for r in category if r['variant']==variant};domain_rows={r['domain']:r for r in domains if r['variant']==variant}
        violations=[]
        if row['confusion_matrix'][1][1] < base_row['confusion_matrix'][1][1]:violations.append('overall')
        for fault in FAULTS:
            if fault_rows[fault]['tp'] < base_fault[fault]['tp']:violations.append(fault)
        for domain in DOMAINS:
            if domain_rows[domain]['tp'] < base_domain[domain]['tp']:violations.append(domain+'_overall')
        constraint.append({'variant':variant,'pass_overall_and_six_fault_recall':not any(v in violations for v in ['overall']+FAULTS),'pass_SIL_HIL_overall_recall':not any(v in violations for v in ['SIL_overall','HIL_overall']),'pass_all_hard_constraints':not violations,'violations':';'.join(violations),'overall_recall_delta_pp':100*(row['recall']-base_row['recall']),'overall_fpr_delta_pp':100*(row['normal_fpr']-base_row['normal_fpr'])})
    paired=[]
    ref_y,ref_pred=saved_predictions['C0']
    for variant in VARIANTS[1:]:
        y,pred=saved_predictions[variant]
        for domain in ['all','SIL','HIL']:
            dmask=np.ones(len(frame),dtype=bool) if domain=='all' else frame.domain.to_numpy()==domain
            fault=dmask&(y==1);normal=dmask&(y==0)
            paired.append({'variant':variant,'domain':domain,'fault_windows':int(fault.sum()),
                'C0_FN_recovered_to_TP':int((fault&(ref_pred==0)&(pred==1)).sum()),
                'C0_TP_lost_to_FN':int((fault&(ref_pred==1)&(pred==0)).sum()),
                'net_fault_TP_delta':int((fault&(pred==1)).sum()-(fault&(ref_pred==1)).sum()),
                'C0_FP_corrected_to_TN':int((normal&(ref_pred==1)&(pred==0)).sum()),
                'new_FP_vs_C0':int((normal&(ref_pred==0)&(pred==1)).sum()),
                'net_normal_FP_delta':int((normal&(pred==1)).sum()-(normal&(ref_pred==1)).sum())})
    selected_pairs=[]
    for reference,candidate in [('Ccap','CO'),('CO','CO9'),('CN','CON'),('CO','CON')]:
        y,ref_pred=saved_predictions[reference]
        cy,pred=saved_predictions[candidate]
        if not np.array_equal(y,cy):raise ValueError('Pairwise labels differ')
        normal=y==0;fault=y==1
        selected_pairs.append({'reference':reference,'candidate':candidate,
            'normal_FP_corrected':int((normal&(ref_pred==1)&(pred==0)).sum()),
            'normal_FP_added':int((normal&(ref_pred==0)&(pred==1)).sum()),
            'fault_FN_recovered':int((fault&(ref_pred==0)&(pred==1)).sum()),
            'fault_FN_added':int((fault&(ref_pred==1)&(pred==0)).sum()),
            'net_normal_FP_delta':int(pred[normal].sum()-ref_pred[normal].sum()),
            'net_fault_TP_delta':int(pred[fault].sum()-ref_pred[fault].sum())})
    pd.DataFrame(selected_pairs).to_csv(a.output/'selected_pairwise_changes.csv',index=False)
    # Static overhead from the already accepted entrypoint gate; no new inference.
    gate_models=read(a.gate)['models'];overhead=[]
    for variant in VARIANTS:
        overhead.append({'variant':variant,'parameters':int(df[df.variant==variant].iloc[0]['parameters']),'batch512_peak_bytes':gate_models[variant]['batch512_peak_bytes'],'cross_device_initial_max_abs_error':gate_models[variant]['cross_device_max_abs_error'],'dense_conv_linear_macs':p['models']['dense_conv_linear_MACs_C0'] if variant=='C0' else p['models']['dense_conv_linear_MACs_auxiliary'],'inference_latency_saved':False,'latency_note':'No trained continuous replay timing was part of this collection; static gate memory/MAC only.'})
    pd.DataFrame(domain_faults).to_csv(a.output/'fault_by_domain_metrics.csv',index=False)
    pd.DataFrame(current).to_csv(a.output/'overall_summary.csv',index=False);pd.DataFrame(category).to_csv(a.output/'fault_category_metrics_same_normal_pool.csv',index=False);pd.DataFrame(domains).to_csv(a.output/'domain_metrics.csv',index=False);pd.DataFrame(states).to_csv(a.output/'normal_state_and_flight_fpr.csv',index=False);pd.DataFrame(curves).to_csv(a.output/'training_curves.csv',index=False);pd.DataFrame(constraint).to_csv(a.output/'hard_constraint_check.csv',index=False);pd.DataFrame(overhead).to_csv(a.output/'model_overhead.csv',index=False);pd.DataFrame(paired).to_csv(a.output/'paired_changes_vs_C0.csv',index=False);pd.DataFrame(files).to_csv(a.output/'result_file_index.csv',index=False)
    pd.DataFrame([{'variant':v,'seed':p['seed'],'arm':p['primary_arm'],'train_windows':p['data']['source_split_identities']['train']['windows'],'val_windows':p['data']['source_split_identities']['val']['windows'],'batch_size':p['training']['batch_size'],'max_epochs':p['training']['max_epochs'],'optimizer':p['training']['optimizer'],'lr':p['training']['lr'],'weight_decay':p['training']['weight_decay'],'loss':p['loss']['name'],'gamma':p['loss']['gamma'],'sampler_seed':p['sampling']['sampler_generator_seed'],'loader_seed':p['sampling']['loader_generator_seed'],'classification_threshold':p['evaluation']['classification_boundary'],'checkpoint_selection':p['validation']['checkpoint'],'validation_early_stop':json.dumps(p['validation']['early_stop'],ensure_ascii=False,separators=(',',':'))} for v in VARIANTS]).to_csv(a.output/'run_configuration.csv',index=False)
    summary={'status':'PASS_COLLECTED_AND_VERIFIED_EIGHT_M3_RESULTS','created_at':datetime.now(timezone.utc).isoformat(),'scope':'Existing saved source-Val results only; no new forward, training, threshold search, Test or Real access. Saved probabilities were used for all reported metrics.','all_eight_complete':True,'common_validation_windows':32381,'common_validation_original_ids_sha256':hashlib.sha256(np.asarray(frame.original_cache_index,dtype=np.int64).tobytes()).hexdigest(),'source_val_residual_cache_sha256':sha(a.features/'cache/val.pt'),'best_selection':'Strict maximum source-Val overall Macro-F1, earliest exact tie, original early stop; independently reconstructed from history.','metrics_precision_recall_definition':'Overall is binary fault-vs-normal. Fault-category rows use that fault directory as positive and the same known-normal pool as negative; other fault directories are excluded from that conditional category metric because the model does not predict fault type.','missing_metrics':['Per-epoch Val loss was not saved. Per-epoch Train accuracy/precision/recall/F1 were not saved. Trained continuous replay latency was not part of this collection; static accepted-gate MAC and batch512 peak memory are reported.'],'constraint_reference':'M3 C0 best saved source-Val checkpoint, fixed argmax/0.5. All overall/six fault and SIL/HIL overall checks use integer TP counts on identical IDs.','hard_constraints':constraint,'best_by_macro_f1':df.sort_values('best_macro_f1',ascending=False).iloc[0]['variant'],'best_by_overall_fpr':df.sort_values('normal_fpr').iloc[0]['variant'],'best_by_overall_recall':df.sort_values('recall',ascending=False).iloc[0]['variant'],'files_verified':len(files),'checks':checks,'input_identities':{str(path):h for path,h in zip(input_paths,input_hashes)},'run_configuration_csv':'run_configuration.csv','paired_changes_csv':'paired_changes_vs_C0.csv','archive_receipt':read(ROOT/'artifacts/M3_results_v1_20261103/remote_archive_receipt.json'),'test_or_real_used':False,'weights_or_prediction_files_overwritten':False}
    write(a.output/'collection_summary.json',summary)
    print(json.dumps({'status':summary['status'],'best_by_macro_f1':summary['best_by_macro_f1'],'all_constraints_pass':[(x['variant'],x['pass_all_hard_constraints']) for x in constraint]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
