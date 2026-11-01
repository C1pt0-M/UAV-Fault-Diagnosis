"""Verify registered source-only M1/M0-matched results and archive once."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
import sys
import tarfile
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/uav_fault'));sys.path.insert(0,str(ROOT/'scripts/research'))
import train_causal_m1_continuous as trainer
from validate_kinematic_residuals import sha
from train_source_normal_sampling import state_digest,selection_step
from smoke_normal_source_sampling import stratified_metrics
from uav_fault.models.train_online_detector import get_warmup_cosine_lr


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,d):
 with path.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('Preserve verified M1 results')
 p=trainer.load_protocol(a.protocol);old=read(ROOT/p['M0_protocol_path']);manifest=read(a.features/'build_manifest.json')
 assert sha(a.features/'build_manifest.json')==p['feature_manifest_sha256']
 # No need to reload source tensors for output verification; explicit source
 # ledger and derived manifest IDs are the authoritative join.
 f=pd.read_csv(a.data/'ledgers/val_included_windows.csv',keep_default_na=False,low_memory=False,dtype={'case_uid':str,'association_group':str});assert sha(a.data/'ledgers/val_included_windows.csv')==old['data_identities']['ledgers/val_included_windows.csv']
 export=torch.load(a.features/'cache/val.pt',map_location='cpu',weights_only=True,mmap=True);indices=export['base_compact_index'].numpy();metadata=f.iloc[indices].copy().reset_index(drop=True)
 metadata['label']=metadata.qualified_y.astype(int);metadata['normal_origin']=np.where(metadata.label==1,'fault_endpoint',np.where(metadata.case_level2_label=='normal','original_normal_case','normal_part_of_fault_case'));metadata['flight_status']=metadata.case_uid.str.split('/').str[1];metadata['cache_index']=np.arange(len(metadata))
 rows=[];common={};files=[];nsteps=math.ceil(p['data_counts']['train']/512);last_predictions={}
 for job in p['formal_jobs']:
  variant,arm=job['variant'],job['arm'];root=a.runs/variant/arm;r=read(root/'report.json');history=read(root/'history.json')
  assert r['status']=='COMPLETE_FORMAL_SOURCE_M1_COMPARISON' and r['variant']==variant and r['seed']==42 and not r['cpu_check']
  assert r['protocol_sha256']==sha(a.protocol) and r['source_train_windows']==p['data_counts']['train'] and r['source_val_windows']==p['data_counts']['val']
  assert len(history)==r['epochs']<=100 and r['checkpoint_reload_matches']
  selected={'best_score':-float('inf'),'best_epoch':None,'significant_reference':-float('inf'),'bad_epochs':0};stop=False
  for i,h in enumerate(history,1):
   assert h['epoch']==i and h['gradient_steps']==nsteps and sum(h['realized_training_origins'].values())==p['data_counts']['train'] and math.isfinite(h['train_loss'])
   lr=get_warmup_cosine_lr(i-1,5,100,3e-4,1e-6);assert math.isclose(lr,h['lr'],rel_tol=0,abs_tol=1e-18)
   selected,_,stop=selection_step(selected,h['source_validation']['overall']['macro_f1'],i,p['validation']['early_stop']);assert selected==h['selection']
   if stop:assert i==len(history)
  assert len(history)==100 or stop
  assert r['best_epoch']==selected['best_epoch'] and r['gradient_steps']==len(history)*nsteps
  assert sha(root/'best.pt')==r['best_checkpoint_sha256']
  with np.load(root/'source_validation_predictions.npz',allow_pickle=False) as z:
   probabilities=z['probabilities'];labels=z['labels']
   assert probabilities.shape==(len(metadata),2) and np.isfinite(probabilities).all()
   assert np.allclose(probabilities.sum(1),1,rtol=0,atol=1e-6)
   assert np.array_equal(labels,metadata.label) and np.array_equal(z['original_indices'],metadata.original_cache_index)
   assert np.array_equal(z['compact_indices'],metadata.cache_index)
   actual=stratified_metrics(labels,probabilities.argmax(1),metadata);assert actual==r['source_validation']==history[r['best_epoch']-1]['source_validation']
   score=probabilities[:,1];order=np.argsort(-score,kind='stable');sorted_score=score[order];sorted_y=labels[order]
   ends=np.r_[np.flatnonzero(sorted_score[:-1]!=sorted_score[1:])+1,len(score)];tp=np.cumsum(sorted_y)[ends-1];precision=tp/ends;recall=tp/sorted_y.sum();ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
   last_predictions[(variant,arm)]=(labels.copy(),probabilities.copy())
  initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=True);best=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
  assert state_digest(initial)==r['initial_state_digest'] and all(torch.isfinite(v).all() for v in best.values()) and any(not torch.equal(initial[k],v) for k,v in best.items())
  state={k:v for k,v in initial.items() if not k.startswith('residual_adapter.')};expected=p['common_M0_pretrained'][arm]['state_digest'];assert state_digest(state)==expected
  if arm in common:assert all(torch.equal(common[arm][k],v) for k,v in state.items())
  else:common[arm]=state
  m=actual['overall'];row={'variant':variant,'arm':arm,'epochs':r['epochs'],'best_epoch':r['best_epoch'],'parameters':r['model_parameters'],'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'fault_recall':m['fault_recall'],'normal_fpr':m['normal_window_fpr'],'original_normal_fpr':actual['normal_origin']['original_normal_case']['normal_window_fpr'],'hover_fpr':actual['original_normal_by_status'].get('hover',{}).get('normal_window_fpr'),'average_precision_step_integral':ap,'confusion_matrix':m['confusion_matrix'],'fault_by_case_class':actual['fault_by_case_class'],'elapsed_s':r['elapsed_s'],'source_val_windows':len(metadata),'best_sha256':r['best_checkpoint_sha256']}
  rows.append(row)
  for file in root.iterdir():
   if file.is_file():files.append((file,'runs/'+variant+'/'+arm+'/'+file.name))
 for arm in p['arms']:
  if ('M0_matched',arm) in last_predictions:
   assert np.array_equal(last_predictions[('M0_matched',arm)][0],last_predictions[('M1',arm)][0])
 for name,h in p['code_identities'].items():assert sha(ROOT/name)==h,name
 a.output.mkdir(parents=True,exist_ok=False)
 result={'status':'PASS_REGISTERED_M1_AND_MATCHED_SOURCE_RESULTS','verified_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),'data_counts':p['data_counts'],'rows':rows,'same_original_IDs_and_pretrained_common_weights_verified':True,'all_metrics_history_schedule_and_early_stop_recomputed':True,'no_test_Real_or_deployment':True,'scope':'Single seed source-val, same pretrained arm starts, matched population. No matched-recall or significance claim. AP explicitly step-integral, not trapezoid PR-AUC.'}
 save(a.output/'summary.json',result);files.append((a.output/'summary.json','summary.json'));files.append((a.protocol,'protocol.json'))
 manifest={'files':{name:{'sha256':sha(path),'bytes':path.stat().st_size} for path,name in files},'source_tensors_in_archive':False};save(a.output/'manifest.json',manifest);files.append((a.output/'manifest.json','manifest.json'))
 archive=a.output/'verified_m1_training.tar.gz'
 with tarfile.open(archive,'x:gz',compresslevel=1) as t:
  for path,name in files:t.add(path,arcname=name,recursive=False)
 save(a.output/'archive_receipt.json',{'status':'COMPLETE_VERIFIED_M1_SOURCE_RESULT_ARCHIVE','path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
 print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
