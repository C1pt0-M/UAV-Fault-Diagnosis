"""Post-run source-only result verification; never starts/retries a trainer."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import tarfile
import sys

import numpy as np
import torch

BASE=Path('/root/autodl-tmp/causal_source_v1_20261031')
CODE=BASE/'unpacked/code';DATA=BASE/'unpacked/data';RUNS=BASE/'runs'
sys.path.insert(0,str(CODE/'scripts/research'));sys.path.insert(0,str(CODE/'packages/uav_fault'))
from smoke_normal_source_sampling import stratified_metrics
from train_source_normal_sampling import selection_step

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

status=read(BASE/'remote_status.json')
if status['phase']!='COMPLETED_TWO_ARM_FORMAL_SEQUENCE':raise RuntimeError('Both formal jobs must complete before publication')
if (BASE/'verified_results_manifest.json').exists():raise FileExistsError('Preserve prior verified results')
config_path=CODE/'configs/experiments/causal_source_development_v1.json';config=read(config_path)
assert sha(config_path)==config_path.with_suffix('.json.sha256').read_text().strip()
for rel,h in config['code_identities'].items():assert sha(CODE/rel)==h,rel
for rel,h in config['data_identities'].items():assert sha(DATA/rel)==h,rel
# Verify all extra code dependencies and deployed source files against archive.
bundle=read(BASE/'unpacked/bundle_manifest.json')
for rel,identity in bundle['identities'].items():assert sha(BASE/'unpacked'/rel)==identity['sha256'],rel
import pandas as pd
metadata=pd.read_csv(DATA/'ledgers/val_included_windows.csv',keep_default_na=False,low_memory=False)
metadata['label']=metadata.qualified_y.astype(int)
metadata['normal_origin']=np.where(metadata.label==1,'fault_endpoint',np.where(metadata.case_level2_label=='normal','original_normal_case','normal_part_of_fault_case'))
metadata['flight_status']=metadata.case_uid.str.split('/').str[1]
metadata['cache_index']=np.arange(len(metadata))
reports={};files=[];rows={}
for arm in config['arms']:
 root=RUNS/arm;r=read(root/'report.json');history=read(root/'history.json')
 assert r['status']=='COMPLETE_FINITE_SOURCE_RETRAIN' and not r['cpu_check'] and r['seed']==42
 assert r['protocol_sha256']==sha(config_path) and r['checkpoint_reload_matches'] and r['parameters_changed']
 assert r['environment']['cuda_available'] and r['environment']['torch']=='2.5.1+cu124' and r['device']=='cuda:0'
 assert r['validation_windows']==34710 and r['source_train_windows']==162069 and r['no_final_test_or_Real_reads']
 assert 1<=r['epochs']<=30 and len(history)==r['epochs']
 state={'best_score':-float('inf'),'best_epoch':None,'significant_reference':-float('inf'),'bad_epochs':0}
 for i,h in enumerate(history,1):
  assert h['epoch']==i and np.isfinite(h['train_loss']) and h['gradient_steps']==317
  assert sum(h['realized_training_origins'].values())==162069
  score=float(h['source_validation']['overall']['macro_f1'])
  state,save,stop=selection_step(state,score,i,config['validation']['early_stop'])
  assert h['selection']==state
  if stop:assert i==len(history)
 assert r['best_epoch']==state['best_epoch']
 assert sha(root/'best.pt')==r['best_checkpoint_sha256']
 with np.load(root/'source_validation_predictions.npz',allow_pickle=False) as d:
  probabilities=d['probabilities'];labels=d['labels']
  assert probabilities.shape==(34710,2) and np.isfinite(probabilities).all()
  assert np.allclose(probabilities.sum(axis=1),1,atol=1e-6,rtol=0)
  assert np.array_equal(labels,metadata.label)
  assert np.array_equal(d['compact_indices'],metadata.cache_index)
  assert np.array_equal(d['original_indices'],metadata.original_cache_index)
  actual=stratified_metrics(labels,probabilities.argmax(1),metadata)
  assert actual==r['source_validation']==history[r['best_epoch']-1]['source_validation']
 initial=torch.load(root/'initial.pt',weights_only=True,map_location='cpu')
 best=torch.load(root/'best.pt',weights_only=True,map_location='cpu')
 assert all(torch.isfinite(v).all() for v in best.values())
 assert any(not torch.equal(initial[k],v) for k,v in best.items())
 reports[arm]=r
 metric=actual['overall']
 rows[arm]={'epochs':r['epochs'],'best_epoch':r['best_epoch'],'macro_f1':metric['macro_f1'],
   'fault_recall':metric['fault_recall'],'normal_window_fpr':metric['normal_window_fpr'],
   'original_normal_window_fpr':actual['normal_origin']['original_normal_case']['normal_window_fpr'],
   'inactive_in_fault_case_normal_fpr':actual['normal_origin']['normal_part_of_fault_case']['normal_window_fpr'],
   'original_normal_hover_fpr':actual['original_normal_by_status']['hover']['normal_window_fpr'],
   'confusion_matrix':metric['confusion_matrix'],'validation_windows':34710,'gradient_steps':r['gradient_steps'],
   'best_checkpoint_sha256':r['best_checkpoint_sha256'],'elapsed_s':r['elapsed_s']}
 for p in sorted(root.iterdir()):
  if p.is_file():files.append((p,'runs/'+arm+'/'+p.name))
files.extend([(BASE/'paired_source_summary.json','paired_source_summary.json'),(BASE/'remote_status.json','remote_status.json'),
 (BASE/'cuda_execution_check.json','cuda_execution_check.json'),(BASE/'preflight/preflight.json','preflight/preflight.json'),
 (BASE/'runs/preflight/preflight.json','runs/preflight/preflight.json'),(BASE/'bundle_verification.txt','bundle_verification.txt'),
 (BASE/'training_sequence.log','training_sequence.log'),(BASE/'runs/uniform_replacement.log','runs/uniform_replacement.log'),
 (BASE/'runs/balanced_source_normal.log','runs/balanced_source_normal.log'),(BASE/'unpacked/bundle_manifest.json','bundle_manifest.json'),
 (config_path,'linux_execution_protocol.json'),(BASE/'unpacked/provenance/original_source_protocol.json','original_source_protocol.json')])
assert reports['uniform_replacement']['initial_state_digest']==reports['balanced_source_normal']['initial_state_digest']
initials=[torch.load(RUNS/a/'initial.pt',weights_only=True,map_location='cpu') for a in config['arms']]
assert all(torch.equal(initials[0][k],initials[1][k]) for k in initials[0])
summary={'status':'PASS_BOTH_FROZEN_FORMAL_SOURCE_DEVELOPMENT_RUNS_VERIFIED','verified_at':datetime.now(timezone.utc).isoformat(),
 'seed':42,'same_initial_state':True,'execution_protocol_sha256':sha(config_path),'rows':rows,
 'balanced_minus_uniform':{k:rows['balanced_source_normal'][k]-rows['uniform_replacement'][k] for k in ['macro_f1','fault_recall','normal_window_fpr','original_normal_window_fpr','inactive_in_fault_case_normal_fpr','original_normal_hover_fpr']},
 'full_source_validation_predictions_and_earliest_best_rule_recomputed':True,
 'internal_test_or_Real_evaluated':False,'deployment_modified':False,'unregistered_reruns':False,
 'interpretation':'One paired seed, SOURCE validation only. Not final testing or statistical significance, not a label-only attribution against historical models. No posthoc rule changes or automatic deployment.',
 'verifier_sha256':sha(Path(__file__))}
(BASE/'verified_source_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
files.append((BASE/'verified_source_summary.json','verified_source_summary.json'))
manifest={'status':'VERIFIED_SOURCE_RESULTS_ONLY','files':{name:{'sha256':sha(p),'bytes':p.stat().st_size} for p,name in files},
 'source_dataset_tensors_in_archive':False,'test_or_Real_payloads':False}
mp=BASE/'verified_results_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');files.append((mp,'verified_results_manifest.json'))
archive=BASE/'verified_source_results.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for p,name in files:tar.add(p,arcname=name,recursive=False)
receipt={'status':'COMPLETE_VERIFIED_SOURCE_RESULTS_ARCHIVE','path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)}
(BASE/'verified_source_results_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(receipt,indent=2))
