from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path('D:/UAV-Fault-Diagnosis');OUT=Path('D:/UAV-Data/causal_input_v1');ART=ROOT/'artifacts/causal_input_full_v1_20261031'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

for _ in range(600):
    if (ART/'BUILD_FAILED.json').exists():raise RuntimeError('Build not complete; do not publish')
    if (ART/'ALL_VERIFIED.json').exists():break
    time.sleep(5)
else:raise RuntimeError('Verification incomplete; cannot mark dataset complete')
if (OUT/'build_manifest.json').exists():raise FileExistsError('Completed version already published')
parts={s:read(OUT/(s+'_cache_complete.json')) for s in ['train','val','test','real']}
checks={s:read(ART/('verification_'+s+'.json')) for s in ['train','val','test','real']}
assert all(v['status']=='PASS_FULL_CAUSAL_SPLIT' for v in checks.values())
assert sum(v['cases'] for v in parts.values())==4153
assert read(ART/'verification_norm.json')['status']=='PASS_SOURCE_TRAIN_ONLY_NORM'
assert read(ART/'verification_parity.json')['status']=='PASS_FULL_VERSION_SOURCE_STREAM_PARITY'
run=read(OUT/'RUNNING.json')
for path,expected in run['identities'].items():assert sha(path)==expected,path
assert sha(OUT/'tasks.json')==run['tasks_sha256']
source_train=parts['train'];assert source_train['original_windows']==168800
assert source_train['excluded_windows']>=3680
norm=read(OUT/'norm_stats.json');assert norm['fit_windows']==source_train['supervised_windows'] and norm['fit_split']=='source_train_only'
report={'status':'COMPLETE_FULL_FROZEN_SCOPE_CAUSAL_PUBLICATION_REPLAY_DATA_WITH_EXCLUSIONS',
    'version':run['version'],'completed_at':datetime.now(timezone.utc).isoformat(),'cases':4153,
    'splits':parts,'normalization':{'file':str(OUT/'norm_stats.json'),'sha256':sha(OUT/'norm_stats.json'),
        'fit_windows':norm['fit_windows'],'fit_frame_contributions':norm['fit_frame_contributions'],'scope':'eligible original cap64 source train only'},
    'source_input_identities':run['identities'],'tasks_sha256':run['tasks_sha256'],
    'legacy_suffix_windows_excluded':3680,'all_original_windows_accounted_for':True,
    'original_raw_or_labeltime_assets_overwritten':False,'physical_source_files_deleted':False,
    'no_future_selection_verified':True,'input_and_label_masks_separate':True,
    'real_roles_preserved':True,'training_or_model_scoring_run':False,'deployment_enabled':False,
    'live_transport_receive_time_certified':False,'engineering_freshness_limits_frozen_not_score_tuned':True,
    'scope_warning':'Complete for 4153 frozen label-reviewed 26-channel cases plus explicit exclusion ledger for 3680 historical suffix windows. Not all 6605 raw cases, open-set, 28 out-of-role Real, 31-channel offline/deployment or physical real-time transport certification.',
    'verification_reports_sha256':{p.name:sha(p) for p in ART.glob('verification_*.json') if p.name not in ['verification_execution.json']},
    'code_sha256':{p:sha(ROOT/p) for p in ['scripts/data/build_full_causal_input.py','scripts/research/verify_full_causal_input.py','packages/uav_fault/uav_fault/data/causal_cache.py',
        'packages/uav_fault/uav_fault/causal_pipeline.py','packages/uav_fault/uav_fault/causal_input.py']}}
with (OUT/'build_manifest.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
shutil.copyfile(OUT/'build_manifest.json',ART/'completion.json')
snap=ART/'source_snapshots';snap.mkdir(exist_ok=False)
for name in report['code_sha256']:shutil.copyfile(ROOT/name,snap/Path(name).name)
print(json.dumps({'status':report['status'],'cases':4153,'splits':{k:{f:v[f] for f in ['original_windows','supervised_windows','excluded_windows','entirely_excluded_cases']} for k,v in parts.items()},'legacy_suffix_excluded':3680,'model_training':False,'live_transport_certified':False},ensure_ascii=False,indent=2))
