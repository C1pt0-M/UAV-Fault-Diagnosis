"""Collect existing M3 outputs into a new archive; never executes a model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile

base = Path('/root/autodl-tmp/M3_fixed_anchor_v1_20261103')
output = base/'collection_v1'
output.mkdir(exist_ok=False)
files = []
for variant in ['C0','Ccap','CO','CO9','CV','CR','CN','CON']:
    run = base/'formal/runs'/variant
    for name in ['initial.pt','best.pt','last.pt','report.json','history.json','progress.json','best_validation_predictions.npz']:
        files.append((run/name, 'runs/'+variant+'/'+name))
    files.append((base/'formal'/(variant+'.log'), 'logs/'+variant+'.log'))
for relative in ['formal/status.json','formal/execution_started.lock','formal_runner.log','gate.json','gate.log']:
    files.append((base/relative, relative))
for relative in ['configs/experiments/M3_preflight_v1.json','configs/experiments/M3_preflight_v1.json.sha256','configs/experiments/M3_fixed_anchor_execution_v1.json','configs/experiments/M3_fixed_anchor_execution_v1.json.sha256','configs/initialization/m3_adapter_seed42_v1.pt','artifacts/M3_preflight_v1_20261103/source_scaling_final/normalization.json','artifacts/M3_preflight_v1_20261103/sampler_cpu_check.json','scripts/research/train_causal_m3.py','scripts/research/execute_m3_formal.py','scripts/research/train_source_normal_sampling.py','packages/uav_fault/uav_fault/models/cnn/mstca_m3.py','packages/uav_fault/uav_fault/data/m3_features.py']:
    files.append((base/'code'/relative, 'code/'+relative))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

missing = [str(path) for path, name in files if not path.is_file()]
if missing:
    raise FileNotFoundError(repr(missing))
manifest = {'status':'COLLECTED_EXISTING_EIGHT_M3_FORMAL_RESULTS',
            'collected_at':datetime.now(timezone.utc).isoformat(), 'remote_root':str(base),
            'files':{name:{'sha256':sha(path),'bytes':path.stat().st_size,'remote_path':str(path)} for path,name in files},
            'training_started_in_collection':False, 'Test_Real_used':False}
manifest_path = output/'source_manifest.json'
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
archive = output/'M3_results.tar.gz'
with tarfile.open(archive,'x:gz',compresslevel=2) as target:
    for path, name in files:
        target.add(path,arcname=name,recursive=False)
    target.add(manifest_path,arcname='source_manifest.json',recursive=False)
receipt = {'archive_path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'members':len(files)+1}
(output/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+chr(10),encoding='utf-8')
print(json.dumps(receipt,indent=2))
