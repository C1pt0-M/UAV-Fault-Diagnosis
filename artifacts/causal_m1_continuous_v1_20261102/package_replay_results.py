"""Archive completed source-val replays, without any training or inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile
BASE=Path('/root/autodl-tmp/causal_m1_continuous_v1_20261102')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    status=json.loads((BASE/'corrected_replays_status.json').read_text())
    assert status['status']=='COMPLETE_CORRECTED_CONTINUOUS_AND_NATURAL_REPLAYS'
    output=BASE/'packaged_replays'
    output.mkdir(exist_ok=False)
    files={}
    for name,count in [('continuous_replay_v2',72),('natural_availability_replay_v2',36)]:
        directory=BASE/name
        assert len(list(directory.glob('*.npz')))==count
        summary=json.loads((directory/'summary.json').read_text())
        assert len(summary['rows'])==count
        for path in directory.iterdir():
            assert path.is_file() and path.suffix in ['.npz','.json']
            files[str(path.relative_to(BASE))]=path
    for name in ['corrected_replays_status.json','recovered_postrun_status.json','postrun_status.json','remote_status.json','paired_results.json','postprocess_recovery_receipt.json','postprocess_recovery_manifest.json','replay_fix_receipt.json','continuous_replay_v2.log','natural_availability_replay_v2.log','recovery_postrun_runner.log','continuous_fixed18_replay.log','training_verification.log','postrun_runner.log','replay_plan.json']:
        files[name]=BASE/name
    sources=['scripts/research/replay_m1_continuous.py','scripts/research/replay_m1_natural_availability.py','scripts/research/verify_m1_training_results.py','packages/uav_fault/uav_fault/online/detection_policy.py','packages/uav_fault/uav_fault/online/event_aggregator.py','packages/uav_fault/uav_fault/online/monitoring_session.py']
    manifest={'status':'COMPLETE_FIXED_SOURCE_VAL_REPLAY_ARCHIVE','created_at':datetime.now(timezone.utc).isoformat(),
              'files':{name:{'sha256':sha(path),'bytes':path.stat().st_size} for name,path in files.items()},
              'replay_source_sha256':{name:sha(BASE/'code'/name) for name in sources},
              'formal_training_jobs_added':0,'raw_logs_or_data_tensor_caches_in_archive':False,
              'final_test_Real_or_deployment':False}
    mp=output/'manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');files['manifest.json']=mp
    archive=output/'verified_source_val_replays.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=1) as t:
        for name,path in files.items():t.add(path,arcname=name,recursive=False)
    receipt={'status':'COMPLETE_SOURCE_VAL_REPLAY_ARCHIVE','path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'NPZ_members':108,'all_members':len(files)}
    (output/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
