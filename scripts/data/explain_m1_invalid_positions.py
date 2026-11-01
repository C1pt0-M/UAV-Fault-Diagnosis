"""Supplement per-position failure provenance without changing M1 sidecars.

Consumes only generated native clocks/flags/counters, no raw file reprocessing.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research'))
from continuous_kinematic_residuals import LocalSupportTrace
from validate_kinematic_residuals import sha
CODES={'endpoint_missing':1,'endpoint_stale':2,'endpoint_flagged':4,'endpoint_nonfinite':8,
    'interior_invalid':16,'interior_stale':32,'reset_or_device_change':64,'invalid_quaternion':128,
    'base_input_history_invalid':256,'stream_prefix':512,'unavailable_native_case':1024}
GROUPS=['baro','velocity','attitude','gyro']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args();plan=read(a.plan);root=Path(plan['output_root']);out=root/'invalid_details';out.mkdir(exist_ok=True);policy=read(Path(plan['source_root'])/'input_contract.json');results={}
    for task in plan['tasks']:
        uid=task['case_uid'];key=uid.replace('/','__');dest=out/(key+'.npz')
        if dest.exists():raise FileExistsError('Preserve per-position invalid explanations')
        receipt=read(root/'case_records'/(key+'.json'));sidefile=root/'cases'/receipt['sidecar_file'];assert sha(sidefile)==receipt['sidecar_sha256']
        with np.load(sidefile,allow_pickle=False) as d:
            n=len(d['timestamp_us']);details=np.zeros((n,4),dtype=np.uint16)
            for flag,code in [(1,512),(2,256)]:
                details[(d['reason']&flag).any(1),:]=code
            if 'baro__native_available_us' not in d:
                details[d['reason'].any(1)]|=1024
            else:
                traces={}
                for g in GROUPS:
                    prefix=g+'__native_';names=[k[len(prefix):] for k in d.files if k.startswith(prefix) and k[len(prefix):] in ['baro_device_id','z_reset_counter','vz_reset_counter','quat_reset_counter']]
                    traces[g]=LocalSupportTrace(d[prefix+'observation_us'],d[prefix+'available_us'],d[prefix+'values'],int(policy['max_age_us'][g]),{name:d[prefix+name] for name in names},d[prefix+'valid'])
                for i in np.flatnonzero((d['reason']&np.uint16(4|8|16)).any(1)):
                    end=int(d['timestamp_us'][i]);start=end-200000
                    for col,g in enumerate(GROUPS):
                        sidecol=0 if g in ['baro','velocity'] else 1
                        if not d['reason'][i,sidecol]:continue
                        tr=traces[g]
                        for t in [start,end]:
                            j=tr.index(t)
                            if j<0:details[i,col]|=1;continue
                            age=t-int(tr.observation_us[j])
                            if age<0 or age>tr.max_age_us:details[i,col]|=2
                            if not tr.valid[j]:details[i,col]|=4
                            if not np.isfinite(tr.values[j]).all():details[i,col]|=8
                        try:
                            _,_,support,changed=tr.support(start,end)
                            if changed:details[i,col]|=64
                            if g=='attitude':
                                with np.errstate(over='ignore',invalid='ignore'):
                                    q=tr.values[support];norm=np.linalg.norm(q,axis=1)
                                if not np.isfinite(q).all() or not np.isfinite(norm).all() or (norm<1e-8).any():details[i,col]|=128
                        except ValueError as exc:
                            if 'interior_invalid' in str(exc):details[i,col]|=16
                            if 'stale_gap' in str(exc):details[i,col]|=32
            invalid=d['reason'].any(1)
            if not (details[invalid].any(1)).all():raise RuntimeError('Unexplained invalid residual:'+uid)
            np.savez_compressed(dest,timestamp_us=d['timestamp_us'],topic_reason_bits=details)
        results[dest.relative_to(root).as_posix()]={'sha256':sha(dest),'bytes':dest.stat().st_size,'source_sidecar_sha256':receipt['sidecar_sha256']}
    manifest={'status':'COMPLETE_EXPLICIT_EVERY_INVALID_GRID_PROVENANCE','groups':GROUPS,'reason_codes':CODES,'files':results,'old_residual_values_or_masks_modified':False}
    p=root/'invalid_details_manifest.json'
    with p.open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    print('INVALID_DETAILS_COMPLETE',len(results),flush=True)


if __name__=='__main__':main()
