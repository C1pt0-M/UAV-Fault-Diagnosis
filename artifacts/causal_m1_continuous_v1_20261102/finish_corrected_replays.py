"""Replay saved best models after correcting JSON scalar and gap timing only."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import traceback
BASE=Path('/root/autodl-tmp/causal_m1_continuous_v1_20261102')
CODE=BASE/'code'
ENV=dict(os.environ,PYTHONPATH=str(CODE/'packages/uav_fault'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')


def main():
    with (BASE/'corrected_replays_started.lock').open('x') as f:f.write(str(os.getpid()))
    assert json.loads((BASE/'verified_training/summary.json').read_text())['status']=='PASS_REGISTERED_M1_AND_MATCHED_SOURCE_RESULTS'
    state={'status':'CONTINUOUS_REPLAY','started_at':datetime.now(timezone.utc).isoformat(),'training_calls':0,'prior_failed_outputs_preserved':True}
    def save():
        p=BASE/'corrected_replays_status.partial.json';p.write_text(json.dumps(state,indent=2));p.replace(BASE/'corrected_replays_status.json')
    try:
        common=['--plan',BASE/'replay_plan.json','--data',BASE/'replay_base','--features',BASE/'features','--runs',BASE/'runs']
        for name,script,args in [
            ('CONTINUOUS_REPLAY','replay_m1_continuous.py',['--output',BASE/'continuous_replay_v2']),
            ('NATURAL_AVAILABILITY_REPLAY','replay_m1_natural_availability.py',['--main-replay',BASE/'continuous_replay_v2','--output',BASE/'natural_availability_replay_v2'])]:
            state['status']=name;save()
            cmd=['/root/miniconda3/bin/python','-u',str(CODE/'scripts/research'/script),*map(str,common+args),'--device','cuda:0']
            with (BASE/(name.lower()+'_v2.log')).open('x') as log:subprocess.run(cmd,cwd=CODE,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status='COMPLETE_CORRECTED_CONTINUOUS_AND_NATURAL_REPLAYS',completed_at=datetime.now(timezone.utc).isoformat());save()
    except Exception as exc:
        state.update(status='FAILED_CORRECTED_REPLAY',error=repr(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
