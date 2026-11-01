"""Freeze M1 training config only after full continuous feature delivery.

No GPU jobs, installations or old asset modifications. One frozen protocol.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/uav_fault'));sys.path.insert(0,str(ROOT/'scripts/research'))
from train_source_normal_sampling import state_digest
from validate_kinematic_residuals import sha


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve frozen M1 protocol')
    oldpath=ROOT/'configs/experiments/causal_pure_mstca_m0_v1.json';old=read(oldpath);manifest=read(a.features/'build_manifest.json')
    assert manifest['status']=='COMPLETE_SOURCE_M1_CONTINUOUS150_FEATURES_PENDING_TRAINING_GATE'
    config={'protocol_name':'causal_m1_continuous150_v1','status':'FROZEN_BEFORE_GPU_FORMAL_UPDATES_AWAITING_GATES','created_at':datetime.now(timezone.utc).isoformat(),
        'M0_protocol_path':oldpath.relative_to(ROOT).as_posix(),'M0_protocol_sha256':sha(oldpath),'feature_manifest_sha256':sha(a.features/'build_manifest.json'),
        'window_frames':150,'horizon_us':200000,'columns':manifest['columns'],'seeds':[42],'arms':old['arms'],
        'training':deepcopy(old['training']),'loss':old['loss'],'validation':dict(old['validation'],population='All frozen full150-continuous eligible source val original IDs, no sampling/reweighting; original34710 denominator and exclusions reported'),'formal_environment':old['formal_environment'],
        'data_counts':{s:manifest['exports'][s]['windows'] for s in ['train','val']},
        'strict_population':'Same full150 valid original IDs for M1 and M0-matched; no refill. Coverage report gates formal launch.',
        'normalization':'Unchanged base26 stats; independent source_train_normal_only residual stats in feature manifest.',
        'model':{'implementation':'OnlineMSTCAKinematicM1','version':'mstca_kinematic_continuous200ms_m1_v1','base_parameters':105938,'adapter_parameters':3168,'total_parameters':109106,'architecture':'Existing26 MSTCA encoder;4->16 k3 ReLU ->16 k3 ReLU ->pool ->linear128; single additive fusion before existing head','adapter_initialization':'seed42 default new convolution weights, zero final projection; initial eval function equals common pretrained M0'},
        'common_M0_pretrained':{},'formal_jobs':[],
        'initialization_interpretation':'Matched M0 and M1 both fine-tune from same arm-specific historical M0 best. New AdamW, no optimizer continuation. Not a from-scratch or unexposed-val comparison.',
        'maximum_formal_jobs':4 if manifest['M0_matched_required'] else 2,
        'code_identities':dict(old['code_identities']),
        'forbidden':['Change frozen base stats, labels, split, source masks or old assets','Use final-test/Real to fit, select or tune','Zero/mean fill unavailable residuals','Unregistered retries, seeds, spans or models','Start GPU before PASS_ALL_M1_PRETRAINING_GATES'],
        'replay_plan_path':'artifacts/causal_m1_continuous_v1_20261102/replay_plan.json',
        'coverage':manifest['coverage']}
    config['training']['parameters']=109106;config['training']['initialization']=config['initialization_interpretation'];config['training']['draws_per_epoch']=config['data_counts']['train']
    for arm in old['arms']:
        path=ROOT/'artifacts/causal_pure_mstca_m0_v1_20261101/results/runs'/arm/'best.pt';r=read(path.parent/'report.json');assert sha(path)==r['best_checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        config['common_M0_pretrained'][arm]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'state_digest':state_digest(state),'epoch':r['best_epoch'],'trained':True}
        if manifest['M0_matched_required']:config['formal_jobs'].append({'variant':'M0_matched','arm':arm})
        config['formal_jobs'].append({'variant':'M1','arm':arm})
    for name in ['scripts/research/train_causal_m1_continuous.py','packages/uav_fault/uav_fault/data/m1_continuous.py','packages/uav_fault/uav_fault/models/cnn/mstca_kinematic_m1.py',
        'scripts/research/continuous_kinematic_residuals.py','scripts/data/build_m1_continuous_features.py','scripts/data/finalize_m1_continuous_features.py',
        'scripts/data/explain_m1_invalid_positions.py']:
        config['code_identities'][name]=sha(ROOT/name)
    config['replay_plan_sha256']=sha(ROOT/config['replay_plan_path'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(config,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
    a.output.with_suffix('.json.sha256').write_text(sha(a.output)+chr(10),encoding='utf-8')
    print(json.dumps({'status':'M1_PROTOCOL_FROZEN_AWAITING_PRETRAIN_GATES','sha256':sha(a.output),'jobs':config['formal_jobs'],'data_counts':config['data_counts'],'coverage':config['coverage']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
