from pathlib import Path
import sys
import json
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
from replay_m1_continuous import aggregate_scores,stream_check
from continuous_kinematic_residuals import compute_continuous
from test_continuous_kinematic_residuals import fixture


def test_actual_default_policy_confirmed_separate_from_final_emitted():
    t=np.arange(101)*.02;result=aggregate_scores(t,np.full(len(t),.8))
    assert result['policy']['inference_every']==1 and result['policy']['k_on']==3
    assert result['confirmations'][0]['candidate_start_t']==0
    assert result['confirmations'][0]['confirmed_t']==.04
    assert result['finalized_events'][0]['emitted_t']==2.
    assert result['finalized_events'][0]['emission_reason']=='end_of_stream_flush'
    assert result['finalized_events'][0]['confirmed_t']!=result['finalized_events'][0]['emitted_t']


def test_gap_does_not_join_two_high_points_into_confirmation():
    t=np.arange(80)*.02;s=np.zeros(80);s[:2]=.8;s[2]=np.nan;s[3:5]=.8
    result=aggregate_scores(t,s)
    assert result['confirmations']==[] and result['finalized_events']==[]


def test_min_duration_and_actual_merge_delay_are_not_shortened():
    t=np.arange(100)*.02;s=np.zeros(100);s[:10]=.8
    result=aggregate_scores(t,s)
    assert len(result['confirmations'])==1 and result['finalized_events']==[]
    s[:40]=.8;result=aggregate_scores(t,s)
    assert len(result['finalized_events'])==1
    assert result['finalized_events'][0]['emitted_t']>result['finalized_events'][0]['end_t']+.5
    assert result['finalized_events'][0]['emission_reason']=='online_update'


def test_flush_emission_uses_current_gap_time_and_end_of_record():
    t=np.arange(150)*.02;s=np.full(150,np.nan);s[:50]=.8
    result=aggregate_scores(t,s)
    event=result['finalized_events'][0]
    assert event['end_t']==.98
    assert event['emitted_t']==1.
    assert event['deployment_timing']['emitted_t']==.98
    assert event['emission_reason']=='invalid_gap_flush'
    json.dumps(result,allow_nan=False)
    # Missing frames after the final available prediction still advance the
    # replay clock; they must not backdate an already gap-flushed event.
    assert len(result['finalized_events'])==1


def test_replay_cli_writes_complete_json_with_measurable_fault(tmp_path,monkeypatch):
    import replay_m1_continuous as replay
    from uav_fault.models.cnn.lstm import RFLY_SEQ_COLUMNS_V2
    class ConstantModel(torch.nn.Module):
        def __init__(self,*args,**kwargs):super().__init__()
        def forward(self,x,*args):return torch.tensor([[0.,2.]]).expand(len(x),2)
    monkeypatch.setattr(replay,'OnlineMSTCA',ConstantModel)
    monkeypatch.setattr(replay,'OnlineMSTCAKinematicM1',ConstantModel)
    monkeypatch.setattr(replay,'stream_check',lambda side,limits:np.ones(len(side['timestamp_us']),dtype=np.int64))
    data=tmp_path/'data';features=tmp_path/'features';runs=tmp_path/'runs';out=tmp_path/'out'
    (data/'cases').mkdir(parents=True);(features/'cases').mkdir(parents=True)
    (data/'norm_stats.json').write_text(json.dumps({'columns':list(RFLY_SEQ_COLUMNS_V2),'mean':[0.]*26,'std':[1.]*26}))
    (features/'residual_norm_stats.json').write_text(json.dumps({'mean':[0.]*4,'std':[1.]*4}))
    (data/'input_contract.json').write_text(json.dumps({'max_age_us':{}}))
    n=180;grid=1000000+np.arange(n,dtype=np.int64)*20000
    cases=[]
    for kind in ['normal','fault']:
        uid='SIL/hover/'+('no_fault' if kind=='normal' else 'motor')+'/sample';key=uid.replace('/','__')+'.npz'
        np.savez(data/'cases'/key,timestamp_us=grid,canonical_rows=np.ones((n,26)),row_input_valid=np.ones(n,dtype=bool),row_label_valid=np.ones(n,dtype=bool),reference_state=np.zeros(n,dtype=np.int8))
        np.savez(features/'cases'/key,timestamp_us=grid,prehistory_timestamp_us=grid[:10],residual=np.zeros((n,4)),valid=np.ones((n,4),dtype=bool))
        cases.append({'case_uid':uid,'domain':'SIL','subtype':'no_fault' if kind=='normal' else 'motor','kind':kind,'gtd_events':[] if kind=='normal' else [{'start_s':4.}],'command_events':[]})
    for arm in ['uniform_replacement','balanced_source_normal']:
        for variant in ['M0_matched','M1']:
            p=runs/variant/arm;p.mkdir(parents=True);torch.save({},p/'best.pt')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'policy':{},'cases':cases}))
    monkeypatch.setattr(sys,'argv',['replay','--plan',str(plan),'--data',str(data),'--features',str(features),'--runs',str(runs),'--output',str(out)])
    replay.main();summary=json.loads((out/'summary.json').read_text())
    assert len(summary['rows'])==8 and not (out/'summary.partial.json').exists()
    assert sum(r['onset_measurable_in_crop'] for r in summary['rows'])==4
    json.dumps(summary,allow_nan=False)


def test_real_arrival_replay_reproduces_saved_continuous_values():
    grid,traces=fixture(80);r=compute_continuous(grid,np.ones(len(grid),dtype=bool),traces)
    side={'timestamp_us':grid[10:],'prehistory_timestamp_us':grid[:10],
        'prehistory_input_valid':np.ones(10,dtype=bool),'original_row_input_valid':np.ones(70,dtype=bool),
        'residual':np.column_stack([r['vertical_m'][10:],r['attitude_rad'][10:]])}
    for g,tr in traces.items():
        side[g+'__native_available_us']=tr.available_us;side[g+'__native_observation_us']=tr.observation_us
        side[g+'__native_values']=tr.values;side[g+'__native_valid']=tr.valid
    elapsed=stream_check(side,{g:tr.max_age_us for g,tr in traces.items()})
    assert elapsed.shape==(70,) and (elapsed>0).all()
