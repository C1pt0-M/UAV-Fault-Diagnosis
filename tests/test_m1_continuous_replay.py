from pathlib import Path
import sys
import numpy as np
import pytest
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
