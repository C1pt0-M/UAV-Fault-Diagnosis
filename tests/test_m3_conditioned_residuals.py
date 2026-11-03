"""Minimal checks for the single frozen conditional residual reference protocol."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
from validate_m3_conditioned_residuals import case_weights, weighted_quantile, fit_reference, score_reference, case_balanced_auc
from collect_m3_formal_results import observed_states


def test_case_weights_and_quantiles_do_not_count_long_cases_as_more_cases():
    weights=case_weights(['a','a','b'])
    assert weights.tolist()==[.5,.5,1.]
    assert weighted_quantile([0,0,10],weights,.5)==0
    assert weighted_quantile([0,0,10],weights,.75)==10
    with pytest.raises(ValueError):weighted_quantile([np.nan],[1],.5)


def test_reference_is_two_sided_and_boundary_tie_is_normal():
    ref=np.tile([-1,0,2],(4,1)).astype(float)
    scores,_=score_reference(np.array([[0]*4,[-1]*4,[2]*4,[2.01]*4]),ref)
    assert scores.tolist()[:3]==[0,1,1] and scores[3]>1
    with pytest.raises(ValueError,match='Degenerate'):fit_reference(np.zeros((5,4)),['a']*5,[.005,.5,.995])


def test_case_balanced_auc_ties_and_causal_state_has_no_metadata_input():
    assert case_balanced_auc([0,1],[.5,.5],['a','b'])==.5
    assert case_balanced_auc([0,0,1],[0,0,1],['a','a','b'])==1
    assert case_balanced_auc([1],[1],['a']) is None
    columns=['vel_x','vel_y','vel_z','gyro_x','gyro_y','gyro_z']
    raw=np.zeros((3,6,150));raw[1,2]=-1;raw[2,3]=.4
    assert observed_states(raw,columns).tolist()==['low_motion_proxy','climb_proxy','maneuver_proxy']
