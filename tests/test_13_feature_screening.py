from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts/research'),str(Path(__file__).resolve().parents[1]/'packages/uav_fault')]
from analyze_13_feature_screening import IDS, observed_motion, corr, balanced_auc
from uav_fault.models.cnn.lstm import RFLY_SEQ_COLUMNS_V2


def physical(vz=0.,vx=0.,gyro=0.):
    rows=np.zeros((150,26))
    rows[:,RFLY_SEQ_COLUMNS_V2.index('vel_z')]=vz
    rows[:,RFLY_SEQ_COLUMNS_V2.index('vel_x')]=vx
    rows[:,RFLY_SEQ_COLUMNS_V2.index('gyro_z')]=gyro
    return rows


def test_motion_proxy_uses_only_available_past_physical_signals():
    assert observed_motion(physical(vz=-1.),RFLY_SEQ_COLUMNS_V2)['observed_state']=='climb_proxy'
    assert observed_motion(physical(vz=1.),RFLY_SEQ_COLUMNS_V2)['observed_state']=='descent_proxy'
    assert observed_motion(physical(),RFLY_SEQ_COLUMNS_V2)['observed_state']=='low_motion_proxy'
    assert observed_motion(physical(vx=3.),RFLY_SEQ_COLUMNS_V2)['observed_state']=='translation_proxy'
    assert observed_motion(physical(gyro=.4),RFLY_SEQ_COLUMNS_V2)['observed_state']=='maneuver_proxy'
    with pytest.raises(ValueError):observed_motion(physical()[:140],RFLY_SEQ_COLUMNS_V2)


def test_two_case_correlations_are_missing_instead_of_perfect_evidence():
    frame=pd.DataFrame({name:[float(i),float(i+1)] for i,name in enumerate(IDS)})
    assert corr(frame).isna().all().all()
    frame=pd.DataFrame({name:np.arange(6,dtype=float)*(i+1) for i,name in enumerate(IDS)})
    frame['N4']=3.
    c=corr(frame)
    assert c.loc['O1','O2']==pytest.approx(1.) and np.isnan(c.loc['N4','N4'])


def test_fault_auc_keeps_direction_and_equal_normal_case_weight():
    frame=pd.DataFrame({'case_uid':['a','a','a','b','f','f'],'O1':[0.,0.,0.,2.,1.,1.]})
    assert balanced_auc(frame,'O1',[4,5],[0,1,2,3])==pytest.approx(.5)
    # Positive is lower than all controls. No post-hoc sign flip.
    frame['O1']=[2.,2.,2.,3.,1.,1.]
    assert balanced_auc(frame,'O1',[4,5],[0,1,2,3])==0.
