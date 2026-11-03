"""Small CPU-only checks for physical-data statistics, not model tests."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
from diagnose_m3_physical_data import label_groups, paired_sign, native_residual_at


def test_pure_vs_mixed_labels_are_not_conflated():
    f=pd.DataFrame({'whole_window_label_valid':[True,False,True,True],
                    'qualified_y':[1,1,0,0], 'fault_fraction':[1,.9,0,0],
                    'case_level2_label':['GPS','GPS','normal','GPS']})
    assert label_groups(f).tolist()==['active','mixed_or_unknown','independent_normal','case_pre_normal']


def test_signed_pairing_does_not_flip_direction():
    assert paired_sign([0,1,2],[1,1,1])==.5
    assert paired_sign([0,0],[1,1])==0
    assert paired_sign([],[]) is None


def test_original_native_support_units_and_reset_rejection():
    times=np.array([100000,200000,300000,400000],dtype=np.int64)
    side={'timestamp_us':times}
    q=Rotation.from_rotvec(np.column_stack([np.zeros(4),np.zeros(4),times/1e6*.1])).as_quat()[:,[3,0,1,2]]
    for group,values in [('baro',np.arange(4)[:,None]*.1),('velocity',np.full((4,1),2.)),
                          ('attitude',q),('gyro',np.tile([0,0,.1],(4,1)))]:
        for name,value in [('observation_us',times),('available_us',times),('values',values),('valid',np.ones(4,dtype=bool))]:
            side[group+'__native_'+name]=value
    policy={'max_age_us':{g:1000000 for g in ['baro','velocity','attitude','gyro']}}
    residual=native_residual_at(side,policy,3)
    assert np.allclose(residual,[.6,0,0,0],atol=1e-12)
    side['attitude__native_quat_reset_counter']=np.array([0,0,1,1])
    with pytest.raises(ValueError,match='Reset'):
        native_residual_at(side,policy,3)
