from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from audit_real_labeltime import hold_integer, read_export, selected_ends


def test_unix_nanoseconds_preserve_last_duplicate_and_unknown_coverage():
    origin=1686794382671454381
    t=np.array([origin,origin+1,origin+1,origin+30],dtype=np.int64)
    q=np.array([origin-1,origin,origin+1,origin+10,origin+30,origin+31],dtype=np.int64)
    state,valid=hold_integer(t,np.array([0,0,1,1]),q)
    assert state.tolist()==[-1,0,1,1,1,-1]
    assert valid.tolist()==[False,True,True,True,True,False]


@pytest.mark.parametrize('t,q',[(np.array([2,1]),np.array([1,2])),
                              (np.array([1.,2.]),np.array([1,2])),
                              (np.array([],dtype=np.int64),np.array([1])),
                              (np.array([1,2]),np.array([2,1]))])
def test_invalid_clocks_rejected(t,q):
    with pytest.raises(ValueError):hold_integer(t,np.zeros(len(t)),q)


def test_export_nanoseconds_not_parsed_through_float(tmp_path):
    p=tmp_path/'times.csv'
    p.write_text('rosbagTimestamp,id'+chr(10)+'1686794382671454381,123450'+chr(10))
    data=read_export(p,['rosbagTimestamp','id'])
    assert int(data['rosbagTimestamp'][0])==1686794382671454381


def test_cap_and_stride_do_not_change_crop_or_create_outside_windows():
    for n in [150,155,250,3500]:
        cap=selected_ends(n,'cap64');replay=selected_ends(n,'stride10')
        assert 1<=len(cap)<=64 and cap[0]==149 and cap[-1]<n
        assert np.all((cap-149)%5==0)
        assert replay[0]==149 and replay[-1]<n and np.all((replay-149)%10==0)


def test_no_extrapolation_of_last_fault_state():
    state,valid=hold_integer(np.array([1,2]),np.array([0,1]),np.array([1,2,3]))
    assert state.tolist()==[0,1,-1] and valid.tolist()==[True,True,False]
