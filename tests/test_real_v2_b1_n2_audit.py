"""Synthetic-only N2 audit checks. No datasets, saved scores or models."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
import audit_real_v2_b1_n2 as a


def test_existing_fixed_rule_reconstruction():
    t=np.arange(1500,dtype=np.int64)*20000;p=np.zeros(len(t));p[100:201]=.95;p[1100:1201]=.95
    arrays={'timestamp_us':t,'row_label_valid':np.ones(len(t),bool),'reference_state':(t>=20000000).astype(np.int8)}
    record,events=a.recover_events({'case_uid':'synthetic','role_v1':'test_fault'},arrays,p,.85)
    assert record['N2_events']==1 and record['detected'] and len(events)==1
    assert events[0]['start_t']==2. and events[0]['confirmed_t']==2.04 and events[0]['end_t']==4.


def test_event_reconstruction_honors_unscored_gaps():
    t=np.arange(1500,dtype=np.int64)*20000;p=np.zeros(len(t));p[100:201]=.95;p[150]=np.nan
    arrays={'timestamp_us':t,'row_label_valid':np.ones(len(t),bool),'reference_state':(t>=20000000).astype(np.int8)}
    record,events=a.recover_events({'case_uid':'synthetic','role_v1':'test_fault'},arrays,p,.85)
    assert record['N2_events']==2 and len(events)==2
    assert all(not (e['start_t']<=3<=e['end_t']) for e in events)


def test_overlap_components_preserve_one_to_many():
    rows=[{'event_id':'R1','case_uid':'x','arm':'R','start_s':0.,'end_s':4.},
          {'event_id':'S1','case_uid':'x','arm':'S_plus_R','start_s':1.,'end_s':2.},
          {'event_id':'S2','case_uid':'x','arm':'S_plus_R','start_s':3.,'end_s':4.},
          {'event_id':'R2','case_uid':'x','arm':'R','start_s':6.,'end_s':7.}]
    out=a.overlap_components(rows)
    assert len(out)==2 and out[0]['shared'] and out[0]['R']==1 and out[0]['S_plus_R']==2
    assert not out[1]['shared']


def test_channel_derivatives_do_not_bridge_invalid_rows():
    t=np.arange(70,dtype=np.int64)*20000;x=np.zeros((70,26));x[:,15]=1.;x[36:,0]=100.;valid=np.ones(70,bool);valid[34:37]=False
    arrays={'timestamp_us':t,'canonical_rows':x,'row_input_valid':valid}
    columns=[str(i) for i in range(26)];v=a.sample_metrics(arrays,(0.,2.),columns)
    assert v['0']['median_abs_delta']==0 and v['0']['zero_delta_fraction']==1
    assert v['quaternion_norm']['median']==1.


def test_insufficient_support_stays_null():
    assert a.summarize(np.arange(24))['median'] is None
    assert a.summarize(np.arange(25))['median']==12.


def test_phase_mapping_in_integer_microseconds_keeps_boundary_rows():
    t=np.array([1000000,1020000,1040000],dtype=np.int64)
    np.testing.assert_array_equal(a.interval_mask_us(t,(1.+1e-15,1.04-1e-15)),[True,True,True])


def test_injection_id_semantics_reused():
    assert a.active_ids([0,1500,123450,123549,123550]).tolist()==[False,False,True,True,False]
    _,_,policy=a.original_functions()
    assert policy==dict(inference_every=1,clear_th=.4,k_on=3,k_off=3,min_event_s=.5,merge_gap_s=.5)
