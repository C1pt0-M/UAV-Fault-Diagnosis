"""Regression checks for actual fixed-crop label repair materialization."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/data'))
from repair_fixed_crop_labeltime import make_split_view, repair_csv, digest, rebuild_window_meta


def overlay():
    return {'cache_index':np.arange(5),'historical_label':np.array([1,0,1,0,0]),
            'reference_label':np.array([0,1,-1,0,0]),
            'label_reference_defined':np.array([1,1,0,1,1],dtype=bool),
            'known_input_clock_review':np.array([0,0,0,1,0],dtype=bool),
            'pre_forward_candidate':np.array([1,1,0,0,1],dtype=bool)}


def test_actual_cache_targets_repaired_and_retained_x_unchanged():
    data=overlay();x=torch.arange(30,dtype=torch.float32).reshape(5,2,3)
    y=torch.tensor(data['historical_label']);before=x.clone();old=y.clone()
    fixed,indices=make_split_view(x,y,data,block=2)
    assert indices.tolist()==[0,1,4]
    assert fixed['y'].tolist()==[0,1,0]
    assert torch.equal(fixed['x'],x[indices])
    assert torch.equal(x,before) and torch.equal(y,old)
    fixed['x'][0,0,0]=99
    assert torch.equal(x,before)


@pytest.mark.parametrize('problem',['old_label','index','allow_uncertain','clock_mask','float_labels','missing_mask'])
def test_bad_overlay_stops_before_label_materialization(problem):
    data=overlay();y=torch.tensor(data['historical_label'].copy());x=torch.zeros(5,2,3)
    if problem=='old_label':data['historical_label'][0]=0
    elif problem=='index':data['cache_index'][1]=0
    elif problem=='allow_uncertain':data['pre_forward_candidate'][2]=True
    elif problem=='clock_mask':data['pre_forward_candidate'][3]=True
    elif problem=='float_labels':data['reference_label']=data['reference_label'].astype(float)
    elif problem=='missing_mask':data['label_reference_defined']=np.ones(4,dtype=bool)
    with pytest.raises(ValueError):make_split_view(x,y,data)


def test_nonfinite_input_never_zero_filled():
    x=torch.zeros(5,2,3);x[0,0,0]=float('nan');d=overlay()
    with pytest.raises(ValueError,match='Nonfinite'):
        make_split_view(x,torch.tensor(d['historical_label']),d)


@pytest.fixture
def csv_fixture(tmp_path):
    original=tmp_path/'old.csv';ref=tmp_path/'ref.npz';target=tmp_path/'new'/'fixed.csv'
    rows=[['Timestamp','value','UAVState_data_cmd','UAVState_data_fault_state'],
          ['1000000','1.0000000000000002','2,6,123450','1'],
          ['1020000','-0.00','a,b','0'],['1040000','NaN','wait','1']]
    with original.open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerows(rows)
    np.savez(ref,timestamp_us=np.array([1000000,1020000,1040000]),existing_state=np.array([1,0,1]),
             gtd_state_reference=np.array([0,1,1]),command_state_reference=np.array([0,0,1]),
             gtd_covered=np.ones(3,dtype=bool),command_covered=np.ones(3,dtype=bool),
             source_references_agree=np.array([1,0,1],dtype=bool))
    record={'processed_csv_sha256':digest(original),'sidecar_sha256':digest(ref),'source_rows':3}
    return original,ref,target,record,rows


def test_repaired_csv_writes_state_and_validity_only(csv_fixture):
    original,ref,target,record,rows=csv_fixture;before=original.read_bytes()
    result=repair_csv(original,target,ref,record,False)
    assert original.read_bytes()==before
    with target.open(encoding='utf-8',newline='') as f:fixed=list(csv.reader(f))
    assert fixed[0]==rows[0]+['labeltime_supervision_valid','labeltime_input_clock_review']
    assert [r[3:] for r in fixed[1:]]==[['0','1','0'],['1','0','0'],['1','1','0']]
    assert [r[:3] for r in fixed[1:]]==[r[:3] for r in rows[1:]]
    assert result['changed_state_rows']==2 and result['uncertain_state_rows']==1
    assert result['nonlabel_fields_exactly_preserved'] is True
    with pytest.raises(FileExistsError):repair_csv(original,target,ref,record,False)


def test_wrong_source_hash_rejected_before_write(csv_fixture):
    original,ref,target,record,_=csv_fixture;record['processed_csv_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):repair_csv(original,target,ref,record,False)
    assert not target.exists()


def test_rebuild_metadata_uses_repaired_endpoint_and_fraction(tmp_path):
    uid='SIL/hover/motor/example';state=np.r_[np.zeros(100,dtype=int),np.ones(100,dtype=int)]
    np.savez(tmp_path/(uid.replace('/','__')+'.npz'),gtd_state_reference=state)
    meta=pd.DataFrame({'cache_index':[0,1],'reference_label':[1,1],
        'normal_origin_reference':['fault_endpoint']*2,'source_family':['motor']*2,
        'case_uid':[uid,uid],'domain':['SIL','SIL'],'window_start_row':[0,50],'window_end_row':[150,200]})
    result=rebuild_window_meta(meta,np.array([0,1]),tmp_path)
    assert result.label.tolist()==[1,1] and result.source_cache_index.tolist()==[0,1]
    assert result.fault_fraction.tolist()==pytest.approx([1/3,2/3])
    assert result.fault_onset_row.tolist()==[100,100]
    assert result.phase.tolist()==['transition','transition']


@pytest.mark.parametrize('flag',['labeltime_supervision_valid','labeltime_input_clock_review'])
def test_legacy_boolean_reader_rejects_repaired_csv_without_mask_support(flag):
    from generate_window_manifest import _infer_fault_mask
    frame=pd.DataFrame({'UAVState_data_fault_state':[0,1],flag:[1,1]})
    with pytest.raises(ValueError,match='versioned window links'):_infer_fault_mask(frame)
    assert _infer_fault_mask(frame.drop(columns=[flag])).tolist()==[False,True]


def test_resume_refuses_complete_or_unrecorded_output(tmp_path):
    from types import SimpleNamespace
    from repair_fixed_crop_labeltime import run
    out=tmp_path/'complete';out.mkdir()
    with pytest.raises(ValueError,match='incomplete recorded'):
        run(SimpleNamespace(output=out,resume=True,workers=1))
    (out/'RUNNING.json').write_text('{}')
    (out/'repair_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='incomplete recorded'):
        run(SimpleNamespace(output=out,resume=True,workers=1))
