from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
from train_causal_source_v1 import source_metadata,load_protocol,run
from train_source_normal_sampling import selection_step
from smoke_normal_source_sampling import sampling_weights,probability_table,ORIGINAL,PREFAULT,FAULT


def test_current_sampling_preserves_class_mass_and_balances_normal_origins():
    labels=np.array([0,0,0,0,1,1]);origins=np.array([ORIGINAL,PREFAULT,PREFAULT,PREFAULT,FAULT,FAULT])
    uniform=sampling_weights(labels,origins,False);balanced=sampling_weights(labels,origins,True)
    a=probability_table(uniform,labels,origins);b=probability_table(balanced,labels,origins)
    assert np.isclose(a['normal_probability'],b['normal_probability']) and np.isclose(a['fault_probability'],b['fault_probability'])
    assert np.isclose(b['origins'][ORIGINAL]['probability'],b['origins'][PREFAULT]['probability'])
    assert balanced.tolist()==[2.,2/3,2/3,2/3,1.,1.]


def make_links(root):
    (root/'ledgers').mkdir()
    frame=pd.DataFrame({'case_uid':['SIL/hover/no_fault/caseA','HIL/circling/motor/caseB','HIL/circling/motor/caseB'],
        'compact_index':[0,1,2],'original_cache_index':[0,4,7],'association_group':['000000001','000000002','000000002'],
        'domain':['SIL','HIL','HIL'],'split':['train']*3,'role':['train']*3,'input_valid':[True]*3,
        'label_valid':[True]*3,'supervised_valid':[True]*3,'qualified_y':[0,0,1],
        'case_level2_label':['normal','motor','motor'],
        'normal_origin':['normal_flight','normal_part_of_fault_case','fault_or_unknown']})
    frame.to_csv(root/'ledgers/train_included_windows.csv',index=False)
    class Data:
        labels=[0,0,1]
        original_cache_indices=torch.tensor([0,4,7])
        def __len__(self):return 3
    return frame,Data()


def test_metadata_uses_new_labels_and_preserves_pair_group_strings(tmp_path):
    frame,ds=make_links(tmp_path);got=source_metadata(tmp_path,'train',ds)
    assert got.normal_origin.tolist()==[ORIGINAL,PREFAULT,FAULT]
    assert got.association_group.tolist()==['000000001','000000002','000000002']
    assert got.flight_status.tolist()==['hover','circling','circling']
    assert got.label.tolist()==[0,0,1]


def test_unknown_or_ineligible_metadata_rejected_before_forward(tmp_path):
    frame,ds=make_links(tmp_path);frame.loc[1,'supervised_valid']=False
    frame.to_csv(tmp_path/'ledgers/train_included_windows.csv',index=False)
    with pytest.raises(ValueError,match='Invalid sample'):
        source_metadata(tmp_path,'train',ds)


def test_wrong_source_original_indices_rejected(tmp_path):
    frame,ds=make_links(tmp_path);ds.original_cache_indices=torch.tensor([0,3,7])
    with pytest.raises(ValueError,match='Original cache IDs'):
        source_metadata(tmp_path,'train',ds)


def test_protocol_hash_lock_prevents_unregistered_overrides(tmp_path):
    p=tmp_path/'config.json';p.write_text(json.dumps({'protocol_name':'causal_source_development_v1','data_version':'rflymad_causal_publication_replay_v1'}))
    p.with_suffix('.json.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
    assert load_protocol(p)['protocol_name']=='causal_source_development_v1'
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='identity mismatch'):load_protocol(p)


def test_formal_no_gpu_rejected_before_data_access_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    args=SimpleNamespace(output=tmp_path/'run',arm='uniform_replacement',seed=42,cpu_check=False)
    with pytest.raises(RuntimeError,match='BLOCKED_NO_CUDA'):
        run(args,{'arms':['uniform_replacement'],'seeds':[42]})
    assert not args.output.exists()


def test_selection_earliest_tie_and_small_strict_improvement_checkpoint():
    rule={'min_epochs':2,'patience':2,'significant_delta':.002}
    state={'best_score':-float('inf'),'best_epoch':None,'significant_reference':-float('inf'),'bad_epochs':0}
    state,best,stop=selection_step(state,.8,1,rule);assert best and not stop
    state,best,stop=selection_step(state,.8,2,rule);assert not best and state['best_epoch']==1 and not stop
    state,best,stop=selection_step(state,.801,3,rule)
    assert best and state['best_epoch']==3 and stop
    with pytest.raises(FloatingPointError):selection_step(state,float('nan'),4,rule)
