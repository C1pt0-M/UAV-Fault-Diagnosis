from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
from uav_fault.models.cnn.mstca_m3 import build_m3_model,load_fixed_adapter
from uav_fault.models.cnn.lstm import OnlineMSTCA
from uav_fault.data.m3_features import M3CandidateDataset,old_candidate_features
from train_causal_m3 import forward,metric_report
from train_source_normal_sampling import state_digest
from replay_causal_m3 import candidate_window


def test_fixed_adapter_loaded_even_with_different_global_seed():
    torch.set_num_threads(2)
    common=OnlineMSTCA(in_channels=26).state_dict()
    for name in ['Ccap','CO','CO9','CV','CR','CN','CON']:
        torch.manual_seed(791)
        model=build_m3_model(name,common)
        assert state_digest(model.adapter.state_dict())=='1edd07581437d47f930040d325d6a22ee34a9a28b095d9c8c40703ca134cebe2'
        assert all(torch.equal(v,model.adapter.state_dict()[k]) for k,v in load_fixed_adapter().items())


def test_vectorized_original_window_matches_scalar_and_O8_reanchors():
    ds=object.__new__(M3CandidateDataset)
    x=torch.randn(5,26,150)
    labels=torch.tensor([0,1,0,1,0]);indices=torch.tensor([1,2,4])
    base=SimpleNamespace(_x=x,_mean=torch.zeros(26,1),_std=torch.ones(26,1),_y=labels,_selected=np.arange(5))
    ds.source=SimpleNamespace(base=base,base_indices=indices,residual=torch.randn(3,4,150),mu=torch.zeros(4,1),std=torch.ones(4,1))
    ds.mean=torch.zeros(9,1);ds.std=torch.ones(9,1)
    batch=ds.__getitems__([2,0,2])
    for result,index in zip(batch,[2,0,2]):
        raw=x[indices[index]];old=old_candidate_features(raw[None])[0]
        assert torch.equal(result[0],raw)
        assert torch.equal(result[1][:9],old)
        assert torch.equal(result[1][9:],ds.source.residual[index])
        assert old[7,0]==raw[21,0].abs()  # Original vel_z column21; diff first=0.
    norm=dict(mean=[0.]*13,std=[1.]*13)
    assert torch.equal(candidate_window(batch[0][0],batch[0][1][9:].numpy(),norm),batch[0][1])


def test_nonfinite_logits_stop_before_loss_or_update():
    class Invalid(torch.nn.Module):
        def forward(self,x):return torch.full((len(x),2),float('inf'))
    with pytest.raises(FloatingPointError,match='logits'):
        forward(Invalid(),'C0',torch.zeros(2,26,150),torch.zeros(2,13,150))
    with pytest.raises(FloatingPointError,match='all13'):
        forward(Invalid(),'C0',torch.zeros(2,26,150),torch.full((2,13,150),float('nan')))


def test_six_fault_and_domain_metrics_include_actual_counts():
    subtypes=['GPS','accelerometer','barometer','gyroscope','magnetometer','motor']
    frame=pd.DataFrame(dict(label=[0,0]+[1]*6,normal_origin=['original_normal_case']*2+['fault_endpoint']*6,
        flight_status=['hover']*8,case_level2_label=['normal']*2+['sensor']*5+['motor'],
        case_uid=['case'+str(i) for i in range(8)],domain=['SIL','HIL']+['SIL','HIL']*3,
        fault_directory=['no_fault']*2+subtypes,window_end_row=[150]*8))
    probabilities=np.array([[.8,.2],[.2,.8]]+[[.2,.8]]*6)
    result=metric_report(frame.label.to_numpy(),probabilities,frame)
    assert set(result['six_faults'])==set(subtypes)
    assert set(result['domains'])=={'SIL','HIL'}
    assert result['fault_precision']==6/7
    assert result['overall']['fault_recall']==1.
