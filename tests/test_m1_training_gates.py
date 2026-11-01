from pathlib import Path
from types import SimpleNamespace
import json
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
import train_causal_m1_continuous as trainer


def test_formal_without_gate_stops_before_data_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer,'prepare',lambda *a:pytest.fail('Data must not be read before gates'))
    a=SimpleNamespace(output=tmp_path/'out',arm='uniform_replacement',seed=42,cpu_check=False,gate=None)
    with pytest.raises(RuntimeError,match='pretraining gate'):trainer.run(a,{'arms':['uniform_replacement']})
    assert not a.output.exists()


def test_failed_gate_stops_before_cuda_or_data(tmp_path,monkeypatch):
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'status':'FAILED'}))
    monkeypatch.setattr(trainer,'prepare',lambda *a:pytest.fail('Failed gates must stop before data'))
    a=SimpleNamespace(output=tmp_path/'out',arm='uniform_replacement',seed=42,cpu_check=False,gate=gate)
    with pytest.raises(RuntimeError,match='blocked by pretraining'):trainer.run(a,{'arms':['uniform_replacement']})
    assert not a.output.exists()


def test_zero_future_history_no_shortcuts_or_extra_horizon():
    from continuous_kinematic_residuals import HORIZON_US,COLUMNS
    assert HORIZON_US==200000 and COLUMNS==['rh_signed_m','rR_x_rad','rR_y_rad','rR_z_rad']
    from uav_fault.online.detection_policy import OnlineDetectionPolicy
    p=OnlineDetectionPolicy();assert p.k_on==3 and p.inference_every==1 and p.onset_th==.6


def test_model_common_pretrained_and_random_initialization_are_not_confused():
    from uav_fault.models.cnn.mstca_kinematic_m1 import OnlineMSTCAKinematicM1
    from uav_fault.models.cnn.lstm import OnlineMSTCA
    torch.manual_seed(42);base=OnlineMSTCA(in_channels=26)
    with torch.no_grad():base.head[1].weight.add_(.25)
    model=OnlineMSTCAKinematicM1();model.load_common_initial(base.state_dict())
    assert torch.equal(model.head[1].weight,base.head[1].weight)
    assert torch.count_nonzero(model.residual_adapter[-1].weight)==0
