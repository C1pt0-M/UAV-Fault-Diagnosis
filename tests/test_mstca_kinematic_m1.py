from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packages/uav_fault'))
from uav_fault.models.cnn.lstm import OnlineMSTCA
from uav_fault.models.cnn.mstca_kinematic_m1 import OnlineMSTCAKinematicM1
from uav_fault.models.cnn.focal_loss import FocalLoss


def pair():
    torch.manual_seed(42);base=OnlineMSTCA(in_channels=26,hidden_size=64,dropout=.2)
    state={k:v.detach().clone() for k,v in base.state_dict().items()}
    torch.manual_seed(42);model=OnlineMSTCAKinematicM1();model.load_common_initial(state)
    return base,model,state


def test_adapter_count_common_identity_and_exact_initial_M0_function():
    base,model,state=pair()
    assert sum(p.numel() for p in base.parameters())==105938
    assert sum(p.numel() for p in model.residual_adapter.parameters())==3168
    assert sum(p.numel() for p in model.parameters())==109106
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    x=torch.randn(3,26,150);r=torch.randn(3,4,150)
    base.eval();model.eval()
    with torch.no_grad():assert torch.equal(base(x),model(x,r))
    assert 'relation_encoder' not in dict(model.named_modules())


def test_forward_backward_and_cpu_small_training_updates_new_module():
    _,model,_=pair();x=torch.randn(8,26,150);r=torch.randn(8,4,150);y=torch.tensor([0,1]*4)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);criterion=FocalLoss(gamma=2)
    old=model.residual_adapter[0].weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);out=model(x,r);loss=criterion(out,y)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
    assert not torch.equal(old,model.residual_adapter[0].weight)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.residual_adapter.parameters())


def test_invalid_residual_is_rejected_before_batchnorm_updates():
    _,model,_=pair();state={k:v.clone() for k,v in model.state_dict().items()}
    r=torch.randn(2,4,150);r[0,1,10]=float('nan')
    with pytest.raises(ValueError,match='before M1/BatchNorm'):model(torch.randn(2,26,150),r)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())


def test_no30_raw_concat_no140_silent_change():
    _,model,_=pair()
    with pytest.raises(ValueError,match='Aligned26x150'):model(torch.randn(2,30,150),torch.randn(2,4,150))
    with pytest.raises(ValueError):model(torch.randn(2,26,140),torch.randn(2,4,140))


def test_no_partial_or_mismatched_common_state_load():
    _,model,state=pair();state.pop(next(iter(state)))
    with pytest.raises(ValueError,match='Exact pure M0'):model.load_common_initial(state)
