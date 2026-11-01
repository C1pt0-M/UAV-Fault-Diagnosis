from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from diagnose_causal_bn_modes import batch_statistics_copy


def test_batch_stats_match_native_without_changing_state_or_dropout():
    torch.manual_seed(3)
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.8), nn.Conv1d(3, 2, 1)).eval()
    model[0].running_mean.fill_(7)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    copied = batch_statistics_copy(model)
    x = torch.randn(8, 3, 20) + 2
    with torch.inference_mode():
        result = copied(x)
        expected = model[2](F.batch_norm(x, None, None, model[0].weight, model[0].bias, training=True, eps=model[0].eps))
        assert torch.equal(result, expected)
        assert torch.equal(result, copied(x))
        assert not torch.allclose(result, model(x))
    assert not model.training and not copied.training and not copied[1].training
    assert copied[0].training and not copied[0].track_running_stats
    for k, value in before.items():
        assert torch.equal(model.state_dict()[k], value)
        assert torch.equal(copied.state_dict()[k], value)
    assert all(p.grad is None for m in [model, copied] for p in m.parameters())


def test_disposable_forward_never_uses_saved_running_mean():
    model = nn.Sequential(nn.BatchNorm1d(1)).eval()
    x = torch.tensor([[[1., 2.]], [[3., 8.]]])
    first = batch_statistics_copy(model)
    model[0].running_mean.fill_(100)
    second = batch_statistics_copy(model)
    with torch.inference_mode():
        assert torch.equal(first(x), second(x))


def test_no_bn_is_rejected():
    with pytest.raises(ValueError, match='BatchNorm1d'):
        batch_statistics_copy(nn.Sequential(nn.Linear(2, 2)))
