"""Independent M3 adapter for controlled preflight; no training entrypoint."""
from __future__ import annotations
from collections.abc import Mapping
import torch
from torch import nn
from .lstm import OnlineMSTCA
from ...data.m3_features import CANDIDATE_IDS
from ...model_contract import RFLY_SEQ_COLUMNS_V2

VARIANTS = {
    'C0': (), 'Ccap': (),
    'CO': ('O1', 'O3', 'O6', 'O7', 'O8', 'O9'),
    'CO9': tuple(f'O{i}' for i in range(1, 10)),
    'CV': ('N1',), 'CR': ('N2', 'N3', 'N4'),
    'CN': ('N1', 'N2', 'N3', 'N4'),
    'CON': ('O1', 'O3', 'O6', 'O7', 'O8', 'O9', 'N1', 'N2', 'N3', 'N4')}
CCAP_COLUMNS = tuple([f'actuator_ctrl_{i}' for i in range(1, 5)]+[f'actuator_out_{i}' for i in range(1, 5)]+
                     ['gyro_x', 'gyro_y', 'gyro_z', 'baro_alt_relative', 'acc_norm'])


class M3AdapterMSTCA(nn.Module):
    """Aligned13-slot auxiliary sequence; one addition before original head.

    Selector zeroes deliberately excluded *valid* candidates. All13 validity is
    checked first; selector never fills missing observations. Ccap uses the same
    module but its13 ordered inputs are selected normalized base channels.
    """
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS or variant == 'C0':
            raise ValueError('C0 is original OnlineMSTCA; auxiliary variants only')
        self.variant = variant
        base = OnlineMSTCA(in_channels=26, hidden_size=64, num_layers=1, num_classes=2, dropout=.2)
        self.encoder, self.head = base.encoder, base.head
        self.adapter = nn.Sequential(nn.Conv1d(13, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, 128))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        selector = torch.zeros(13, dtype=torch.float32)
        for name in VARIANTS[variant]:selector[CANDIDATE_IDS.index(name)] = 1
        if variant == 'Ccap':selector.fill_(1)
        self.register_buffer('feature_selector', selector.view(1, 13, 1), persistent=False)
        self.register_buffer('ccap_indices', torch.tensor([RFLY_SEQ_COLUMNS_V2.index(name) for name in CCAP_COLUMNS], dtype=torch.int64), persistent=False)

    def forward(self, x, features):
        if x.ndim != 3 or features.ndim != 3 or x.shape[1:] != (26, 150) or features.shape[1:] != (13, 150) or len(x) != len(features):
            raise ValueError('Aligned Bx26x150 and Bx13x150 required')
        if not torch.isfinite(x).all() or not torch.isfinite(features).all():
            raise ValueError('Invalid full13 inputs rejected before selector/BatchNorm')
        auxiliary = x.index_select(1, self.ccap_indices) if self.variant == 'Ccap' else features*self.feature_selector
        return self.head(self.encoder(x)+self.adapter(auxiliary))


def build_m3_model(variant: str, common_untrained_state: Mapping[str, torch.Tensor]):
    """Shared untrained encoder/head plus deterministic identical adapter seed42.

    Artifact hashes/digest are enforced by the caller; no checkpoint file is
    read here. Per-build fork_rng avoids consuming caller/sampler RNG state.
    """
    if variant not in VARIANTS:raise ValueError('Unregistered M3 variant')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = OnlineMSTCA(in_channels=26, hidden_size=64, num_layers=1, num_classes=2, dropout=.2) if variant == 'C0' else M3AdapterMSTCA(variant)
    state = model.state_dict()
    common = {k: v for k, v in state.items() if k.startswith(('encoder.', 'head.'))}
    if set(common_untrained_state) != set(common):raise ValueError('Exact pure-M0 common-state keys required')
    for key, expected in common.items():
        value = common_untrained_state[key]
        if value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
            raise ValueError('Bad untrained common tensor: '+key)
        state[key] = value.detach().clone()
    model.load_state_dict(state, strict=True)
    return model
