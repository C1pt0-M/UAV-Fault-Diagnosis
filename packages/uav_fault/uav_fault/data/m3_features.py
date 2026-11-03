"""M3 candidate calculation/scaling, separate from every historical experiment."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch
from .causal_cache import sha
from .m1_continuous import M1ContinuousDataset
from ..model_contract import RFLY_SEQ_COLUMNS_V2
from ..models.cnn.lstm import OnlineKGMSTCA

CANDIDATE_IDS = tuple([f'O{i}' for i in range(1, 10)]+[f'N{i}' for i in range(1, 5)])
OLD_CONTEXT = SimpleNamespace(
    _ctrl_idx=[RFLY_SEQ_COLUMNS_V2.index(f'actuator_ctrl_{i}') for i in range(1, 5)],
    _out_idx=[RFLY_SEQ_COLUMNS_V2.index(f'actuator_out_{i}') for i in range(1, 5)],
    _gyro_idx=[RFLY_SEQ_COLUMNS_V2.index('gyro_'+axis) for axis in 'xyz'],
    _quat_idx=[RFLY_SEQ_COLUMNS_V2.index('att_q_'+axis) for axis in 'xyz'],
    _acc_norm_idx=RFLY_SEQ_COLUMNS_V2.index('acc_norm'),
    _vel_z_idx=RFLY_SEQ_COLUMNS_V2.index('vel_z'),
    _alt_idx=RFLY_SEQ_COLUMNS_V2.index('baro_alt_relative'), _effort_idx=None)


def old_candidate_features(x):
    """Use the unchanged historical deterministic26->9 function, no model."""
    if x.ndim != 3 or x.shape[1:] != (26, 150) or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError('Finite normalized float32 Bx26x150 inputs required')
    result = OnlineKGMSTCA._relation_features(OLD_CONTEXT, x)
    if result.shape != (len(x), 9, 150) or not torch.isfinite(result).all():
        raise ValueError('Nonfinite old candidate result; no filling/clipping')
    return result


class M3CandidateDataset(torch.utils.data.Dataset):
    """All variants consume one identical full13-valid original-ID population.

    Old9 are computed on demand. This intentionally avoids a second giant input
    cache and never relaxes M1's continuous native-history/mask qualification.
    """
    def __init__(self, base_root, residual_root, normalization_path, *, split, normalization_sha256):
        if split not in ['train', 'val']:
            raise ValueError('M3 preflight permits only source Train/Val')
        if not normalization_sha256 or sha(Path(normalization_path)) != normalization_sha256:
            raise ValueError('M3 normalization identity mismatch')
        stats = json.loads(Path(normalization_path).read_text(encoding='utf-8'))
        if stats['status'] != 'FROZEN_SOURCE_TRAIN_NORMAL_M3_SCALING' or stats['candidate_ids'] != list(CANDIDATE_IDS):
            raise ValueError('Wrong M3 normalization status/order')
        if stats['fit_split'] != 'source_train_known_normal_only' or stats['residual_stats_sha256'] != sha(Path(residual_root)/'residual_norm_stats.json'):
            raise ValueError('Source-Train-only parent residual statistics required')
        if stats['base_norm_sha256'] != sha(Path(base_root)/'norm_stats.json'):
            raise ValueError('Frozen base26 normalization changed')
        self.source = M1ContinuousDataset(base_root, residual_root, split=split)
        old = stats['old9'];self.mean = torch.tensor(old['mean'], dtype=torch.float32).view(9, 1)
        self.std = torch.tensor(old['std'], dtype=torch.float32).view(9, 1)
        if self.mean.shape != (9, 1) or self.std.shape != (9, 1) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError('Finite old9 means and strictly positive std required')
        expected = stats['source_split_identities'][split]
        if len(self.source) != expected['windows'] or sha(Path(residual_root)/'cache'/(split+'.pt')) != expected['residual_cache_sha256']:
            raise ValueError('M3 common population/identity changed')
        self.original_cache_indices = self.source.original_cache_indices
        self.labels = self.source.labels
        self.base_indices = self.source.base_indices

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        x, residual_normalized, label = self.source[index]
        old = old_candidate_features(x.unsqueeze(0))[0]
        features = torch.cat([(old-self.mean)/self.std, residual_normalized])
        if features.shape != (13, 150) or not torch.isfinite(features).all() or int(label) not in [0, 1]:
            raise ValueError('All13 valid before any selector or BatchNorm')
        return x, features, label

    def __getitems__(self, indices):
        """Vectorize the same per-window formulas for DataLoader batches."""
        positions = torch.as_tensor(indices, dtype=torch.int64)
        source = self.source
        compact = source.base_indices[positions]
        raw_positions = source.base._selected[compact.numpy()]
        x = (source.base._x[raw_positions]-source.base._mean)/source.base._std
        residual = (source.residual[positions]-source.mu)/source.std
        old = old_candidate_features(x)
        features = torch.cat([(old-self.mean)/self.std, residual], dim=1)
        labels = source.base._y[raw_positions]
        if not torch.isfinite(features).all() or not torch.isin(labels, torch.tensor([0, 1])).all():
            raise ValueError('All13 valid before any selector or BatchNorm')
        return [(x[i], features[i], labels[i]) for i in range(len(positions))]
