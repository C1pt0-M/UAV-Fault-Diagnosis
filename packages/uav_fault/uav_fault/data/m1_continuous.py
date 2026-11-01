"""M1 and matched-M0 source datasets with one identical full150 population."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .causal_cache import CausalCacheDataset, VERSION, sha

RESIDUAL_COLUMNS = ['rh_signed_m', 'rR_x_rad', 'rR_y_rad', 'rR_z_rad']
M1_VERSION = 'm1_continuous_publication_residual_v1'


class M1ContinuousDataset(Dataset):
    def __init__(self, base_root, feature_root, *, split):
        if split not in ['train', 'val']:
            raise ValueError('M1 development allows only source train/val')
        root = Path(feature_root)
        manifest = json.loads((root/'build_manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'COMPLETE_SOURCE_M1_CONTINUOUS150_FEATURES_PENDING_TRAINING_GATE' or manifest['version'] != M1_VERSION:
            raise ValueError('M1 feature version/completion mismatch')
        if manifest['columns'] != RESIDUAL_COLUMNS or manifest['window_frames'] != 150 or manifest['horizon_us'] != 200000:
            raise ValueError('M1 exact four signed columns/150-window/200ms required')
        base = CausalCacheDataset(base_root, expected_version=VERSION, split=split, purpose='source_development')
        source_info = json.loads((Path(base_root)/(split+'_cache_complete.json')).read_text(encoding='utf-8'))
        if source_info['cache_sha256'] != manifest['source_data_identities']['cache/'+split+'.pt']:
            raise ValueError('Residuals belong to a different base cache')
        stats_path = root/'residual_norm_stats.json'
        if sha(stats_path) != manifest['residual_norm_sha256']:
            raise ValueError('Residual statistics identity mismatch')
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        if stats['fit_split'] != 'source_train_normal_only' or stats['columns'] != RESIDUAL_COLUMNS or stats['source_base_norm_sha256'] != sha(Path(base_root)/'norm_stats.json'):
            raise ValueError('Only source-train normal residual scaling allowed')
        export = manifest['exports'][split]
        path = root/export['file']
        if path.resolve().parent != (root/'cache').resolve() or sha(path) != export['sha256']:
            raise ValueError('Sidecar identity/location mismatch')
        p = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        if p['version'] != M1_VERSION or p['split'] != split or p['columns'] != RESIDUAL_COLUMNS or p['normalization_applied'] is not False:
            raise ValueError('Residual cache contract mismatch')
        if p['source_cache_sha256'] != manifest['source_data_identities']['cache/'+split+'.pt'] or p['residual_norm_sha256'] != sha(stats_path):
            raise ValueError('Residual cache/source identity mismatch')
        indices, ids, labels, residual = p['base_compact_index'], p['original_cache_index'], p['y'], p['residual']
        if indices.dtype != torch.int64 or ids.dtype != torch.int64 or labels.dtype != torch.int64 or residual.dtype != torch.float32 or residual.ndim != 3 or tuple(residual.shape[1:]) != (4,150):
            raise ValueError('M1 tensor layout mismatch')
        n = len(residual)
        masks = p.get('full_window_valid')
        if masks is None or masks.dtype != torch.bool or masks.shape != (n,150) or not masks.all():
            raise ValueError('Strict full150 residual validity required before forward/BN')
        if any(t.shape != (n,) for t in [indices, ids, labels]) or n != export['windows'] or len(base) != export['base_population']:
            raise ValueError('M1 population mismatch')
        if n == 0 or indices.min() < 0 or indices.max() >= len(base) or (n > 1 and not torch.all(indices[1:] > indices[:-1])):
            raise ValueError('Invalid or duplicate source positions')
        if not torch.equal(ids, base.original_cache_indices[indices]) or not torch.equal(labels, torch.tensor(base.labels,dtype=torch.int64)[indices]):
            raise ValueError('M0/M1 original IDs or labels differ')
        mu = torch.tensor(stats['mean'], dtype=torch.float32).view(4,1)
        std = torch.tensor(stats['std'], dtype=torch.float32).view(4,1)
        if not torch.isfinite(mu).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError('Invalid residual scales; no fallback')
        for start in range(0,n,512):
            if not torch.isfinite((residual[start:start+512]-mu)/std).all():
                raise ValueError('Unavailable/nonfinite residual reached M1 forward population')
        self.base = base
        self.base_indices = indices.clone()
        self.original_cache_indices = ids.clone()
        self.labels = labels.tolist()
        self.residual = residual
        self.mu, self.std = mu, std
        self.split = split

    def __len__(self):return len(self.base_indices)

    def __getitem__(self, index):
        x,y = self.base[int(self.base_indices[index])]
        r = (self.residual[index]-self.mu)/self.std
        return x,r,y


class M1IndexedDataset(Dataset):
    def __init__(self, source, indices=None):
        self.source=source
        self.indices=np.arange(len(source),dtype=np.int64) if indices is None else np.asarray(indices,dtype=np.int64)
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        idx=int(self.indices[index]);x,r,y=self.source[idx]
        return x,r,y,idx
