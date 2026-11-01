"""Explicit version/split/role reader for causal publication-replay caches.

Only loads frozen data, filters before any model call, and normalizes using its
source-train statistics. Does not authorize training or physical live deployment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from torch.utils.data import Dataset

VERSION='rflymad_causal_publication_replay_v1'
ROLES={'adapt_normal':0,'calibrate_normal':1,'test_normal':2,'test_fault':3,'supplemental_normal_test':4}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class CausalCacheDataset(Dataset):
    """Requires explicit split and, for Real, a single frozen role.

    Compact source indices remain original cache IDs; evidence invalid rows
    never leave this reader. No max-sample replacement, augmentation or role mix.
    """
    def __init__(self, root, *, expected_version, split, purpose, role=None):
        root=Path(root)
        if expected_version!=VERSION or split not in {'train','val','test','real'}:
            raise ValueError('Explicit supported data version and split required')
        if purpose not in {'source_development','evaluation','normal_adaptation','normal_calibration'}:
            raise ValueError('Explicit consumption purpose required')
        if split=='real':
            if role not in ROLES:raise ValueError('Real requires a single explicit frozen role')
            allowed={'adapt_normal':'normal_adaptation','calibrate_normal':'normal_calibration',
                     'test_normal':'evaluation','test_fault':'evaluation','supplemental_normal_test':'evaluation'}
            if purpose!=allowed[role]:raise ValueError('Real role cannot be used for requested purpose or source training')
        elif role is not None:
            raise ValueError('Real role not applicable to source split')
        elif (split in {'train','val'} and purpose!='source_development') or (split=='test' and purpose!='evaluation'):
            raise ValueError('Split/purpose mismatch; final tests not used for training/model selection')
        manifest=json.loads((root/'build_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status')!='COMPLETE_FULL_FROZEN_SCOPE_CAUSAL_PUBLICATION_REPLAY_DATA_WITH_EXCLUSIONS' or manifest.get('version')!=expected_version:
            raise ValueError('Full independently verified dataset build is not complete')
        info=json.loads((root/(split+'_cache_complete.json')).read_text(encoding='utf-8'))
        contract=json.loads((root/'input_contract.json').read_text(encoding='utf-8'))
        stats=json.loads((root/'norm_stats.json').read_text(encoding='utf-8'))
        if contract['version']!=expected_version or stats['data_version']!=expected_version:
            raise ValueError('Dataset/normalization version mismatch')
        if stats['fit_split']!='source_train_only' or len(stats['columns'])!=26 or stats['columns']!=contract['columns']:
            raise ValueError('Source-only statistics or ordered schema mismatch')
        if stats['input_contract_sha256']!=sha(root/'input_contract.json'):
            raise ValueError('Statistics input contract identity mismatch')
        # Recorded provenance can name an absolute Windows build path. Relocate
        # only its basename into this explicit version root, never follow the
        # original machine path or weaken the content/containment checks.
        recorded=info['cache_file']
        if not isinstance(recorded,str) or not recorded or '\x00' in recorded:
            raise ValueError('Invalid recorded cache location')
        portable=PureWindowsPath(recorded) if '\\' in recorded or PureWindowsPath(recorded).drive else Path(recorded)
        if '..' in portable.parts or portable.name in ('','.','..'):
            raise ValueError('Cache location traversal not allowed')
        path=root/'cache'/portable.name
        if path.resolve().parent!=(root/'cache').resolve() or sha(path)!=info['cache_sha256']:
            raise ValueError('Cache identity or location mismatch')
        payload=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        if payload['data_version']!=expected_version or payload['split']!=split or payload['normalization_applied'] is not False:
            raise ValueError('Cache version/split or preprocessing mismatch')
        if payload['input_contract_sha256']!=sha(root/'input_contract.json') or payload['norm_sha256']!=sha(root/'norm_stats.json'):
            raise ValueError('Cache input contract/normalization mismatch')
        x=payload['x'];indices=payload['original_cache_index']
        if x.dtype!=torch.float32 or x.ndim!=3 or tuple(x.shape[1:])!=(26,150) or indices.dtype!=torch.int64 or indices.shape!=(len(x),):
            raise ValueError('Invalid x/index layout')
        if len(indices)>1 and not bool(torch.all(indices[1:]>indices[:-1])):
            raise ValueError('Original index identity is not strictly increasing')
        self.original_population=int(info['original_windows'])
        if split in {'train','val'}:
            selected=np.arange(len(x),dtype=np.int64);labels=payload['y']
        else:
            if 'y' in payload:raise ValueError('Evidence cannot have ambiguous y')
            masks=[payload[key] for key in ['input_valid','label_valid','supervised_valid']]
            if any(m.dtype!=torch.bool or m.shape!=(len(x),) for m in masks):raise ValueError('Explicit boolean masks required')
            if not torch.equal(masks[2],masks[0]&masks[1]):raise ValueError('Input/label/supervision masks disagree')
            mask=masks[2].clone()
            if split=='real':
                role_ids=payload['role_id']
                if role_ids.dtype!=torch.int64 or role_ids.shape!=(len(x),) or not torch.isin(role_ids,torch.tensor(list(ROLES.values()))).all():
                    raise ValueError('Invalid Real role IDs')
                self.original_population=int((role_ids==ROLES[role]).sum())
                mask &= role_ids==ROLES[role]
            selected=np.flatnonzero(mask.numpy()).astype(np.int64);labels=payload['qualified_reference_y']
            if len(labels)!=len(x) or not torch.all(labels[~payload['supervised_valid']]==-1):raise ValueError('Unknown targets must remain -1')
        if labels.dtype!=torch.int64 or labels.shape!=(len(x),) or not torch.isin(labels[selected],torch.tensor([0,1])).all():
            raise ValueError('Selected targets must be int64 binary, never -1')
        mu=torch.tensor(stats['mean'],dtype=torch.float32).view(-1,1);sigma=torch.tensor(stats['std'],dtype=torch.float32).view(-1,1)
        if mu.shape!=(26,1) or sigma.shape!=(26,1) or not torch.isfinite(mu).all() or not torch.isfinite(sigma).all() or not (sigma>0).all():
            raise ValueError('Invalid versioned normalization')
        for start in range(0,len(selected),256):
            chunk=x[selected[start:start+256]]
            if not torch.isfinite((chunk-mu)/sigma).all():raise ValueError('Selected normalized inputs not finite')
        self._x=x;self._y=labels;self._selected=selected;self._mean=mu;self._std=sigma
        self.original_cache_indices=indices[selected].clone();self.labels=labels[selected].tolist()
        self.excluded_population=self.original_population-len(selected)
        self.data_version=expected_version;self.split=split;self.role=role;self.purpose=purpose

    def __len__(self):return len(self._selected)

    def __getitem__(self,index):
        selected=int(self._selected[index])
        return (self._x[selected]-self._mean)/self._std,self._y[selected]
