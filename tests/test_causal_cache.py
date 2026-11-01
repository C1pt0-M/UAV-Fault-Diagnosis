import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_fault.data.causal_cache import CausalCacheDataset, VERSION, sha
from uav_fault.models.cnn.lstm import RFLY_SEQ_COLUMNS_V2


def make_cache(root,split):
    (root/'cache').mkdir()
    (root/'build_manifest.json').write_text(json.dumps({'status':'COMPLETE_FULL_FROZEN_SCOPE_CAUSAL_PUBLICATION_REPLAY_DATA_WITH_EXCLUSIONS','version':VERSION}))
    contract={'version':VERSION,'columns':list(RFLY_SEQ_COLUMNS_V2)}
    (root/'input_contract.json').write_text(json.dumps(contract))
    stats={'data_version':VERSION,'fit_split':'source_train_only','columns':list(RFLY_SEQ_COLUMNS_V2),
        'mean':[0.]*26,'std':[1.]*26,'input_contract_sha256':sha(root/'input_contract.json')}
    (root/'norm_stats.json').write_text(json.dumps(stats))
    x=torch.zeros(3,26,150);idx=torch.arange(3)
    payload={'x':x,'data_version':VERSION,'split':split,'normalization_applied':False,
        'original_cache_index':idx,'input_contract_sha256':sha(root/'input_contract.json'),'norm_sha256':sha(root/'norm_stats.json')}
    if split in ['train','val']:payload['y']=torch.tensor([0,1,0])
    else:
        payload.update(input_valid=torch.tensor([True,False,True]),label_valid=torch.tensor([True,True,True]),
            supervised_valid=torch.tensor([True,False,True]),qualified_reference_y=torch.tensor([0,-1,1]))
        payload['x'][1]=float('nan')
        if split=='real':payload['role_id']=torch.tensor([3,3,2])
    path=root/'cache'/('dataset.pt');torch.save(payload,path)
    (root/(split+'_cache_complete.json')).write_text(json.dumps({'cache_file':str(path),'cache_sha256':sha(path),'original_windows':3}))
    return path,payload


def test_compact_source_reader_uses_versioned_stats_without_training(tmp_path):
    make_cache(tmp_path,'train')
    ds=CausalCacheDataset(tmp_path,expected_version=VERSION,split='train',purpose='source_development')
    assert len(ds)==3 and ds.labels==[0,1,0] and ds.original_population==3
    assert ds[0][0].shape==(26,150) and torch.isfinite(ds[0][0]).all()


def test_internal_test_unknown_inputs_filtered_before_any_forward(tmp_path):
    make_cache(tmp_path,'test')
    ds=CausalCacheDataset(tmp_path,expected_version=VERSION,split='test',purpose='evaluation')
    assert len(ds)==2 and ds.excluded_population==1 and ds.original_cache_indices.tolist()==[0,2]
    assert ds.labels==[0,1] and torch.isfinite(ds[1][0]).all()


def test_real_requires_explicit_single_role_and_blocks_fault_supervision(tmp_path):
    make_cache(tmp_path,'real')
    with pytest.raises(ValueError,match='single explicit'):
        CausalCacheDataset(tmp_path,expected_version=VERSION,split='real',purpose='evaluation')
    with pytest.raises(ValueError,match='requested purpose'):
        CausalCacheDataset(tmp_path,expected_version=VERSION,split='real',purpose='source_development',role='test_fault')
    ds=CausalCacheDataset(tmp_path,expected_version=VERSION,split='real',purpose='evaluation',role='test_fault')
    assert len(ds)==1 and ds.original_population==2 and ds.excluded_population==1 and ds.labels==[0]


def test_wrong_version_or_final_test_training_rejected(tmp_path):
    with pytest.raises(ValueError,match='version'):
        CausalCacheDataset(tmp_path,expected_version='wrong',split='train',purpose='source_development')
    with pytest.raises(ValueError,match='purpose mismatch'):
        CausalCacheDataset(tmp_path,expected_version=VERSION,split='test',purpose='source_development')


def test_incomplete_build_is_not_consumable(tmp_path):
    make_cache(tmp_path,'train')
    (tmp_path/'build_manifest.json').write_text(json.dumps({'status':'RUNNING','version':VERSION}))
    with pytest.raises(ValueError,match='not complete'):
        CausalCacheDataset(tmp_path,expected_version=VERSION,split='train',purpose='source_development')


@pytest.mark.parametrize('recorded',[r'D:\UAV-Data\causal_input_v1\cache\dataset.pt','/old/machine/cache/dataset.pt','cache/dataset.pt'])
def test_portable_recorded_path_still_requires_identical_local_cache(tmp_path,recorded):
    make_cache(tmp_path,'train')
    marker=tmp_path/'train_cache_complete.json'
    info=json.loads(marker.read_text());info['cache_file']=recorded;marker.write_text(json.dumps(info))
    ds=CausalCacheDataset(tmp_path,expected_version=VERSION,split='train',purpose='source_development')
    assert len(ds)==3 and ds.labels==[0,1,0]


@pytest.mark.parametrize('recorded',[r'D:\cache\..\dataset.pt','../dataset.pt'])
def test_portable_reader_rejects_recorded_traversal(tmp_path,recorded):
    make_cache(tmp_path,'train')
    marker=tmp_path/'train_cache_complete.json';info=json.loads(marker.read_text())
    info['cache_file']=recorded;marker.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='traversal'):
        CausalCacheDataset(tmp_path,expected_version=VERSION,split='train',purpose='source_development')


def test_cache_tampering_rejected_before_consumption(tmp_path):
    path,payload=make_cache(tmp_path,'test');payload['qualified_reference_y'][0]=-1;torch.save(payload,path)
    with pytest.raises(ValueError,match='identity'):
        CausalCacheDataset(tmp_path,expected_version=VERSION,split='test',purpose='evaluation')
