from pathlib import Path
import json
import pytest
import torch
from test_causal_cache import make_cache
from uav_fault.data.causal_cache import sha
from uav_fault.data.m1_continuous import M1ContinuousDataset, M1_VERSION, RESIDUAL_COLUMNS


def fixture(tmp_path, mutate=None):
    base=tmp_path/'base';base.mkdir();cp,original=make_cache(base,'train')
    root=tmp_path/'features';(root/'cache').mkdir(parents=True)
    stats={'fit_split':'source_train_normal_only','columns':RESIDUAL_COLUMNS,
        'mean':[1.,2.,3.,4.],'std':[2.]*4,'source_base_norm_sha256':sha(base/'norm_stats.json')}
    (root/'residual_norm_stats.json').write_text(json.dumps(stats))
    payload={'version':M1_VERSION,'split':'train','columns':RESIDUAL_COLUMNS,'normalization_applied':False,
        'source_cache_sha256':sha(cp),'residual_norm_sha256':sha(root/'residual_norm_stats.json'),
        'base_compact_index':torch.tensor([0,2]),'original_cache_index':torch.tensor([0,2]),'y':torch.tensor([0,0]),
        'residual':torch.ones(2,4,150),'full_window_valid':torch.ones(2,150,dtype=torch.bool)}
    if mutate:mutate(payload)
    path=root/'cache/train.pt';torch.save(payload,path)
    manifest={'status':'COMPLETE_SOURCE_M1_CONTINUOUS150_FEATURES_PENDING_TRAINING_GATE','version':M1_VERSION,
        'columns':RESIDUAL_COLUMNS,'window_frames':150,'horizon_us':200000,'residual_norm_sha256':sha(root/'residual_norm_stats.json'),
        'source_data_identities':{'cache/train.pt':sha(cp)},
        'exports':{'train':{'file':'cache/train.pt','sha256':sha(path),'windows':2,'base_population':3}}}
    (root/'build_manifest.json').write_text(json.dumps(manifest))
    return base,root,manifest


def test_matched_M0_M1_exact_original_IDs_and_base_normalization_unchanged(tmp_path):
    base,root,_=fixture(tmp_path);ds=M1ContinuousDataset(base,root,split='train')
    assert ds.original_cache_indices.tolist()==[0,2] and ds.base_indices.tolist()==[0,2]
    x,r,y=ds[1];assert torch.equal(x,ds.base[2][0]) and int(y)==0
    assert r.shape==(4,150) and r[0,0]==0 and r[3,0]==-1.5


@pytest.mark.parametrize('mutate',[
    lambda p:p['original_cache_index'].__setitem__(0,1),
    lambda p:p['y'].__setitem__(0,1),
    lambda p:p['base_compact_index'].__setitem__(1,0),
    lambda p:p['full_window_valid'].__setitem__((0,0),False),
    lambda p:p['residual'].__setitem__((0,0,0),float('nan'))])
def test_wrong_ids_labels_duplicate_mask_or_nonfinite_rejected_before_model(tmp_path,mutate):
    base,root,_=fixture(tmp_path,mutate)
    with pytest.raises(ValueError):M1ContinuousDataset(base,root,split='train')


def test_no_test_Real_reads_or_val_fitted_scales(tmp_path):
    with pytest.raises(ValueError,match='only source'):M1ContinuousDataset(tmp_path,tmp_path,split='test')
    with pytest.raises(ValueError,match='only source'):M1ContinuousDataset(tmp_path,tmp_path,split='real')
    base,root,manifest=fixture(tmp_path)
    stats=json.loads((root/'residual_norm_stats.json').read_text());stats['fit_split']='val';(root/'residual_norm_stats.json').write_text(json.dumps(stats))
    manifest['residual_norm_sha256']=sha(root/'residual_norm_stats.json');(root/'build_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='source-train normal'):M1ContinuousDataset(base,root,split='train')
