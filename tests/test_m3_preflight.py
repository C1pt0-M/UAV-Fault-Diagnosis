"""M3 checks that do not perform optimizer updates or read final populations."""
from pathlib import Path
import hashlib
import json
import sys
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
from uav_fault.models.cnn.lstm import OnlineMSTCA
from uav_fault.models.cnn.mstca_m3 import VARIANTS, build_m3_model
from uav_fault.data.m3_features import old_candidate_features, M3CandidateDataset
from analyze_m3_recall_constraints import SUBTYPES, allowed_extra_misses, recall_gate
from train_source_normal_sampling import selection_step


def test_recall_margin_is_all_scopes_and_percentage_points():
    assert allowed_extra_misses(1228, .5) == 6
    assert allowed_extra_misses(16498, .25) == 41
    assert allowed_extra_misses(9875, .1) == 9
    loss = dict(overall=.20, **{s:0.0 for s in SUBTYPES})
    loss['GPS'] = .60
    assert recall_gate(loss, .25, {s:.5 for s in SUBTYPES})['failed_recall_scopes'] == ['GPS']
    assert recall_gate(loss, None, {s:None for s in SUBTYPES})['pass'] is None
    with pytest.raises(ValueError):recall_gate(loss, -.1, {s:.5 for s in SUBTYPES})
    with pytest.raises(ValueError):allowed_extra_misses(1228, float('nan'))


def test_macro_F1_selection_cannot_hide_recall_failure_with_second_checkpoint():
    state = dict(best_score=.95, best_epoch=1, significant_reference=.95, bad_epochs=0)
    rule = dict(min_epochs=30, patience=20, significant_delta=.002)
    state, save, stop = selection_step(state, .96, 2, rule)
    loss = dict(overall=.75, **{s:1.0 for s in SUBTYPES})
    assert save and not stop and state['best_epoch'] == 2
    assert recall_gate(loss, 0., {s:0. for s in SUBTYPES})['pass'] is False
    _, save, _ = selection_step(state, .96, 3, rule)
    assert not save  # Earliest exact tie; recall cannot override epoch selection.


def test_Ccap_has_no_candidate_dependence_and_rejects_before_BN():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    common = OnlineMSTCA(in_channels=26, hidden_size=64, dropout=.2).state_dict()
    model = build_m3_model('Ccap', common).eval()
    with torch.no_grad():model.adapter[-1].weight.fill_(.001)
    x, r = torch.randn(2,26,150), torch.randn(2,13,150)
    with torch.no_grad():assert torch.equal(model(x,r), model(x,r*7+9))
    before = {k:v.clone() for k,v in model.state_dict().items()}
    model.train();r[0,12,4] = float('nan')
    with pytest.raises(ValueError, match='before selector/BatchNorm'):model(x,r)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    with pytest.raises(ValueError):model(x[:,:,:140],r[:,:,:140])
    with pytest.raises(ValueError):model(torch.randn(2,39,150),r)


def test_selector_does_not_repair_invalid_excluded_feature_and_has_no_gradient():
    common = OnlineMSTCA(in_channels=26, hidden_size=64, dropout=.2).state_dict()
    model = build_m3_model('CV', common).eval()
    x, r = torch.randn(3,26,150), torch.randn(3,13,150)
    invalid = r.clone();invalid[0,0,4] = float('nan')  # O1 excluded but still invalid.
    with pytest.raises(ValueError):model(x,invalid)
    with torch.no_grad():model.adapter[-1].weight.copy_(torch.linspace(-.01,.01,2048).reshape(128,16))
    r.requires_grad_(True)
    torch.nn.functional.cross_entropy(model(x,r),torch.tensor([0,1,0])).backward()
    assert r.grad[:,9].abs().sum() > 0
    assert r.grad[:,:9].abs().sum() == 0 and r.grad[:,10:].abs().sum() == 0
    assert (model.adapter[0].weight.grad[:,:9] == 0).all()


def test_frozen_protocol_cannot_be_mistaken_for_launch_approval():
    path = ROOT/'configs/experiments/M3_preflight_v1.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix('.json.sha256').read_text().strip()
    config = json.loads(path.read_text(encoding='utf-8'))
    assert config['ready_for_formal_training'] is False
    assert config['authorization']['authorized_training_jobs'] == 0
    assert config['recall_acceptance']['overall_allowed_loss_pp'] is None
    assert set(config['recall_acceptance']['per_fault_allowed_loss_pp'].values()) == {None}
    assert {row['id']:tuple(row['features']) for row in config['primary_configurations']} == VARIANTS
    assert config['training']['draws_per_epoch'] == 151833
    assert config['training']['steps_per_epoch'] == 297 and config['training']['max_epochs'] == 100
    assert config['evaluation']['classification_boundary'] == .5
    for rel, expected in dict(config['input_identities'],**config['code_identities']).items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected
    stats = ROOT/config['normalization']['path']
    assert hashlib.sha256(stats.read_bytes()).hexdigest() == config['normalization']['sha256']


def test_dataset_forbids_final_splits_and_unpinned_scaler_without_reading_data(tmp_path):
    for split in ['test','real']:
        with pytest.raises(ValueError,match='only source'):M3CandidateDataset(tmp_path,tmp_path,tmp_path/'absent',split=split,normalization_sha256='a')
    with pytest.raises(ValueError,match='identity mismatch'):M3CandidateDataset(tmp_path,tmp_path,tmp_path/'absent',split='train',normalization_sha256='')
    with pytest.raises(ValueError):old_candidate_features(torch.zeros(1,26,140))
    x = torch.zeros(1,26,150)
    x[0,2,20] = float('inf')
    with pytest.raises(ValueError,match='Finite'):old_candidate_features(x)
