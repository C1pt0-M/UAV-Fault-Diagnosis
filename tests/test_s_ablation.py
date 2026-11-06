"""Synthetic CPU checks only; no source dataset access or remote/GPU execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
import train_s_ablation as train
import evaluate_s_ablation as evaluate
import preflight_s_ablation as preflight


@pytest.fixture(scope='module',autouse=True)
def cpu_threads():
    old=torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def ledger(groups,labels=None):
    labels=np.asarray(labels if labels is not None else [i%2 for i in range(len(groups))])
    n=len(groups)
    return pd.DataFrame({'association_group':groups,'case_uid':[f'SIL/cruise/{"GPS" if y else "normal"}/{i}' for i,y in enumerate(labels)],
        'qualified_y':labels,'compact_index':np.arange(n),'original_cache_index':np.arange(n,dtype=np.int64),
        'normal_origin':np.where(labels==1,'fault_or_unknown','normal_flight'),
        'case_level2_label':np.where(labels==1,'sensor','normal'),'domain':'SIL','split':'train','role':'train',
        'window_end_row':1000,'input_valid':True,'label_valid':True,'supervised_valid':True})


def test_fp97_ties():
    y=np.r_[np.ones(100,dtype=int),np.zeros(3,dtype=int)]
    p=np.r_[np.full(97,.8),np.full(3,.2),[.8,.799,.9]]
    metric=evaluate.fp_at_recall(y,p)
    assert metric['threshold']==.8 and metric['fp']==2


def test_fp97_exactly_97_percent():
    y=np.r_[np.ones(100,dtype=int),np.zeros(3,dtype=int)]
    p=np.r_[np.arange(1,101)/100,[.04,.039,.9]]
    m=evaluate.fp_at_recall(y,p)
    assert m['threshold']==.04 and np.mean(p[:100]>=m['threshold'])==.97 and m['fp']==2


def test_fp97_all_values_equal_and_no_fault():
    m=evaluate.fp_at_recall([1,1,0,0,0],[.5]*5)
    assert m['threshold']==.5 and m['fp']==3 and m['ap']==.4
    assert evaluate.fp_at_recall([0,0],[.5,.5])['threshold'] is None


def test_weighted_metrics_equivalent_to_repeated_clusters():
    y=np.array([0,1,0,1]);p=np.array([.7,.8,.8,.6]);w=np.array([2,2,1,1])
    weighted=evaluate.MetricCache(y,p).basic(w)
    repeated=evaluate.fp_at_recall(np.repeat(y,w),np.repeat(p,w))
    assert weighted==repeated


def test_balanced_ce_not_window_weighted_and_accumulates_across_batches():
    logits=torch.tensor([[2.,0.],[2.,0.],[2.,0.],[2.,0.]],dtype=torch.float64)
    y=torch.tensor([0,0,0,1])
    expected=.5*(-np.log(torch.softmax(logits,1)[0,0].item())-np.log(torch.softmax(logits,1)[3,1].item()))
    assert train.class_balanced_ce(logits,y).item()==pytest.approx(expected)
    assert train.class_balanced_ce(logits,y).item()!=pytest.approx(torch.nn.functional.cross_entropy(logits,y).item())
    loader=DataLoader(TensorDataset(logits,y),batch_size=2,shuffle=False)
    assert train.inner_ce(torch.nn.Identity(),loader,'cpu')==pytest.approx(expected)


def test_balanced_ce_missing_class_rejected():
    with pytest.raises(ValueError,match='both classes'):
        train.class_balanced_ce(torch.zeros(3,2),torch.zeros(3,dtype=torch.long))


def initial_selection():
    return {'best_loss':float('inf'),'best_epoch':None,'reference':float('inf'),'bad_epochs':0}


def test_early_stop_min30_then_20_consecutive_epochs():
    state=initial_selection()
    for epoch in range(1,51):
        state,best,stop=train.selection_step(state,1.,epoch)
        assert stop==(epoch==50)
        assert best==(epoch==1)
        if epoch==30:
            assert state['bad_epochs']==0
    assert state['best_epoch']==1 and state['bad_epochs']==20


def test_reference_only_changes_on_strict_significant_improvement():
    state,_,_=train.selection_step(initial_selection(),1.,1)
    state,best,_=train.selection_step(state,.995,31)
    assert best and state['reference']==1. and state['bad_epochs']==1
    state,best,_=train.selection_step(state,.996,32)
    assert not best and state['reference']==1. and state['bad_epochs']==2
    state,best,_=train.selection_step(state,.994,33)
    assert best and state['reference']==.994 and state['bad_epochs']==0
    state,best,_=train.selection_step(state,.994,34)
    assert not best and state['best_epoch']==33


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_selection_rejected(bad):
    with pytest.raises(FloatingPointError):
        train.selection_step(initial_selection(),bad,1)


def test_paired_bootstrap_shared_draws_all_arms_and_seeds(monkeypatch):
    y=np.array([0,0,0,1,1,1]);score=np.array([.8,.85,.9,.6,.7,.8])
    scores={a:{s:score.copy() for s in evaluate.SEEDS} for a in evaluate.ARMS}
    seen=[]
    original=evaluate.MetricCache.basic
    def record(self,weights=None,recall=.97):
        seen.append((id(weights),weights.copy()))
        return original(self,weights,recall)
    monkeypatch.setattr(evaluate.MetricCache,'basic',record)
    result=evaluate.paired_bootstrap(y,scores,np.array(['001','002','003']*2),replicates=10)
    assert len(seen)==10*12
    for start in range(0,len(seen),12):
        assert len({x[0] for x in seen[start:start+12]})==1
        assert all(np.array_equal(seen[start][1],x[1]) for x in seen[start:start+12])
    for comparison in result['comparisons'].values():
        assert comparison['aggregate_ratio_ci']==[1.,1.]
        assert comparison['aggregate_delta_ap_ci']==[0.,0.]


@pytest.mark.parametrize('arm,ci,ratios,expected',[
    ('S1',[1.02,1.08],[1.03,1.04,.99],'COMPONENT_HELPS'),
    ('S2',[1.2,1.3],[1.2,1.3,1.1],'COMPONENT_HELPS'),
    ('S3',[1.02,1.08],[1.03,1.04,.99],'MSTCA_BETTER'),
    ('S1',[.8,1.05],[.9,.95,1.01],'REMOVABLE'),
    ('S2',[.8,1.05],[.9,.95,1.01],'REMOVABLE'),
    ('S3',[.8,1.05],[.9,.95,1.01],'NO_ADVANTAGE'),
    ('S1',[.9,1.2],[.9,1.1,1.2],'INCONCLUSIVE'),
])
def test_each_primary_verdict_and_overlap_precedence(arm,ci,ratios,expected):
    result=evaluate.decide(arm,ratios,ci,[-.02,.02],{'GPS':.5})
    assert result['verdict']==expected
    assert result['effect_below_margin']==(expected in ('COMPONENT_HELPS','MSTCA_BETTER') and ci[1]<1.1)


def test_seed_consistency_precedes_class_constraint():
    assert evaluate.decide('S1',[.9,1.1,1.],[1.02,1.08],[-.02,.02],{'GPS':2.})['reason']=='seed_consistency'
    assert evaluate.decide('S1',[1.2,1.1,.9],[1.02,1.08],[-.02,.02],{'GPS':1.01})['verdict']=='TRADE_OFF'


@pytest.mark.parametrize('ratios,ci,ap',[
    ([.9,.95,1.],[.8,1.05],[-.04,-.01]),
    ([1.2,1.3,.99],[1.1,1.3],[.01,.04]),
])
def test_ap_opposite_direction_downgrades(ratios,ci,ap):
    result=evaluate.decide('S1',ratios,ci,ap,{'GPS':0.})
    assert result['verdict']=='INCONCLUSIVE' and result['reason']=='AP_consistency'


def test_exact_margin_is_not_noninferiority_and_missing_ci_not_conclusion():
    assert evaluate.decide('S3',[.9,.95,1.],[.9,1.10],[-.02,.02],{'GPS':1.})['verdict']=='INCONCLUSIVE'
    assert evaluate.decide('S1',[.9,.95,1.],None,[-.02,.02],{'GPS':0.})['verdict']=='INCONCLUSIVE'
    assert evaluate.ratio(1,0) is None and evaluate.percentile([1.,None]) is None


def test_late_hover_split_and_strict_half_rule():
    frame=ledger(['a','b','c','d','e'],[0,0,1,1,1])
    frame['case_uid']=['SIL/hover/normal/0','SIL/hover/sensor/1','SIL/hover/GPS/2','SIL/cruise/GPS/3','SIL/cruise/barometer/4']
    frame['window_end_row']=[1600,1600,1600,1000,1000]
    frame.loc[1,'normal_origin']='normal_part_of_fault_case'
    metrics=evaluate.score_metrics(frame,np.full(5,.5))
    assert metrics['primary']['fp']==2
    assert metrics['fp_r97_outside_late_hover_normal']==1
    assert metrics['fixed_p_gt_half']['all_normal_fp']==0
    assert metrics['fixed_p_gt_half']['late_hover_fault_recall']['GPS']==0.
    assert metrics['matched_class_recall_outside_late_hover_fault']['GPS']==1.


def test_hash_splits_preserve_leading_zero_strings(tmp_path):
    groups=[f'{i:05}' for i in range(120)]+['1','42']
    frame=ledger(groups)
    (tmp_path/'ledgers').mkdir()
    frame.to_csv(tmp_path/'ledgers/train_included_windows.csv',index=False)
    read=train.read_ledger(tmp_path)
    assert read.association_group.tolist()==groups
    for fold in range(3):
        idx,outer=train.split_ledger(read,fold)
        ref_outer=np.array([int(hashlib.sha256(('h1cv|'+g).encode()).hexdigest(),16)%3 for g in groups])
        ref_inner=np.array([int(hashlib.sha256(('s_ablation_inner_v1|'+g).encode()).hexdigest(),16)%10==0 for g in groups])
        assert np.array_equal(outer,ref_outer)
        assert np.array_equal(idx['fit'],np.flatnonzero((ref_outer!=fold)&~ref_inner))
        assert np.array_equal(idx['inner'],np.flatnonzero((ref_outer!=fold)&ref_inner))
        assert not set(read.iloc[idx['fit']].association_group)&set(read.iloc[idx['inner']].association_group)
    assert train.hash_group('00001','h1cv|',3)!=train.hash_group('1','h1cv|',3) or train.hash_group('00001','s_ablation_inner_v1|',10)!=train.hash_group('1','s_ablation_inner_v1|',10)
    with pytest.raises(ValueError):
        train.hash_group(1,'h1cv|',3)


def args_for(output,phase='cv',smoke=True):
    return SimpleNamespace(output=output,phase=phase,cpu_smoke=smoke,phase1_verdicts=None,
        phase1_verdicts_sha256=None,protocol=train.DEFAULT_PROTOCOL,data=output/'must_not_read',
        arm='S0',seed=42,fold=0,implementation_manifest=None,implementation_manifest_sha256=None)


def test_phase_val_without_verdict_sha_refused_before_any_data(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):
        pytest.fail('No protocol/data/cache access permitted before phase-val gate')
    monkeypatch.setattr(train,'load_protocol',forbidden)
    monkeypatch.setattr(train,'verify_data',forbidden)
    monkeypatch.setattr(train,'CacheWindows',forbidden)
    with pytest.raises(RuntimeError,match='Phase val requires'):
        train.run(args_for(tmp_path/'unused',phase='val'))
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('key,bad',[
    ('cuda_available',False),('torch','2.8.0+cpu'),('numpy','2.5.1'),('cublas_workspace_config',None)])
def test_formal_refusal_precedes_data(key,bad,monkeypatch,tmp_path):
    env={'torch':'2.5.1+cu124','numpy':'2.1.3','torch_cuda':'12.4','cuda_available':True,'cublas_workspace_config':':4096:8'}
    env[key]=bad
    monkeypatch.setattr(train,'environment',lambda:env)
    monkeypatch.setattr(train,'load_protocol',lambda *a:pytest.fail('Data/code setup should not run'))
    with pytest.raises(RuntimeError,match='Formal environment'):
        train.run(args_for(tmp_path/'unused',smoke=False))


def test_existing_output_refused(tmp_path):
    with pytest.raises(FileExistsError):
        train.run(args_for(tmp_path))


def test_formal_slots_enforce_frozen_order_and_manifest(tmp_path,monkeypatch):
    monkeypatch.setattr(train,'ROOT',tmp_path)
    parent=tmp_path/'artifacts/s_ablation_mstca_v1/cv'
    first=SimpleNamespace(seed=42,fold=0,arm='S0',phase='cv',output=parent/'S0_seed42_fold0')
    train.require_formal_slot(first,'manifest')
    second=SimpleNamespace(seed=42,fold=0,arm='S1',phase='cv',output=parent/'S1_seed42_fold0')
    with pytest.raises(RuntimeError,match='preceding run'):
        train.require_formal_slot(second,'manifest')
    train.write_json(parent/'S0_seed42_fold0/report.json',{'status':'COMPLETE',
        'protocol_sha256':train.PROTOCOL_SHA,'implementation_manifest_sha256':'manifest'})
    train.require_formal_slot(second,'manifest')
    with pytest.raises(RuntimeError,match='different manifest'):
        train.require_formal_slot(second,'changed')
    second.output=tmp_path/'alternate_completed_run'
    with pytest.raises(ValueError,match='canonical'):
        train.require_formal_slot(second,'manifest')


def test_local_preflight_refuses_ssh_before_access(tmp_path):
    with pytest.raises(RuntimeError,match='never connect'):
        preflight.preflight(SimpleNamespace(output=tmp_path/'unused',local=True,check_ssh=True))


def test_phase_gate_rejects_synthetic_and_hash_mismatch(tmp_path):
    path=tmp_path/'verdicts.json'
    train.write_json(path,{'status':'COMPLETE_SYNTHETIC_DEMO_NOT_RESULT','synthetic_demo':True})
    with pytest.raises(ValueError,match='SHA'):
        train.phase1_gate(path,'0'*64)
    with pytest.raises(ValueError,match='complete, real'):
        train.phase1_gate(path,train.sha(path))


def test_val_identity_files_are_hashed_without_parsing(tmp_path,monkeypatch):
    names=['val_cache_complete.json','cache/val.pt','ledgers/val_included_windows.csv','cache/train.pt']
    identities={}
    for name in names:
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'not a valid JSON/CSV/torch artifact')
        identities[name]=train.sha(path)
    monkeypatch.setattr(train,'read_json',lambda *a:{'data_identities':identities})
    result=train.verify_data(tmp_path,smoke=True)
    assert set(result['matched_sha256'])==set(names)-{'cache/train.pt'}
    assert result['skipped_sha256']==['cache/train.pt']


def test_incomplete_evaluation_has_no_conclusions(tmp_path,monkeypatch):
    if not tmp_path.resolve().is_relative_to(train.SMOKE_ROOT.resolve()):
        pytest.skip('Use the task artifact --basetemp')
    frame=ledger(['00001','00002','00003','00004'],[0,1,0,1])
    path=tmp_path/'ledger.csv';frame.to_csv(path,index=False)
    args=SimpleNamespace(output=tmp_path/'evaluation',phase='cv',synthetic_demo=True,ledger=path,
        protocol=train.DEFAULT_PROTOCOL,runs=tmp_path/'missing',phase1_verdicts=None,phase1_verdicts_sha256=None)
    result=evaluate.evaluate(args)
    assert result['status']=='INCOMPLETE' and result['decisions']=={} and not result['performance_claim']
    assert len(result['missing_runs_or_files'])==36


def test_synthetic_end_to_end_never_accesses_heldout_during_training(tmp_path,monkeypatch):
    if not tmp_path.resolve().is_relative_to(train.SMOKE_ROOT.resolve()):
        pytest.skip('Use the task artifact --basetemp')
    # One group of each inner status in every outer fold; each group has all origins.
    chosen={}
    for i in range(10000):
        g=f'{i:05}'
        key=(train.hash_group(g,'h1cv|',3),train.hash_group(g,'s_ablation_inner_v1|',10)==0)
        chosen.setdefault(key,g)
        if len(chosen)==6:
            break
    frame=ledger([g for g in chosen.values() for _ in range(3)],[0,0,1]*6)
    for i in range(1,len(frame),3):
        frame.loc[i,'normal_origin']='normal_part_of_fault_case'
        frame.loc[i,'case_level2_label']='sensor'
    events=[]
    positions,_=train.split_ledger(frame,0)
    class FakeWindows(TensorDataset):
        def __init__(self,root,fr,indices,split='train'):
            role=next(k for k,v in positions.items() if np.array_equal(v,indices))
            if role=='heldout':
                assert events.count('inner_selection')==2
            events.append(role+'_cache')
            self.x=torch.zeros(len(indices),26,150)
            self.y=torch.tensor(fr.iloc[indices].qualified_y.to_numpy(),dtype=torch.long)
            self.ids=torch.tensor(fr.iloc[indices].original_cache_index.to_numpy(),dtype=torch.long)
            super().__init__(self.x,self.y)
    monkeypatch.setattr(train,'CacheWindows',FakeWindows)
    monkeypatch.setattr(train,'verify_data',lambda *a,**k:{'matched_sha256':{},'skipped_sha256':['cache/train.pt']})
    monkeypatch.setattr(train,'read_ledger',lambda *a:frame)
    monkeypatch.setattr(train,'fold_sizes',lambda *a:{})
    original=train.inner_ce
    def inner(*a):
        events.append('inner_selection')
        return original(*a)
    monkeypatch.setattr(train,'inner_ce',inner)
    result=train.run(args_for(tmp_path/'run'))
    assert result['epochs']==2 and result['status']=='COMPLETE_CPU_SMOKE_NOT_FORMAL'
    assert events==['fit_cache','inner_cache','inner_selection','inner_selection','heldout_cache']
    assert result['heldout_prediction_passes']=={'best':1,'last':1}
