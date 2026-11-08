"""Synthetic-only checks; no Real/source/test model or payload is opened."""
import ast
from pathlib import Path
import subprocess
import sys
import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
import real_v2_b0_diagnostics as b


def test_stats_and_recipe_exactly_match_template_without_importing_it():
    path = b.ROOT / 'artifacts/ens_internal_test_v1_20261105/run_test.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    func = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'stats')
    scope = {'np': np}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), 'exec'), scope)
    x = np.random.default_rng(2).normal(size=(3, 26, 150))
    np.testing.assert_array_equal(b.stats(x), scope['stats'](x))
    call = next(x for x in ast.walk(tree) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == 'HistGradientBoostingClassifier')
    assert b.RECIPE == {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    assert b.stats(x).shape == (3, 130) and b.stats(x).dtype == np.float32


def test_split_identity_quarantine_and_group_safety(tmp_path):
    rows = [{'case_uid': f'REAL/hover/GPS/{i}', 'association_group': str(i), 'flight': 'hover',
        'family': 'sensor', 'stratum_family': 'sensor', 'role_v1': 'test_fault',
        'label_quarantine': i < 5, 'partition': 'dev', 'dev_fold': i % 5} for i in range(320)]
    rows.append({**rows[-1], 'case_uid': 'REAL/hover/GPS/sealed', 'association_group': 'sealed', 'partition': 'sealed', 'dev_fold': ''})
    path = tmp_path / 'split.csv';pd.DataFrame(rows).to_csv(path, index=False)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    eligible, info = b.eligible_split(path, h)
    assert len(eligible) == 315 and info['dev_quarantine'] == 5
    assert not eligible.partition.eq('sealed').any()
    with pytest.raises(ValueError, match='SHA'):
        b.eligible_split(path, 'wrong')
    rows[-1]['association_group'] = '319';pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match='group-disjoint'):
        b.eligible_split(path, hashlib.sha256(path.read_bytes()).hexdigest())


def ledger_row(uid='REAL/hover/GPS/dev', y=0):
    return {'case_uid': uid, 'association_group': 'g', 'original_cache_index': 8, 'qualified_y': y,
        'domain': 'REAL', 'split': 'real', 'role': 'test_fault', 'supervised_valid': True,
        'input_valid': True, 'label_valid': True, 'window_start_row': 0, 'window_end_row': 150,
        'normal_origin': 'normal_part_of_fault_case', 'phase': 'pre_fault', 'fault_onset_row': 500}


def metadata():
    return pd.DataFrame([{'case_uid': 'REAL/hover/GPS/dev', 'association_group': 'g', 'family': 'sensor',
        'flight': 'hover', 'stratum_family': 'sensor', 'role_v1': 'test_fault', 'dev_fold': 0}])


def test_ledger_drops_sealed_rows_before_any_validation(tmp_path):
    rows = [ledger_row(), {**ledger_row('REAL/hover/GPS/sealed', y=999), 'original_cache_index': 8,
                         'association_group': 'not-dev', 'supervised_valid': False}]
    p = tmp_path / 'ledger.csv';pd.DataFrame(rows).to_csv(p, index=False)
    f = b.read_ledger(p, metadata())
    assert len(f) == 1 and f.qualified_y.tolist() == [0]


def test_window_alignment_and_qualified_label_mismatch(tmp_path):
    p = tmp_path / 'win.npz';x = np.stack([np.full((26, 150), 2, np.float32), np.ones((26, 150), np.float32)])
    np.savez(p, x=x, y=np.array([1, 0]), original_cache_index=np.array([9, 8]), window_start_row=np.array([1, 0]))
    f = pd.DataFrame([ledger_row(), {**ledger_row(y=1), 'original_cache_index': 9, 'window_start_row': 1}])
    result = b.window_features(p, f)
    assert result[0, 0] == 1 and result[1, 0] == 2
    f.loc[1, 'qualified_y'] = 0
    with pytest.raises(ValueError, match='qualified'):
        b.window_features(p, f)


def test_real_weights_preserve_faults_and_normal_origin_masses():
    y = np.array([0, 0, 0, 0, 1, 1]);origin = np.array(['normal_flight']*3+['normal_part_of_fault_case']+['fault_or_unknown']*2)
    w = b.real_weights(y, origin)
    assert w[:3].sum() == 2 and w[3] == 2 and w[4:].tolist() == [1, 1]
    with pytest.raises(ValueError, match='Missing normal'):
        b.real_weights(y, np.array(['normal_flight']*6))


@pytest.mark.parametrize('p', [[0, 0, 1, 1, .5], [1, 1, 1, 1, 1], [1, 0, .5, .5, 0]])
def test_weighted_auc_ties_match_sklearn(p):
    y=np.array([0, 1, 0, 1, 1]);w=np.array([2, 1, 4, 3, 2]);p=np.array(p)
    plan=b.auc_plan(p,y,np.ones(5,bool))
    assert b.weighted_auc(plan,w)==pytest.approx(roc_auc_score(y,p,sample_weight=w),abs=1e-15)
    assert b.weighted_auc(plan,np.zeros(5,int)) is None
    assert b.weighted_auc(plan,np.array([1,0,1,0,0])) is None


def test_bootstrap_shared_plan_matches_explicit_group_resampling():
    p=np.array([.1,.5,.5,.9]);y=np.array([0,1,0,1]);groups=np.array(['a','b','c','d'])
    plans={'a':b.auc_plan(p,y,np.ones(4,bool)),'identical':b.auc_plan(p,y,np.ones(4,bool))}
    result,meta=b.auc_intervals(plans,groups,groups,reps=40,seed=4)
    rng=np.random.default_rng(4);sample=[]
    for _ in range(40):
        w=np.bincount(rng.integers(0,4,4),minlength=4)
        sample.append(None if w[y==0].sum()==0 or w[y==1].sum()==0 else roc_auc_score(y,p,sample_weight=w))
    assert result['a']==result['identical'] and meta['shared_plan']
    assert result['a']['undefined_replicates']==sum(x is None for x in sample)
    assert result['a']['ci95'] is None  # Undefined replicates are never dropped/redrawn.


def test_normal_full_vs_supplement_is_role_not_stratum_family():
    f=pd.DataFrame({'qualified_y':[0,0,0,0,0,1], 'family':['normal','normal','motor','sensor','motor','motor'],
        'role_v1':['test_normal','supplemental_normal_test','test_fault','test_fault','test_fault','test_fault'],
        'stratum_family':['normal_full','motor','motor','sensor','motor','motor'],
        'phase':['normal_flight','normal_flight','pre_fault','post_fault_normal','transition','transition'],
        'normal_origin':['normal_flight','normal_flight']+['normal_part_of_fault_case']*3+['fault_or_unknown']})
    m=b.populations(f)
    assert m['normal_full'].tolist()==[True,False,False,False,False,False]
    assert m['normal_segment'].tolist()==[False,True,False,False,False,False]
    np.testing.assert_array_equal(np.sum(np.stack(list(m.values())),axis=0),[1,1,1,1,1,0])


def test_recall_threshold_keeps_ties_and_empty_normal_is_null():
    p=np.array([.9,.8,.8,.1,.8]);y=np.array([1,1,1,0,0])
    r=b.fpr_at_recall(p,y,{'normal':y==0,'empty':np.zeros(5,bool)},.5)
    assert r['threshold']==.8 and r['actual_recall']==1
    assert r['normal_FPR']=={'normal':.5,'empty':None}


def test_recommendation_exact_boundaries_and_no_automatic_launch():
    assert b.recommendation({'auc':.75,'ci95':[.5,.9]}, {})['recommendation']=='GO'
    assert b.recommendation({'auc':.6,'ci95':[.5,.64999]}, {})['recommendation']=='STOP'
    r=b.recommendation({'auc':.7,'ci95':[.5,.65]}, {'motor':{'auc':.65},'sensor':{'auc':.6499}})
    assert r['recommendation']=='交用户决定' and r['not_supported_in_B1']==['sensor'] and not r['automatic_B1_launch']


def test_oof_only_predicts_held_groups_and_never_refits_sensitivity(monkeypatch):
    f=pd.DataFrame({'dev_fold':np.repeat(np.arange(5),4),'association_group':np.repeat(list('abcde'),4),
        'normal_origin':np.tile(['normal_flight','normal_part_of_fault_case','fault_or_unknown','fault_or_unknown'],5)})
    x=np.arange(20).reshape(-1,1);y=np.tile([0,0,1,1],5);calls=[]
    class Fake:
        def __init__(self,**kw):assert kw==b.RECIPE
        def fit(self,xx,yy,sample_weight):
            self.ids=set(xx[:,0]);calls.append(self.ids);self.n_iter_=1
            return self
        def predict_proba(self,xx):
            assert not self.ids & set(xx[:,0]);return np.tile([.4,.6],(len(xx),1))
    monkeypatch.setattr(b,'HistGradientBoostingClassifier',Fake)
    p,log=b.oof(x,y,f,np.ones(20,bool),'real',lambda *a:None)
    assert len(calls)==5 and np.isfinite(p).all() and len(log)==5


def test_exposure_sentinel_and_prefault_validity(tmp_path):
    e=metadata();data=tmp_path/'data';(data/'cases').mkdir(parents=True)
    np.savez(data/'cases/REAL__hover__GPS__dev.npz',row_input_valid=np.array([1,1,0,1,1],bool),row_label_valid=np.array([1,0,1,1,1],bool))
    f=pd.DataFrame([ledger_row()]).assign(family='sensor', dev_fold=0);f.loc[0,'fault_onset_row']=4
    rows,r=b.audits(f,e,data)
    assert rows[0]['pre_fault_exposure_h']==pytest.approx(2*.02/3600)
    assert rows[0]['pre_fault_length_s']==.08
    f.loc[0,'fault_onset_row']=-1
    rows,r=b.audits(f,e,data)
    assert rows[0]['pre_fault_length_s'] is None and rows[0]['pre_fault_exposure_h'] is None
    assert r['missing_onset_fault_cases']==['REAL/hover/GPS/dev']


def test_boundary_rejects_sealed_quarantine_source_test_and_writes(tmp_path):
    code=r'''
from pathlib import Path
import sys,pandas as pd
sys.path.insert(0,sys.argv[1]);import real_v2_b0_diagnostics as b
p=Path(sys.argv[2]);data=p/'inputs/causal';(data/'windows').mkdir(parents=True)
models=p/'models';models.mkdir();source=models/'gbt.pkl';source.write_bytes(b'model')
for name in ['REAL__hover__GPS__dev.npz','REAL__hover__GPS__sealed.npz','REAL__hover__GPS__quarantine.npz']:(data/'windows'/name).write_bytes(b'synthetic')
e=pd.DataFrame({'case_uid':['REAL/hover/GPS/dev']});a=b.install_boundary(e,data,source)
assert (data/'windows/REAL__hover__GPS__dev.npz').read_bytes()==b'synthetic'
blocked=0
for work in [lambda:(data/'windows/REAL__hover__GPS__sealed.npz').read_bytes(),lambda:(data/'windows/REAL__hover__GPS__quarantine.npz').read_bytes(),lambda:(data/'cache/test.pt').read_bytes(),lambda:(data/'windows/REAL__hover__GPS__dev.npz').write_bytes(b'bad'),lambda:(models/'test_predictions.npz').read_bytes()]:
 try:work()
 except PermissionError:blocked+=1
assert blocked==5 and len(a['windows'])==1 and len(a['blocked_accesses'])==5
'''
    q=subprocess.run([sys.executable,'-X','utf8','-E','-P','-B','-c',code,str(b.ROOT/'scripts/research'),str(tmp_path)],capture_output=True,text=True,encoding='utf-8')
    assert q.returncode==0,q.stdout+q.stderr
