"""Synthetic persistence/restart tests; never read Real/model checkpoints."""
from pathlib import Path
import sys,json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
import resume_real_v2_b1_learning_curve as r
import real_v2_b1_learning_curve as b


def test_complete_model_checkpoint_is_never_trained_again(tmp_path):
    rows={};calls=[];journal=[];registered=[]
    def train():calls.append('fit');return {'weights':[1.,2.]},{'name':'slot123','n_iter':400}
    def append(row):journal.append(row)
    p=tmp_path/'slot123.checkpoint.pkl'
    m,path,new=r.recover_model('slot123',rows,p,train,registered.append,append)
    assert calls==['fit'] and new and path==p and m=={'weights':[1.,2.]}
    def forbidden():raise AssertionError('A completed fit must never restart')
    m,path,new=r.recover_model('slot123',rows,p,forbidden,registered.append,append)
    assert m=={'weights':[1.,2.]} and not new and calls==['fit'] and len(journal)==1


def test_published_model_with_missing_journal_is_adopted_without_fit(tmp_path):
    p=tmp_path/'model.pkl';r.commit_model(p,{'weights':[3.]},{'name':'interrupted_receipt','n_iter':317})
    rows={};log=[]
    def forbidden():raise AssertionError('Published model was already fitted')
    model,_,_=r.recover_model('interrupted_receipt',rows,p,forbidden,lambda p:None,log.append)
    assert model=={'weights':[3.]} and log[0]['n_iter']==317 and rows['interrupted_receipt']['sha256']


def test_atomic_failure_preserves_old_complete_output_and_partial(tmp_path):
    p=tmp_path/'state.json';p.write_bytes(b'old complete')
    def fail(f):f.write(b'incomplete');raise RuntimeError('simulated reboot during write')
    with pytest.raises(RuntimeError):r.atomic_write(p,fail,replace=True)
    assert p.read_bytes()==b'old complete' and list(tmp_path.glob('*.partial'))
    r.atomic_write(p,lambda f:f.write(b'new complete'),replace=True)
    assert p.read_bytes()==b'new complete'


def test_npz_and_metric_cache_reused_without_compute_or_rehash(tmp_path,monkeypatch):
    local=tmp_path/'local';local.mkdir();journal=tmp_path/'files.jsonl';store=r.Checkpoints(local,journal)
    p=local/'oof.npz';count=[]
    first=store.npz(p,lambda:(count.append(1) or {'probabilities':np.array([.2,.8]),'original_cache_index':np.array([10,11])}))
    meta=local/'metric.json';assert store.json(meta,lambda:{'AP':.9})=={'AP':.9}
    reopened=r.Checkpoints(local,journal)
    def forbidden(*args):raise AssertionError('Do not redo cached computation/hash')
    monkeypatch.setattr(r.b.b0,'sha',forbidden)
    np.testing.assert_array_equal(reopened.npz(p,forbidden)['probabilities'],first['probabilities'])
    assert reopened.json(meta,forbidden)=={'AP':.9} and count==[1]


def test_atomic_journal_and_csv_keep_existing_bytes(tmp_path):
    log=tmp_path/'fits.jsonl';r.append_jsonl(log,{'name':'old'});prefix=log.read_bytes()
    r.append_jsonl(log,{'name':'new'});assert log.read_bytes().startswith(prefix)
    assert [v['name'] for v in r.read_jsonl(log)]==['old','new']
    csv=tmp_path/'metrics.csv';csv.write_bytes(b'arm,fraction,repeat,AP\r\nR,0.1,0,0.12345678901234567\r\n');prefix=csv.read_bytes()
    r.append_csv(csv,{'arm':'S_plus_R','fraction':.1,'repeat':0,'AP':.8})
    assert csv.read_bytes().startswith(prefix)
    bad=tmp_path/'truncated.jsonl';bad.write_bytes(b'{"name":')
    r.append_jsonl(bad,{'name':'recover without losing interrupted evidence'})
    assert r.read_jsonl(bad)==[{'name':'recover without losing interrupted evidence'}]
    assert any(p.read_bytes()==b'{"name":' for p in tmp_path.glob('*.interrupted'))
    corrupt=tmp_path/'corrupt.jsonl';corrupt.write_bytes(b'bad complete row\n')
    with pytest.raises(ValueError):r.append_jsonl(corrupt,{'name':'do not skip corrupt complete records'})


def test_atomic_replace_permission_retry_is_bounded_and_reported(tmp_path,monkeypatch):
    original=r.os.replace;calls=[];r.IO_EVENTS.clear()
    def temporarily_locked(a,b):
        calls.append(1)
        if len(calls)<3:raise PermissionError('synthetic Windows sharing contention')
        return original(a,b)
    monkeypatch.setattr(r.os,'replace',temporarily_locked);monkeypatch.setattr(r.time,'sleep',lambda _:None)
    target=tmp_path/'output.json';r.atomic_write(target,lambda f:f.write(b'complete'))
    assert target.read_bytes()==b'complete' and len(calls)==3 and len(r.IO_EVENTS)==2


def test_single_runner_lock_releases_on_exception(tmp_path):
    p=tmp_path/'resume.lock'
    with r.single_runner(p):
        with pytest.raises(RuntimeError):
            with r.single_runner(p):pass
    with pytest.raises(ValueError):
        with r.single_runner(p):raise ValueError('simulated stop')
    with r.single_runner(p):pass


def test_prediction_is_same_fixed_4096_batch_path():
    class Fake:
        def predict_proba(self,x):return np.column_stack([1-x[:,0],x[:,0]])
    x=np.linspace(0,1,8200).reshape(-1,1)
    np.testing.assert_array_equal(r.predict(Fake(),x),x[:,0])
    assert r.predict(Fake(),np.empty((0,1))).shape==(0,)


def test_resume_uses_original_numeric_protocol_helpers():
    assert r.b is b and r.b.FRACTIONS==(.1,.25,.5,1.)
    assert r.b.GRID==(.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95)
    assert r.b.select_onset is b.select_onset and r.b.h1_bootstrap is b.h1_bootstrap
    assert r.b.event_record is b.event_record and r.b.event_bootstrap is b.event_bootstrap
