"""Synthetic server unit tests; no smoke or forbidden dataset access."""
from itertools import product
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/research'))
import numpy as np
import pandas as pd
import pytest
import torch
import train_w_phase_balance as train
import evaluate_w_phase_balance as e
from smoke_normal_source_sampling import sampling_weights,ORIGINAL,PREFAULT,FAULT

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)

@pytest.fixture
def ledger():
    rows=[]
    for f in train.FLIGHTS:
        for case,onset,end,y,origin in [('A',100,50,0,'normal_part_of_fault_case'),('A',100,150,1,'fault_or_unknown'),
            ('B',200,250,1,'fault_or_unknown'),('N',-1,100,0,'normal_flight'),('N',-1,250,0,'normal_flight')]:
            rows.append({'case_uid':f'SIL/{f}/GPS/{case}','fault_onset_row':onset,'window_end_row':end,
                'qualified_y':y,'normal_origin':origin})
    return pd.DataFrame(rows)

@pytest.mark.parametrize('arm',train.ARMS)
def test_force_one_is_elementwise_identical_to_original_sampler(ledger,arm):
    y=ledger.qualified_y.to_numpy()
    origins=np.where(y==1,FAULT,np.where(ledger.normal_origin.eq('normal_flight'),ORIGINAL,PREFAULT))
    expected=sampling_weights(y,origins,True)
    actual,stats=train.phase_weights(ledger,arm,force_one=True)
    assert torch.equal(actual,expected)
    assert all(c['multiplier']==1 for c in stats['cells'].values())

def test_unique_case_median_excludes_minus_one_and_boundary_is_pre(ledger):
    refs=train.onset_references(ledger)
    assert refs=={f:3. for f in train.FLIGHTS}
    flight,phase=train.cells(ledger,refs)
    assert phase[:5].tolist()==['pre','pre','post','pre','post']
    # Repeating windows from one case must not reweight its onset.
    duplicate=pd.concat([ledger,ledger.iloc[[0]]]*3,ignore_index=True)
    assert train.onset_references(duplicate)==refs
    assert set(flight)==set(train.FLIGHTS)

@pytest.mark.parametrize('arm',train.ARMS)
def test_formula_faults_unchanged_no_renormalization_probability_and_ess(ledger,arm):
    baseline,_=train.phase_weights(ledger,arm,force_one=True)
    weights,stats=train.phase_weights(ledger,arm)
    y=ledger.qualified_y.to_numpy();flight,phase=train.cells(ledger,stats['onset_reference_s'])
    assert torch.equal(weights[torch.from_numpy(y==1)],baseline[torch.from_numpy(y==1)])
    assert len(stats['cells'])==12
    for f in train.FLIGHTS:
        for p in ('pre','post'):
            c=stats['cells'][f+'/'+p]
            expected=max(1,c['r_c'] if arm=='W1' else np.sqrt(c['r_c']))
            assert c['multiplier']==expected
            mask=(flight==f)&(phase==p)&(y==0)
            assert torch.equal(weights[torch.from_numpy(mask)],baseline[torch.from_numpy(mask)]*expected)
    assert stats['effective_sample_size']==float(weights.sum())**2/float(weights.square().sum())
    assert sum(c['sampling_probability'] for c in stats['cells'].values())==pytest.approx(1.)
    assert not stats['renormalized'] and stats['draws_per_epoch']==len(ledger)


def test_no_normal_cell_uses_ratio_one(ledger):
    ledger=ledger.drop(index=[4])
    _,stats=train.phase_weights(ledger,'W1')
    assert stats['cells']['acce/post']['N_c']==0 and stats['cells']['acce/post']['r_c']==1
    assert stats['cells']['acce/post']['multiplier']==1


def test_missing_or_inconsistent_reference_rejected(ledger):
    bad=ledger[~ledger.case_uid.str.contains('/acce/')]
    with pytest.raises(ValueError,match='Every flight'):train.onset_references(bad)
    ledger.loc[1,'fault_onset_row']=101
    with pytest.raises(ValueError,match='Inconsistent'):train.onset_references(ledger)


def inputs(flags):
    c1,c2,c3,c4=flags
    point={'per_seed_ratios':{str(s):.8 if c1 else 1. for s in train.SEEDS},
        'C2_delta_pp':{'GPS':-1. if c2 else -1.001},'C3_drop_pp':{'GPS':2. if c3 else 2.001}}
    ci={'ratio_ci':[.7,.9],'delta_ap_ci':[-.1,0. if c4 else -.001]}
    return point,ci

@pytest.mark.parametrize('flags',list(product((False,True),repeat=4)))
def test_all_pass_fail_combinations_and_exact_caps(flags):
    result=e.decide(*inputs(flags))
    assert result['criteria']==dict(zip(('C1','C2','C3','C4'),flags))
    assert result['verdict']==('PASS' if all(flags) else 'FAIL')
    assert result['failed_criteria']==[c for c,b in zip(('C1','C2','C3','C4'),flags) if not b]

@pytest.mark.parametrize('first,second,ratios,wanted',[
    ('FAIL','FAIL',{'W1':.7,'W2':.8},'KEEP_T0'),('PASS','FAIL',{'W1':.7,'W2':.8},'ADOPT_W1'),
    ('FAIL','PASS',{'W1':.7,'W2':.8},'ADOPT_W2'),('PASS','PASS',{'W1':.7,'W2':.8},'ADOPT_W1'),
    ('PASS','PASS',{'W1':.8,'W2':.7},'ADOPT_W2')])
def test_adoption_every_registered_branch(first,second,ratios,wanted):
    assert e.adopt({'W1':{'verdict':first},'W2':{'verdict':second}},ratios)==wanted


def test_equal_pass_ratios_do_not_invent_unregistered_tie_break():
    with pytest.raises(ValueError,match='tie-break'):
        e.adopt({a:{'verdict':'PASS'} for a in train.ARMS},{a:.8 for a in train.ARMS})


def frame():
    return pd.DataFrame({'case_uid':['SIL/hover/GPS/'+str(i) for i in range(4)],'qualified_y':[0,1,0,1],
        'normal_origin':['normal_flight','fault_or_unknown']*2,'window_end_row':2000,'association_group':['a','a','b','b']})

def test_primary_class_windows_include_late_hover_but_v_descriptive_excludes():
    f=frame();scores={a:{sd:np.array([.8,.7,.8,.7]) for sd in train.SEEDS} for a in e.ARMS}
    prepared=e.prepare(f,scores)
    assert prepared['masks']['GPS'].sum()==2 and prepared['outside_masks']['GPS'].sum()==0
    assert e.measure(prepared)['W1']['C3_drop_pp']['GPS']==0
    assert e.measure(prepared,outside=True)['W1']['C3_drop_pp']['GPS'] is None


def test_bootstrap_single_weight_plan_for_all_fifteen_vectors(monkeypatch):
    f=frame();scores={a:{sd:np.array([.8,.7,.8,.7]) for sd in train.SEEDS} for a in e.ARMS}
    prepared=e.prepare(f,scores);original=e.m.MetricCache.basic;calls=[]
    def record(self,weights=None,recall=.97):
        calls.append(id(weights));return original(self,weights,recall)
    monkeypatch.setattr(e.m.MetricCache,'basic',record)
    result=e.bootstrap(prepared,f.association_group.to_numpy(),replicates=4)
    assert len(calls)==60 and all(len(set(calls[i:i+15]))==1 for i in range(0,60,15))
    for arm in train.ARMS:
        assert result['comparisons'][arm]['ratio_ci']==[1.,1.]
        assert result['comparisons'][arm]['delta_ap_ci']==[0.,0.]


def test_undefined_intervals_fail_without_pseudocounts():
    point,ci=inputs((True,True,True,True));ci['ratio_ci']=None;ci['delta_ap_ci']=None
    result=e.decide(point,ci)
    assert result['verdict']=='FAIL' and result['failed_criteria']==['C1','C4']
