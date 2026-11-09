"""Synthetic-only B1 checks. Never execute the real entry point."""
from pathlib import Path
import sys,subprocess
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
import real_v2_b1_learning_curve as b


def fake_metadata():
    rows=[]
    for fold in range(5):
        for family in ('motor','sensor','normal_full','normal_segment'):
            for i in range(7):
                rows.append({'case_uid':f'{fold}/{family}/{i}','association_group':f'g{fold}_{family}_{i}',
                             'stratum_family':family,'dev_fold':fold})
    rows.append({**rows[0],'case_uid':'a_second_case_same_group'})
    return pd.DataFrame(rows)


def test_subsets_nested_hash_fixed_rep_and_outer_group_exclusion():
    e=fake_metadata();held=set(e.loc[e.dev_fold.eq(2),'association_group'])
    for rep in range(5):
        previous=set()
        for frac,n in zip(b.FRACTIONS,[12,28,56,112]):
            selected=set(b.subset_groups(e,2,frac,rep))
            assert len(selected)==n and previous<=selected and not selected&held
            assert selected==set(b.subset_groups(e,2,frac,rep))
            previous=selected
    assert b.subset_groups(e,2,.1,0)!=b.subset_groups(e,2,.1,1)
    assert b.subset_groups(e,2,1.,0)==b.subset_groups(e,2,1.,4)
    with pytest.raises(ValueError):b.subset_groups(e,2,.2,0)


def test_source_real_equal_total_weight_and_preserved_real_ratios():
    sw=np.array([.5,.5,1,1]);rw=np.array([1.,2.,3.])
    w,scale=b.domain_weights(sw,rw)
    np.testing.assert_array_equal(w[:4],sw)
    assert w[:4].sum()==pytest.approx(w[4:].sum()) and scale==.5
    np.testing.assert_allclose(w[4:]/rw,scale)
    with pytest.raises(ValueError):b.domain_weights(sw,np.array([0.,1.]))


@pytest.mark.parametrize('p',[[.1,.5,.5,.9],[1,1,1,1],[.8,.1,.8,.1]])
def test_weighted_ap_matches_sklearn_with_ties_zero_weight_blocks(p):
    y=np.array([0,1,0,1]);p=np.array(p)
    for w in [np.array([2,3,4,1]),np.array([0,0,1,2]),np.ones(4,int)]:
        assert b.weighted_ap(b.ap_plan(p,y),w)==pytest.approx(average_precision_score(y,p,sample_weight=w),abs=1e-15)
    assert b.weighted_ap(b.ap_plan(p,y),np.array([1,0,1,0])) is None


def test_H1_is_mean_of_paired_repeat_delta_not_averaged_predictions():
    y=np.array([0,1,0,1]);frame=pd.DataFrame({'qualified_y':y,'association_group':['a','a','b','b']})
    meta=pd.DataFrame({'association_group':['a','b']});probs={}
    for frac in b.FRACTIONS:
        for rep in range(1 if frac==1 else 5):
            probs[('R',frac,rep)]=np.full(4,.5)
            probs[('S_plus_R',frac,rep)]=y.astype(float) if frac==.25 and rep==0 else np.full(4,.5)
    out=b.h1_bootstrap(probs,frame,meta,reps=31,seed=4)
    assert out['fractions']['0.25']['repeat_mean_delta_AP']==pytest.approx(.1)
    np.testing.assert_allclose(out['fractions']['0.25']['ci95'],[.1,.1])
    assert out['fractions']['0.25']['holds']
    assert out['fractions']['0.1']['ci95']==[0.,0.]


def choice_row(th,m=8,s=8,events=2,frames=100):
    return {'onset_th':th,'N1':{'events':events,'normal_frames':frames},'N2':{'events':0,'normal_frames':frames},
            'motor':{'cases':10,'detected':m},'sensor':{'cases':10,'detected':s}}


def test_feasible_selection_rate_then_detection_then_lower_onset():
    rows=[choice_row(.45,8,8),choice_row(.5,9,8),choice_row(.55,9,8)]
    selected=b.select_onset(rows)
    assert selected['onset_th']==.5 and not selected['no_feasible_onset']
    assert selected['tied_onsets_primary']==[.45,.5,.55]
    assert selected['tied_onsets_after_detection']==[.5,.55]
    rows.append(choice_row(.6,8,8,events=1))
    assert b.select_onset(rows)['onset_th']==.6


def test_supplement1_infeasible_maxmin_total_then_lower_with_full_tie_list():
    rows=[choice_row(.45,6,8),choice_row(.5,7,6),choice_row(.55,6,9),choice_row(.6,9,6)]
    selected=b.select_onset(rows)
    assert selected['no_feasible_onset'] and selected['onset_th']==.55
    assert selected['tied_onsets_primary']==[.45,.5,.55,.6]
    assert selected['tied_onsets_after_detection']==[.55,.6]
    assert selected['supplement1_fallback_A']


def arrays():
    return {'timestamp_us':np.arange(10,dtype=np.int64)*1000000,
            'reference_state':np.array([0,0,0,0,0,1,1,0,0,0]),'row_label_valid':np.ones(10,bool)}


def fake_aggregate(t,p,policy):
    assert policy.inference_every==1 and policy.k_on==policy.k_off==3 and policy.clear_th==.4
    assert policy.min_event_s==policy.merge_gap_s==.5
    return {'finalized_events':[{'start_t':1.,'end_t':2.},{'start_t':7.,'end_t':8.},{'start_t':4.,'end_t':7.}],
            'confirmations':[{'candidate_start_t':1.,'confirmed_t':1.04},{'candidate_start_t':5.,'confirmed_t':5.04}]}


def test_event_N2_only_prefault_post_separate_and_new_confirmation_not_spans():
    meta={'role_v1':'test_fault','family':'sensor','case_uid':'synthetic'};a=arrays();p=np.full(10,.1)
    row=b.event_record(meta,a,p,.45,aggregate=fake_aggregate)
    assert row['N1_events']==0 and row['N2_events']==1 and row['post_events']==1
    assert row['N2_frames']==5 and row['post_frames']==3 and row['post_onset_scored']
    assert row['detected'] and row['latency_s']==pytest.approx(.04) and row['spans_onset']
    p[1]=np.nan;row=b.event_record(meta,a,p,.45,aggregate=fake_aggregate)
    assert row['N2_events']==0 and row['N2_frames']==4  # Entire event must be in scored normal_time.


def test_N1_and_supplement_are_role_distinct_not_directory_or_stratum():
    a=arrays();a['reference_state'][:]=0;p=np.full(10,.1)
    base={'family':'normal','case_uid':'synthetic'}
    n=b.event_record({**base,'role_v1':'test_normal'},a,p,.5,aggregate=fake_aggregate)
    s=b.event_record({**base,'role_v1':'supplemental_normal_test'},a,p,.5,aggregate=fake_aggregate)
    assert n['N1_events']==3 and n['N1_frames']==10 and n['N2_events']==0
    assert s['N1_events']==s['N2_events']==0 and s['supplement_events']==3
    assert not n['detected'] and not s['detected']


def test_H2_inclusive_boundaries_and_consequences_never_autolaunch():
    x={'N1':{'rate':5.5},'N2':{'rate':5.5},'motor':{'detection':.8},'sensor':{'detection':.8}}
    assert b.h2(x)['holds']
    x['N2']['rate']=5.500001
    assert b.h2(x)['failed']==['N2']
    for a,c,key in [(True,True,'H1_and_H2'),(False,True,'H2_only'),(True,False,'H2_fails')]:
        out=b.consequences(a,c);assert out['key']==key and not out['automatic_next_step']


def test_event_bootstrap_groups_rates_and_latency_empty_are_not_zero():
    rows=[]
    for group in ('a','b'):
        for family in ('normal','motor','sensor'):
            normal=family=='normal';motor=family=='motor'
            rows.append({'case_uid':group+family,'association_group':group,'family':family,
                'N1_events':1 if normal else 0,'N1_frames':100 if normal else 0,
                'N2_events':0,'N2_frames':0 if normal else 100,'post_events':0,'post_frames':0,
                'supplement_events':0,'supplement_frames':0,'post_onset_scored':not normal,
                'detected':motor,'latency_s':1. if motor else None})
    point=b.event_summary(rows)
    assert point['N1']['rate']==1800 and point['N2']['rate']==0
    assert point['motor']['detection']==1 and point['sensor']['detection']==0
    assert point['sensor']['latency_median_s'] is None
    out=b.event_bootstrap({'R':rows,'S_plus_R':rows},pd.DataFrame({'association_group':['a','b']}),reps=23,seed=5)
    assert out['arms']['R']==out['arms']['S_plus_R']
    assert out['arms']['R']['sensor/latency_median_s']['undefined_replicates']==23
    assert out['arms']['R']['sensor/detection']['ci95']==[0.,0.]


def test_boundary_source_train_allowed_but_sealed_test_and_source_writes_denied(tmp_path):
    code=r'''
from pathlib import Path
import sys,pandas as pd
sys.path.insert(0,sys.argv[1]);import real_v2_b1_learning_curve as b
p=Path(sys.argv[2]);data=p/'inputs/causal';local=p/'inputs/runtime';source=p/'models/S.pkl'
(data/'cache').mkdir(parents=True);(data/'windows').mkdir();local.mkdir();source.parent.mkdir();source.write_bytes(b'model')
(data/'cache/train.pt').write_bytes(b'train');(data/'windows/REAL__hover__GPS__dev.npz').write_bytes(b'dev')
a=b.boundary(pd.DataFrame({'case_uid':['REAL/hover/GPS/dev']}),data,local,source)
assert (data/'cache/train.pt').read_bytes()==b'train'
assert (data/'windows/REAL__hover__GPS__dev.npz').read_bytes()==b'dev'
(local/'scores.npz').write_bytes(b'new')
blocked=0
for work in [lambda:(data/'cache/test.pt').read_bytes(),lambda:(data/'windows/REAL__hover__GPS__sealed.npz').read_bytes(),lambda:(data/'cache/train.pt').write_bytes(b'bad'),lambda:(source.parent/'test_predictions.npz').read_bytes()]:
 try:work()
 except PermissionError:blocked+=1
assert blocked==4 and len(a['blocked'])==4
'''
    result=subprocess.run([sys.executable,'-X','utf8','-E','-P','-B','-c',code,str(b.ROOT/'scripts/research'),str(tmp_path)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
