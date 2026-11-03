"""Minimal read-only checks for M3 saved-result summarization."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/research'))
from collect_m3_formal_results import cm_metrics,hover_segment_masks,metrics_match,check_selection,ap_score


def test_hover_segments_do_not_include_other_scenarios_or_unknowns():
    frame=pd.DataFrame({'flight_status':['hover','hover','velocity','waypoint','hover'],
                        'window_end_row':[150,1600,150,1700,1700]})
    masks=hover_segment_masks(frame,np.array([True,True,True,True,False]))
    assert masks['first30s'].tolist()==[True,False,False,False,False]
    assert masks['after30s'].tolist()==[False,True,False,False,False]


def test_metric_counts_exact_AP_only_tolerates_roundoff():
    assert metrics_match({'average_precision':.9},{'average_precision':.9+2e-16})
    assert not metrics_match({'tp':9},{'tp':8})
    assert not metrics_match({'recall':.9},{'recall':.9+2e-16})
    assert not metrics_match({'average_precision':.9},{'average_precision':.91})


def test_conditional_fault_precision_uses_shared_normal_pool():
    # Two target faults, three normals; another fault type is excluded.
    y=np.array([1,1,0,0,0]);pred=np.array([1,0,1,0,0])
    m=cm_metrics(y,pred)
    assert m['confusion_matrix']==[[2,1],[1,1]]
    assert m['precision']==m['fault_recall']==m['fault_f1']==.5
    assert cm_metrics([1,0],[0,0])['fault_f1']==0
    assert ap_score([1,0],[.5,.5])==.5


def test_strict_best_retains_earliest_tie_and_requires_stop():
    rule=dict(min_epochs=2,patience=2,significant_delta=.002)
    history=[]
    for i,bad in [(1,0),(2,1),(3,2)]:
        history.append({'epoch':i,'train_loss':.2,'source_validation':{'overall':{'macro_f1':.8}},
                        'selection':dict(best_score=.8,best_epoch=1,significant_reference=.8,bad_epochs=bad)})
    selected,stop=check_selection(history,rule)
    assert selected['best_epoch']==1 and stop==3
    with pytest.raises(ValueError,match='without registered early stop'):
        check_selection(history[:2],rule)
