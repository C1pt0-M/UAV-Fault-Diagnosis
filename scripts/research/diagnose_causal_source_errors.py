"""Explain the completed source-validation errors, no new model/data audit.

Consumes saved predictions and versioned source ledgers only. Original crop
length comes from existing rebuilt-case metadata, not raw flight reinspection.
Never changes labels, windows, thresholds, checkpoint selection or deployment.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('D:/UAV-Data/causal_input_v1')
RESULTS=ROOT/'artifacts/causal_source_remote_v1_20261031/results'
OUT=ROOT/'artifacts/causal_source_error_diagnosis_v1_20261031'
ARMS=['uniform_replacement','balanced_source_normal']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def counts(frame):
    n=len(frame);fp=int(frame.fp.sum());fn=int(frame.fn.sum())
    return {'windows':n,'cases':int(frame.case_uid.nunique()),'false_positives':fp,'false_negatives':fn,
            'false_positive_rate':fp/n if n else None}


def main():
    if OUT.exists():raise FileExistsError('Preserve earlier diagnosis')
    ledger=DATA/'ledgers/val_included_windows.csv'
    meta=pd.read_csv(ledger,dtype={'case_uid':str,'association_group':str},keep_default_na=False,low_memory=False)
    trainpath=DATA/'ledgers/train_included_windows.csv'
    train=pd.read_csv(trainpath,dtype={'case_uid':str,'association_group':str},keep_default_na=False,low_memory=False)
    for f in [meta,train]:f['mode']=f.case_uid.str.split('/').str[1]
    pure=meta.case_level2_label.eq('normal')
    hover=pure & meta['mode'].eq('hover')
    assert len(meta)==34710 and int(pure.sum())==4565 and int(hover.sum())==754
    assert len(train)==162069
    length={}
    for uid in meta.loc[pure,'case_uid'].unique():
        path=DATA/'case_records'/(uid.replace('/','__')+'.json')
        record=read(path);length[uid]=int(record['summary']['rows'])
    # Relative progress means position within retained processing crop, NOT an
    # authenticated flight phase or physical landing/takeoff annotation.
    meta['crop_progress']=(meta.window_end_row-1)/(meta.case_uid.map(length)-1)
    OUT.mkdir(parents=True,exist_ok=False)
    input_ids={str(ledger):sha(ledger),str(trainpath):sha(trainpath)}
    summary={'status':'COMPLETED_SAVED_SOURCE_PREDICTION_ERROR_DIAGNOSIS','created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Saved source train/val metadata and completed source-val predictions only. No raw logs, test/Real, new inference, training or threshold changes.',
        'arms':{},'normal_hover_train_support':{
            'cases':int(train.loc[(train.case_level2_label=='normal') & train['mode'].eq('hover'),'case_uid'].nunique()),
            'windows':int(((train.case_level2_label=='normal') & train['mode'].eq('hover')).sum())},
        'source_val_population':{'all_windows':34710,'normal_windows':17740,'inactive_fault_case_normal':13175,
            'original_normal_windows':4565,'original_normal_cases':72,'original_normal_hover_windows':754,'original_normal_hover_cases':12}}
    predictions={}
    for arm in ARMS:
        path=RESULTS/'runs'/arm/'source_validation_predictions.npz';input_ids[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as d:
            assert np.array_equal(d['labels'],meta.qualified_y)
            assert np.array_equal(d['original_indices'],meta.original_cache_index)
            assert np.array_equal(d['compact_indices'],np.arange(len(meta)))
            probabilities=d['probabilities'];labels=d['labels'];pred=probabilities.argmax(1)
            assert probabilities.shape==(34710,2) and np.isfinite(probabilities).all()
        predictions[arm]=pred
        frame=meta.copy();frame['p_fault']=probabilities[:,1];frame['fp']=(labels==0)&(pred==1);frame['fn']=(labels==1)&(pred==0)
        cases=frame.loc[pure].groupby(['domain','mode','case_uid'],sort=True).agg(
            windows=('fp','size'),false_positives=('fp','sum'),fault_score_median=('p_fault','median'),fault_score_max=('p_fault','max'))
        cases['false_positive_rate']=cases.false_positives/cases.windows
        cases.reset_index().to_csv(OUT/(arm+'_normal_case_errors.csv'),index=False)
        pos=[]
        for j in range(4):
            mask=hover & (frame.crop_progress>=j/4) & (frame.crop_progress<(j+1)/4 if j<3 else frame.crop_progress<=1)
            pos.append(dict(crop_progress_quarter=j+1,**counts(frame.loc[mask])))
        assert sum(c['windows'] for c in pos)==754
        domains={dom:counts(g) for dom,g in frame.loc[hover].groupby('domain')}
        confidence={}
        for group,mask in [('all_original_normal_false_positives',pure & frame.fp),('original_hover_false_positives',hover & frame.fp)]:
            scores=probabilities[mask,1]
            confidence[group]={'windows':len(scores),'median_fault_score':float(np.median(scores)),
                'min_fault_score':float(scores.min()),'max_fault_score':float(scores.max()),
                'fault_score_below_0_6':int((scores<.6).sum()),'fault_score_at_least_0_9':int((scores>=.9).sum())}
        faults={family:{'windows':len(g),'missed_faults':int(g.fn.sum()),'recall':1-float(g.fn.mean())}
                for family,g in frame.loc[labels==1].groupby('case_level2_label')}
        report=read(RESULTS/'runs'/arm/'report.json')['source_validation']
        assert int(frame.fp.sum())==report['overall']['confusion_matrix'][0][1]
        assert int(frame.loc[hover,'fp'].sum())==report['original_normal_by_status']['hover']['confusion_matrix'][0][1]
        result={'all_normal':counts(frame.loc[labels==0]),'original_normal':counts(frame.loc[pure]),
            'inactive_fault_case_normal':counts(frame.loc[(labels==0)&~pure]),'normal_hover':counts(frame.loc[hover]),
            'hover_by_domain':domains,'hover_by_original_crop_quarter':pos,'confidence':confidence,
            'normal_cases_with_any_false_positive':int((cases.false_positives>0).sum()),
            'normal_hover_cases_with_any_false_positive':int((cases.loc[cases.index.get_level_values('mode')=='hover','false_positives']>0).sum()),
            'fault_family':faults}
        history=read(RESULTS/'runs'/arm/'history.json')
        result['epoch_trajectory']=[{'epoch':h['epoch'],'training_loss':h['train_loss'],
            'overall_macro_f1':h['source_validation']['overall']['macro_f1'],
            'original_normal_hover_fpr':h['source_validation']['original_normal_by_status']['hover']['normal_window_fpr']}
            for h in history]
        summary['arms'][arm]=result
    a=hover & (predictions[ARMS[0]]==1);b=hover & (predictions[ARMS[1]]==1)
    summary['paired_hover_errors']={'both_false_positive':int((a&b).sum()),'uniform_only':int((a&~b).sum()),
        'balanced_only':int((~a&b).sum()),'both_correct':int((hover&~a&~b).sum())}
    summary['interpretation']={
        'confirmed':'False positives concentrate on original-normal flights, especially later parts of hover crops; both SIL/HIL affected. Reweighting removes some false positives with fault-recall tradeoff.',
        'supported_hypothesis':'Current learned boundary distinguishes source flight backgrounds/stages and normal sources insufficiently; pooled source-val Macro-F1 hides worst normal operating condition.',
        'not_proven':'Exact sensor/feature or model branch causing errors; physical landing/ground state; dataset label corruption; fundamental inability of dataset; improvement versus old-data models. Train-normal inference not performed, so fit failure vs held-out normal generalization is not settled.',
        'no_posthoc_changes':'0.5 decision threshold and checkpoint/early stop unchanged. Confidence summaries are descriptive, not a recommendation to raise threshold. Do not delete misclassified valid cases or retune using final tests.'}
    for path,expected in input_ids.items():assert sha(path)==expected
    summary['input_identities']=input_ids;summary['script_sha256']=sha(__file__)
    with (OUT/'summary.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['input_identities','arms']},ensure_ascii=False,indent=2))
    for arm in ARMS:print(arm,json.dumps({k:summary['arms'][arm][k] for k in ['normal_hover','hover_by_original_crop_quarter','confidence']},ensure_ascii=False))


if __name__=='__main__':main()
