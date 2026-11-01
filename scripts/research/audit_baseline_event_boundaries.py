"""Post-run diagnostic only: do not change predictions, thresholds, or labels."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/2026-10-08/uav_baselines_v1'

def label_comparison(r):
    payload=Path(r['final_output_path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=r['current_sha256']:raise ValueError('Source changed')
    import io
    d=pd.read_csv(io.BytesIO(payload),usecols=['Timestamp','_rfly_ctrl_lxl_0_id','UAVState_data_fault_state'])
    t=(d.Timestamp-d.Timestamp.iloc[0]).to_numpy()/1e6
    ids=d['_rfly_ctrl_lxl_0_id']
    command=(ids.between(123450,123459)|ids.between(123540,123549)).to_numpy()
    state=d.UAVState_data_fault_state.to_numpy()>0
    result={k:r[k] for k in ['case_uid','domain','split','level3_label','flight_status']}
    result.update(command_active=bool(command.any()),state_active=bool(state.any()),observed_control_ids=ids.unique().tolist())
    if command.any() and state.any():
        result.update(command_onset_s=float(t[command][0]),state_onset_s=float(t[state][0]),
                      state_minus_command_s=float(t[state][0]-t[command][0]))
    return result

def main():
    metas=json.loads((OUT/'case_metadata.json').read_text(encoding='utf-8'))
    thresholds=json.loads((OUT/'thresholds.json').read_text(encoding='utf-8'))
    diagnostics=[]
    for m in metas:
        if not m['intervals']:continue
        s=np.load(OUT/m['score_file']);t=s['t']
        for model in ['pca','gru','persistence']:
            q=s[model]>thresholds[model];alarm=q.copy();alarm[:2]=False;alarm[2:]&=q[1:-1]&q[:-2]
            edges=alarm & ~np.r_[False,alarm[:-1]]
            begin,end=m['intervals'][0];idx=np.searchsorted(t,begin)
            at_start=bool(idx<len(t) and alarm[idx])
            previous_edges=np.flatnonzero(edges & (t<=t[idx])) if idx<len(t) else np.array([],dtype=int)
            preceding_edge=float(t[previous_edges[-1]]) if len(previous_edges) else None
            ongoing=at_start and preceding_edge is not None and preceding_edge<begin
            diagnostics.append(dict(case_uid=m['case_uid'],split=m['split'],model=model,
                any_alarm_during_fault=bool(alarm[(t>=begin)&(t<end)].any()),
                ongoing_alarm_from_before_onset=ongoing,
                ongoing_alarm_lead_s=begin-preceding_edge if ongoing else None))
    pd.DataFrame(diagnostics).to_csv(OUT/'event_definition_sensitivity.csv',index=False,encoding='utf-8-sig')
    manifest=pd.read_csv(OUT/'frozen_case_index.csv',keep_default_na=False)
    hil=manifest[(manifest.domain=='HIL')&manifest.split.isin(['val','test'])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        labels=list(pool.map(label_comparison,hil.to_dict('records')))
    d=pd.DataFrame(labels);d.to_csv(OUT/'hil_control_state_onset_comparison.csv',index=False,encoding='utf-8-sig')
    delta=d.state_minus_command_s.dropna()
    summary=dict(hil_cases=len(d),hil_both_active=len(delta),
        state_active_without_command=int((d.state_active & ~d.command_active).sum()),
        state_minus_command_s_quantiles={str(q):float(delta.quantile(q)) for q in [0,.25,.5,.75,1]},
        state_later_than_command=int((delta>1e-6).sum()),state_earlier_than_command=int((delta< -1e-6).sum()),
        caution='Two processed event markers disagree. This comparison cannot identify true physical onset or prove model early warning; original labels and metrics remain unchanged.')
    (OUT/'event_boundary_diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
