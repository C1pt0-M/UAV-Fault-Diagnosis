from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import pandas as pd
R=Path.cwd();O=R/'artifacts/real_v2_b1_n2_audit_v1';assert not O.exists();O.mkdir()
S=R/'configs/data/real_v2_split.csv';assert hashlib.sha256(S.read_bytes()).hexdigest()=='9232ee35e7a59f4fac4534fc8e98fc454380b51e6e5d8439aaa0f769cd1090ee'
s=pd.read_csv(S,keep_default_na=False);s=s[s.partition.eq('dev') & s.label_quarantine.astype(str).str.lower().eq('false')].copy()
d=pd.read_csv(R/'artifacts/real_v2_b1_learning_curve/events_per_case.csv');assert set(d.case_uid)<=set(s.case_uid)
d=d[d.arm.isin(['R','S_plus_R']) & d.role_v1.eq('test_fault')].copy()
p=d.pivot(index='case_uid',columns='arm',values='N2_events');alerts=sorted(p.index[p.sum(axis=1).gt(0)]);zero=set(p.index[p.sum(axis=1).eq(0)])
base=d[d.arm.eq('R')].set_index('case_uid');pairs=[]
for uid in alerts:
 a=base.loc[uid];sub=uid.split('/')[2]
 candidate=base.loc[base.index.isin(zero) & base.flight.eq(a.flight) & base.family.eq(a.family) & base.association_group.ne(a.association_group)].copy()
 candidate['same_subtype']=[v.split('/')[2]==sub for v in candidate.index]
 candidate['same_fold']=candidate.dev_fold.eq(a.dev_fold);candidate['length_difference']=(candidate.N2_frames-a.N2_frames).abs()
 candidate['uid']=candidate.index
 candidate=candidate.sort_values(['same_subtype','same_fold','length_difference','uid'],ascending=[False,False,True,True])
 groups=set()
 for v in candidate.itertuples():
  if v.association_group in groups:continue
  pairs.append({'alert_uid':uid,'control_uid':v.Index,'same_subtype':bool(v.same_subtype),'same_fold':bool(v.same_fold),'pre_scored_frame_difference':int(v.length_difference)})
  groups.add(v.association_group)
  if len(groups)==2:break
 assert len(groups)==2,uid
selected=sorted(set(alerts)|{v['control_uid'] for v in pairs});records={}
for uid in selected:
 pth=Path('D:/UAV-Data/causal_input_v1/case_records')/(uid.replace('/','__')+'.json');v=json.loads(pth.read_text(encoding='utf-8'));assert v['case_uid']==uid and not bool(v.get('label_quarantine',False));records[uid]=v
files=[]
for folder in [R/'artifacts/real_v2_b1_learning_curve',R/'artifacts/real_v2_b1_learning_curve_checks',R/'artifacts/real_v2_b1_learning_curve_checks_resume']:
 for path in folder.iterdir():
  if path.is_file():files.append({'file':path.relative_to(R).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mtime_ns':path.stat().st_mtime_ns})
plan={'created_utc':datetime.now(timezone.utc).isoformat(),'authority':'user B1 N2 audit PI v1; no new model inference or training','eligible_cases':315,
 'alert_cases':alerts,'selected_cases':selected,'controls':pairs,'selection':'all20 alerts; 2 controls per alert, sameflight/family, different group, prioritize exact subtype then sameouterfold then closest N2_frames then UID; no raw-signal/score-based picking',
 'signal_windows':'baseline start-5s to start-1s; event inclusive; post end+1s to end+5s; require >=25 valid rows per interval; controls use identical lead to own onset, no extrapolation/interpolation',
 'channels':'all26 raw canonical channels; ch25 recomputed norm is derived, not independent sensor; show physical raw altitude and note model windows use relative-first altitude',
 'summary_only':'perchannel median, IQR, p05,p95, median_abs_delta, zero_delta_fraction, relative baseline shift; descriptive, no new statistical selection or model optimization',
 'classification':'A requires independent physical/injection corroboration beforereference; B needs identified maneuver and suitable normal controls, not label alone; C needs clock/command mismatch; D needs independent acquisition/batch metadata; otherwise E; hypotheses separate from primary supported attribution',
 'event_overlap':'samecase intervals overlap inclusively; graph components allow one-to-many; no tolerance or onset retuning',
 'typical_plot_selection':'top waypoint case by max per-arm N2 count, top other waypoint not already chosen, top waypoint SR-only case, first lexicographic waypoint sensor alert; other20 cases all quantified',
 'no_label_edits':True,'no_predictions':True,'no_training':True,'records':records}
for name,value in [('plan.json',plan),('B1_protected_before.json',files)]:
 (O/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pd.DataFrame(pairs).to_csv(O/'control_pairs.csv',index=False)
print('SELECTED',len(selected),'alerts',len(alerts),'unique_controls',len(set(selected)-set(alerts)))
print(pd.DataFrame(pairs).to_string(index=False))
uid=next(v for v in alerts if '/waypoint/' in v);root=Path('D:/RflyMAD').joinpath(*uid.split('/'))
print('SAMPLE_TESTINFO',uid,[str(p) for p in root.glob('TestInfo*')])
