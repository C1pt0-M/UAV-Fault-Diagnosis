"""Integer-clock diagnostic tables and fixed-selection figures; no event replay/model."""
from pathlib import Path
import sys,os,json,collections
import numpy as np
import pandas as pd
r=Path.cwd();o=r/'artifacts/real_v2_b1_n2_audit_v1';c=r/'artifacts/real_v2_b1_n2_audit_v1_checks'
sys.path[:0]=[str(r/'scripts/research'),str(r/'packages/uav_fault')]
import audit_real_v2_b1_n2 as a
plan=a.read_json(o/'plan_r2.json');events=pd.read_csv(o/'events.csv').to_dict('records');selected=plan['selected_cases'];records=plan['records']
b1=pd.read_csv(a.B1/'events_per_case.csv');b1=b1[b1.arm.isin(['R','S_plus_R'])];base=b1[b1.arm.eq('R')].set_index('case_uid');registry=a.read_json(a.B1/'receipt.json')['local_artifacts_sha256']
counts=collections.Counter(v['case_uid'] for v in events)
way=[u for u in plan['alert_cases'] if '/waypoint/'in u]
def rank(u):
 row=b1[b1.case_uid.eq(u)];return (-int(row.N2_events.max()),-int(row.N2_events.sum()),u)
first=sorted(way,key=rank)[0];second=sorted([u for u in way if u!=first],key=rank)[0]
third=sorted([u for u in way if int(base.loc[u,'N2_events'])==0],key=rank)[0]
fourth=sorted([u for u in way if base.loc[u,'family']=='sensor'])[0];typical=list(dict.fromkeys([first,second,third,fourth]))
svg='--export-svg'in sys.argv;plot_only='--plot-only'in sys.argv;revision='r2' if (plot_only or svg) else 'r1'
if svg:
 assert a.read_json(c/'visual_review.json')['all_four_reviewed_pass']
elif not plot_only:
 assert not(o/'channel_period_statistics_integer_us.csv').exists()
allowed={a.DATA/'input_contract.json':{'kind':'input_contract'}}
for uid in selected:
 key=uid.replace('/','__')+'.npz';row=base.loc[uid];allowed[Path(records[uid]['frame_file'])]={'kind':'canonical_case','case_uid':uid}
 for arm in ['R','S_plus_R']:
  name=f'outer_R_f1p00_rep0_k{int(row.dev_fold)}' if arm=='R' else f'outer_S_plus_R_f1.00_rep0_k{int(row.dev_fold)}'
  allowed[a.RUNTIME/'scores'/name/key]={'kind':'existing_frame_score','case_uid':uid,'arm':arm}
# One metadata inconsistency is checked against the original received command, not inferred from scores.
uid124='REAL/waypoint/motor/124_1';allowed[Path(records[uid124]['raw_ulog'])]={'kind':'raw_ULog_parameter_check','case_uid':uid124}
audit=a.install_guard(allowed);columns=a.read_json(a.DATA/'input_contract.json')['columns'];arrays={};scores={}
needed=set(selected) if not (svg or plot_only) else set(typical)
controls={}
for uid in typical:
 pairs=[v for v in plan['controls'] if v['alert_uid']==uid]
 control=min(pairs,key=lambda v:(not v['same_subtype'],v['pre_scored_frame_difference'],not v['same_fold'],v['control_uid']))['control_uid']
 controls[uid]=control;needed.add(control)
for uid in sorted(needed):
 rec=records[uid];arrays[uid]=a.npz(rec['frame_file'],rec['output_identities'][rec['frame_file']]);row=base.loc[uid];key=uid.replace('/','__')+'.npz'
 for arm in ['R','S_plus_R']:
  name=f'outer_R_f1p00_rep0_k{int(row.dev_fold)}' if arm=='R' else f'outer_S_plus_R_f1.00_rep0_k{int(row.dev_fold)}';path=a.RUNTIME/'scores'/name/key
  scores[uid,arm]=a.npz(path,registry[path.relative_to(a.RUNTIME).as_posix()])['probabilities']
if not (svg or plot_only):
 stats=[];comparisons=[]
 for ev in events:
  uid=ev['case_uid'];ref_us=int(round(ev['reference_onset_s']*1e6));s_us=int(round(ev['start_s']*1e6));e_us=int(round(ev['end_s']*1e6))
  cases=[(uid,'alert')]+[(v['control_uid'],'control') for v in plan['controls'] if v['alert_uid']==uid]
  for target,role in cases:
   shift_us=int(round(float(base.loc[target,'onset_s'])*1e6))-ref_us;aa=arrays[target];lo=s_us+shift_us;hi=e_us+shift_us
   for period,(left,right) in {'baseline':(lo-5000000,lo-1000000),'event':(lo,hi),'post':(hi+1000000,hi+5000000)}.items():
    for channel,v in a.sample_metrics(aa,(left/1e6,right/1e6),columns).items():stats.append({'event_id':ev['event_id'],'case_uid':target,'role':role,'period':period,'channel':channel,'interval_start_us':left,'interval_end_us':right,**v})
   mask=(aa['timestamp_us']>=lo)&(aa['timestamp_us']<=hi);expected=(hi-lo)//20000+1
   for arm in ['R','S_plus_R']:
    p=scores[target,arm];finite=mask&np.isfinite(p);normal=finite&aa['row_label_valid']&(aa['reference_state']==0);q=p[normal];th=float(b1[b1.case_uid.eq(target)&b1.arm.eq(arm)].onset_th.iloc[0])
    comparisons.append({'event_id':ev['event_id'],'case_uid':target,'role':role,'arm':arm,'mapped_start_us':lo,'mapped_end_us':hi,'expected_grid_frames':expected,
     'present_grid_frames':int(mask.sum()),'scored_frames':int(finite.sum()),'known_normal_scored_frames':len(q),
     'full_scored_normal_interval':bool(int(mask.sum())==expected and len(q)==expected),'score_median':float(np.median(q)) if len(q) else None,
     'score_max':float(max(q)) if len(q) else None,'at_or_above_own_onset_fraction':float(np.mean(q>=th)) if len(q) else None,'own_fold_onset':th})
 a.csv(o/'channel_period_statistics_integer_us.csv',stats);a.csv(o/'score_control_comparison_integer_us.csv',comparisons)
 log=a.ULog(records[uid124]['raw_ulog'],message_name_filter_list=['rfly_ctrl_lxl']);topic=next(v.data for v in log.data_list if v.name=='rfly_ctrl_lxl'and v.multi_id==0)
 active=np.flatnonzero(a.active_ids(topic['id']));idx=int(active[0]);entry={key:float(value[idx]) for key,value in topic.items()}
 a.save(o/'motor124_original_received_parameters.json',{'case_uid':uid124,'first_active_original_ULog_record':entry,'not_a_reference_relabel':True});del log
# Report goal: inspect raw time evolution and overlap, not assert physical causation.
os.environ['MPLCONFIGDIR']=str(c/'mpl_config');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
sys.path.insert(0,'C:/Users/23119/.agents/skills/scipilot-figure-skill/scripts');from visual_qa import audit_layout
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'svg.fonttype':'none'})
figdir=o/'figures';figdir.mkdir(exist_ok=True);qa=[]
names=['ctrl1','ctrl2','ctrl3','ctrl4','output1','output2','output3','output4','gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z','baro_native','quat_w','quat_x','quat_y','quat_z','vel_x','vel_y','vel_z','mag_x','mag_y','mag_z','acc_norm_derived']
for uid in typical:
 control=controls[uid];aa=arrays[uid];bb=arrays[control];onset=float(base.loc[uid,'onset_s']);ct=float(base.loc[control,'onset_s']);t=aa['timestamp_us']/1e6-onset;tc=bb['timestamp_us']/1e6-ct
 xx=aa['canonical_rows'].copy();xx[:,25]=np.linalg.norm(xx[:,11:14],axis=1);yy=bb['canonical_rows'].copy();yy[:,25]=np.linalg.norm(yy[:,11:14],axis=1)
 xx[~aa['row_input_valid']]=np.nan;yy[~bb['row_input_valid']]=np.nan
 fig,axes=plt.subplots(7,4,figsize=(9.2,11.8),layout='constrained');axes=axes.ravel();key=uid.replace('/','__')
 fig.suptitle(uid+'\ncontrol: '+control+'; time relative to each reference onset; native units',fontsize=9)
 for i in range(26):
  ax=axes[i];ax.plot(t,xx[:,i],color='#333333',lw=.65,label='alert case');ax.plot(tc,yy[:,i],color='#999999',lw=.65,ls='--',label='control')
  ax.axvline(0,color='#000000',lw=.6,ls=':');ax.set_title(names[i]);ax.grid(alpha=.15);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
  if i>=24:ax.set_xlabel('seconds to reference')
  ax.set_xlim(min(float(t[0]),float(tc[0]))-1,10)
 for j,arm in enumerate(['R','S_plus_R']):
  ax=axes[26+j];color=['#0072B2','#D55E00'][j];ax.plot(t,scores[uid,arm],color=color,lw=.8,label=arm+' alert');ax.plot(tc,scores[control,arm],color='#777777',ls='--',lw=.7,label='control')
  ax.axhline(float(b1[b1.case_uid.eq(uid)&b1.arm.eq(arm)].onset_th.iloc[0]),color=color,ls=':',lw=.7)
  for ev in events:
   if ev['case_uid']==uid and ev['arm']==arm:ax.axvspan(ev['start_s']-onset,ev['end_s']-onset,color=color,alpha=.15)
  ax.axvline(0,color='#000000',lw=.6,ls=':');ax.set_title(arm+' existing score (shading = N2)');ax.set_ylim(-.03,1.03);ax.set_xlim(min(float(t[0]),float(tc[0]))-1,10);ax.set_xlabel('seconds to reference')
 handles,labels=axes[0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=7,frameon=False)
 issues=audit_layout(fig);qa.append({'case_uid':uid,'control':control,'issues':issues});assert not any(v[0]=='FAIL'for v in issues)
 if svg:fig.savefig(figdir/(key+'_'+revision+'.svg'))
 else:fig.savefig(figdir/(key+('' if revision=='r1' else '_'+revision)+'.png'),dpi=170)
 plt.close(fig);print('FIGURE',uid,'issues',issues,flush=True)
a.save(c/('figure_svg_qa_'+revision+'.json' if svg else 'figure_png_qa_'+revision+'.json'),qa)
