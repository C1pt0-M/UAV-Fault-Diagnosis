"""Join audit evidence into tables/report; never load arrays or run models/events."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib
import pandas as pd
R=Path.cwd();O=R/'artifacts/real_v2_b1_n2_audit_v1';C=R/'artifacts/real_v2_b1_n2_audit_v1_checks';B=R/'artifacts/real_v2_b1_learning_curve'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
e=pd.read_csv(O/'events.csv');labels=pd.read_csv(O/'label_clock_checks.csv').set_index('case_uid');co=pd.read_csv(O/'cohort_metadata.csv');plan=load(O/'plan_r2.json');audit=load(O/'audit_results.json');b1=pd.read_csv(B/'events_per_case.csv');stat=pd.read_csv(O/'channel_period_statistics_integer_us.csv');cmp=pd.read_csv(O/'score_control_comparison_integer_us.csv');logs=load(O/'flight_status_and_messages.json');info=load(O/'TestInfo_records.json');infomap={v['case_uid']:v['metadata'] for v in info};paired=pd.read_csv(B/'paired_case_AUC.csv');original_result=load(B/'results.json')
assert len(e)==45 and e.groupby('arm').size().to_dict()=={'R':18,'S_plus_R':27}
assert len(labels)==52 and not labels[['label_file_disagreement_rows','fresh_ROS_hold_disagreement_valid_rows','fresh_ULog_hold_disagreement_valid_rows']].to_numpy().any()
assert (labels.ULog_active_records_before_crop==0).all()
metadata_notes=[];case_notes={
 'REAL/acce/motor/429_1':'事件水平速度中位2.129、倾角13.95°；与加减速/姿态工况相容，但无独立物理故障起点。',
 'REAL/circling/gyroscope/376_1':'圆周飞行，事件速度2.748、角速度范数0.347；运动与角速度变化并存，不能据此确诊陀螺故障。',
 'REAL/circling/magnetometer/385_1':'圆周飞行，速度2.647、倾角6.08°；磁场分量随姿态变化不单独作为磁力计损坏证据。',
 'REAL/circling/magnetometer/386_1':'两臂共同区间；速度约2.94，事件加速度范数中位10.14–10.50；运动与传感器效应不能单靠该值区分。',
 'REAL/hover/barometer/72_1':'低水平速度0.087，倾角9.43°；没有足够证据将短时姿态/高度波动归因于气压计提前故障。',
 'REAL/hover/gyroscope/61_1':'两臂共同短段；速度0.16–0.18、角速度范数0.28–0.31；命令仍未注入，实际硬件状态无独立参照。',
 'REAL/hover/magnetometer/89_1':'速度0.128、倾角7.05°；磁场变化可与姿态/环境有关，未认证真实磁异常。',
 'REAL/hover/motor/17_1':'速度0.174、倾角7.30°；未注入接收记录早于参考，无维修/转速等独立故障证据。',
 'REAL/velocity/barometer/301_1':'速度1.758、倾角25.13°；大倾角运动支持工况线索，不等同已定位误报因果通道。',
 'REAL/velocity/motor/267_1':'速度约5.0；R事件长8.36s，S+R仅覆盖其后段2.68s；工况和振动都应保留为线索。',
 'REAL/waypoint/accelerometer/195_1':'与186_1/188_1同类对照速度相近；26通道演化相似但局部振动幅度不同，S+R独有0.70s事件，不能证明前兆。',
 'REAL/waypoint/motor/114_1':'S+R独有3段；107s计划注入，80s航线等待；112_1同日期组长前段对照无N2，两臂score形状也不同。',
 'REAL/waypoint/motor/116_1':'S+R独有2段；107s计划注入、长航线；低速后期对照不能自动视为同一物理飞行阶段。',
 'REAL/waypoint/motor/117_1':'R1段/S+R3段；姿态、速度、磁场分量均有分段变化；与117_2同关联组，不计作独立重复。',
 'REAL/waypoint/motor/117_2':'R4段/S+R2段；一个S+R长段覆盖3个R段；正常对照112_1也有转向脉冲，但速度和振动并非完全匹配。',
 'REAL/waypoint/motor/118_1':'R独有0.88s段，角速度范数中位1.119；发生于较强转动，但不能据峰值直接判定正常或故障。',
 'REAL/waypoint/motor/120_1':'计划注入87s、长于后期对照；S+R最早事件提前60.32s，时间远大于命令/接收差，不能用毫秒级对齐差解释。',
 'REAL/waypoint/motor/122_1':'两臂几乎共同区间；角速度范数约0.50、速度2.836；与早期长航线日期组关联。',
 'REAL/waypoint/motor/123_1':'R2段/S+R1段；共同长段及R独有短段，角速度/速度均变化；不将关联当成因果认证。',
 'REAL/waypoint/motor/124_1':'R2段/S+R4段；gyro_y与加速度振动明显，但112_1也有部分共同转向结构；TestInfo参数字段与首条接收参数不逐项一致，非起点错位证明。'}
classified=[]
for row in e.to_dict('records'):
 uid=row['case_uid'];lab=labels.loc[uid];waymotor=row['flight']=='waypoint'and row['family']=='motor'
 classified.append({**row,'primary_class':'E','primary_reason':'缺少独立物理故障真值或可隔离工况/日期效应的对照，不能认证唯一根因',
   'possible_causes':'B;D' if waymotor else 'B','B_status':'工况同时出现的线索，不是已确认因果','D_status':'早期目录日期/长航线联合关联' if waymotor else '本事件无独立批次归因证据',
   'A_supported':False,'C_timing_supported':False,'parameter_annotation_review':uid.endswith('/124_1'),
   'end_before_original_ROS_command_s':float(lab.ROS_first_active_in_crop_s)-row['end_s'],
   'case_evidence':case_notes[uid]})
pd.DataFrame(classified).to_csv(O/'event_classification.csv',index=False)
case_rows=[]
for uid in sorted(e.case_uid.unique()):
 f=e[e.case_uid.eq(uid)];r=f[f.arm.eq('R')];s=f[f.arm.eq('S_plus_R')];md=infomap[uid]
 messages=next(v['pre_reference_log_messages'] for v in logs if v['case_uid']==uid);armed=next((v['timestamp_s'] for v in messages if 'Armed by external command'in v['message']),None)
 injection=float(md['Fault injection time'])
 case_rows.append({'case_uid':uid,'association_group':f.association_group.iloc[0],'flight':f.flight.iloc[0],'family':f.family.iloc[0],
  'R_events':len(r),'S_plus_R_events':len(s),'lead_start_min_s':float(f.lead_start_s.min()),'lead_start_max_s':float(f.lead_start_s.max()),
  'primary_class':'E','possible_causes':'B;D' if f.flight.iloc[0]=='waypoint'and f.family.iloc[0]=='motor'else'B',
  'planned_injection_s_TestInfo':injection,'reference_minus_armed_log_s':float(f.reference_onset_s.iloc[0])-armed if armed is not None else None,
  'reference_minus_armed_minus_planned_s':float(f.reference_onset_s.iloc[0])-armed-injection if armed is not None else None,
  'TestInfo_time_zero_not_independently_certified':True,'evidence':case_notes[uid]})
pd.DataFrame(case_rows).to_csv(O/'case_evidence.csv',index=False)
co=co.merge(b1[b1.arm.eq('R')][['case_uid','N2_frames']],on='case_uid',validate='one_to_one');w=co[co.flight.eq('waypoint')&co.family.eq('motor')]
cohort=[]
for date,f in w.groupby('recorded_log_date',sort=True):
 h=float(f.N2_frames.sum())*.02/3600
 cohort.append({'directory_date_proxy':date,'cases':len(f),'groups':f.association_group.nunique(),'N2_hours':h,'R_event_cases':int(f.R_events.gt(0).sum()),
 'S_plus_R_event_cases':int(f.S_plus_R_events.gt(0).sum()),'R_events':int(f.R_events.sum()),'S_plus_R_events':int(f.S_plus_R_events.sum()),
 'R_events_per_hour':float(f.R_events.sum())/h,'S_plus_R_events_per_hour':float(f.S_plus_R_events.sum())/h})
pd.DataFrame(cohort).to_csv(O/'waypoint_motor_date_exposure.csv',index=False)
components=audit['overlap_components'];shared=[v for v in components if v['shared']]
unique={arm:[v['event_id'] for v in classified if v['arm']==arm and not v['shared_interval_with_other_arm']]for arm in ['R','S_plus_R']}
common_cases=set(e[e.arm.eq('R')].case_uid)&set(e[e.arm.eq('S_plus_R')].case_uid)
class_counts={arm:{letter:int(sum(v['arm']==arm and v['primary_class']==letter for v in classified))for letter in 'ABCDE'}for arm in ['R','S_plus_R']}
# Aggregate only the recorded access journal. Sources imported as code are not data payload reads.
access=[json.loads(v)for v in(O/'input_access.jsonl').read_text(encoding='utf-8').splitlines()if v.strip()];by_path={}
for v in access:
 if v['path']not in by_path:by_path[v['path']]={**v,'open_calls':0}
 by_path[v['path']]['open_calls']+=1
# Metadata preparation happened before the runtime guard; list it explicitly, not as guarded calls.
preflight=[{'path':str(R/'configs/data/real_v2_split.csv'),'purpose':'eligible whitelist metadata'},
 {'path':str(R/'artifacts/real_labeltime_repair_v1_20261031/bag_verification/case_results.jsonl'),'purpose':'select only52 case records before consulting reference values'},
 {'path':str(R/'docs/research/2026-10-31/real_labeltime_repair_v1.md'),'purpose':'label-reference convention and provenance; no performance data'},
 {'path':str(R/'scripts/data/repair_real_labeltime.py'),'purpose':'read label convention only, not executed'},
 {'path':str(R/'scripts/data/build_full_causal_input.py'),'purpose':'read build convention only, not executed'}]
pd.DataFrame(list(by_path.values())).to_csv(O/'input_files.csv',index=False)
save(O/'preflight_metadata_reads.json',{'metadata_note':'All52 selected case_records were read during preparation;61_1 once additionally for schema; all265 eligible fault-case metadata read during audit. No nonselected payloads opened.',
 'selected_case_record_preparation_unique':52,'additional_schema_sample':'REAL/hover/gyroscope/61_1','other_metadata':preflight,
 'B1_input_files':[str(B/n)for n in ['events_per_case.csv','results.json','paired_case_AUC.csv','execution_notes.md','receipt.json']],
 'input_contract_read_by_read_tool_once_before_guard':str(Path('D:/UAV-Data/causal_input_v1/input_contract.json'))})
unique_counts=Counter(v['kind']for v in by_path.values());open_counts=Counter()
for v in by_path.values():open_counts[v['kind']]+=v['open_calls']
for key in ['canonical_case','existing_frame_score','label_reference','verified_ROS_bag_reference','raw_ULog_injection_and_flight_status','TestInfo']:
 assert unique_counts[key] in (52,104)
assert all(v['end_before_original_ROS_command_s']>0 for v in classified)
# Verify every original B1 text artifact, including its timestamp; never open any model/array here.
protected=load(O/'B1_protected_before.json');unchanged=[]
for v in protected:
 p=R/v['file'];assert p.stat().st_mtime_ns==v['mtime_ns'] and sha(p)==v['sha256'],str(p)
 unchanged.append(v['file'])
summary={'status':'COMPLETE_ROOT_CAUSE_AUDIT_UNRESOLVED_PHYSICAL_CAUSATION','created_utc':datetime.now(timezone.utc).isoformat(),
 'audited_signal_cases':52,'alert_cases':20,'alert_groups':e.association_group.nunique(),'control_cases':32,'events':45,'R_events':18,'S_plus_R_events':27,
 'primary_class_counts':class_counts,'possible_D_association_events':{'R':12,'S_plus_R':19},'primary_E_not_a_label_or_metric_change':True,
 'shared_cases':len(common_cases),'shared_components':len(shared),'shared_interval_R_events':13,'shared_interval_S_plus_R_events':11,
 'exclusive_R_events':len(unique['R']),'exclusive_S_plus_R_events':len(unique['S_plus_R']),'exclusive_event_ids':unique,
 'reference_checks':{'valid_row_disagreements':0,'all52_no_prior_received_injection_before_crop':True,
 'max_reference_minus_ROS_first_s':float(labels.reference_minus_ROS_first_s.max()),'max_reference_minus_ULog_first_s':float(labels.reference_minus_ULog_first_s.max()),
 'min_N2_event_end_before_ROS_first_s':min(v['end_before_original_ROS_command_s']for v in classified),'physical_onset_certified':False,'original_ROS_bag_redecoded_now':False},
 'waypoint_motor_date_exposure':cohort,'input_unique_counts_guarded':dict(unique_counts),'input_open_counts_guarded':dict(open_counts),
 'input_access_note':'preparation metadata separately listed; log path unique counts can overlap raw ULog roles (124 parameter check rereads same file).',
 'B1_protected_text_files_unchanged':len(unchanged),'blocked_accesses':audit['blocked'],'new_fits':0,'new_predictions':0,'new_thresholds':0,
 'event_timestamp_replay':'originalaggregate/originalevent_record AST only, saved scores at previously fixed eachfold onset;104 casearm counts matched',
 'scalar_diagnostic_correction':'original float-boundary tables kept; integer-us variants are authoritative, strict full scored-normal support not mere crop coverage',
 'tests':'6 initial pass;7 including integer-clock boundary regression pass','future_experiment_started':False}
save(O/'results.json',summary)
# Markdown tables, no new performance metric computation.
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,row))+' |'for row in rows]])
cat=table(['因果主分类','R事件','S+R事件','含义'],[[k,class_counts['R'][k],class_counts['S_plus_R'][k],{'A':'未认证真实前兆','B':'没有认证工况为唯一/可分离根因','C':'没有确认标签时间错误','D':'日期/分布有直接关联，但未隔离其因果','E':'物理根因证据不足，保留B/D候选'}[k]]for k in 'ABCDE'])
listing=table(['臂','案例（REAL/前缀省略）','关联组','start / confirm（PX4秒）','起始/确认提前秒','持续秒','waypoint','主类/候选'],
 [[v['arm'],v['case_uid'][5:],v['association_group'].split(':')[-1],f'{v["start_s"]:.6f} / {v["confirmed_s"]:.6f}',f'{v["lead_start_s"]:.2f} / {v["lead_confirmation_s"]:.2f}',f'{v["duration_s"]:.2f}','是'if v['waypoint']else'否','E / '+v['possible_causes']]for v in classified])
cohort_table=table(['原始目录日期proxy','案例/关联组','N2小时','R事件/事件案例','S+R事件/事件案例','R/S+R次每小时'],[[v['directory_date_proxy'],f'{v["cases"]}/{v["groups"]}',f'{v["N2_hours"]:.6f}',f'{v["R_events"]}/{v["R_event_cases"]}',f'{v["S_plus_R_events"]}/{v["S_plus_R_event_cases"]}',f'{v["R_events_per_hour"]:.3f}/{v["S_plus_R_events_per_hour"]:.3f}']for v in cohort])
case_table=table(['案例','R/S+R次数','起始提前范围秒','事实与限制'],[[v['case_uid'][5:],f'{v["R_events"]}/{v["S_plus_R_events"]}',f'{v["lead_start_min_s"]:.2f}–{v["lead_start_max_s"]:.2f}',v['evidence']]for v in case_rows])
clk=table(['案例','TestInfo计划秒','参考−armed日志秒','两者差秒（非认证时钟误差）'],[[v['case_uid'][5:],v['planned_injection_s_TestInfo'],f'{v["reference_minus_armed_log_s"]:.6f}',f'{v["reference_minus_armed_minus_planned_s"]:.6f}']for v in case_rows if '/waypoint/'in v['case_uid']])
report='''# B1 N2误报事件根因审计 v1

**状态：审计完成；B1的18/27次N2判定不变。没有认证真实故障前兆或能解释这些事件的标签时间错位。最明确的新发现是waypoint早期日期组、长前故障航线覆盖和动态输入分布相互混杂，不能把它们单独当作已证实根因。**

本次是零训练、零新预测、零阈值调整的诊断。H1成立/H2不成立以及“不进入B2”均保持原结论。报告中的E表示物理根因未确定，不是把原N2改成unknown、剔除或豁免。

## 1. 范围、输入与方法

- 全部20个有N2事件的开发案例（19关联组），45个“臂×事件”：R18次/13例，S+R27次/16例。另审查32个不同案例的无N2对照，共52例；均eligible dev、非quarantine。
- 先保存方法`plan.json`；原40个配对优先同目录/家族/故障子型、同外层折，再按N2覆盖长度排序。仅看元数据即发现部分waypoint同折对照前故障覆盖短，保留全部原对照，再增加10个跨折最近覆盖配对（新增3个独立案例），形成50个配对关系；没有按原始信号或高分择对照。方法修订在任何npz/分数读取前记录为`plan_r2.json`。
- B1 CSV仅有逐案例次数，故读取已有104份逐帧分数；只抽取原`aggregate_scores`和`event_record`函数，按原各折onset固定回放一次恢复时间。104个案例×臂（包括全部对照）计数和参考onset全部与B1一致；无模型加载、fit、predict或新onset。
- 所有52例均审查26个canonical输入通道、输入/标签mask、现成分数、TestInfo、保存的ROS参考及原ULog的注入/状态消息。26通道加速度模长按B1从加速度重算；气压通道图展示native值，但模型每3秒窗会减去首行，不能把绝对气压高度差直接说成模型依赖。
- 诊断取样：事件前5至1秒、事件本段、结束后1至5秒，原20ms整数µs网格；至少25有效行才给分位数，缺失不补零、不插值、不跨缺口算差分。控制映射以各自参考onset的相同提前时间为proxy，不等价于已认证的同一转弯/航段。事件后检查窗可能越过注入起点，仅作上下文。
- 120个事件×控制映射中，每臂96个完整可评分且已知正常区间、116个至少有正常评分；每个原事件至少有一个完整支持的对照。对照案例/配对不是新增独立样本，不做因果显著性检验。
- 批次分母使用全部265个eligible故障开发案例的`case_records`元数据，不加载额外213个案例的时序或分数；未读封存/quarantine payload或source/internal test。

## 2. 全45事件清单与分类

时间均为保存PX4流时钟，提前量=参考onset−事件时刻；start为候选起点、confirm为3点确认，不混同最终emit。关联组数字G表示`testinfo_case_id_candidate:G`；完整UID、组、家族、目录、起止/确认/emit/提前/持续时间见`events.csv`及`event_classification.csv`。45个确认均早于参考起点；重放生成的主机Unix时间没有当作历史证据。

'''+listing+'''

### 因果分类统计（保守主分类，与B1事件判定分离）

'''+cat+'''

B是工况候选，D是目录日期/航线/分布候选，**不是已认证因果标签**。D关联线索涉及R12和S+R19个waypoint/motor事件；其余事件保留工况线索。没有用“标签正常”证明B，也没有用“模型高分”证明A。所有事件主类保留E，避免用生态相关或不可比对照强行给出根因。124_1的6个事件另带参数注释复核标记，不据此认定C时间错误。

## 3. R与S+R的共同与独有误报

- 案例层：共同9例；R独有4例；S+R独有7例，共20例。
- 时间区间层：11个跨臂重叠连通分量，涉及R13个事件、S+R11个事件；R独有5个事件、S+R独有16个事件。`event_overlap_components.csv`列出每个成员。
- 重叠按同案例区间相交定义，无额外时间容差；允许一对多，不强行做一对一配对。117_2中一个S+R 26.80s长事件覆盖R的3段；另一个S+R段与R第4段相交。R总持续20.96s、S+R34.52s，**S+R计数更少不等于报警占用更少**。
- 两臂onset本来就不同：R折0–4为.85/.85/.85/.85/.90；S+R为.70/.85/.80/.85/.85。此次没有统一阈值反事实重评，不能把全部计数差只归因于源监督或某个输入通道。

## 4. waypoint专项：日期、航线和覆盖混杂

32/45事件来自waypoint，其中31为motor（R12、S+R19），另1为accelerometer195_1。waypoint/motor共有51个开发案例；31事件全部出自目录日期2023-05-22/23的15例，其余36例0事件。目录日期不等于经过独立认证的实验批次，ULog内部UTC日志可能与文件夹本地日期跨日；这里始终使用同一目录proxy，不混合两种日期。

'''+cohort_table+'''

按曝光量归一后，早期两日仍为R36.630、S+R57.998次/h（0.327600h）；后续日期0次/0.489728h，但这**不是控制飞行阶段后的日期效应**。早期告警案例TestInfo多为107s注入（120_1为87s）、航线等待80s（120_1较短）；已审后期motor对照多为约58s注入、30s等待。相对起点−40秒在早期可能正飞航线，在后期却接近起飞/尚未进入同一航段，缺乏可比支持。

同早期日期组、长等待的112_1两臂均无N2，是重要反例；它的计划航速约2，与多数告警案例约3、114_1约2.5不同。112_1又被多个告警案例重复匹配，不能当作许多独立健康重复。因此日期/协议共同关联得到直接支持，**D作为独立因果解释尚未证实，B也未被单独隔离**。

## 5. 典型案例的26通道时序证据

典型展示按事前规则确定：waypoint最多事件的117_2、其后124_1、S+R独有最多事件114_1，以及字典序首个waypoint sensor195_1。其余20例全部量化，不以可解释性挑展示。图中黑实线为告警案例、灰虚线为对照，各自参考起点对齐；底部为既有R/S+R分数和原N2阴影。展示完整前段及参考后10秒，不代表两条曲线已精确对齐相同航点；图线断开处不插值。

### 5.1 117_2：分段与长事件合并，不是“源数据减少了误报”

参考onset264.497355s。R4段提前40.08/28.00/18.48/11.96s，S+R2段提前39.44/12.14s。26通道中姿态、水平速度、磁场分量分段变化，gyro/accel有不同程度振动；112_1也有转向脉冲，但并非完全相同速度/振动。首个R区间水平速度中位2.848，对照112_1为2.087；不可把该对照当作精确工况匹配。没有证实是哪一通道造成模型告警。

![117_2与112_1](../../../artifacts/real_v2_b1_n2_audit_v1/figures/REAL__waypoint__motor__117_2_r2.png)

### 5.2 124_1：有振动差异，但不能认证“真实前兆”

R2段、S+R4段，最早提前46s；第一共同段gyro_y IQR约0.683，对照112_1约0.198，加速度模长IQR约3.334对1.281（对应事件窗；支持行数不同已记录）。这是真实记录到的输入差异，但风、振动、机体/采集状态和潜在机械问题没有独立真值区分。气压native层面也有明显偏移，不能直接归因到使用窗内相对高度的模型。

![124_1与112_1](../../../artifacts/real_v2_b1_n2_audit_v1/figures/REAL__waypoint__motor__124_1_r2.png)

### 5.3 114_1：S+R独有，不是两臂共同确认的前兆

3次S+R事件提前42.72/13.48/6.66s，R没有N2。raw输入相同，现成分数与原onset共同决定两臂差别；没有做改阈值重评或特征归因实验。其首段速度中位2.434，112_1约1.964，对照不能证明完全相同的飞行状态。

![114_1与112_1](../../../artifacts/real_v2_b1_n2_audit_v1/figures/REAL__waypoint__motor__114_1_r2.png)

### 5.4 195_1：同类型对照也有相似运动，不支持“看到振动即故障”

S+R独有0.70s事件，提前27.90s；R未形成N2。对照186_1/188_1同为waypoint accelerometer，事件映射水平速度中位2.934/2.816，对案例2.939；gyro_y和加速度局部离散程度仍有不同。姿态、速度和磁场整体演化较相似，但这些条件下分数差别不能认证传感器已经故障。

![195_1与188_1](../../../artifacts/real_v2_b1_n2_audit_v1/figures/REAL__waypoint__accelerometer__195_1_r2.png)

### 全20告警案例的逐例复核摘要

'''+case_table+'''

每个事件及各匹配对照的全26通道基线/事件/后窗中位、IQR、P05/P95、相邻差分和重复值比例见`channel_period_statistics_integer_us.csv`；存在至少25行的支持才给统计，未据任何通道阈值自动判健康/故障。已有配对AUC只作背景，不代替事件/物理证据。

## 6. 标签时间与故障注入核查

- 保存标签参考、按原整数首timesync锚点重取的ROS发送状态，以及本次新读原ULog接收状态：52例所有有效行分歧均0。所有52例在裁剪前都没有已记录的ULog active注入，未发现“前一次注入被当成N2”的证据。
- 首个**可监督且双方一致**故障帧相对ROS发送首active最多晚0.290309s，相对ULog接收首active最多晚0.025293s；这是通信/网格/有效性共同参考，不是证明标签与发送时刻零误差。
- 45次事件全部在原ROS首active之前结束，最小间隔0.285834s；候选起始提前1.32–60.32s。当前看到的亚秒参考差不能解释这些事件，更没有移动标签使其变成检出。
- ROS侧本次使用既有逐case、带SHA并曾独立核验原bag相关字段的参考，**未重新解码原bag**；ULog与TestInfo本次实际重新读取。缺少独立物理损伤/传感器校准真值，不能由ID正常推论硬件必正常，也不能由高分推论提前故障。

TestInfo是计划相对时间，原文未单独认证其零点等于ULog armed消息。下表只列一致性旁证，不拿差值直接判错或拟合新时钟：

'''+clk+'''

124_1的TestInfo Control Sequence包含`...,123450,11,10,1,0.8,...`，Fault parameter字段也不是接收首条的直接逐项副本；原ULog首active为269.565740s、id123450、mode1、controls[0]≈0.8，其余前三电机系数1。此处需要原控制脚本/字段语义复核，**不是已确认的起点或故障标签错误**，也不能解释提前5–46秒的全部事件。没有修改任何参数注释或标签。

ULog在所查前段没有severity≤warning日志（日志ASCII等级已解码），事件状态均nav_state14/arming_state2；这些只说明记录状态，不能用“没报warning”认证健康，也不将nav_state14直接解释为某个具体航段。

## 7. 证据强度、能回答与不能回答

| 问题 | 结论与强度 |
|---|---|
| 18/27事件是否真实存在于冻结事件规则 | 强：全104案例臂计数、原onset和保存分数SHA核对一致 |
| 是否源于秒级保存标签/ROS/ULog错位 | 本次没有支持：有效行一致，全部事件在发送active前结束；不是独立物理真值认证 |
| 是否真实故障前兆A | 未认证：没有独立损伤/注入已提前的证据；振动与模型分数不足以证明 |
| 是否正常工况造成B | 有工况共现与正常对照线索；没有单独隔离到因果，不能全部改判B |
| 是否批次/分布差异D | 早期目录日期与长前段、速度/姿态/振动分布共同关联为直接事实；真实批次ID、环境/载荷/机体效应尚未隔离 |
| 模型到底使用了哪个通道 | 不能回答；没有做新预测、消融、SHAP或阈值反事实 |
| S+R为何多9次 | 已定位独有区间、不同原onset及部分事件合并；不能唯一归因于源域负迁移 |

**可确定**：问题不是简单的全局标签平移；事件高度集中且伴随阶段/日期分布支持不一致；只看pooled AUC或事件计数不足以定位物理根因。**不能确定**：45个事件里哪些是没有注入记录的真实机械前兆，哪一个输入/批次因素独立导致告警；因此保持E主分类与原B1评价。

## 8. 下一步建议（仅建议，不执行）

1. 优先索取原控制脚本、TestInfo时间零点/字段语义和真实实验批次、机体/载荷、天气、传感器校准记录；专项核对124_1参数注释。没有新证据前不改标签。
2. 围绕早期waypoint长航线补独立健康对照：相同路径、速度、80s等待和同机体/环境；不能用30s后就注入的后期案例替代完整前段。112_1是反例但单一且速度不同，不能重复使用它伪造独立性。
3. 若要检验真实机械前兆，须有独立转速/电流/维修/视频或受控健康基线；不能用GBT高分给物理故障定时。
4. 如原始记录最终支持时间/标签修正，另交版本化审计方案，保留旧B1结果；不得按本次模型效果回填标签、事后删例或宣布H2通过。
5. 不启动B2/T1/GPU，不训练新模型，不自动统一/重扫阈值。由用户决定是否开展上述数据取证。

## 9. 完整偏差与实现选择

1. 未产生数值协议、标签或既有事件判定变更。已有摘要没有逐事件时间，新增固定原规则回放仅恢复时间并核对，未调用模型/新预测/搜索阈值，B1所有原结果及历史仍在。
2. 对照先按元数据冻结，发现覆盖不匹配后在任何payload读取前保留原配对并补最近覆盖关系；两版计划都保留，最终52例/50配对。相同相对onset只是阶段proxy，跨折对照还混有不同模型/onset，不作完美匹配声明。
3. 原始统计第一版用浮点秒选区间，“covers_full_interval”只检查裁剪覆盖且有分数，不能证明完整评分。发现边界取样问题后保留原表，另生成整数µs权威表，并区分expected/present/scored/known-normal及完整支持；增加边界回归，7项合成检查通过。未重放事件或修改其端点。
4. 仅前后窗等诊断统计使用25行最小支持，非模型/事件阈值；缺失保持null，不插值、不补零，不跨缺口算相邻差分。
5. 首轮26通道PNG机器QA无问题，但AI发现图例可能压住第一轴尖峰；保留r1，移到图外生成r2，4图重新机器和AI检查通过，再导SVG，严格矢量检查通过；灰度预览亦通过。这是展示修改，不改数据/事件。
6. 全20案例量化，4典型图按预先元数据规则选取。没有按“好解释”挑案例、没有加反事实评分/消融；新增病例事实表、目录日期曝光表均为已有数据的描述，不是模型优化或独立确认。
7. 原bag本次未重解码，使用已有认证字段参考；新读52原ULog与52 TestInfo，124参数另读一次。TestInfo计划秒与armed起点只作旁证，目录日期不等于认证批次，未构造有利新时钟。
8. 主分类全E是证据门槛下的未决结论，并非漏审；B/D候选及124注释问题单列，未把相关性、参考正常或分数升高当作物理因果。
9. B1原文本证据全部逐文件SHA/mtime保护验证未变；没有读取模型权重、source train/val、内部test、封存/quarantine payload，没有训练、预测、标签修改、病例删除、规则调整、依赖安装或CPU dispatch变化。
10. 实际读取文件和数量见`input_files.csv`（逐路径/用途/打开次数）及`preflight_metadata_reads.json`。准备阶段52个selected case_records及61_1一次额外schema读取在运行时guard之外，明确补记；运行时265故障元数据用于完整日期分母，不打开其余213例payload。重复出图/统计只重新读取既有输入，不生成预测，绝不把重复打开计作新案例。
11. 初始6项、整数边界修正后7项合成测试均首轮通过，0真实数据/模型访问；正式提取一次退出0，69.58秒，无失败/模型重试。仅诊断表与图的修订如上，旧版本不覆盖。
12. 报告和唯一维护入口更新属于本次用户明确授权；其他历史结项报告/协议、无关case-relative脚本与两份临时日志不修改。审计完成后只提交/push本次文件，不启动后续实验。

## 10. 产物和复现

- `scripts/research/audit_real_v2_b1_n2.py`及`tests/test_real_v2_b1_n2_audit.py`；仅合成单测。
- `artifacts/real_v2_b1_n2_audit_v1/`：两版方法与对照、45事件、分类、20案例证据、时钟核查、全26通道诊断表（整数µs版为准）、对照分数摘要、目录日期曝光、四例PNG/SVG、输入回执与results。
- `artifacts/real_v2_b1_n2_audit_v1_checks/`：实现/检查/出图配方、数据剖析、视觉QA及执行日志。旧浮点表、首轮PNG保留，不能替代权威整数表/最终r2图。
- `docs/文档索引与实验台账.md`已增加B1完成状态、N2审计状态和本报告链接；未改写B1原结论。
- Git精确提交清单见checks目录的`submission_files.txt`，commit/push验证由Git记录与任务回复给出，避免自引用哈希。
'''
path=R/'docs/research/2026-11-09/N2误报事件根因审计_v1.md';path.parent.mkdir(exist_ok=True,parents=True)
if path.exists():raise FileExistsError('Preserve original report')
path.write_text(report,encoding='utf-8',newline='\n')
save(O/'execution_audit.json',{'status':summary['status'],'report_sha256':sha(path),'B1_protected_files_unchanged':unchanged,
 'script_sha256':sha(R/'scripts/research/audit_real_v2_b1_n2.py'),'test_sha256':sha(R/'tests/test_real_v2_b1_n2_audit.py'),
 'data_array_or_model_outputs_written':0,'no_source_internal_or_sealed_payload_access':True,'blocked_accesses':audit['blocked'],
 'input_file_manifest_sha256':sha(O/'input_files.csv'),'results_sha256':sha(O/'results.json'),'main_extract_exit_code':0,'synthetic_tests':7,
 'primary_causal_labels':'E for all45; evidence and candidate B/D flags separated; no B1 relabel',
 'formatting_only_report_phase':True,'next_experiment_started':False})
print('AUDIT_COMPLETE',len(e),'events; primary',class_counts,'input_unique',dict(unique_counts),'report_lines',len(report.splitlines()))
