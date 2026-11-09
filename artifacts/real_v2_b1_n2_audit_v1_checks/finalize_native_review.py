"""Preserve pre-native summaries, add native health evidence and final access counts."""
from pathlib import Path
from collections import Counter
import json,hashlib
import pandas as pd
R=Path.cwd();O=R/'artifacts/real_v2_b1_n2_audit_v1';C=R/'artifacts/real_v2_b1_n2_audit_v1_checks';report=R/'docs/research/2026-11-09/N2误报事件根因审计_v1.md'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['results.json','execution_audit.json','input_files.csv']:
 backup=O/(Path(name).stem+'_before_native_sensor_review'+Path(name).suffix)
 with backup.open('xb') as f:f.write((O/name).read_bytes())
d=pd.read_csv(O/'native_IMU_interval_review.csv');alerts=d[d.role.eq('alert')&d.topic.eq('sensor_combined')]
assert len(alerts)==45 and alerts['nonfinite_gyro_accel_values'].sum()==0 and d.timestamp_reversals_entire_topic.max()==0
flag_columns=[k for k in d if k.endswith('_nonzero_records') or k.endswith('_last_minus_first')]
assert all(d[k].dropna().eq(0).all() for k in flag_columns)
result=load(O/'results.json');result['native_sensor_review']={'cases':52,'intervals_per_topic':165,'topics':sorted(d.topic.unique().tolist()),
 'alert_intervals':45,'native_sensor_combined_samples_per_alert_interval_range':[int(alerts.samples.min()),int(alerts.samples.max())],
 'nonfinite_alert_gyro_accel_values':0,'topic_timestamp_reversals':0,'recorded_clipping_and_error_fields_nonzero_or_increase':False,
 'absence_of_flags_not_a_health_certificate':True,'no_new_models_or_scores':True}
records=[json.loads(s) for s in(O/'input_access.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()];by={}
for v in records:
 if v['path']not in by:by[v['path']]={**v,'open_calls':0,'purposes':set()}
 by[v['path']]['open_calls']+=1;by[v['path']]['purposes'].add(v['kind'])
for v in by.values():v['purposes']=';'.join(sorted(v['purposes']))
pd.DataFrame(list(by.values())).to_csv(O/'input_files.csv',index=False)
result['input_unique_counts_guarded']=dict(Counter(v['kind'] for v in by.values()));counts=Counter()
for v in by.values():counts[v['kind']]+=v['open_calls']
result['input_open_counts_guarded']=dict(counts)
(O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
text=report.read_text(encoding='utf-8')
text=text.replace('**可确定**：问题不是简单的全局标签平移；','**可确定**：现有命令/参考证据不支持用简单全局标签平移解释全部事件；')
insert='''
### 实际输入文件与原生IMU复核

| 输入类别 | 唯一文件 | 实际打开次数（运行时guard记录） |
|---|---:|---:|
| 26通道连续案例NPZ | 52 | 116 |
| 已有逐帧分数NPZ | 104 | 232 |
| 保存标签参考NPZ | 52 | 52 |
| 已认证ROS命令/timesync参考NPZ | 52 | 52 |
| 原ULog（注入/状态、124参数及原生IMU复核合计） | 52 | 105 |
| 原TestInfo XLSX/CSV | 52 | 52 |
| eligible故障case_records元数据 | 265 | 265（另有准备阶段52例及61_1一次schema读取） |

重复取诊断统计、复核图和导SVG会重新读取同一现成输入，不是新预测，也不增加案例。输入路径、用途集合和次数在`input_files.csv`；准备阶段与代码/协议元数据单列`preflight_metadata_reads.json`。分数、案例、标签及ROS参考都以已有身份SHA核对，原模型文件一次未打开。

26通道图是50Hz因果发布回放视图，不等价于未加工的原生高频传感器流。为防止把回放尖峰直接叫作传感器损坏，另对同52个原ULog读取sensor_combined、sensor_accel、sensor_gyro和vehicle_imu_status；165个事件/对照映射区间逐topic记录（不是165个独立病例）。45个告警区间原生gyro/accel均有限，相关topic的发布timestamp无倒序；实际记录的accelerometer_clipping、clip_counter、accel_clipping及gyro/accel error_count在检查区间均0、无计数增加。原生检查仍没有提供确诊硬件故障的证据，**缺少这些flag也不认证健康**。原生采样存在至少98、最多5291个/告警区间，未用稀疏50Hz视图推断削顶；完整字段与支持样本数见`native_IMU_interval_review.csv`及topic inventory。

'''
text=text.replace('## 2. 全45事件清单与分类',insert+'## 2. 全45事件清单与分类')
text=text.replace('12. 报告和唯一维护入口更新属于本次用户明确授权；其他历史结项报告/协议、无关case-relative脚本与两份临时日志不修改。审计完成后只提交/push本次文件，不启动后续实验。',
 '12. 报告和唯一维护入口更新属于本次用户明确授权；其他历史结项报告/协议、无关case-relative脚本与两份临时日志不修改。审计完成后只提交/push本次文件，不启动后续实验。\n13. 补充原生IMU健康/剪切检查仍使用同52案例，未扩展样本或生成预测；ULog累计105次打开、52唯一文件。保存补充前results/audit/access快照，更新最终读取回执；原始B1全部证据不变。缺少clipping/error flag只降低特定硬故障解释的支持，不能排除真实机械前兆或把事件强行归为B。')
report.write_text(text,encoding='utf-8',newline='\n')
audit=load(O/'execution_audit.json');audit.update(report_sha256=sha(report),results_sha256=sha(O/'results.json'),input_file_manifest_sha256=sha(O/'input_files.csv'),native_sensor_review=result['native_sensor_review'])
(O/'execution_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Reconfirm original B1 evidence only; never models/arrays.
for v in load(O/'B1_protected_before.json'):
 p=R/v['file'];assert sha(p)==v['sha256'] and p.stat().st_mtime_ns==v['mtime_ns']
print('FINALIZED_NATIVE_REVIEW',result['input_open_counts_guarded'],'B1_79_TEXT_INPUTS_UNCHANGED')
