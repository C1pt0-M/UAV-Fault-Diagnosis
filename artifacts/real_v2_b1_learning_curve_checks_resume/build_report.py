"""Render completed JSON/CSV only. No model, npz, metric or bootstrap rerun."""
from pathlib import Path
from datetime import datetime,timezone
import collections,csv,hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/real_v2_b1_learning_curve'
CHECK=ROOT/'artifacts/real_v2_b1_learning_curve_checks_resume'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
x=load(OUT/'results.json');receipt=load(OUT/'receipt.json')
assert x['status']=='COMPLETE' and receipt['status']=='COMPLETE' and x['fit_count']==200
old=lines(CHECK/'attempt1_fit_receipts.jsonl');fits=lines(OUT/'fit_receipts.jsonl');starts=lines(OUT/'resume_fit_attempts.jsonl')
assert len(old)==122 and old==fits[:122] and len(starts)==78
assert not set(v['name'] for v in old)&set(v['name'] for v in starts)
assert sha(ROOT/'scripts/research/real_v2_b1_learning_curve.py')=='312b4c70b647dffb4c7aa4113df1fb6af3ec1d496d9f71db389571a216ed7d12'
assert load(CHECK/'exit_2.json')['exit_code']==0
h1=x['H1']['fractions']['0.25'];selections=x['onset_selection'];arms=('R','S_plus_R','S')
counts=dict(collections.Counter(k.split('/')[0] for k in receipt['local_artifacts_sha256']))
# These are formatting arithmetic over already completed per-repeat statistics.
def fmt(v,n=6):return 'null' if v is None else f'{v:.{n}f}'
def interval(v):return '['+', '.join(fmt(t) for t in v)+']' if v else 'null'
def group_summary(rows,key,percent=False):
 values=[v[key] for v in rows if v[key] is not None];m=100 if percent else 1
 return f'{statistics.fmean(values)*m:.6f} [{min(values)*m:.6f}, {max(values)*m:.6f}]' if values else 'null'
def ci(a,key):return interval(x['event_CI']['arms'][a][key]['ci95'])
curve=['| 臂 | 比例 | 重复 | AP 均值 [最小, 最大] | AUC(b) 均值 [最小, 最大] |','|---|---:|---:|---|---|']
fpr=['| 臂 | 比例 | 完整正常 FPR% | 补充正常 FPR% | pre_fault FPR% | post_fault_normal FPR% | transition_normal FPR% |','|---|---:|---|---|---|---|---|']
for arm in arms[:2]:
 for frac in (.1,.25,.5,1.):
  rows=[v for v in x['window_metrics'] if v['arm']==arm and v['fraction']==frac]
  curve.append(f'| {arm} | {frac:.2f} | {len(rows)} | {group_summary(rows,"AP")} | {group_summary(rows,"AUC_b")} |')
  fpr.append(f'| {arm} | {frac:.2f} | '+' | '.join(group_summary(rows,k,True) for k in ['FPR_normal_full','FPR_normal_segment','FPR_pre_fault','FPR_post_fault_normal','FPR_transition_normal'])+' |')
h1table=['| 比例 | 平均 ΔAP | 95% CI | 地位 |','|---:|---:|---|---|']
for frac,point in x['H1']['fractions'].items():h1table.append(f'| {float(frac):.2f} | {fmt(point["repeat_mean_delta_AP"])} | {interval(point["ci95"])} | '+('H1 成立' if point['primary'] else '仅描述')+' |')
onsets=['| 外层折 | R onset | R 可行 | S+R onset | S+R 可行 |','|---:|---:|---|---:|---|']
for k in range(5):
 r=next(v for v in selections if v['arm']=='R' and v['outer_fold']==k);s=next(v for v in selections if v['arm']=='S_plus_R' and v['outer_fold']==k)
 onsets.append(f'| {k} | {r["onset_th"]:.2f} | {not r["no_feasible_onset"]} | {s["onset_th"]:.2f} | {not s["no_feasible_onset"]} |')
inner_rates=['| 臂 | 外层折 | 所选点内层N1次/h | 内层N2次/h | 内层motor检出 | 内层sensor检出 |','|---|---:|---:|---:|---:|---:|']
for v in selections:
 point=v['inner'];inner_rates.append(f'| {v["arm"]} | {v["outer_fold"]} | {fmt(point["N1"]["rate"])} | {fmt(point["N2"]["rate"])} | {fmt(point["motor"]["detection"])} | {fmt(point["sensor"]["detection"])} |')
events=['| 臂 | N1 次数 / h / 次·h⁻¹ [CI] | N2 次数 / h / 次·h⁻¹ [CI] | motor 新确认 [CI] | sensor 新确认 [CI] | H2 |','|---|---|---|---|---|---|']
latency=['| 臂 | motor median 秒 [CI] | motor P90 秒 [CI] | sensor median 秒 [CI] | sensor P90 秒 [CI] |','|---|---|---|---|---|']
for a in arms:
 e=x['event'][a]
 def normal(n):
  v=e[n];return f'{v["events"]} / {v["hours"]:.9f} / {v["rate"]:.6f} {ci(a,n+"/rate")}'
 def detected(f):
  v=e[f];return f'{v["detected"]}/{v["cases"]} = {v["detection"]:.6f} {ci(a,f+"/detection")}'
 events.append(f'| {a} | {normal("N1")} | {normal("N2")} | {detected("motor")} | {detected("sensor")} | '+('仅参照' if a=='S' else '不成立：N1、N2 失败')+' |')
 latency.append(f'| {a} | '+' | '.join(f'{e[f][k]:.6f} {ci(a,f+"/"+k)}' for f,k in [('motor','latency_median_s'),('motor','latency_p90_s'),('sensor','latency_median_s'),('sensor','latency_p90_s')])+' |')
paired=['| 臂 | 类别 | 有定义/全部案例 | AUC 中位数 | Q25 / Q75 | 最小 / 最大 |','|---|---|---:|---:|---|---|']
for a,fs in x['descriptive']['paired_case_AUC'].items():
 for f,v in fs.items():paired.append(f'| {a} | {f} | {v["defined"]}/{v["cases"]} | {fmt(v["median"])} | {fmt(v["q25"])} / {fmt(v["q75"])} | {fmt(v["min"])} / {fmt(v["max"])} |')
byflight=['| 臂 | 飞行目录 | N1 次/h | N2 次/h | motor 检出 | sensor 检出 |','|---|---|---:|---:|---:|---:|']
for a,flights in x['descriptive']['by_flight'].items():
 for f,v in flights.items():byflight.append(f'| {a} | {f} | {fmt(v["N1"]["rate"])} | {fmt(v["N2"]["rate"])} | {fmt(v["motor"]["detection"])} | {fmt(v["sensor"]["detection"])} |')
post=['| 臂 | post_fault_normal 次数 / h / 次/h | 补充正常片段 次数 / h / 次/h |','|---|---|---|']
for a in arms:
 e=x['event'][a];post.append('| '+a+' | '+' | '.join(f'{e[k]["events"]} / {e[k]["hours"]:.9f} / {e[k]["rate"]:.6f}' for k in ['post','supplement'])+' |')
notes='''# B1 执行记录、偏差与实现选择

1. 原协议、PI 原指令和补充1保持不变；不可行 onset 的并列规则按补充1澄清，不算偏差。本次10个内层选择均可行，回退规则未触发；并列列表均保存在回执。
2. 原次执行在122/200模型、23份合并OOF后因用户告知的本机重启中断；保存进度不代表进程仍活跃。没有捕获到原进程退出码或系统关机事件，不能虚构。旧日志、快照和中断回执保留。
3. 补充2明确批准本机断点恢复，覆盖原“输出目录必须全新”的执行要求。新增独立恢复入口，原B1代码字节不改；数值函数仍调用原B1/B0实现。
4. 核对从简：只复核冻结配置、划分、同一解释器/版本和sklearn版本；没有再次逐个哈希已有122模型、23 OOF，也没有扫描原始Real npz做身份检查。原输入数值没有本次新哈希证明，不作此声称。
5. 第123槽位 outer_S_plus_R_f0.50_rep1_k2 的本次物理启动为第2次，明确记录为“本机重启后的基础设施恢复”，不是重跑指标。其余77新槽位各1次物理启动；原122完成模型无重训，原拟合回执前122行对象完全相等。
6. R1.00原5模型补回丢失的合并OOF；S+R50% rep1 k0/k1只补预测。23已完成合并OOF直接复用；其115折预测从对应OOF按折提取，不重预测。原已完成窗口指标不重算，旧CSV前缀保留。丢失的特征矩阵按原实现重建并缓存。
7. 恢复尝试1在第87个窗口输入打开前，访问审计日志原子替换触发Windows PermissionError [WinError 5]。已完成86案例窗口特征缓存，0个新拟合，0个新模型；当次原日志、异常回执及.partial全部保留。未证明具体是哪一个进程锁住文件。
8. 对恢复控制修复：JSONL改为追加、flush、fsync，避免反复替换活跃日志；若重启截断尾记录，先保留完整原文件到.interrupted旁证，再仅修复尾部；损坏的完整记录仍拒绝跳过。其他原子提交遇PermissionError最多3次，0.1/0.25秒等待，逐次记入stdout/IO事件；只重试文件替换，不重做拟合/预测。恢复尝试2实际IO事件为空。
9. 恢复控制合成检查首轮21通过；针对上述修复增补检查后22通过（含原B1的13项），均无实际Real或模型访问。原首次实现的28项检查（含B0 15项+B1 13项）保留。没有测试失败后放宽断言或改数值规则。
10. 新模型采用单个原子pickle检查点内嵌拟合回执；若模型已提交但日志未写完，再启动会收养该模型、不调用fit。每次拟合后立即缓存对应预测；逐案例分数/事件及完整分析阶段也缓存。临时文件不删，不作为已完成检查点。Windows原生字节锁拒绝并发恢复，进程退出/重启后由OS释放。
11. 特征（源train及Real130维窗口、逐帧）、模型、OOF、逐帧分数仅放 D:/UAV-Data/；Git只含数值汇总、事件/指标CSV、方法/回执/代码及SHA元数据。source监督权重、Real正常来源权重、S+R域总质量相等、抽组嵌套和GBT默认auto早停全部沿用原实现；原已早停模型也作为完整模型复用。
12. 前3个新拟合均值18.435421秒，按剩余75槽给拟合预算23.044277分钟，另以未实测阶段预留估计总46.09–92.18分钟；已写run.log并告知用户。实际明显超出，已纠正为偏乐观、不承诺时限。恢复2记录墙钟30880.683427秒（约8小时34分41秒）；多次单拟合墙钟异常延长，未确认睡眠/争用等具体原因，不称作数值错误、不因此重训。
13. 一个只读内核等待的工具6600秒超时；该次没有重启训练。随后OS查询监督/工作PID均仍活跃（259），另一个等待得到退出0。原次的观察超时和重连记录也保留；不把观察者超时当模型失败。
14. 原次访问计数只在内存，重启后丢失；不能声称恢复它。补充2起，白名单原始npz打开即时记盘，累计315窗口+315连续案例、630次，全部eligible dev且非quarantine。原次根据完成代码路径加载315窗口是推断，不加到“实际记录打开次数”中；原次未到连续案例阶段。
15. 窗口原始float32统计130维；逐帧仍用case_windows（输入valid150行、重算通道25、通道14相对首行）。source脚本run_test.py有顶层训练/test副作用，从未导入。N1、N2暴露量按B1 finite-score mask，区别于B0原始有效行时间，且N1/N2绝不混合用于H2。
16. H1按5重复ΔAP的均值、共享288关联组/2000 bootstrap重算，非平均预测。H2为full-budget外层OOF点判据，CI仅描述；内层逐折选点固定后不在bootstrap重选。没有未定义重采样丢弃、重抽、伪计数或概率裁剪；本次所有已注册CI未定义重复数均0。
17. 逐案例配对AUC中4个motor案例缺少y=1窗口，R/S+R均null（不是排除案例）；保持在事件检出分母154中。其他可定义案例的分位数仅作描述。时延是新确认成功案例的条件分布，不把未检出伪装成0秒。
18. sealed Real始终只可读取划分/台账元数据后立即筛除，未读其npz；不称为blind（以往465案例汇总已知）。未读取内部test、source val、不跑M0/MSTCA/GPU/T1/B2。允许的冻结S pickle并不授权读取同目录内部test结果。没有标签、排除规则、CPU16或dispatch变化。
19. 不对外层扫onset、不因H2失败改阈值、指标或加权。无未完成注册项；H2失败是完整科学结果，不是恢复失败。用户“无法继续才全部重来”的条件未触发，不从头重跑，不因结果未通过而重试。
20. 整理报告只读取完成的JSON/CSV，做重复均值/范围排版，不再训练、预测、聚合事件或运行bootstrap。原无关脚本、两份临时审计日志、golden、结项报告和台账未由本任务修改。
21. 正式完成后对恢复入口执行一次完成态幂等检查，返回B1_ALREADY_COMPLETE_NO_ACTION；只检查配置/版本并读取完成标记，不读取原始npz，不新建恢复尝试、不训练/预测/评价。此为运行控制检查，不是指标重跑。
22. 提交前普通git diff --cached --check退出2：仓库.gitattributes规定artifacts/** -text，Windows原始证据中的CRLF被默认规则报作行尾空白。完整失败日志保留；不改证据字节、不改Git配置，改用单次命令-c core.whitespace=trailing-space,space-before-tab,cr-at-eol后检查通过。没有发生提交失败或实验重跑；这只是提交前样式检查失败。
'''
(OUT/'execution_notes.md').write_text(notes,encoding='utf-8',newline='\n')
report='''# 跨域 B1 学习曲线结果 v1

**结论：H1成立，H2不成立；不进入B2，回到数据/标签问题与用户讨论。**

按原协议完成200个逻辑拟合。首次本机重启后复用122完成模型和23合并OOF；补充2恢复只新增78拟合，无需全部重来。全部冻结数值规则不变。

## 1. 启动、复现与输入边界

- 配置 SHA256 `114ac5427187a6cb78ed1a77e39d8861194b4d25340f07c1debf43f3d1e1bd3c`；划分 `9232ee35e7a59f4fac4534fc8e98fc454380b51e6e5d8439aaa0f769cd1090ee`。
- 原次启动检查通过；dev 320去quarantine 5，eligible **315案例/288组/18480窗口**。冻结S SHA256 `df468524e2047205dd8422ab3e486ce7692f0ac312276e20394d6f88a1bb6774`。
- R1.00与B0复现：18480窗口标签/索引一致，最大概率差 **1.1102230246251565e-16 ≤ 1e-6，PASS**。
- 恢复按补充2仅核对配置、划分、Python3.12.4原路径及sklearn1.7.1；`-X utf8 -E -P -B`、CPU native16，不改dispatch。无旧145产物重复哈希、无原始npz身份扫描。
- 补充2起即时记录的实际加载：**315个windows、315个cases、630次原始npz打开**，均eligible dev、非quarantine；原次加载315窗口的计数在内存中丢失，只按代码路径推断，不能计作测得的累计次数。
- 原始源train162069窗口，仅train，缓存/台账身份见`source_train_receipt.json`。未读内部test/source val/封存或quarantine payload，无GPU/M0/MSTCA/T1/B2。
- runtime全部留在`D:/UAV-Data/real_v2_b1_learning_curve_runtime/`；特征/模型/OOF/逐帧概率均不入库。原模型、OOF的近期字节审计依照补充2沿用，不声称本次再次证明其哈希。

## 2. 学习曲线与H1

以下为各重复合并外层OOF指标的均值[最小,最大]，不是均值±SD，也不是先平均预测再算指标。

'''+ '\n'.join(curve)+f'''

S仅作冻结参照：AP={x['S_window_reference']['AP']:.9f}，AUC(b)={x['S_window_reference']['AUC_b']:.9f}；不计入R/S+R拟合预算。

'''+ '\n'.join(h1table)+'''

25%档平均ΔAP=0.00815332138581586，95% CI [0.002775112482339398, 0.013715974146905346]，下界>0，**H1成立**。288关联组、2000次、rng20261108、每次重算每重复AP差再取均值；其余档只描述，不能替代主检验。

### 固定召回0.8时各正常总体FPR

下表仍为重复均值[最小,最大]，单位%；每重复阈值与实际召回在`window_metrics.csv`。这些窗口阈值不是事件部署onset，不用于H2选点。

'''+ '\n'.join(fpr)+'''

## 3. 内层onset、外层事件与H2

内层仅用非外层的4折OOF，选定后固定在外层评价。全部10个选择可行；每个主/检出并列列表只有所选点；补充1的无可行回退未触发。可行只要求两类检出≥0.8，并不意味着内层误报≤5.5/h。

'''+ '\n'.join(onsets)+'''\n\n以下为各选中点的内层OOF数值；11点全扫描与并列列表在`inner_selection.json`/`receipt.json`，不是在外层反选点。\n\n'''+ '\n'.join(inner_rates)+'\n\n'+ '\n'.join(events)+'''

H2四条件是**N1≤5.5/h且N2≤5.5/h且motor≥0.8且sensor≥0.8**，按点值，CI仅描述。两臂都仅N1/N2失败；检出两项均通过。整条路线H2不成立。S@0.95完全冻结，仅作参照，不参与择优或后续门槛。

N1全正常飞行有效可评分120259帧/0.668105556h；N2故障前440022帧/2.444566667h。暴露量含finite-score mask，不能用B0原始有效行时长替换；不得把N1/N2合并以绕过两个约束。

### 新确认检出的时延（秒）

'''+ '\n'.join(latency)+'''

上述为成功新确认案例的条件时延，相对参考故障onset，不等价于物理损伤起点或实时系统端到端时延。跨onset持续报警不算新确认；未检出不填0秒。所有已注册CI的未定义重复数=0，未重抽/丢弃/补伪计数；onset不在bootstrap里重选。

## 4. 描述性核查

### 逐案例故障对pre_fault配对AUC

'''+ '\n'.join(paired)+'''

每臂4个motor案例无y=1窗口，AUC保持null；没有删案例，事件检出分母仍154。高配对AUC说明窗口区分不仅来自完整正常飞行与故障会话的差异，但不证明物理因果，也不保证低事件误报。

R的N1=11.974/h，N2=7.363/h；S+R的N1=8.981/h，N2=11.045/h。S+R减少独立正常上的事件，却增加故障前段的事件；不能把两类正常或两个臂混合后宣称通过。仅窗口AP提高不足以证明上线可用。

'''+ '\n'.join(post)+'''

### 按飞行目录拆分（均为描述性，无额外判据）

'''+ '\n'.join(byflight)+'''

S+R的N2事件中20/27来自waypoint；R为12/18。acce/velocity的sensor检出在两臂均弱于pooled结果，未据此事后删案例或改变主判据。具体事件和案例见`events_per_case.csv`。

## 5. 协议结论与下一步边界

`consequences.key = H2_fails`：**不进入B2**，回到数据/标签问题与用户讨论。H1说明25%标注预算下源监督增加了AP，但H2未证明事件可用性；不能据此宣布跨域部署成功，也不启动T1/GPU或任何新实验。

## 6. 完整偏差、失败、恢复与审计限制

以下执行记录完整列出实现选择与失败/重试；数值协议没有调整。按补充1澄清不算偏差，补充2明确授权的恢复不隐瞒为一次未中断运行。

'''+notes.split('\n\n',1)[1]+'''

## 7. 产物与复现

- 原入口：`scripts/research/real_v2_b1_learning_curve.py`（原SHA `312b4c70…d12`，未改字节）；其全新运行保护仍保留。
- 恢复入口：`scripts/research/resume_real_v2_b1_learning_curve.py`。同解释器、原flags再次执行可从持久化检查点继续；完成后直接返回`B1_ALREADY_COMPLETE_NO_ACTION`，不重跑指标。
- 单测：`tests/test_real_v2_b1_learning_curve.py`与`tests/test_real_v2_b1_resume.py`；22项恢复合成检查通过，无真实数据访问。
- 结果/回执：`artifacts/real_v2_b1_learning_curve/`的JSON、CSV、run.log、`execution_notes.md`、`execution_audit.json`。`fit_receipts.jsonl`逐项保留200拟合的n_iter、Real/source窗口数、权重质量与墙钟耗时；正常早停不视为未完成。
- 原次证据：`artifacts/real_v2_b1_learning_curve_checks/`；恢复证据与不可覆盖的旧快照：`artifacts/real_v2_b1_learning_curve_checks_resume/`。
- 中断记录：`docs/research/2026-11-08/跨域B1学习曲线_中断现状与补跑评估_v1.md`；补充2已由上游commit b8af03a入库，本提交仍保留它，未人为改写。
- 本提交精确文件列表见`artifacts/real_v2_b1_learning_curve_checks_resume/submission_files.txt`；commit与push验证在Git记录和任务回复中给出，避免报告自引用commit哈希。
'''
path=ROOT/'docs/research/2026-11-08/跨域B1学习曲线结果_v1.md';path.write_text(report,encoding='utf-8',newline='\n')
audit={'status':'COMPLETE','built_from_completed_JSON_CSV_only':True,'no_analysis_rerun':True,
 'scientific_fits':200,'reused_completed_fits':122,'new_fits':78,'old_first_122_fit_records_equal':True,'old_completed_fit_names_restarted':[],
 'first_restart_physical_start':starts[0],'resume1':'PermissionError during feature/audit cache; 86 cached, no fits; retained failure and partial',
 'resume2_exit_code':0,'resume2_wall_time_s':x['wall_time_this_resume_s'],'io_events':receipt['io_events_this_attempt'],
 'python':'D:/python3.12.4/python.exe','python_version':'3.12.4','sklearn':'1.7.1','flags':['-X','utf8','-E','-P','-B'],'CPU_native_threads':16,
 'config_sha256':x['config_sha256'],'split_sha256':x['startup']['split_sha256'],'original_B1_code_sha256':sha(ROOT/'scripts/research/real_v2_b1_learning_curve.py'),
 'resume_code_sha256':sha(ROOT/'scripts/research/resume_real_v2_b1_learning_curve.py'),
 'recorded_payload_access':{k:v for k,v in x['access'].items() if k!='recorded_since_supplement2'},'local_artifact_counts':counts,
 'old_122_models_and_23_OOF_not_rehashed':True,'raw_real_npz_identity_rescan':False,'original_access_counter_not_reconstructed':True,
 'protected_user_files_and_frozen_reports_not_modified_by_task':True,'internal_test_not_read':True,'no_sealed_quarantine_payload':True,
 'no_GPU_T1_B2_or_full_restart':True,'H1':h1,'H2':x['H2'],'consequences':x['consequences'],
 'results_sha256':sha(OUT/'results.json'),'receipt_sha256':sha(OUT/'receipt.json'),'report_sha256':sha(path),
 'tests':['original 28 passed','resume initial21 passed','resume fix22 passed'],'all_synthetic_access_attempts':[],
 'submission_whitespace_check':load(CHECK/'submission_whitespace_receipt.json') if (CHECK/'submission_whitespace_receipt.json').exists() else None,
 'report_created_utc':datetime.now(timezone.utc).isoformat(),'formal_result_created_utc':datetime.fromtimestamp((OUT/'results.json').stat().st_mtime,timezone.utc).isoformat()}
(OUT/'execution_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print('REPORT_AND_AUDIT_WRITTEN',path,'rows',len(report.splitlines()),'bytes',len(report.encode('utf-8')))
