# 内部SIL/HIL测试集标签时间修复v1

As of 2026-10-31。版本`rflymad_internal_test_labeltime_v1`。

## 交付结果

- 原553案例（SIL276／HIL277）、35392窗口全部保留，原cap64索引、裁剪、50Hz连续输入与划分不变。
- 553份实际CSV副本位于cases/SIL与cases/HIL，按与源train／val修复一致的原始GTD时间保持规则改正28644个逐时刻状态。命令字符串／连续输入等其他原字段字符串不变。
- 旧标签与GTD诊断参考不同的窗口末端628个；其中603个有两来源一致支持，已作为可判定标签修正。其余窗口不硬判。
- 35365个末端可参与明确版本的监督分类评估；27个GTD／ULog来源分歧末端保留未知。27中25个属于旧／新差异、2个虽然旧标签与GTD同值，两来源仍不同。
- 本批553例未发现原始惯性时钟倒退。456个窗口跨历史描述切点0.1秒采样间隙，保留输入并记录，不按此任意剔除。
- 5个案例的原始注入在官方裁剪之后发生，原片段不因此延长，也不把来自故障目录等同片段内已经故障。

## 使用文件

- repair_manifest.json：最终完成清单、统计与来源／输出哈希。
- case_manifest.csv、parts/part*.jsonl：新旧CSV身份、原始GTD／ULog哈希、原划分与逐例统计。
- cases/SIL、cases/HIL：独立实际CSV；新增labeltime_supervision_valid与labeltime_input_clock_review。其他命令列保留历史值，不代表已修监督。
- references：553份完整行时间轴、历史标签、GTD／命令参考、来源覆盖与资格；不删除时间轴。
- test_window_links.csv：35392个原缓存索引对应新旧标签、资格、输入诊断、重建phase／fault_fraction／fault_onset_row。
- evaluation_ineligible_window_ledger.csv：27个来源分歧窗口及原因，未从证据缓存删除。
- internal_test_evidence.pt：全部原输入与新旧标签／mask，故意没有通用y键，不是可以无视mask的旧test.pt。
- norm_stats.json：历史源训练统计字节一致复制，测试数据不拟合归一化。

证据缓存字段：x、historical_y、gtd_reference_y、command_reference_y、qualified_reference_y、endpoint_evaluation_valid、source_cache_index。qualified_reference_y仅在mask为真时为0／1，其余为-1；-1不是第三类，不可直接送旧损失或旧二分类指标。

只能作冻结内部最终测试，不能用这553例训练、选模型、选阈值或重新拟合统计。后续评估必须先应用endpoint_evaluation_valid，报告原35392分母和27个无效数，不把覆盖变化与模型提升混为一谈。事件／延迟评估还需逐行mask感知处理，不能跨未知时段硬判连续事件；本版本未认证event scoring。

## 旧资产与边界

原D:/RflyMAD、旧CSV、online_cache_v2/test.pt、旧划分、源修复train／val、Real修订、模型和部署均未修改。本轮没有训练／推理、没有查看模型概率、没有新性能结论。输入插值、严格因果在线与物理故障真值不在本标签收尾范围。

HIL划分清单中的旧输出哈希早已过期：本轮复用了先前已核实的2026-08-13处理日志与2026-10-08全量审计一致哈希，保留旧清单哈希和成员；再用全部旧test X/y逐元素确认实际版本与缓存对应。它不是新发现的数据损坏，原冻结清单未改。

构建首批在101条完成后触及200秒工具时限，保存中断／脚本记录，核验完成文件后续跑，未登记输出另存interrupted_files；后续分批日志均留档。完成状态以repair_manifest为准。

## 复现

仓库入口scripts/data/repair_internal_test_labeltime.py；初始化、5个case分批、最后cache阶段。输出已存在时拒绝重新初始化，已完分批拒绝覆盖，未完分批只在已完成身份验证后续建。

独立复核入口scripts/research/verify_internal_test_labeltime.py；全部CSV／原始事件／mask按5批检查，另对全部X、旧y、索引、划分和统计核查。证据位于D:/UAV-Fault-Diagnosis/artifacts/internal_test_labeltime_v1_20261031/。本轮相关91项回归检查通过；最终复核状态见该目录completion.json。
