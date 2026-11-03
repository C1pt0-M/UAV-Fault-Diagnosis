# M3针对性错误样本分析 v1

报告：[M3_targeted_error_samples_v1.md](../../docs/research/2026-11-03/M3_targeted_error_samples_v1.md)。

本轮仅读取C0、CO、CN、CON、CO9保存的best Val概率和已有Val台账；无训练、模型前向、连续回放、阈值搜索或Test／Real使用。输出与原训练结果分开；不修改权重、预测和日志。

## 表与检查

- `verification.json`：五组原预测SHA、32381个同序原ID／标签／窗口起止行身份核验，输入未改。
- `independent_check.json`：用保存NPZ直接布尔计数复核六对模型；七对模型全部分组和迁移集合合计一致，四对C0比较与此前核验结果一致；3项针对性测试通过。
- `transition_totals.csv`：四对C0主比较及CN→CON、CO→CO9、CO→CON补充对照。
- `transition_groups.csv`：故障类、域、目录工况、标签phase、正常来源、hover时间段及交叉分组；含参照FN/TP/FP/TN的风险分母及迁移率。
- `window_comparison.csv`：全32381原窗元数据、五组故障概率与预测；原ID不等于本表行号。
- `changed_windows.csv`：七对比较所有FN→TP、TP→FN、FP→TN、TN→FP窗口，包含原ID、case UID、标签、概率及其变化。相同窗口可能因比较对不同出现多次，不是新的样本人口。
- `probability_changes.csv`：每个比较对／迁移类型的概率及配对差值P10／中位数／P90；正常和故障总体也报告。不将两组各自中位数之差误作逐窗差值中位数。
- `case_concentration.csv`、`event_concentration.csv`：各案例计数、案例／association group覆盖及高贡献案例集中度。
- `CN_CON_linked_patterns.csv`：CN新增漏检被CON恢复、CN纠正误报被CON撤销等对应关系，含分类分组。
- `typical_examples.csv`：每对／事件贡献最多的两案例，各自取最接近案例内中位概率差的窗口，原ID解平；描述性选例，不改变评价人口。
- `report_examples.csv`：报告采用的9个上述代表窗，加1个CN新增漏检中C0概率最高的窗口。不是有利样本筛选，完整变化窗仍保留。
- `completion.json`：此次交付文件SHA及范围。

已有工况仅是目录场景，不视为实时飞行意图真值。`phase`采用原标签政策；hover前后段按原裁剪窗口末端20ms网格相对30秒划分，不跨目录混入其他工况。`normal_origin=normal_flight`对应独立正常案例，不能仅靠同名phase识别独立正常：部分故障案例也会含normal_flight标签phase。

预测本身不能给出分支margin、逐维因果作用或校准证明。本轮未计算输入特征或观测运动state，也未读取故障物理起点／实机时钟。重叠窗、单seed、开发Val和SIL/HIL配对限制结论。

复算入口：

```bash
python scripts/research/analyze_m3_error_samples.py --output <一个不存在的新目录>
python -m pytest -q tests/test_m3_error_samples.py
```

入口只用NumPy／pandas和标准库，不导入模型／torch；拒绝覆盖已有分析目录。复算入口生成全量机器表和自动代表窗，简报及其精选代表窗是基于这些表的本次交付，未伪造缺失元数据。

分析生成阶段仅在本地完成，`completion.json`保留该阶段的原始回执。现按用户后续授权将报告、CSV、核验文件、脚本和测试归档至GitHub；[发布清单](publication_manifest.json)记录本次实际发布文件SHA，并说明README发布说明的更新。原权重、NPZ和训练日志未修改，也不包含在此次新增提交中。
