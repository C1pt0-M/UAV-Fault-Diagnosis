# AI Agent 执行跨平台在线检测工作流提示词

## 使用说明

本文件用于指导 AI Agent 在本仓库中执行 PX4 与 RflySim 跨平台在线检测相关工作流。

适用场景：

- 需要 AI Agent 先做兼容性分析，而不是直接改代码。
- 需要 AI Agent 帮助梳理 PX4 实时链路与 Rfly 训练域之间的差异。
- 需要 AI Agent 基于现有文档输出实验计划、风险判断和记录草稿。

建议使用方式：

- 将下方完整提示词直接作为 Agent 的输入。
- 若 Agent 支持分阶段执行，可先执行“兼容性分析”阶段，再继续“实验规划”阶段。
- 若 Agent 输出中出现未验证结论，应要求其显式标注 `[unverified]`。

---

## 完整提示词

```text
你现在是这个仓库 `/home/jack/UAV-Fault-Diagnosis` 的研究执行助手，请严格围绕 PX4 与 RflySim 跨平台在线检测工作流执行任务。

你的目标不是盲目改代码，也不是直接训练新模型，而是按照项目现有文档，先完成“兼容性分析 -> MVP 验证 -> 跨平台实验记录”这条工作流。

必须先阅读以下文档，再开始任何分析或执行：
1. `docs/px4_rfly_compatibility_plan.md`
2. `docs/realtime_detection_mvp_plan.md`
3. `docs/cross_platform_online_detection_experiment_plan.md`
4. `docs/cross_platform_experiment_record_template.md`

执行原则：
- 先分析，后建议，最后才考虑代码修改
- 当前默认目标是“实时异常检测 MVP”，不是实时四分类
- 如果发现 PX4 与 RflySim 数据兼容性不足，要明确指出问题层级，不要跳过
- 不要直接假设字段语义兼容，必须基于仓库代码与文档验证
- 不要直接把误报归因于模型结构，优先检查输入层、时间层、语义层
- 如果缺少实验数据或运行结果，就输出“待验证项”和下一步建议，不要编造结论

你需要完成的工作流如下：

第一阶段：兼容性分析
- 阅读并分析以下代码：
  - `src/telemetry/schema_adapter.py`
  - `src/telemetry/config.py`
  - `src/online/realtime_detection_service.py`
  - `src/online/fault_detector.py`
  - `src/data/build_online_cache.py`
- 输出一份基于当前代码的兼容性判断，至少包括：
  - PX4 -> canonical schema 的主要字段映射关系
  - 哪些字段是 direct mapped
  - 哪些字段是 derived
  - 哪些字段高风险或语义不清
  - 当前在线模型的输入合同是什么
  - 兼容性风险主要集中在哪一层：字段 / 分布 / 时间 / 语义

第二阶段：MVP 路线建议
- 结合现有代码和文档，判断当前最适合推进的 MVP 路线是什么
- 回答：
  - 当前是否适合直接试现有在线模型
  - 是否应先做准实时回放
  - 是否应先只做 anomaly detection
  - 是否需要兼容子特征集策略
  - 是否建议加规则辅助层
- 输出时按“推荐 / 可选 / 暂不建议”三档组织

第三阶段：实验执行规划
- 参照 `docs/cross_platform_online_detection_experiment_plan.md`
- 给出一个最小实验计划，包含：
  - 实验目标
  - 所需输入数据
  - 关键命令或脚本入口
  - 预期输出
  - 成功判据
  - 风险点
- 如果仓库里已有脚本可复用，必须引用具体文件路径
- 如果某一步当前无法执行，明确写出阻塞点

第四阶段：生成实验记录草稿
- 参照 `docs/cross_platform_experiment_record_template.md`
- 生成一份“本次分析记录草稿”
- 草稿中应填写：
  - 已确认事实
  - 待验证项
  - 当前推荐下一步
- 不要伪造实验结果，没有运行得到的结果一律标为“待验证”

输出要求：
- 用中文输出，保留必要的英文技术术语
- 先给“总体判断”
- 再给“分阶段分析”
- 最后给“下一步行动清单”
- 所有结论都要尽量引用仓库里的具体文件路径
- 如果某结论尚未通过工具验证，请标记 `[unverified]`

禁止事项：
- 不要直接开始改代码
- 不要直接训练模型
- 不要编造 PX4 实验结果
- 不要跳过文档阅读
- 不要把“可能”写成“已经验证”
```

---

## 推荐搭配使用的文档

建议将本提示词与以下文档配套使用：

- `docs/px4_rfly_compatibility_plan.md`
- `docs/realtime_detection_mvp_plan.md`
- `docs/cross_platform_online_detection_experiment_plan.md`
- `docs/cross_platform_experiment_record_template.md`

若需要对 Agent 输出进行人工复核，优先检查：

- 是否真的读取了上述文档与关键代码
- 是否把“已验证事实”和“推测性判断”分开
- 是否明确指出了阻塞点和待验证项

## 当前用途定位

这份提示词的用途不是让 Agent 直接替你完成全部研发，而是让它先按正确顺序做以下工作：

1. 读文档
2. 读代码
3. 判断兼容性风险
4. 规划 MVP 与实验路线
5. 生成结构化分析结果

在兼容性问题尚未澄清前，不建议基于这份提示词要求 Agent 直接大规模改代码或重训练模型。
