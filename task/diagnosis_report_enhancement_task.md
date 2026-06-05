# 诊断报告增强任务交接文档

## 背景

当前系统已经具备离线诊断任务、报告生成、报告中心、PDF 下载和 AI 解读入口。现有报告可以展示模型结论、置信度、严重度、TOP-K 概率、规则证据和建议动作，但整体仍偏向“模型结果展示”，还没有形成完整的工程诊断闭环。

本轮任务目标是把诊断报告增强为“可追溯的故障处置报告”，让报告不仅说明模型判断了什么，还能说明输入数据是否可靠、判断依据来自哪里、使用了哪个模型和规则版本、建议如何处置，以及后续人工复核状态。

## 当前边界

本任务只处理诊断报告相关功能，不处理仿真和实时链路。

本轮不做：

- RflySim / PX4 实时链路开发
- ULG 转 CSV 链路
- 旧 6 分类模型恢复
- 训练流水线调整
- 文档 archive 清理

当前运行主线按离线三分类理解：

```text
normal / motor / sensor
```

如代码中仍保留历史标签或扩展标签，优先保证当前三分类主线行为正确，不要因为兼容历史材料而扩大本轮范围。

## 关键代码入口

后端报告生成：

```text
apps/dashboard/backend/uav_dashboard_api/services/diagnosis_report.py
```

报告数据模型：

```text
apps/dashboard/backend/uav_dashboard_api/models/diagnosis_report.py
```

严重度与建议规则：

```text
apps/dashboard/backend/uav_dashboard_api/configs/severity_rules_v1.yaml
```

报告 API：

```text
apps/dashboard/backend/uav_dashboard_api/routes/diagnosis_reports.py
```

前端报告详情：

```text
apps/dashboard/frontend/src/modules/reports/components/ReportDetailDialog.vue
```

前端报告视图转换：

```text
apps/dashboard/frontend/src/utils/diagnosisReportView.js
```

报告列表和报告中心：

```text
apps/dashboard/frontend/src/modules/reports/components/ReportsWorkspacePanel.vue
apps/dashboard/frontend/src/modules/reports/composables/useReportsWorkspace.js
apps/dashboard/frontend/src/modules/reports/reportsView.js
```

AI 报告解读：

```text
apps/dashboard/backend/uav_dashboard_api/services/ai_assistant.py
apps/dashboard/frontend/src/modules/ai/composables/useAIAssistantWorkspace.js
```

## 目标效果

增强后的报告应能回答以下问题：

- 本次诊断结论是什么？
- 这个结论的风险等级是什么？
- 模型有多确定？
- 输入数据质量是否足够支撑该结论？
- 哪些指标或规则支撑了该结论？
- 使用的是哪个模型、规则版本和输入文件？
- 建议如何处置？
- 是否需要人工复核？
- 人工复核或维修处理后，结果如何记录？

## 建议新增报告结构

报告当前主要内容保存在 `DiagnosisReport.summary_json` 中。建议继续使用 `summary` 扩展字段，避免本轮大规模数据库迁移。

建议新增字段：

```json
{
  "data_quality": {
    "level": "good | warning | poor | unknown",
    "sample_count": 0,
    "window_count": 0,
    "feature_count": 0,
    "missing_columns": [],
    "missing_value_ratio": null,
    "invalid_value_ratio": null,
    "time_range": {
      "start": null,
      "end": null
    },
    "warnings": []
  },
  "traceability": {
    "run_id": null,
    "asset_id": null,
    "asset_filename": "",
    "input_csv": "",
    "model_name": "",
    "model_version": "",
    "model_hash": "",
    "rules_version": "rule_v1",
    "generated_at": ""
  },
  "model_card": {
    "task": "offline_multiclass",
    "class_set": ["normal", "motor", "sensor"],
    "intended_use": "飞行日志 / CSV 回放离线故障诊断辅助分析",
    "limitations": [],
    "metrics": {}
  },
  "report_confidence": {
    "level": "high | medium | low | unknown",
    "score": null,
    "reasons": []
  },
  "review": {
    "status": "pending | confirmed | false_alarm | ignored",
    "reviewer": "",
    "reviewed_at": "",
    "conclusion": "",
    "maintenance_notes": ""
  }
}
```

字段可以按实际可获取数据逐步填充。拿不到的数据不要硬编码假值，可以使用空值或 `unknown`，但前端和 PDF 必须能稳定展示。

## 任务拆分

### 1. 后端报告 summary 增强

在报告生成时补充：

- 数据质量摘要
- 模型与规则追溯信息
- 报告可信等级
- 人工复核预留字段

注意事项：

- 旧报告没有这些字段时，`to_dict()` 和前端展示不能报错。
- 不要破坏现有 `basic`、`conclusion`、`severity`、`topk`、`key_params`、`recommendations`、`recommendation_meta` 字段。
- `json.dumps` 保持 `ensure_ascii=False`，避免中文被转义成 `\uXXXX`。

### 2. PDF 报告增强

当前 PDF 已有：

- 诊断结论
- 主要判断依据
- 建议动作
- 技术摘要

建议新增或调整为：

1. 诊断结论
2. 数据质量
3. 主要判断依据
4. 风险等级与可信度
5. 建议动作
6. 模型与规则追溯
7. 人工复核记录
8. 技术摘要

要求：

- 中文正常显示。
- 字段缺失时显示 `-`、`未知` 或 `待补充`，不要抛异常。
- PDF 内容应和前端详情页保持一致，不要出现两套互相矛盾的解释。

### 3. 前端报告详情增强

在报告详情弹窗中增加：

- 数据质量区域
- 模型与规则信息区域
- 报告可信度区域
- 人工复核状态区域

建议展示方式：

- 数据质量：等级 + 关键指标列表
- 证据链：指标名称、当前值、状态、贡献分
- 模型信息：模型名、任务类型、类别集、规则版本
- 人工复核：状态标签和备注，占位即可

要求：

- 旧报告兼容。
- 移动端和较窄窗口不溢出。
- 不把原始 JSON 作为主要展示内容，JSON 只保留在技术详情中。

### 4. AI 报告解读增强

AI 助手应作为“报告解释器”，不要重新生成独立诊断。

推荐回答结构：

```text
结论：
证据：
风险：
建议动作：
不确定性与限制：
需要人工确认：
```

要求：

- AI 回答只能围绕报告字段解释。
- 如果数据质量差或字段缺失，要明确提示“不足以支持强结论”。
- 不要编造报告中不存在的传感器、工况、模型指标或维修结果。

### 5. 人工复核字段预留

本轮可以先只做字段和展示，不一定实现完整编辑流程。

最低要求：

- 报告详情能显示复核状态。
- 默认状态为 `pending` 或“待复核”。
- 后续可以扩展为报告确认、误报标记、维修记录和复测结果。

## 验收标准

功能验收：

- 成功离线诊断任务可以生成报告。
- 报告详情页可以展示新增的数据质量、追溯信息、可信度和复核状态。
- PDF 可以下载并展示新增章节。
- AI 报告解读能引用新增字段。
- 旧报告没有新增字段时，列表、详情、PDF、AI 解读都不崩溃。

质量验收：

- 中文不乱码。
- 页面不出现 `\uXXXX` 形式的中文转义。
- 不引入旧 6 分类作为当前主线。
- 不修改 RflySim/PX4/ULG 相关链路。
- 不删除历史文档或实验材料。

## 建议测试

后端建议补充或更新：

```text
apps/dashboard/backend/tests/test_main_flows.py
apps/dashboard/backend/tests/test_ai_assistant.py
apps/dashboard/backend/tests/test_ai_assistant_answer_style.py
```

前端建议补充或更新：

```text
apps/dashboard/frontend/src/modules/reports/reportsView.regression.test.js
apps/dashboard/frontend/src/modules/reports/components/ReportsWorkspacePanel.regression.test.js
apps/dashboard/frontend/src/modules/ai/useAIAssistantWorkspace.regression.test.js
```

可优先覆盖：

- 新字段存在时的展示
- 新字段缺失时的兼容
- AI 解读不编造报告外信息
- 中文内容不出现 Unicode 转义

## 参考资料

- ISO 17359: Condition monitoring and diagnostics of machines - General guidelines
- ISO 13374: Condition monitoring and diagnostics of machines - Data processing, communication and presentation
- NIST AI Risk Management Framework
- Model Cards for Model Reporting
- Datasheets for Datasets

这些资料的共同点是：诊断报告需要兼顾数据来源、模型/规则说明、证据链、限制说明和后续决策，而不是只输出一个分类标签。
