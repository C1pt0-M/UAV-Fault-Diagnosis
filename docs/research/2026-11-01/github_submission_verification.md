# GitHub归档前验证记录

日期：2026-11-01。本记录对应近期源码、测试、配置、原创报告及轻量证据的集中归档，不是训练、数据重建或部署验收。

## 本地检查

|检查|实际结果|
|---|---|
|Python全范围测试，补齐源码导入路径后|922 passed，1 failed，1 skipped，22 deselected|
|前端Vitest|37个测试文件、156项测试全部通过|
|前端Vite生产构建|成功；保留部分chunk大于500kB的警告|
|Python compileall|通过|
|新增／修改暂存文件的凭据模式筛查|匹配项均为测试夹具字符串；未发现真实凭据。不是安全认证|
|当前冻结代码与归档证据的暂存字节检查|通过；相关路径通过.gitattributes保留原字节，未改历史hash|

第一次Python运行出现3项失败，其中2项是子进程无法继承pytest的pythonpath配置，未找到uav_fault／uav_dashboard_api。仅为复测命令设置以下PYTHONPATH后，这2项通过；没有修改应用代码或测试来规避失败。新环境应按README进行editable安装，或为子进程明确提供源码路径。

```powershell
$env:PYTHONPATH = "$PWD/apps/dashboard/backend;$PWD/packages/uav_fault;$PWD/packages/rfly_bridge"
python -m pytest -q tests apps/dashboard/backend/tests packages/uav_fault/uav_fault/online packages/uav_fault/uav_fault/telemetry packages/rfly_bridge/rfly_bridge
```

唯一仍失败：`tests/test_source_normal_sampling_training.py::test_frozen_protocol_and_source_code_pins`。历史2026-10-30协议锁定trainer哈希`0580a8106ca942edb6c91119ad65e2d064047b8fb32e1a9311c906fb628783f6`，当前加安全门禁后的文件哈希为`aa4d64f6c79a8b568e8a5f3f79953dd9572d9d074b7f7ee7d699d4894e5eddc3`，因此正确拒绝历史代码身份混用。此问题已有留档，本次不改旧pin、不删除测试、不把整个测试集写成全绿。历史实验源码快照另存于artifacts相应结果目录。

跳过项为Windows无权限创建目录符号链接的检查；22项external按现有pytest配置排除。测试夹具短JWT密钥警告保留，不代表新生产凭据。前端依赖安装、打包优化等无关变更本次不做。

另：git diff --check报告冻结证据原CRLF、历史源码快照末尾空行及旧归档Markdown行尾空格。按CRLF规则重新检查后仍保留旧归档格式告警；不为消除格式告警改写有hash的历史证据，也不宣称该检查全绿。

## 上传范围与可复现边界

- 包含工作区现有近期源码改动、测试、配置、先前已暂存删除／移动，以及本轮补充的交接和历史模型比较说明。
- 研究报告与轻量证据原先被.gitignore忽略，本次按显式清单纳入；被跟踪后后续修改仍可见。清单见`github_evidence_manifest.json`。
- 不新增上传原始日志、训练缓存、权重、NPZ概率／张量、大型压缩包、第三方论文PDF或OCR全文、运行时数据库、本机配置和凭据。这些保留本地。旧有跟踪模型资产不删除。
- 因未上传全部数据和checkpoint，GitHub快照不是整个实验盘备份。历史hash及原索引记录可用于匹配用户已有本地／服务器资产，不能声称无外部数据就可复现实验。
- 文献笔记与方案包含此前记录的参考信息，本次仅归档，不重新进行文献查新；第三方全文入口可能指向本地排除内容。
- 本轮未连接AutoDL、未分配资源、未启动推理／训练，未修改原冻结协议及模型权重。

## 历史模型比较

已确认31通道fewshot旧协议下存在纯MSTCA与KG-MSTCA资产及聚合结果，见`github_handoff.md`。不能再将其描述为从未做过对比，也不能将其与当前26通道因果输入源验证直接混为同一实验。后续先复用已有证据，再决定是否需要新实验。
