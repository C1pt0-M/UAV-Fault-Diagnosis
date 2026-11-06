# MSTCA 基线审查输出

正式报告：`docs/research/2026-11-06/MSTCA_baseline_audit_v1.md`。

## 取数入口

- `checks_v2/results.json`：正式引用的主检查结果。
- `checks_v2/source_evidence.json`：源码行号、SHA 和冻结 pin 核对。
- `edge_cases.json`：持续阶跃、浮点近并列补充。
- `pytest.txt`：22 个选定用例通过；没有执行 test/Real fixture 用例。
- `checks_v2_console.txt`、`edge_cases_console.txt`：运行输出。
- `checks_v2/reader_fixture/`：每 split 64 个源窗口的本地审查子集，不是正式数据版本。
- `checks_v2/runtime_fixture/best.pt`：归档未训练 M0 状态，为测试真实 FaultDetector 加载器而写，绝非训练后权重或部署资产。

根目录的 `results.json`、`source_evidence.json`、`console.txt` 及两个根级 fixture 目录保留第一轮探索；该轮绝对高度合成样本未同步更新 acc_norm，导致 canonical 比较混入通道不一致，不作为报告正式证据。正式版在 `checks_v2` 修正此合成样本，并加入本机 fresh KG 对照以区分版本差异和 RNG 构造顺序。

## 边界

仅 CPU 前向/反向，0 优化步，0 GPU 训练；只访问源 train/val，mmap 实取特征各64窗，完整标签/metadata用于准入和索引抽样。未读取 test/Real。没有运行历史训练/回放入口，没有修改已有源码、冻结配置、数据或权重，没有 commit/push。
