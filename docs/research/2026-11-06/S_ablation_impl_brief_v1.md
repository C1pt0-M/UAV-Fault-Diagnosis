# 任务书：S 组结构消融——实现与本地冒烟（v1）

委托方：架构设计（Claude）。执行方：PI agent。日期：2026-11-06。

## 依据

- 冻结协议：`configs/experiments/s_ablation_mstca_v1.json`（以它为准）；设计说明：`docs/research/2026-11-06/S组结构消融协议_v1.md`。
- 可复用：`scripts/research/train_m0_groupcv.py`（折训练框架）、`train_causal_mstca_m0.py` / `train_causal_source_v1.py`（`source.prepare`、`sampling_weights`、`IndexedSource`、`FocalLoss`、`get_warmup_cosine_lr`、`save_state`）、`artifacts/feature_groupcv_v1_20261105/evaluate_groupcv.py`（分层口径）、`ablation_factory.py`。

## 本轮范围：只实现和做 CPU 冒烟，不做正式训练

服务器还没有，所以本轮**不连接远程，不使用 GPU，不做正式训练**。

## 硬性边界

- 禁止：读取 source val、内部 test、任何 Real 数据；修改已有文件（包括冻结协议 JSON、factory、已有测试）；GPU；git commit/push。
- 允许：新建 `scripts/research/train_s_ablation.py`、`scripts/research/evaluate_s_ablation.py`、`scripts/research/preflight_s_ablation.py`、`tests/test_s_ablation.py`；输出到 `artifacts/s_ablation_mstca_v1_smoke_20261106/`。
- 读取 source train 时只读冒烟所需的窗口（每臂不超过 2048 个 fit 窗口、不超过 1024 个 inner 窗口、不超过 1024 个留出窗口），以及 train ledger 的全部元数据列。

## 要实现的内容

1. **`train_s_ablation.py`**：参数 `--protocol --data --arm --seed --fold --phase {cv,val} --output`，以及冒烟专用的 `--cpu-smoke`（限制窗口数、2 轮、允许 CPU）。
   - 启动时校验协议 SHA 锁、代码 SHA、数据 SHA。冒烟模式可以跳过 `cache/train.pt` 的全量 SHA，但必须在 report 中注明跳过了哪些。
   - 按协议计算外层折和 inner（`association_group` 用字符串读取）。正式模式断言窗口数与协议中的期望值完全一致。
   - 模型只通过 factory 构造和初始化；断言 `full_state_digest` 等于 golden 值；保存 factory receipt。
   - 训练配方、选点、早停严格按协议的 `training` 和 `selection` 两节实现：类平衡 CE；显著改善 = 低于参考值 × 0.995；min 30 轮、patience 20。训练过程中**不得**触碰留出折。
   - 训练结束后，用 best 和 last 权重各对留出折预测一次，保存 npz（概率 + `original_cache_index`）。`report.json` 记录每轮训练损失、inner CE、best/last 轮数、用时、参数量、MAC、环境、各项 SHA。
   - `--phase val` 的代码路径要实现，但本轮**不执行**（它会读 val）。在测试里用 mock 确认：未提供阶段 1 结论文件的 SHA 时拒绝运行。
   - 正式模式下：无 CUDA、环境版本不符、输出目录已存在、`CUBLAS_WORKSPACE_CONFIG` 不是 `:4096:8`，都直接拒绝。
2. **`evaluate_s_ablation.py`**：读取全部 run 的留出预测，计算协议 `evaluation` 一节的全部指标、bootstrap 和 `decision_rules` 的结论，输出 `evaluation.json` 和一份可读摘要。缺少任何 run 时拒绝出结论，只输出"不完整"状态。
3. **`preflight_s_ablation.py`**：把协议 `preflight_gates` 中可以自动化的项做成一个命令，供服务器上运行（golden 测试、环境、SHA、折大小、GPU 2 轮确定性）。本地只运行其中不需要 GPU 的部分。
4. **`tests/test_s_ablation.py`**（合成数据，CPU）：
   - FP@R97 阈值规则的边界：并列值，恰好 97%，全部同值；
   - 类平衡 CE 的计算；
   - 早停规则：用合成损失序列验证 min_epochs、patience、显著改善参考值只在显著改善时更新；
   - bootstrap 的配对性：同一个 replicate 对所有臂、所有 seed 使用同一组重采样；
   - 判读规则：每种结论至少一个构造样例，包括"seed 方向不一致 → 不确定"和"AP 方向相反 → 不确定"；
   - 用合成 ledger 验证折和 inner 的哈希规则，包括带前导零的组名。

## 本地冒烟

四个臂各跑一次：seed 42、fold 0、`--cpu-smoke`。然后用这 4 个 run 加上合成的其余 run，演示 `evaluate_s_ablation.py` 的完整输出（明确标注为合成，不是结果）。另外确认 4 个臂的初始 digest 与 golden 值一致。

## 交付与回复

- 报告 `docs/research/2026-11-06/S_ablation_impl_smoke_v1.md`：实现与协议逐条对照表（协议条目 → 代码位置）、冒烟输出、pytest 汇总、偏离清单、你认为协议中有歧义的地方（只列出，不要自行修改协议）。
- 回复时只给：pytest 汇总、四臂冒烟是否通过、偏离清单和歧义清单。
