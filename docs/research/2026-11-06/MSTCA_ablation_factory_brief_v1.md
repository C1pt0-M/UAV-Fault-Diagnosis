# 任务书：MSTCA 消融臂统一构造与初始化（实现 v1）

委托方：架构设计（Claude）。执行方：PI agent。日期：2026-11-06。
前置：`docs/research/2026-11-06/MSTCA_baseline_audit_v1.md`（你上一轮的审查报告），本任务实现其第 7 节设计，并补上 P2、P5 指出的缺口。

## 已拍板的决定（不要再改）

1. **消融使用统一初始化政策 `module_seed_v1`，所有 seed（含 42）都用它。** 不在不同 seed 之间混用"KG 兼容"和新政策。S0（完整 M0）在新协议下会重新训练，所以不需要新政策复现旧 seed42 锚点。
2. **旧 seed42 锚点只做兼容核对**：把它转换成纯 M0 文件，digest 必须等于 `a68551099cba00575f1eec6edc31ccf47b39ca8e0fa204ae2fe065d3400be646`。这个文件不参与新消融，只证明"不依赖 KG 类也能精确复现历史 M0 起点"。
3. 消融臂固定为四个：`mstca`（直接复用现有 `OnlineMSTCA`）、`mstca_no_se`、`mstca_single_k7`（三路都是 k7，通道仍为 24/24/16，后续结构不变）、`cnn1d_matched`。
4. **公共层共享规则**：同一 seed 下，各臂中语义相同、形状相同的模块，初始值必须与 `mstca` 臂逐位相同；删除或改变某个模块，不得让其他模块的初始值漂移。
5. 选点规则、评估人口、阈值不在本任务中决定，留给之后的消融协议。本任务不训练。

## 硬性边界

- **禁止**：任何训练（GPU 或 CPU 优化器步）；读取内部 test 或任何 Real 数据；修改已有源码、测试、配置、冻结协议、数据、权重（包括 `fault_detector.py`，P3 本轮不修）；git commit/push。
- **允许**：
  - 新建 `packages/uav_fault/uav_fault/models/cnn/ablation_factory.py`（新文件）；
  - 新建 `scripts/research/build_pure_m0_anchor.py`（一次性转换脚本）；
  - 新建 `tests/test_mstca_ablation_factory.py`；
  - 输出写到 `artifacts/mstca_ablation_factory_v1_20261106/`（拒绝覆盖已存在的文件）。
- 只用合成数据和 CPU。Python 加 `-I`。

## 要实现的内容

### 1. 纯 M0 锚点转换（`build_pure_m0_anchor.py`）

- 输入：`configs/experiments/causal_pure_mstca_m0_v1.json` 中 `initialization_anchor` 指向的归档 `initial.pt`，先核对文件 SHA 和状态 digest。
- 只按审查报告 B7 中已核对的映射（`raw_encoder.*→encoder.*`、`head.*→head.*`，41 个张量）提取，保存为新文件；不修改原文件。
- 输出 receipt：原文件 SHA、原 digest、映射表、新文件 SHA、纯状态 digest（必须等于上面的值，不等则失败）。

### 2. 构造接口（`ablation_factory.py`）

```python
model = build_variant(variant, in_channels=26, num_classes=2, dropout=0.2)
receipt = initialize_variant(model, variant, seed, policy="module_seed_v1")
```

- `module_seed_v1`：每个**语义模块**（例如 `encoder.branch_s.0`、`encoder.branch_m.1`、`encoder.se.*`、`head.1`；现有代码中 k3/k7/k15 分别是 `branch_s`/`branch_m`/`branch_l`；名称表写死在代码里）的子 seed = `int(sha256(f"module_seed_v1|{seed}|{module_name}|{shape}"))` 截取 63 位；用局部 `torch.Generator` 初始化，不触碰全局 RNG。
  - Conv1d/Linear：与 PyTorch 默认相同的分布（kaiming_uniform(a=√5) 权重 + fan-in 均匀偏置），但使用局部 generator。
  - BN：weight=1、bias=0、running_mean=0、running_var=1、num_batches_tracked=0。
  - 子 seed 名称里含 shape，所以 `mstca_single_k7` 中 `branch_s`/`branch_l` 换成 k7 后形状改变，自然得到新值（模块名不改，`branch_s` 与 `branch_m` 形状虽同但名称不同，因此不会重复）；`branch_m` 名称和形状不变，因此与 `mstca` 臂逐位相同。
- 构造时不得调用 `torch.manual_seed`，构造前后全局 CPU RNG 状态必须不变。
- `cnn1d_matched`：静态枚举整数宽度，选出参数量最接近 105938 的结构（容差 ±2%），encoder 输出保持 128 维，使 head 形状与 M0 相同。具体候选表和最终选择写进报告。如果做不到 ±2% 或 128 维输出，报告原因，不要硬凑。它与 MSTCA 没有共享语义模块（head 除外：若形状相同，head 视为共享）。
- receipt 字段：`variant`、`seed`、`policy`、`param_count`、`macs`（与审查报告相同的计法：只计 Conv/Linear）、`shared_keys`、`variant_specific_keys`、`full_state_digest`、`shared_subset_digest`。digest 沿用 `train_source_normal_sampling.state_digest` 的定义（只 import，不修改）。
- 另外提供 `load_pure_m0_anchor(path)`：加载第 1 步的纯锚点到 `mstca` 臂，校验 digest。

### 3. 测试（`tests/test_mstca_ablation_factory.py`，全部 CPU、合成输入）

1. 四臂：输出形状 (B,2)，B=1 和 B=32 的 eval 前向有限；参数量和 MAC 与报告一致；`mstca` 臂为 105938 / 6764928。
2. 同 seed 两次构造：full digest 相同，前向逐位一致；不同 seed：digest 不同。
3. 构造前后全局 CPU RNG 状态不变。
4. 对 seed 42/43/44：`mstca_no_se` 的全部 key 与 `mstca` 的对应 key 逐位相同，只缺 SE 的 key；`mstca_single_k7` 中 `branch_m`、SE、block2/block3、head 与 `mstca` 逐位相同，只有 `branch_s`/`branch_l` 的 Conv 权重和偏置不同（它们的 BN 形状不变、取值为固定常数，因此也相同）；key 清单与 receipt 一致。
5. `load_pure_m0_anchor` 得到 digest `a6855109…`（使用转换脚本的输出；若该文件不存在则 skip，并在报告中说明）。
6. 一个**标记为 `xfail(strict=True)`** 的回归，记录审查报告 P3：26 维、高度随时间变化的合成 records，`FaultDetector.predict_records_batch` 第二个及之后窗口的输入应等于逐窗 `predict` 的输入。它现在应失败；以后修好 P3 时这个测试会变成 XPASS 并提醒删掉 xfail。如果构造 FaultDetector 需要的夹具太重，可以只比较 `_build_feature_matrix` 加切窗的结果与逐窗相对化的结果。

## 交付物

1. 上述三个新文件。
2. 报告 `docs/research/2026-11-06/MSTCA_ablation_factory_v1.md`（中文，简短）：
   - 四臂表：参数量、MAC、共享 key 数、臂专属 key 数、seed 42/43/44 的 full digest；
   - `cnn1d_matched` 的候选表和最终结构；
   - 纯锚点转换 receipt；
   - pytest 输出（通过、失败、xfail 的数目）；
   - 与本任务书不一致或做不到的地方，逐条列出。
3. 回复时只给：四臂表、pytest 汇总、偏离清单。
