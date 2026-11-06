# MSTCA 消融臂统一构造与初始化 v1

完成范围：三个新增实现/测试文件；全部 seed（含 42）统一使用 **module_seed_v1**。旧 seed42 纯锚点仅用于历史兼容核对，不参与新消融。未训练、未执行优化器步、未读取任何数据集（包括 train/val/test/Real），未修改已有文件或 commit/push。

## 1. 四臂实测表

MAC 按单窗 26×150、只计 Conv/Linear 的乘加，1 MAC=1 次乘加；不计 BN/激活/池化/SE 广播乘法。key 数指 state_dict 张量，包含 BN buffers；“共享”均以相同 seed 的 `mstca` 为参照，不要求四臂全部 state digest 相同。

|臂|参数量|MAC/窗|共享 key|臂专属 key|seed 42 full digest|seed 43 full digest|seed 44 full digest|
|---|---:|---:|---:|---:|---|---|---|
|mstca|105938|6764928|41|0|`c86a31c99c3858e6b0d96262fb92dac4c472cb8376c0356df8a3c8e35fec521d`|`227dc429536953fae4a0d8d63069617ed09abe39a4842e60cde7b375738686f5`|`f53dd1b285f7586146402a7cdcfcee2c2dde3b8886346c66c64961727388a6bd`|
|mstca_no_se|103810|6762880|37|0|`40cdc82c75161257d4ec2f1d3174f475d825b2bce7401c49a7bac27e38e4c297`|`869af3abf771b0889110dcdce320c9946c491f885935f038833839489be740e0`|`819fe00c9175e3aaf387c61272b29eb15cfecca1329cc3b8ac2a441ba63dcac6`|
|mstca_single_k7|105106|6640128|37|4|`2f4cf7a1eb6d917158be670044481951ac2e533579dccc2d4ef84a109b406509`|`3e343db34359531698597b7c6d9cf25d12af8fc63c76edcecd909e4ff3916729`|`653bc50a62a4d6a9a61d5f8dafd89ebf71cee042562f4e5b22c005690a9d4d28`|
|cnn1d_matched|105929|7097374|2|21|`61c67e76cdf337df0bd6eb05bc0f375e84a207d859878e0ded2381dc30fbf48a`|`8860d857fc60d2ebc35c1dd5081762af725de88a5b2ea193ce68549002f97041`|`2c96ed1bb3b595622e5a0b4ce5fc4a6e8a4f85ed16b547a5c9be12fdc6af6076`|

完整机器可读结果：`artifacts/mstca_ablation_factory_v1_20261106/four_arms.json`；12 份 receipt（含所有 key 清单、完整和共享子集 digest）：同目录 `receipts.json`。

## 2. 实现与初始化约定

新增：

- `packages/uav_fault/uav_fault/models/cnn/ablation_factory.py`
- `scripts/research/build_pure_m0_anchor.py`
- `tests/test_mstca_ablation_factory.py`

完整 M0 直接使用现有 `OnlineMSTCA`；去 SE 只将 SE 换为 Identity；单尺度仅将 branch_s/branch_l 的 Conv 改为 k7，保持 24/24/16 及后续结构。CNN 复用现有 `OnlineCNN1D` 前向与层序，仅替换对应宽度的 Conv/BN/head，输出固定为 128。

语义模块名表写死在 factory 中，MSTCA 的 branch/SE/block/head 名称不改；CNN 的 `features.*` 不与 MSTCA 编码器共享语义，只有 `head.1` 共享。

每个 Conv/Linear 使用一个本地 CPU Generator：

```text
shape = tuple(module.weight.shape)
payload = f"module_seed_v1|{seed}|{module_name}|{shape}"
sub_seed = int(sha256(payload.encode("utf-8")).hexdigest(), 16) & ((1 << 63) - 1)
```

例如 shape 字符串为 `(24, 26, 7)`，取 SHA256 整数的**低 63 位**。同一 generator 依次初始化 weight 和 bias，使用 kaiming_uniform(a=√5) 及 ±1/√fan_in 均匀偏置；因此换 kernel 时，Conv 的 weight 和 bias 都变化。BN 全部用固定默认常数并重置 buffers。

构造使用 CPU device 和 `fork_rng(devices=[])` 保存/恢复默认构造器消耗的 CPU RNG，不调用 `torch.manual_seed`；政策初始化只使用局部 generator。测试同时检查 build 和 initialize 前后全局 CPU RNG 不变，并把 `torch.manual_seed` 替换为禁止调用函数验证。

`state_digest` 直接从既有 `train_source_normal_sampling` 导入，没有复制/修改它。研究入口需像现有脚本一样将 `packages/uav_fault`、`scripts/research` 加入 sys.path；转换脚本与新测试已完成此引导。新构造/初始化/转换/加载逻辑均不调用 KG 类；历史 digest 工具的传递导入不参与模型构造。

seed 42/43/44 实测：去 SE 的全部剩余状态逐位等于 M0；单尺度仅 branch_s/branch_l Conv 的四个 weight/bias key 专属，其余包括 BN/SE/后续块/head 全部逐位相同；CNN 的 head 两 key 逐位相同。新政策 seed42 的 digest 明确不同于旧锚点。

## 3. CNN1D 参数匹配

冻结本次静态搜索族：三层卷积 `26→c1→c2→128`，k7/k5/k3，每层 Conv+BN+ReLU，前两层后 MaxPool(2)，最后 GAP、Dropout、Linear(128,2)。穷举 **1≤c1≤c2≤128，共 8256 个整数宽度组合**；不添加虚假参数，不用数据/训练选宽度。

参数公式：`5*c1*c2 + 185*c1 + 387*c2 + 642`。排名依次为参数绝对差、较低参数量、宽度字典序。以下为前十项，完整候选表见 artifacts 的 `cnn1d_candidates.json`。

|排名|c1/c2/c3|参数量|距 105938 的绝对差|MAC/窗|
|---:|---|---:|---:|---:|
|1（选择）|74/121/128|105929|9|7097374|
|2|80/115/128|105947|9|7268176|
|3|75/120/128|105957|19|7127716|
|4|81/114/128|105915|23|7294018|
|5|79/116/128|105969|31|7241584|
|6|96/101/128|105969|31|7692064|
|7|76/119/128|105975|37|7157308|
|8|78/117/128|105981|43|7214242|
|9|77/118/128|105983|45|7186150|
|10|73/122/128|105891|47|7066282|

最终 **74/121/128**，比 M0 少 9 参数（约 **−0.00850%**），满足 ±2%；输出 128，head 与 M0 同形同义。这里只主张在上述明确搜索族内参数最接近，不主张遍历了任意 CNN 拓扑。参数匹配不代表 MAC 匹配，实际 MAC 已单列。

## 4. 历史纯锚点转换 receipt

执行：

```text
python -I -B scripts/research/build_pure_m0_anchor.py
```

|字段|结果|
|---|---|
|原文件|`artifacts/causal_source_longbudget_v1_20261101/results/runs/uniform_replacement/initial.pt`|
|原 SHA256|`b666b8a28129319f0b58e1c62ab7fa0e8698d61ec6aa32243824cda243f3e956`|
|原 state digest|`7772ccc1a20ce8b6d97c7ce83fc4927a8ff9a5dc7bef17038ca767806e561c55`|
|映射|`raw_encoder.*→encoder.*`，`head.*→head.*`，41 张量|
|新文件|`artifacts/mstca_ablation_factory_v1_20261106/pure_anchor/pure_m0_seed42.pt`|
|新文件 SHA256|`3e7f20226b5ede3b182055607c7deb9a450f6792050ef7d3ca24273f3b26ec1f`|
|纯 state digest|`a68551099cba00575f1eec6edc31ccf47b39ca8e0fa204ae2fe065d3400be646`|
|用途|历史 seed42 兼容核对，**不参与 module_seed_v1 消融**|

原协议 lock、原 SHA、原 digest、公共张量 shape/dtype、41 张量和目标 digest 全部在创建输出前核对。输出使用新目录和独占创建，拒绝覆盖；`load_pure_m0_anchor` 检查固定 digest 后严格加载 M0。完整 41 项映射及 receipt：`artifacts/mstca_ablation_factory_v1_20261106/pure_anchor/receipt.json`。本次文件存在，兼容加载测试已运行，**未 skip**。

## 5. pytest 与边界核验

架构验收修订后的最终输出：`artifacts/mstca_ablation_factory_v1_20261106/pytest_acceptance_r1.txt`，JUnit：同目录 `acceptance_r1/junit.xml`。

```text
44 passed, 1 xfailed in 5.83s
```

**通过 44，失败 0，错误 0，xfail 1，XPASS 0，skip 0。**

保留原有全部覆盖：四臂 B=1/B=32 eval 有限前向、hooks 独立 MAC、参数量、重复/异 seed、全局 RNG、42/43/44 公共 key 与 receipt、局部 seed/默认分布/BN 常数、CNN 候选排名、纯锚点加载及错误拒绝、非法 seed/policy/布局拒绝。

本轮新增 17 项通过用例：

1. **包独立导入（1 项）：**在新 Python 子进程中使用 `-I -B`，只将 `packages/uav_fault` 加入 sys.path，即可导入 factory 并构造、初始化 M0；同时断言 `train_source_normal_sampling`、`smoke_normal_source_sampling` 未进入 sys.modules。
2. **digest 字节契约（4 项）：**factory 本地 `_state_digest` 与历史 `train_source_normal_sampling.state_digest` 对空 state、含非零 running_mean/running_var 和 int64 counter=17 的 BN state、含 float64 非连续/需梯度 tensor 与 int64 counter 的混合 state、完整 M0 state 逐一比较；另反转 key 插入顺序验证一致。metadata 编码、detach/CPU/contiguous 和 tensor 字节顺序与原实现相同。
3. **完整 golden（12 项）：**4 臂×seed42/43/44 的完整 64 位 digest 均写死在测试中。注释明确基准为 **PyTorch 2.8.0+cpu**，用于在正式 **2.5.1+cu124** 环境检测漂移；不会按版本 skip 或自动重生成。实算结果与本报告第 1 节上一版的全部 12 值及既有 `four_arms.json` **完全一致**。本轮没有运行正式 CUDA 环境，也不声称跨版本必然相同。

**修订说明：本段替代第 2 节上一版关于导入历史 digest 工具的描述。** factory 现已移除所有研究脚本导入，使用模块内 `_state_digest`；历史 helper 仅在测试中作为对照。转换脚本未改，初始化政策、语义名称、分布、模型结构和第 1 节 digest 表均未变。

P3 仍通过真实 `predict_records_batch`/`predict` 捕获变化高度合成 records 的输入，按预期在后续窗口失败；保留 **xfail(strict=True, raises=AssertionError)**，`fault_detector.py` 未改。未来修复后严格 XPASS 会使测试失败，提醒移除 xfail。

重跑示例（先建全新输出父目录，拒绝覆盖）：

```text
python -I -B -c "from pathlib import Path; Path('artifacts/mstca_ablation_factory_v1_20261106/pytest_next').mkdir(exist_ok=False)"
python -I -B -m pytest -q -p no:cacheprovider --basetemp=artifacts/mstca_ablation_factory_v1_20261106/pytest_next/tmp tests/test_mstca_ablation_factory.py
```

实际验收运行禁用了 pytest 自动插件加载、隐藏 CUDA，并用 audit hook 拒绝 `D:/UAV-Data` 下任何文件读取。旧 `pytest_run1.txt`（临时父目录未预建导致 5 个 setup errors）和 `pytest_run2.txt`（27 passed / 1 xfailed）原样保留，未覆盖任何历史输出。

环境仍为 Python 3.12.4、PyTorch 2.8.0+cpu。边界与 12 个 digest 的对照结果见 `acceptance_r1/verification.json`；上一轮 21 个受保护文件身份记录保持不变。

## 6. 偏离清单

**相对本次架构验收要求，无偏离。**

- 按本轮明确授权，仅修改 factory、本任务测试文件及本报告第 5、6 节；没有改动其他已有源码、测试、配置、冻结协议、数据或权重，也没有修改转换脚本。
- 本地 `_state_digest` 替代研究脚本依赖是本轮明确要求，取代上一份任务书的直接导入约定；字节契约已与历史 helper 对照验证。
- 12 个完整 golden 均通过且与上一版报告完全一致；所有 seed 仍统一 module_seed_v1，历史纯锚点只做兼容核对。
- P3 未修，严格 xfail 保留；未决定选点、评估人口或阈值。
- 未训练、未执行优化器步、未读取任何数据集、未 commit/push；新增运行输出均位于指定 artifacts 的新路径，未覆盖旧输出。
