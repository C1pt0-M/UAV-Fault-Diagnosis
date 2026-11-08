# PI 指令补充 1：内部 test 最终评价重新启动

> 依据：`内部test最终评价_PI指令_v1.md` 和冻结协议 `configs/experiments/internal_test_final_v1.json`，两者都不变。

## 架构方核对结论

第 1 次启动（commit 4516868）的失败，核对如下：

- 失败发生在导入阶段：`-I` 隔离模式屏蔽了用户 site-packages，导致找不到 threadpoolctl。
- 失败时还没有加载模型，也没有读取 test 案例数据：逐帧分数目录是空的，新锁没有写，`results.json` 里所有指标都是 null。
- 因此 test 结果**没有被任何人看到**。这次重新启动属于用户已批准的那**一次**评价，不算第二次评价，也不算重跑指标。偏差清单里要写明这一点。

## 重新启动步骤

1. **保留失败现场。**
   - 用 `git mv` 把 `artifacts/internal_test_final_v1/` 改名为 `artifacts/internal_test_final_v1_attempt1_env_failure/`；
   - 用 `git mv` 把 `docs/research/2026-11-08/内部test最终评价结果_v1.md` 改名为 `内部test最终评价_第1次启动失败记录_v1.md`；
   - `artifacts/internal_test_final_v1_checks/` 不动，本次新的检查记录放到 `artifacts/internal_test_final_v1_checks_r2/`；
   - 这些改名和下面的修复一起单独提交，并 push。
2. **修复解释器参数。** 把 `-I` 换成 `-X utf8 -E -P -B`。这组参数仍然忽略 PYTHON* 环境变量，也不会把脚本目录加入搜索路径，但会保留用户 site-packages。不安装依赖，不切换环境。
3. **补一项不接触 test 的完整链路预检。** 在正式启动前完成，任何一步不符就停止：
   - 用正式参数，在子进程里导入评分和聚合链路上的全部模块，至少包括 numpy、pandas、torch、sklearn.ensemble、threadpoolctl、uav_fault 相关模块、replay_ens_events_val、uav_fault.online.detection_policy；
   - 加载两个冻结模型，加载前后都核对 sha256；
   - 用同一条评分代码路径，给 **1 个 val 案例**打分（任选一个 SIL、一个 HIL 更好）；
   - 与 `D:/UAV-Data/ens_event_threshold_v1_scores/` 中同名 npz 的 m0、gbt 比对：GBT 最大差 ≤ 1e-6，M0 ≤ 0.002；
   - 把预检结果写进 `artifacts/internal_test_final_v1_checks_r2/`。
4. **正式启动，只启动一次。**
   - `D:/UAV-Data/internal_test_final_v1_scores/` 必须为空，否则停止；
   - 之后完全按原指令第 3 步执行：启动检查 → 逐帧打分 → 一致性门 → 写锁 → 计算 F1–F4 和描述性指标；
   - 如果再次在读取 test 数据之前失败，照这次的办法保留现场并报告，不要自行再次启动；
   - 如果已经读到 test 数据之后失败，直接停止并报告。

## 回复格式

与原指令第 5 节相同。另外：

- 补充预检结果：导入列表、val 案例 ID、两项最大差；
- 偏差清单中写明第 1 次启动失败、现场如何保留、本次启动的依据。
