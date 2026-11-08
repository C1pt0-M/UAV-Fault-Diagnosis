# 补充1：保留现场与运行参数修复（正式test启动前独立提交）

依据用户提供的《内部test最终评价_PI指令_补充1.md》，原协议与原PI不变。

- 第1次启动4516868在导入阶段失败，没有加载模型、没有读取test案例payload、没有产生逐帧分数或指标，新锁未写。架构方核对确认结果未被看到。本次属于已批准的一次评价的重新启动，不是第二次评价或重跑指标。
- 已按要求git mv整个artifacts/internal_test_final_v1到internal_test_final_v1_attempt1_env_failure；结果报告git mv为内部test最终评价_第1次启动失败记录_v1.md。内容、SHA、mtime原样保留。旧checks目录不动，全部新证据在checks_r2。
- 新launcher与预检使用同一解释器的-X utf8 -E -P -B；不安装依赖，不切换环境，不改CPU dispatch。
- 原外部评分目录仍存在但为空，按补充1允许空目录；只将拒绝条件改为“已有canonical输出/锁，或评分目录非空/不是目录”。已有输出和任何评分文件仍禁止覆盖。没有改评分、模型、策略、工作点或指标代码。
- 合成测试一次26 passed、0 failed、0 skipped；访问审计0次真实数据/模型访问尝试。单测覆盖空评分目录接受、非空/文件路径拒绝、旧锁和四工作点限制等。
- 完整链路val预检代码随本修复提交，必须等独立提交并push后运行；要求17个模块导入、模型加载前/后SHA检查、SIL与HIL各1个val案例用正式score_arrays比较全部有效帧。该预检禁止打开test预测/tasks/台账/payload及Real。
- 只有预检PASS且外部评分目录仍为空，才允许launcher正式启动一次。任何失败停下，不自动修复重启。按原门控，parity通过后先写锁再计算F1–F4。
