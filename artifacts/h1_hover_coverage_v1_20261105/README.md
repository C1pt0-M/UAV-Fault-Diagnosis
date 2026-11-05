# H1 悬停后段正常覆盖实验产物

报告：[结果](../../docs/research/2026-11-05/H1_hover_normal_coverage_results.md)，[协议](../../docs/research/2026-11-05/H1_hover_normal_coverage_protocol_draft.md)。配置：`configs/experiments/h1_hover_normal_coverage_v1.json`。

单 seed 42、Balanced、M0 相同初始化与预算，唯一改动是 train 中 56 个独立正常 hover 案例追加 19573 个步长 5 补窗（正常总采样质量固定为 M0 的 83051）。服务器 autodl_UAV3，torch 2.5.1+cu124。结论：按冻结判据失败（1/7）。

## 入库文件

- `run/report.json`、`run/history.json`、`run/h1_config.json`、`run/trainer.py`：训练回执、逐 epoch 验证、配置与脚本副本
- `run.log`：训练输出
- `evaluate_h1.py`、`evaluation.json`：冻结判据计算
- `diag_predict.py`：epoch 49 last.pt 诊断预测脚本（不参与判定）
- `refill_manifest.json`：补窗数据身份
- `../hover_coverage_preflight_v1_20261105/`：第 0 步只读统计脚本

## 未入库（本地与服务器保留）

|文件|SHA256|
|---|---|
|run/best.pt（epoch 16）|9a9cfb417104d484f54b6bea6de763e0b87d33440b6819d7599de97cf41a0ac0|
|run/source_validation_predictions.npz|b934bfa2f1718522785025c2df09ceafc47d3e17ab5dfa642ac0b3e52b3fd979|
|diag_last_epoch49.npz|de80c67ce24e3cf71a17040347e8cfcc32c603b601fd065248d8725715a161bb|
|D:/UAV-Data/h1_hover_refill_v1/x.npy|0d39b9c676da1dbf662430f0a9e9506416353889339584ea37cd7b923c9c3999|
|D:/UAV-Data/h1_hover_refill_v1/windows.csv|3321f11e7f542e8a589b016922cc5c89534e228ef23ee55aeebb2c288ba45523|

服务器路径：`/root/autodl-tmp/h1_hover_coverage_v1_20261105/`。
