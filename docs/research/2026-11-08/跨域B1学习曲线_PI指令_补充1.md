# PI 指令补充 1：B1 无可行 onset 时的并列规则

> 依据：`configs/experiments/real_v2_b1_learning_curve.json` 与 `跨域B1学习曲线_PI指令_v1.md`，两者都不变。本补充在 B1 任何训练和评价开始之前发出，只澄清口径，不改判据。

## 结论：采用 A

内层没有可行 onset 时，按以下顺序选：

1. min(motor 检出率, sensor 检出率) 最大；
2. 并列时，取总检出案例数（motor + sensor）多的；
3. 仍并列时，取较低的 onset。

## 理由

这与可行情况下的并列规则（先比检出多，再取低 onset）保持一致，不另引入新的偏好。

## 偏差清单

这一条写成“按补充 1 澄清”，不算偏差。该折仍标记 `no_feasible_onset`，回执中给出并列的 onset 列表。
