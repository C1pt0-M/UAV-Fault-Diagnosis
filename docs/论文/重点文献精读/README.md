# 重点文献：MinerU 全文与分批精读

资料归档：2026-10-30；阅读状态更新：2026-10-31。范围为上一轮推荐的 **6 篇方法论文＋RflyMAD＋2 篇健康目标域论文，共 9 篇**；不是 Zotero 全部 75 条记录。

**当前完成：7 篇 PDF、87 页全部经过 MinerU 提取并完成首轮全文精读：第 1 批 3 篇／48 页，第 2 批 4 篇／39 页。另 2 篇尚未取得可用 PDF。全文精读不等于作者代码复现或实验认证。**

## 先打开精读笔记

- **[第 1 批：DDCNN、Meta-GAD、MSETL-AD 精读](第1批_DDCNN_MetaGAD_MSETLAD精读.md)**
- **[第 2 批：AeroTSBoost、RflyMAD、两篇 PHM 健康目标域论文精读](第2批_AeroTSBoost_RflyMAD_PHM精读.md)**；4 篇／39 页已读，含关键原图、公式与10组混淆矩阵算术核对。
- 第 3 批：Embedded-EKF、TC-SGDN；需先取得全文，不能用摘要或代码仓库说明代替。

第 1 批的重要修正：后两篇是参数回归＋残差检测，主要评价真实记录上的人工偏置／漂移，不是源故障监督多类分类；两篇指标正类解释也有需核实之处，不直接把 TPR 当作故障召回。

第 2 批的重要结论：两篇PHM明确采用健康子集MMD，但不保证目标故障迁移；AeroTSBoost的事件F1使用测试best-F1阈值，不能当部署验收；RflyMAD的AdaBN演示混入目标训练数据，未限定健康。当前项目优先标签时间修复与大创结项，不因读到新结构自动启动实验或换数据集。

## 全文与状态

| 论文／简称 | 原 PDF | 页数 | MinerU 全文 | 图片文件数 | 精读状态 |
|---|---|---:|---|---:|---|
| DDCNN | [PDF](PDF/DDCNN.pdf) | 7 | [full.md](MinerU/DDCNN/full.md) | 23 | 第 1 批已读 |
| Meta-GAD | [PDF](PDF/Meta-GAD.pdf) | 20 | [full.md](MinerU/Meta-GAD/full.md) | 73 | 第 1 批已读 |
| MSETL-AD | [PDF](PDF/MSETL-AD.pdf) | 21 | [full.md](MinerU/MSETL-AD/full.md) | 118 | 第 1 批已读 |
| AeroTSBoost | [PDF](PDF/AeroTSBoost.pdf) | 17 | [full.md](MinerU/AeroTSBoost/full.md) | 17 | 第 2 批已读 |
| RflyMAD | [PDF](PDF/RflyMAD.pdf) | 6 | [full.md](MinerU/RflyMAD/full.md) | 13 | 第 2 批已读 |
| PHM-Healthy-MMD | [PDF](PDF/PHM-Healthy-MMD.pdf) | 9 | [full.md](MinerU/PHM-Healthy-MMD/full.md) | 19 | 第 2 批已读 |
| PHM-CDANet | [PDF](PDF/PHM-CDANet.pdf) | 7 | [full.md](MinerU/PHM-CDANet/full.md) | 10 | 第 2 批已读 |
| Embedded-EKF | 本轮未取得 | — | — | — | 无 Zotero PDF，公开请求未返回 PDF |
| TC-SGDN | 本轮未取得 | — | — | — | 无 Zotero PDF，出版方请求被阻止；仓库不代替全文 |

页数是文件页面数，不是期刊卷期页码；图片数包含版面裁切，不等于论文图数。

## 来源、版本与 Zotero 定位

| 简称 | 题名／身份 | 来源与版本 |
|---|---|---|
| DDCNN | _DDCNN: A Promising Tool for Simulation-To-Reality UAV Fault Diagnosis_ | Zotero `HIJILBCV`，附件 `29BDSHVJ`；arXiv 2302.08117 v2；现有引用 key `zhang_ddcnn_2024` |
| Meta-GAD | _Meta-learning-aided generalized anomaly detection for unmanned aerial vehicles from simulation to unseen reality_ | Zotero `GY47MIW7`，附件 `LSKYHR5B`；[10.1016/j.aei.2025.103866](https://doi.org/10.1016/j.aei.2025.103866)；现有引用 key `noauthor_meta-learning-aided_2026`，作者字段待整理 |
| MSETL-AD | _Multi-source ensemble transfer learning-based unmanned aerial vehicle flight data anomaly detection with limited data: From simulation to reality_ | Zotero `IPAFPTB4`，附件 `6EB9WF65`；[10.1016/j.aei.2025.103255](https://doi.org/10.1016/j.aei.2025.103255)；现有引用 key `noauthor_multi-source_2025`，作者字段待整理 |
| AeroTSBoost | _AeroTSBoost: Temporal-Statistical Boosting for Real-World UAV Telemetry Anomaly Mining_ | Zotero `V4B4NPIC`，附件 `I9KK3NDC`；arXiv 2605.25639 **v2**，与上轮网上查看 v1 不同，不默认所有数字一致；`wei_aerotsboost_2026` |
| RflyMAD | _RflyMAD: A Dataset for Multicopter Fault Detection and Health Assessment_ | Zotero `R5U7ZGB5`，附件 `LNYPG8Z2`；arXiv 2311.11340 **v2**，不是 2025 期刊版 PDF；`le_rflymad_2024` |
| PHM-Healthy-MMD | _Partial Domain Adaptation for Intelligent Machinery Fault Diagnosis: Leveraging Healthy-Only Target Data for Multi-Class Classification_ | [官方 PDF](https://papers.phmsociety.org/index.php/phmconf/article/download/4341/phmc_25_4341)；[10.36001/phmconf.2025.v17i1.4341](https://doi.org/10.36001/phmconf.2025.v17i1.4341)；本轮未导入 Zotero |
| PHM-CDANet | _Multi-Class Gearbox Fault Diagnosis via Pre-Trained Model-based Domain Adaptation with Healthy-Only Target Data_ | [官方 PDF](https://www.papers.phmsociety.org/index.php/phmap/article/download/4519/phmap_25_4519)；网页 citation 元数据为 `10.36001/phmap.2025.v5i1.4519`，会议信息与网页上线日期应分别核对；本轮未导入 Zotero |
| Embedded-EKF | _Embedded real-time UAV sensor fault diagnosis with a multirate EKF based lightweight network_ | Zotero `A4QJDVIN`，无附件；[10.1088/1361-6501/ae45c0](https://doi.org/10.1088/1361-6501/ae45c0)；`li_embedded_2026` |
| TC-SGDN | _Digital Twin-Guided Spatiotemporal Graph Representation Learning for Reliable UAV Fault Diagnosis under Complex Flight Conditions_ | Zotero `TXXD5IZ2`，无附件；[10.1109/JIOT.2026.3709310](https://doi.org/10.1109/JIOT.2026.3709310)；`tu_digital_2026` |

作者／期刊等完整书目背景仍见 [上一轮筛选记录](../无人机相关论文筛选_2026-10-30.md) 和本批精读。上表中的 Zotero 条目 key、附件 key 与 BibTeX 引用 key 不是同一种标识。

## 提取参数与检查范围

- 使用 MinerU API，`vlm`、`en`、开启 OCR／公式／表格，7 篇全部页面提交，无页范围截取。
- 5 份本地 Zotero PDF 只复制、不覆盖附件原件；另 2 份从 PHM 官方公开链接下载。未修改 Zotero 库、没有自动导入。
- 已核对 7 份原输入 PDF 哈希、MinerU 内容块页索引全覆盖、内容列表 JSON 可解析及 Markdown 图片引用全部存在。合计 273 个图片文件。
- 原输入 SHA-256、页数、附件 key、Markdown 哈希及当前阅读状态见 [reading_manifest.json](../../../artifacts/uav_literature_screening_20261030/reading_manifest.json)。
- [MinerU 执行日志](../../../artifacts/uav_literature_screening_20261030/mineru_logs)；[关键算术检查](../../../artifacts/uav_literature_screening_20261030/reading_checks.json)；[原公式／指标视觉复核](../../../artifacts/uav_literature_screening_20261030/visual_checks)。
- 提取成功不等于所有公式、数字和列顺序均正确。两批已对选定原公式、指标定义、结构图和关键结果图视觉复核；第二批另有[最小核查脚本](../../../artifacts/uav_literature_screening_20261030/check_batch2_reading.py)及[算术结果](../../../artifacts/uav_literature_screening_20261030/reading_checks_batch2.json)，记录健康MMD的T4图／表小幅差异及CDANet六矩阵一致性。未做全部公式／表格的独立认证，未运行作者代码。
- `MinerU/<简称>/` 保留 `full.md`、`images/`、结构化内容和服务返回的 `*_origin.pdf`。**服务返回 PDF 不作为原始输入身份凭据**，引用原件及核哈希以 `PDF/` 为准。

本轮不训练、不部署、不重新访问最终 Real 测试特征或概率，不修改冻结协议。项目状态见 [项目进程](../../../项目进程.md)。
