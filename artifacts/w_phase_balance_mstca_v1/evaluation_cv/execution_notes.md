# W r1.1 完整执行偏差、失败、重试、实现选择和歧义清单

## 结论与范围
- 协议澄清提交4d3b5538597469cba7b707eb7850c49f8ee91e25；仅source train CV。30/30 run COMPLETE、每run100轮、共3000轮。W1/W2均FAIL且仅失败C1，adoption=KEEP_T0。这是有效科学结果，不是执行失败；未据此重跑或改判。
- 没有SHA锁、implementation manifest或本地smoke。新训练器/评估器/测试首次落盘并执行于服务器；本地仅保存传输/安装脚本、只读监控记录和回传结果。既有六项train数据身份认证仍复用T helper，属于输入信任校验，不是新增W锁；三个val文件连字节哈希也不读。

## 每次失败/中断/重试
1. 初始bundle同步的保护检查中断一次：发现服务器已有未跟踪 artifacts/v_confirm_fixed100_mstca_v1/architect_verify.py，sync.log保留AssertionError；此前只fetch，没有merge或覆盖该文件。Git hash-object与incoming1856ba2的blob相同。随后仅git add登记相同内容到索引，再ff-only；文件字节未改，全部冻结目录大小/mtime不变。用户随后明确认可，并要求列入偏差。此为同步操作恢复，不是实验重跑。
2. 元数据核查发现fault_onset_row=-1而非空；notna()参考会纳入无有效起始案例。未实施/训练旧口径，先报告并停等澄清。用户发布r1.1，拉取4d3b553，采用>=0排除-1后才实现和训练。旧notna()预算保留在操作记录中，不作为W结果。fold0新乘子全部匹配用户更新参考。
3. Git身份探测git var GIT_AUTHOR_IDENT失败一次（服务器未配置身份，fatal unable to auto-detect email）；没有尝试失败commit。提交时仅用进程级git -c复用本地仓库既有user.name/user.email，不改服务器全局/仓库配置，不虚构默认身份。
4. 第三个本地只读监控窗口被用户中断（工具Command aborted）；用户指出不必持续盯住。停止本地轮询后后台nohup照常运行，未给训练进程发中断/终止信号；之后只在用户询问时读取状态。前两个窗口各10次、第三窗口4次SSH观测均成功，传输/观测网络失败0。无训练重启、目录改名、移动、覆盖、seed替换或补跑。
5. 单元测试首轮154 passed、0 failed、0 skipped、1原有P3 xfailed，14golden均通过；没有测试失败重试、golden更新或CPU dispatch变更。W0复现门首轮PASS（42/43/44=3595/3588/4610）。正式训练失败0、训练重跑0、评估失败0；评估只运行一次。未触发基础写报告函数的异常/临时文件恢复。

## 预先确定的实现/报告选择
6. 只用每run FIT中的唯一case_uid且onset>=0；先将onset转秒再取中位数。每case有多个不同有效onset时报错，不任取一行；缺失任一flight的有效FIT参考时报错，不从held-out补参考。N_c=0时r=1；故障权重保持原b；不重新归一化。强制乘子1的逐元素相等测试、toy ledger、W1/W2公式和判定/adoption分支均覆盖。
7. 保留T的100轮trajectory、float32每轮inner/held-out概率、未加权inner CE和T0 replay；初始化、优化器、调度、focal loss、裁剪、sampler抽样数/seed、loader及推理独立generator均复用/保持。阶段/时间仅用于采样权重，不加入任何模型输入，输入仍是原26个缓存通道。
8. 选择串行seed→fold→W1/W2，而非可选的双并行。磁盘开始约31GB可用，保留全部100个checkpoint（指令允许磁盘够时保留全部），并保留全部预测npz；大文件只留服务器，绝不提交。全过程30run按顺序一次完成。
9. 为运行既有factory/T basetemp测试，仅在pytest进程内把train_s_ablation.SMOKE_ROOT与build_pure_m0_anchor.OUTPUT_ROOT指向新的W临时scope，不改冻结文件或生产路径常量。Python -I -B、禁用pytest插件自动加载/cache、防止pyc写入；均未改变训练计算。运行始终显式设置并assert CUBLAS_WORKSPACE_CONFIG=:4096:8、ATEN_CPU_CAPABILITY=default；原runtime torch2.5.1+cu124/numpy2.1.3，无新依赖。
10. W0只读取原T CV T0轮次，没有新W0 run。W评估C2/C3使用全部故障窗口，明确包含late-hover；V式排除late-hover口径仅描述。所有162069窗口按原outer folds配对拼接，原original_cache_index/labels/epoch顺序/dtype和100轮inner CE replay均校验。
11. 复用V的criteria与matched_fp_threshold。匹配阈值kth-normal且>=，记录实际FP；本次两臂所有seed实际FP均等于k（3595/3588/4610/3740/4243），没有并列超额。C2/C3按整数配对TP差汇总为pp，不加边界epsilon；类约束是seed均值而非逐seed约束。C2/C3 CI描述性，不加未登记的CI veto。
12. 两臂/5seed/15轨迹共用2000次association_group重采样，rng20261106，1296组；每draw重算各自t*和匹配点。原尺度比值/平均AP差、全部重复保留，无伪计数、丢弃或重抽；所有需要的CI均定义。两项主比较不作多重性修正。W1/W2比较及epoch100仅描述，不因epoch100较低FP改变adoption。
13. 协议未定义“两臂PASS且几何比值完全相等”的tie-break：在看到任何W结果前于readiness中声明遇到精确tie就停下询问，不能擅选W1/W2；本次两臂FAIL，未触发。没有实际结果受到该未触发分支影响。
14. 时间段用右闭区间[0,10],(10,20],...,(90,inf]，首格包含0，和诊断pd.cut口径一致。无窗口格子率记null，不记0。独立正常的phase标签按该窗口的outer-fold FIT参考生成，所有arm/seed用同样标签，不能用全train或held-out算参考。
15. 描述性prefault沿用原normal_origin=normal_part_of_fault_case，含无有效onset故障案例中的正常窗口；>=0排除仅用于onset参考，不偷偷改变该stratum。独立正常飞行报警数给每seed和seed均值；诊断文档276是跨seed联合报警数，不与seed均值直接混比。log-FP SD分母未登记，ddof0/1同时标明，均不决策。
16. T式float32概率出现0/1端点，15个arm×seed的概率域class-balanced CE均为null（未裁剪/伪计数）；这是沿用T数值路径的描述性项，不是选点用的稳定logit域inner CE，也不影响C1–C4的FP/AP/类召回。完整指标原样保留。

## 时间、保存与提交
17. START2026-11-07T15:55:17Z；30run结束22:30:42Z；评估+检查COMPLETE22:34:38Z。含评估6h39m21s，较7.5h预算短；没有提前停止run或省略epoch。后台完成后服务器保持开机、GPU空闲。
18. 完成检查确认原11938个S/T/V文件大小及mtime不变，冻结tracked文件没有修改。数据访问钩子限制源目录只读六项train文件，禁止val/internal/Real及源文件写入。未做任何新的val确认。
19. 任务运行期间本地main前进到ad7f9a1（E组提交），服务器另有未跟踪E组配置/脚本。它们不是本任务内容，不stage、不修改、不移动。W提交仅含3个新源码/测试、60份report/selections、门控证据和评估输出；checkpoint、预测npz、pytest临时模型不提交。W提交将以cherry-pick整合到已有main，保留用户其他改动；最终hash单独回报，不循环写入commit自身。

除以上已列事项，没有其他失败、重跑、传输问题、环境变更、条件放宽或未报告偏差。若后续传输/整合出现异常，须另留日志并在最终回报追加，不能用此段覆盖后续事实。
