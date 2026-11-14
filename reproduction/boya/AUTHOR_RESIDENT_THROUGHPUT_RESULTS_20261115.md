# 常驻双路采集实测与采满时间（2026-11-15）

常驻执行、共享 CGR 和两场景并行已经接通，固定 fit 及 internal 小对照全部完成、9 次首次执行数值审计通过。**4000 fit＋800 internal 正式扩采没有启动，正式训练仍为 0。当前成本不支持承诺数小时采满。** 新入口只作诊断，不能直接把这些已知场景的重复结果加入训练。

两场 fit 的完整流程实测 **1765.611 s＝29.427 分钟**，包括新场景构建、新 RGB-D、官方 CGR、四型原首选准备、resident 执行、首次完整数值审计和进程退出。旧相同 scene1＋4 的历史 stage 跨度分别计时后合计 2688.671 s＝44.811 分钟；观察比值约 1.523。旧值不含其外层边界工作，且没有本轮重新跑一个冷启动串行组，因此它是历史比较，不是严格新配对因果倍率，更不能迁移为随机场景的固定提速。

按此前 11 个 fit 场景的有效记录产出率，以及本轮实测阶段成本和两组调度系数，四型各 1000 条 fit 的条件情景约 **15.02 天连续运行**；同工作量的常驻执行完全不重叠情景约 **20.20 天**。internal 只有 5 个历史场景，g1/g6/g8 均为 0 条、g7 为 1 条，当前没有可靠的整体采满 ETA。不能把 fit 的产率借给 internal，也不能把 null 算负例。全部 4800 条的完成时间仍未知；不建议据此直接启动全量。

## 固定预算与实际范围

主协议：`/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/resident_collection_control_20261115_v1/MASTER_PROTOCOL.json`。运行前固定 3 场景、12 手型槽、最多 12 次执行、0 重试；历史预期 9 ready 不作为修改预算的依据。fit 取原 serial queue 的 index1、4，internal 取原 fresh-wave child 的 local index0。每次都从冻结 scene spec/seed 重新 build/capture/CGR/prepare，不复制旧候选或动作文件。

| 对照 | 槽数 | 审计后诊断结果 | 完整 outer wall |
|---|---:|---|---:|
| fit scene1＋4 | 8 | 6 正、2 负；8 次首次数值审计通过 | 1765.611 s |
| internal local0 | 4 | 1 正、2 null_geometry、1 no_candidate；1 次首次数值审计通过 | 578.404 s |

internal 全流程为 9.640 分钟，仅 g7 ready，一次冷启动/一次建场/零热 reset，不能作为内部热复用或并行收益证据。两路合计 12 终态槽、9 次实际执行、7 正 2 负诊断、3 个无计划；共 **28898 执行步＝25656＋3242**，另有 2250 初始化 tick/6750 初始化对象步，未知执行成本 0。根归档 helper 首次把总步数误加为 28998，原两份 PROGRESS 和各条数值回执不受影响；[算术更正](evidence/author_resident_throughput_20261115_v1/ERRATUM_TOTAL_STEPS.json)保留原值并明确修正，没有重跑物理。

fit 协议 SHA `8da19e12bc23bc8ceb7ca70416de00a6ca2ae1eac5eff9ae635fbf95bdc95e96`；internal 协议 SHA `8ad9fe561f81b237d54433966e22947933459c8facc86808b7c891b779b6e99b`。实际入口源码、测试和初始实现证据已在用户 origin 的 `fadc12a` 提交冻结，后续不修改这些运行源码。

## 实际复用了什么

参考已检出的 kywch fork 常驻环境/模型循环，沿已有 `ResidentArmScene` 和官方 IsaacLab context 生命周期接薄入口；没有另建物理或 IPC 框架，也没有继续局部数学优化。fit 共享一个官方 CGR 模型处理两帧，0 手型评分头；每组三体仅启动一次 Kit、构建一次执行场景，按原顺序执行四型，其间恢复全部三体及机器人初始状态，共两次 Kit、两次建场、六次边界重置，重置不增加积分步。

两组实际并发由 host PID/start ticks、PhysX/libcarb 映射、同区间双 journal 增长、全局 lease 和独立 kit 槽的 FD 锁独立确认。GPU0 的 NVML 编号对应本机 `/dev/nvidia1`，证据按 UUID/PCI/minor 绑定；没有占用另一 GPU 的任务。

这仍是 **CPU 物理＋GPU0 场景/RTX/CGR 服务**。capture 每场景仍单独启动，prepare 沿原预算，首次独立 numeric audit 仍串行。没有接通官方多环境 GPU 批量积分，不能把两个 Kit 称为 GPU 向量化采集。改变三体几何的同一 Kit 重建分支仅经过离线测试，本轮只验证了同组三体不同手型之间的实际复用。

## 成本和一致性

fit 的 build 合计 14.099 s，capture 121.323 s，官方 CGR 6.375 s，prepare 各任务之和 518.667 s，执行组各任务之和 1625.969 s、实际并行段 868.020 s，首次独立数值审计 475.586 s。阶段任务和存在重叠，不能直接求和当总耗时；主数值使用 Python 最外层 wrapper 的完整 wall，包含 imports、协议预检、循环、执行子进程退出和 lease 释放，shell/PTY 建立不在其内。实现、开发、安装和事后只读报告时间不在采集 wall 内。

两场 fit 的初始化 RGB/depth、pose/velocity、34 个 CGR 输出数组逐值一致。8 个条件的 36 项候选语义字段、原 arm 计划及实际 lift 指令数值、phase/time/joint-name 均与旧对应记录相同；来源路径参与 SHA，因此新旧 provenance 字符串和整个文件 SHA 可自然不同。独立审核使用真实 source OBJ 顶点/三角面、原生 hull/plane 和可见源射线，未把原 PLY 的 SHA 核验说成重新计算其几何。

实际读回的机器人及全三体初态与旧记录逐值一致，两个冷启动 g1 的全部保存状态与旧轨迹一致；六次热复用的机器人 settle/HOME/approach 关节状态一致，关节差异从 close 开始。物体在 preclose 已有至多 43.738 µm 的微差；接触后的最大轨迹差出现在原本失败的 scene4 g8，为 149.905 mm 平移、179.916° 旋转。8/8 原门、步数和正负标签保持，但**不能据此声称热启动物理轨迹等价或统计无偏**，差异原因未证明。这一限制保留在独立动作审计中，没有修改容差、物理或标签来消除差异。

各执行仍接受原全部控制门、全三体源网格底面/承载与接触重算、24→28 指令重建和 source/input SHA 审计。原生 20 µm 门只证明导入保持，不证明凸代理等于精确 CAD。外部独立 capture 审核与采集重叠，fit 两帧一次性完整解压共 42.315 s、internal 第三帧 23.230 s，合计 65.545 s 单列为审核成本；CPU/IO 竞争自然反映在实测 wall，未为了提速移除。internal 的 17 个 CGR 数组、存在候选的三型语义及唯一 ready 指令也与旧对应记录一致，g8 两边均无候选；唯一冷执行的机器人和全三体保存轨迹一致。

## 估算的边界和接续

原随机开发数据累计仍为 **15 条有效新标签＝14 fit＋1 internal**，不混旧 435；物体池仍为 S40 派生的 40 件、32 fit/8 internal，未冒称作者完整 144 件或本轮已扩大物体池。本轮所有父账本 `label=null`、`formal_training_eligible=false`、`exclude_training=true`，原 numeric certificate 的内层准入不能覆盖父协议排除。重复已知场景不能提高随机产出率样本量，诊断正负比例也不是新模型成功率。

11 个 fit 场景的四型有效数为 4/3/3/4，最慢型经验率 3/11，对各 1000 条需要约 3667 场景的期望工作量。15.02 天情景包含 placement/null、原准备、执行、每次首次审计、两路调度与未归因开销，并计入其它型超过配额时继续执行的成本；它不保证各型届时填满。单纯把两场全 ready 的高负载均摊到所有随机场景会得到约 37.46 天，单列为控制负载迁移情景，不作为主估算。新资产转换、后续训练、故障/机器停机和实际样本率变化没有可靠量化，不纳入承诺。

正式扩采还需独立冻结有限总预算、顺序和生产账本准入；这版入口硬性排除训练，不是可直接启动正式训练数据的大采集器。现阶段不新增扩样、不调候选/阈值、不训练或进行新的外部成功率评测。若要求显著缩短为小时级，仍需实际接通并验证官方 GPU 多环境物理/记录路径及内部物体有效产出，不能靠理论并发数给出承诺。

独立证据：[动作与热复用审计](AUTHOR_RESIDENT_EXECUTION_RESULTS_20261115.md)、[输入与初始化审计](AUTHOR_RESIDENT_COLLECTION_INPUT_AUDIT_20261115.md)、[只读吞吐和 ETA](AUTHOR_RESIDENT_ETA_RESULTS_20261115.md)。根回执和启动前零步失败分别归档在 `evidence/author_resident_throughput_20261115_v1/`；实现的 31 项 CPU 测试通过。元数据 freeze 类型错误及缺少 `/usr/bin/time` 的 shell 退出均在正式 dispatch 前、0 Kit/0 积分，保留原记录；不计作物理重试。最终 13 份旧数据/八权重 SHA 不变，host 只读确认 34 个相关 stage/parent PID 已退出、13 槽 released、14 个排他锁可重新取得，无后台物理/训练。只提交相关轻量源码/文档/证据，原始输出留 public/outputs 忽略目录，CanonDex 只读。
