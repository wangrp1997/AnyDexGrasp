# 常驻完整链路输入与初始化独立审计（2026-11-15）

本报告只审已有产物，不启动 Kit、规划、模型前向或积分。冻结采集入口与历史证据均不修改。实际执行/reset 由单独审计负责；这里不会用输入相同推断物理结果相同。

目前已审新 fit scene 1、4，均从原 scene/spec/seed 重新 build、capture、CGR 和 prepare。两场新旧初始化 pose/velocity 与原生 RGB/depth 四个数组逐 dtype/shape/字节相同；两场共 34 个 CGR 输出数组也全部相同。原 persistent CGR 每帧恢复固定 RNG，没有改变这两帧的候选数值。

八个条件均已准备 ready。每个条件的 36 个候选语义字段与历史参考完全相同，包括原行、排名、flip、随机深入、手型/slot、2261 特征、canonical480、实际掌位姿、动作合同与碰撞配置。八套原 arm 命令及最终 lift 的实际数值、phase/time/joint-name 数组也完全相同。NPZ 中仅 `collection_candidate_sha256`、`collection_execution_spec_sha256` 来源字符串不同：新观测/表征文件路径参与绑定，应该生成新的 SHA，不能复制旧 SHA 冒充新采集。`.mat` 的时间头不作为数值输入差异，K 与深度比例另行核验。

两场 compact capture 分别只完整解压一次 initialization journal。scene 1 的 1,892,396 条接触/750 帧通过；全三体最后 250 帧均逐帧有桌面支持。它包含 889,808 个实际 source OBJ 顶点、1,779,636 个三角面及 1,111 个 native hull，96 条独立源三角面射线重算与保存值最大误差 `4.44e-16 m`，native-plane 重算与回执误差 0。审计壁钟为 22.706 秒。scene 4 的 1,595,112 条接触/750 帧、807,761 个顶点、1,616,114 个三角面、1,135 个 native hull 同样通过，96 条射线误差最大 `8.88e-16 m`，耗时 19.609 秒。

两场共同核验：751×3 的全部初态/末态 pose、velocity、finite/unit quaternion；原稳定门及 native `.1 kg/.6` 材料回读；冻结源码中逐体 `-.98 N` 的施加语义；全部接触数值/身份/完整逻辑 SHA/桌面支持计数；每个保存 native source/cooked/plane 数组；全实际 source OBJ 顶点/三角面；原生 RGBD 到 PNG 的完整量化、有效深度掩码与 meta；96 条完整 source 几何射线；所有输入与源绑定及前后 SHA。

这里完整几何计算使用实际 `source_mesh` OBJ；原数据集 `source_visual_mesh` PLY 身份以 SHA 验证。native 20 µm 门表示原生导入保持；本次仅独立复算保存 plane inequalities，没有再次运行凸距离优化器，不声称代理等于精确 CAD。稳定门在全部保存 trace 上复算；source bounds 在观测末态计算，不声称逐 tick 重新计算所有源顶点。

外部只读审计与实际采集重叠运行，其耗时和 CPU/IO 竞争单列，不能当作 collector 原内置阶段。旧 7 capture 的既有轻量审核不重复；原 runtime numeric audit 主要覆盖执行，不能单独代称独立初始化审核。

fit 完整主循环现已结束，8 条原 runtime numeric audit 全部通过。独立父账本审计复核 716 个绑定文件：两场各四型的终态完整，8 条诊断结果为 6 正 2 负，执行共 25,656 步，初始化 1,500 tick / 4,500 对象步；未知执行成本 0。逐条全三体身份/split、canonical480/选中输出、候选与命令 SHA、原 record/evaluation/runtime audit、账本成本及外层训练排除一致。全部父账本 `label=None`、`formal_training_eligible=False`、`exclude_training=True`；这些诊断结果不加入此前 15 条有效新采集标签。账本复核耗时 0.426 秒，不重复解压执行接触。

fit 主循环壁钟 1,765.078 秒，含两场 build/capture、一次批量 CGR、全部 prepare、两常驻执行组退出、八条原 numeric audit 与入账；不含入口进程启动及 preflight SHA 时间。该数值对应已知场景控制，不能外推整个对象集有效率或最终 4,000 fit 采集 ETA。分项和双组实际重叠由单独吞吐报告核算。

第三场 internal capture 也已一次完整审核通过：750 帧、2,754,208 条接触，全部三体最后 250 帧均逐帧受桌面支持；737,688 个实际 source OBJ 顶点、1,475,380 个三角面、2,172 个 native hull，96 条完整源射线误差最大 `3.33e-16 m`，native-plane 回执误差 0。独立壁钟为 23.230 秒；三场合计 65.545 秒。这三个新 capture 各只完整解压一次。

internal 新旧初始化/RGBD 与 17 个 CGR 数组均 exact；g1/g6 仍为 null_geometry、g7 ready，它们各 36 个已选候选语义字段及所有已有命令实际数值一致。g8 两边均为 no_candidate，均没有已选候选文件，未将无候选当作失败标签。

internal 完整主循环已结束，g7 的原首次 numeric audit 通过，为 1 条正诊断结果、3,242 执行步；两条 null_geometry 和一条 no_candidate 仍没有标签。最终账本独立复核 608 个绑定文件通过，耗时 0.346 秒。internal parent loop 为 577.872 秒，外层入口时间由根报告另计。所有 4 槽终态完整，未知执行成本 0、正式训练标签 0。

三场总预算 12 槽全部终态：9 次执行产生 7 正 2 负诊断，另 2 条 null_geometry、1 条 no_candidate，0 infra、0 未知执行成本；初始化合计 2,250 tick / 6,750 对象步，执行合计 28,898 步。全部三体初始 velocity 为 0，fit/internal 的全体原始 source 身份及 digest 无交集。三场共 51 个 CGR 数组 exact，11 条实际已选候选各 36 项语义 exact；其中 9 条最终可执行 lift 的所有实际数值指令 exact。其余两条仅原 rejected arm 准备产物一致，不能称为可执行计划。

最终轻证据 `evidence/author_resident_collection_inputs_20261115_v1/EVIDENCE.json` 绑定三次 capture 审核、两次输入/计划对照、两次候选语义对照、两次最终账本审核及所有独立脚本。执行接触数值审核沿用本轮九条原首次审计的有效 PASS，没有重复解码。审计期间新增模型前向、Kit 启动、规划与物理积分均为 0；本轮正式可训练新标签仍不增加。

脚本 `audit_author_resident_collection.py` 及 `evidence/author_resident_collection_inputs_20261115_v1/{compare_semantics.py,audit_ledger.py}`。大审计产物位于 `outputs/boya_author_resident_collection_audit_20261115_v1/`，轻量汇总在同名 evidence 目录。本阶段不会把这轮旧场景诊断标签加入正式训练集。
