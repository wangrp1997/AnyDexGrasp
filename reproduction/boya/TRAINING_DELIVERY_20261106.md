# Boya评分器训练与实际对照交付：2026-11-06

核心学习算法复用官方：冻结原CGR表示/权重；从原源文件AST加载Inference、单索引MSE及PolyLR，官方训练/推理结构与数值等价检查通过。新增代码负责Boya记录格式、Isaac执行、数据/运行校验与自动调度，不把这些工程适配称为新的学习算法。原作者源码保持。

## 数据和训练

`prepare_boya_training_data.py`读取已完成的正式续跑RESULTS与实际训练/验证JSONL，复用已由原函数规范化一次的480输入，不二次翻转/转角，不截前100，不把1次试抓展开成20标签。默认未完成即拒绝，显式development snapshot只能产生benchmark_only数据。40训练mesh事前固定32内部拟合+8检查，外部10mesh不进入训练或调参。

每手型独立480→20评分头；20项为**5个原两指raw depth×4个额外深入**，angle维=1。这是对旧COLLECTION_PROTOCOL_20261106.md“5角度”文档口误的勘误，旧冻结文件不热改；采数、target_index和实现始终为raw_depth*4+extra。

`train_boya_decision.py`正式双阶段各20epoch：先32组拟合并仅报告8组末轮检查，再全部40组从头重训四head。固定Adam初始1e-4取论文，WD5e-4/PolyLRpower.99取公开代码；该显式组合不同于原公开脚本100epoch/5e-4，不称逐行原脚本复现。epoch/config不按外部验证选择。BN单样本尾batch并入前batch，不丢样本/不复制记录；仅实际执行的一个输出索引受MSE监督。

真实116条记录的工程smoke（98内部拟合/18内部检查，外部0）已在GPU0跑通，每阶段2epoch、共16次optimizer更新，trainer内部3.393s，四个最终和四个内部state_dict均安全严格重载。benchmark_only=true，不作为正式模型或性能证据，零CGR/零新物理。正式模型尚未训练。

## 自动后台链

`run_boya_training_delivery.py`已实际启动，PID4050257，启动北京时间2026-11-06 11:35:53；活动目录 `outputs/boya_decision_delivery_20261106_v1/`，计划SHA48fcfa71b97edbe02c25e5251c7433d6cde258dc7a4ae36b24109cc2881ec8d0。源文件与计划冻结，每15秒监控原采集进度/PID，最长等待12小时，正常成败均可完成采集；基础设施异常、producer退出、未知执行或源码变化明确写blocked，不自动修复/重放。

自动顺序为正式dataset→四型训练（内部检查+全40最终拟合）→评分排名冻结→两策略各对10个外部物体一次真实top1试抓→最多4段实际RTX轨迹回放视频。物理对照两策略各自独立初态恢复，最多两worker；无重抓择优、无oracle成功筛选，记录benchmark_only不写回训练。视频失败独立报告，不掩盖已完成的模型/物理结果。

`evaluate_boya_decision.py`在原431已几何/臂准入的同一候选池内比较原CGR分数最大与四型学习分数最大，平分按原queue顺序，选型无阈值。排名阶段只hash含标签的manifest，不解析样本/读取结果；先冻结selection/scores/policy_manifests，随后才能读既有标签作离线配对证据。默认拒绝工程smoke模型；实际再执行两策略推荐的top1，不能拿旧标签离线重放称新抓取。该比较受此前候选/臂过滤限制，10件是初步验证，不能替代广泛未知物体泛化测试。

## 进度与交付文件

- `DELIVERY_STATUS.json`：waiting_for_collection及各阶段/blocked原因。
- `dataset/manifest.json`：正式数据、32/8/10分组与SHA。
- `training/TRAINING_RESULTS.json`与`heads/g01.pt,g06.pt,g07.pt,g08.pt`：四个最终head及官方来源/config。
- `ranking/selection.json,scores.json,RANKING_FREEZE.json,policy_manifests/`：无标签参与的推荐冻结。
- `physical_comparison/{baseline,learned}/RESULTS.json`及汇总`RESULTS.json`：真正的新试抓与物理步数。
- `videos/index.html`：原模型/原实际轨迹视频；回放零物理。
- `DELIVERY_RESULTS.json`：最后模型、同候选池top1成功率和媒体链接。

上列阶段文件依赖后台完成后逐步创建。新源码/轻量证据可提交；模型、原始记录、NPZ、轨迹、图像、视频和outputs不提交Git。当前采集源码与旧试验目录全部保持。
