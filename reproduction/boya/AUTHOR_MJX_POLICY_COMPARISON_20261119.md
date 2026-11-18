# 新布局CGR／小样本评分头对照

用户授权“对照下再说”。比较固定pilot模型是否改善候选选择，不追加大规模采集、不训练、不录像。

固定seed2026111901抽16个原internal对象池的新布局，8个placement_null保留，8个ready全部参与比较。每个布局生成cgr／head两个源物体初始坐标完全相同的副本，固定RNG随机分配相邻lane，共16world、8对。布局未参与fit或选模；对象仍来自原internal池，不称新物体域外测。只按建场几何准入，没有按CGR、IK或抓取结果挑场。

baseline使用原碰撞筛选后的CGR降序top1。head在**同一套筛选规则产生的原CGR top10**内，用冻结best_0.5.pt的对应raw_depth×4＋extra_depth输出重排，平分保留原CGR次序。不是全部2048候选重新搜索或分数融合。baseline的head分数仅shadow记录。两边都只尝试一个候选，规划失败不换候选，不改动作、物理、碰撞门和标签。

各world独立稳定、实拍与CGR；原persistent engine每帧恢复相同post-init RNG。相同源布局不自动等于稳定后观测或候选逐值相同，最终逐对报告网络points/coords、原raw候选／特征和语义top10候选池是否exact，差异对不能当纯排序收益。比较双方各8个assigned布局的任务成功率，同时列出规划通过、执行、合格及成对差异；空/无效不改负标签。

16env一wave，最多2380800原生环境积分，最多16抓与原控制回位，0重试／0训练／0录像。沿原v9/native_batch v1、750以上稳定、250Hz控制、0.5ms×8、完整17字段接触、功能标签及原完整独立审核。1mm诊断与5mm硬停止保持，所有数据仅比较诊断、不并入训练。

新代码：author_mjx_policy_compare_worker、author_mjx_policy_compare_collection、run_author_mjx_policy_compare。两项ranking单测（稳定平分、拒绝非有限分数）及旧真实候选池接续检查通过，0新增积分。接续检查中head改选原rank1，baseline保持原rank0，候选SHA／姿态／trace未变；不是效果评估。

root `outputs/boya_mjx_policy_compare_20261119_v1`。启动看PIPELINE_STARTED／collection/RUN_STARTED；运行进度看collection/BATCH_COUNTERS／PROGRESS。完整审核和独立排名复算通过后生成POLICY_COMPARISON_RESULTS、PIPELINE_FINISHED；异常保留PIPELINE_FAILURE，不对已派发root重试。当前单组8对仅探索性对照，不足以证明稳定泛化收益。
