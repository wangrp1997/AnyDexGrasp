# S40全40件采集已后台启动

2026-11-07，Asia/Shanghai。用户授权继续扩。真实启动时间2026-11-07T11:05:27.373395+08:00，PID 1658959；PID和以下阶段是启动凭据，不替代当前运行状态。

运行根：outputs/s40_collection_20261107_v1/。入口 run_s40_collection.py --phase all。固定40件全train、四型1/6/7/8、四extra0/10/20/30mm，共640槽；继承原000/002的8槽(5实际正例/10500步、3零步prepare null)，新增632槽。原两帧观测/CGR只读复用，另38件新观测/表示；不加载旧DGN head，不启动EGAD或新评分器训练。

目标每型100条可用正负标签、共400，只是有限640槽的目标；原缺额/null和所有额外标签保留，不能保证达标。按固定源SHA预先冻结S40内部32/8网格诊断分组；40件仍全train，内部8不是论文外部测试。每物体当前一个固定源轴姿态，不声称真实多姿态/杂乱场景采集分布已复现。

CPU32–95：native4/geometry4/arm8/object-execution4，CGR用GPU0单模型38新forward；物理为对象内CPU serial resident wave，不是GPU物理。不同命令长度可拆多cohort/多Kit，不保证每对象仅建场一次。源码、源资产、物理参数及原门已冻结；勿在运行中改py或重复启动。

已修扩大前两处：新renderer强制source_inventory路径/hash freeze；合法geometry no_candidate经源/物体/手型/深入/零步和子命令绑定后记null，其他非零异常停止。执行用v2完整原计划长度，已知两种早停需原证书认证才消费为terminal null；unknown停止新派发并收尾，提交意图不作未执行证明，绝不自动重播。

训练记录与实际binding/精确label/steps及eval/commands SHA重新核对，原collect_record重新派生比对；对象行总步和各serial cohort时钟一致。41项检查和原5条artifact导出复用通过。冻结核验573源码/1162输入/3manifest均通过。完整协议见S40_COLLECTION_PROTOCOL_20261107.md，轻量凭据evidence/s40_expansion_20261107，原始审阅在运行目录launch_review。

观察入口：pipeline_progress.json为阶段/完成项，logs/和根目录旁launcher.log为实际子程序；planning/queue_manifest.json为准入/null；execution/PROGRESS.json为对象执行；最终execution/RESULTS.json、train_records.jsonl与collection_counts.json为实际正负/null、所有物理步和每型/深入欠额。阶段尚未到时文件可以不存在，不能据此补0。

本轮为GraspNet派生S40原尺度的Boya仿真采集移植。原作者CAD逐字节身份、真实物理/控制、完整测试集合仍有缺口；整体论文复现未完成。旧DGN统计、原两件pilot、所有失败与成本保持，不能以启动凭据报告新成功率。
