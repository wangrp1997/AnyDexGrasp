# 真实小样本评分头训练与推理验证

用户要求先跑通流程，失败物体后续再改，且明确不需要录像。上一批100场已经完整收尾（20有效：16fit／4internal），没有新采集在后台。本轮用这20条独立MJX域数据运行一次工程pilot，不等待40种物体均有成功，不修改旧采集、源物理或原正式100/300/1000训练规模定义。

新入口`author_mjx_head_pilot.py`，产物`outputs/boya_mjx_head_pilot_20261118_v1/`。输入16fit为14正2负、4internal为2正2负，原全三体对象身份隔离、canonical480、目标深度列及标签均通过绑定验证。没有合并旧域或平衡样本。所有原source与archive保持。

调用原作者模型／loss／train_one_epoch／eval_one_epoch／save_model，100epoch、batch128、Adam lr0.0005/weight_decay0.0005、PolyLR100与原selected-output MSE不变。小于正式100条的数据规模在新协议中显式标记limited_data_pilot；不冒用synthetic development模式，也不改原sample-cap函数。

CPU单线程实跑100次optimizer更新，主体训练及推理验证2.869812秒，参数SHA已变化。按原验证规则选出0.5／0.7 checkpoint，另存显式last_pilot便于核验。三份权重均weights_only安全重载，在4条留出特征上与原source模型eval输出[4,1,20]逐值一致；20条原真实特征推理输出有限，保存完整深度分数及所执行深度分数。

0.5所选checkpoint在4条内部验证中TP1、FP0、TN2、FN1。这只是小样本流程证据；4条内部数据还用于选模，不代表外测泛化或新场景物理抓取成功率。原ReLU分数也不是校准成功概率。

现在已打通真实采集／标签／导出→训练更新→权重保存重载→已有真实候选评分。训练头尚未接入新场景候选排序和实抓，`live_candidate_ranking_integrated=false`、`new_scene_execution_complete=false`；不能称训后完整物理闭环完成。没有自动改变原采集候选分布或派发新物理／扩采任务。

本轮0视频、0新物理、0新CGR，训练进程已结束。checkpoint/数据/分数在ignored outputs；轻证据见[结果](evidence/author_mjx_head_pilot_20261118_v1/SUMMARY.json)。

空间核查：上一批没有视频，文件逻辑体积约28.777GB，其中25.506GB是完整逐步轨迹／接触JSONL。必要RGBD快照和完整物理审核日志不同于录像；历史已冻结原始日志保留。本轮没有进行删文件或压缩改写。用户尚未选择新的有效fit总目标，不能把问题中的“采集完”当自动恢复旧4800或1000配额。
