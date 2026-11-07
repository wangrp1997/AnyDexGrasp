# 双flip语义及单次top1执行接线（2026-11-07）

结论：当前`deploy_score_first.make_queries(...,'both')`语义与collection一致。8项纯测试通过；对内部8件现有**27条planning-ready binding**全部重建，canonical_CGR480、其float32字节身份、raw depth、actual/official flip、extra depth逐项一致。没有读执行标签/训练记录/EGAD，没有改domain或主源码，零模型前向与物理。

## exactly once的准确含义

`make_queries:75–89`每个原raw先False、再True，每个flip调用一次`canonical_candidate`，再让四手型复用该480向量，故每raw8个query但只做2次独立规范化。`candidate_catalog:188–199`同样按False/True、从原raw2261特征独立调用。`prepare_binding:110–128`随后从已冻结NPZ与原source_row_index**重新读取原raw**，再次独立重建作为binding核验。这不是把已经canonical的480再次规范化：每一条从raw到head/binding的特征链都只规范化一次。

原两函数语义：actual flip反转CGR R的第2/3列（接近第1列不变）；对480特征分别交换score与width的120维前后半块，再按原angle×5在各240维块内循环移位。两种flip通常得到不同的480与非对称手部落位。`if_flip=False/True`必须始终表示**相对原raw的绝对选择**，不是“在现有姿态上再切换一次”。

纯测试包括：原raw×False/True枚举/调用次数、四手型复用、原official策略不改、独立half-swap+roll数值参考、不修改源数组/重复调用不累积flip、二次canonical会改变结果、2262已追加flip特征拒绝、depth错配拒绝、同depth错配须靠源字节身份发现，以及27条真实planning binding逐字节对照。

## 必须保留的身份和配对检查

- mask和topK后保留**原NPZ source_row_index**，不能拿压缩后局部index索引原raw/features。raw与2261 features必须同一行且有原NPZ SHA、原row byte SHA、feature byte SHA。仅shape/depth相同不足以发现错配。
- fresh raw副本上执行实际flip；不要先调用official翻面、又对其输出应用actual flip。extra depth只参与target `raw_depth*4+extra`和palm insertion，不能改变CGR480或追加后再canonical。
- 查询身份为`(sourceNPZ_SHA, source_row_index, gesture, actual_flip, extra_depth)`；canonical480 byte SHA辅助核验，不能单凭相同feature就合并False/True两种实际姿态。相同480可能来自对称/重复特征，几何姿态仍可能不同。
- `official_policy_if_flip`作为原策略参考保留，但在双flip口径中不得覆盖actual flip。原姿态/源特征不可原地修改；每次构造都从原raw重新开始。
- `prepare_binding`最后重建的canonical480 byte SHA必须等于排名冻结的query identity，actual flip及target index也必须完全相等；现有`admit:228–235`已有这三项检查，建议再核source_row_index、NPZ SHA、gesture与extra。规范化后的435条训练480直接输入head，不再经过旧legacy angle/flip转换。

## 不新增训练采集的geometry→有限arm→最终top1路径

1. 新目录冻结：现有观测/原始NPZ、单一原domain policy、明确20epoch或公开100epoch训练配方的某一份checkpoint、双flip口径、阈值及统一预算。将两配方分别报告，不按已知成功标签选择、拼接或按物体换checkpoint。
2. 在domain内按原CGR topK形成False/True×四head×四extra组合；只凭新head分全局排序，冻结`DOMAIN/head_outputs/shortlist/RANKING_FREEZE`，然后才接触几何/IK。不要读旧ready队列或历史成功标签选候选。本文只把旧planning binding用于独立语义核验，不能传给部署排序。
3. 复用`deploy_score_first.admit(args.stage='arm')`现有单候选pool/generation构造和`prepare_collection_queue.prepare_one(max_arm_plans=1)`。按已冻结global rank检查原几何；几何准入后，完整臂准备共享最多4候选额度。对每个正常IK/路径拒绝保留零物理拒绝回执，基础设施错误则停止。遇第一条完整臂ready即停，它就是**当前有限候选/门下按新学习分最高的可执行top1**，不是物理试抓过后挑成功。
4. 对该selected row用`execute_collection_queue_v3.py`新建**只含这一条**的manifest，原commands/binding SHA不改，命令按源各phase原长度执行。先做v3 prepare-only绑定核对，再执行一次原完整抓取。不要把全部候选排队逐个物理试抓来选top1。若没有ready就诚实报“有限预算内没有可执行计划”，不启动物理。
5. 新实际执行的结果与录像只放新deployment-evaluation目录，单列实际步骤/所有规划成本。当前训练435条和模型保持冻结，不自动把部署结果并回训练或重训练。单次物理失败仍是本轮结果，不事后换成另一条成功覆盖；若需要现场重抓，必须另立明确协议。

这里的top1是评分排序经有限可行性筛选的首个准入，不能称全空间最优。上述前几步不需要新增训练采集；最后一步是一次新的部署物理验收，本文未启动。

测试脚本：`/tmp/test_boya_both_flip_semantics_20261107.py`（8 passed）。27条逐条核对：`/tmp/boya_both_flip_binding_matches_20261107.json`。
