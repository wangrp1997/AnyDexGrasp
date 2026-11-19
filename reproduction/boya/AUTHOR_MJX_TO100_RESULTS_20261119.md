# 累计100有效fit扩采与正式评分头训练结果

原CGR／g1常驻采集已达到目标并停止。累计正式数据为101条fit＋16条internal，共117条。随后按固定前100条fit完成原配方正式训练；内部验证正例召回仍低，目前没有模型增益结论。

## 采集与审核

根目录`outputs/boya_mjx_cgr_to100_20261119_v1`，原调度工具session79363已exit0。父流程实际耗时19009.471秒，即5小时16分49.471秒，其中采集＋审核18907.213秒；不是派发到用户晚查的间隔。此次新增97条有效标签，约18.37条／小时，约195.97秒／整机新增有效标签，包含新增内部验证数据；不同于此前重复布局一小时39条的诊断样本流，也不是单环境单条延迟。

| 项目 | 数量 |
| --- | ---: |
| 预声明场景容量 | 800 |
| 实际处理场景 | 432 |
| 达标后未派发 | 368 |
| placement_null | 164 |
| planning_no_candidate | 108 |
| planning_null_geometry | 50 |
| 实际抓取并完成回位 | 110 |
| 无效执行，标签null | 13 |
| 新有效正例／负例 | 63／34 |
| 新fit／internal | 85／12 |
| 合并后fit正例／负例 | 68／33 |
| 合并后internal正例／负例 | 11／5 |

旧正式16fit＋4internal与本批新数据按同域引用合并。所有trial／record／源hash保留，不合并旧诊断、对照、PhysX数据或435旧数据。最后同wave产生的第101条fit负例保留，正式cap100采用固定前缀，不按结果挑100条。未派发368槽不计失败；场景中未被尝试的旁观物体不当负例，也不声称40种均有正负抓取样本。

27个wave完整只读raw审核全部PASS，完整匹配28,920,360接触和873,952条trial观测，native palm FK最大差6.645028e-7m。97个有效标签全部审核通过。13个invalid保持null，因此原总汇`all_complete_grasps_pass=false`保留；审核PASS不表示110次抓取都满足所有物理门。原1mm穿透诊断、5mm硬停止及其他六门不改，不宣称与PhysX轨迹等价。

实际202,268个global control ticks、25,890,304次native环境积分；404,537行NativeCall日志begin/end完整，未知成本0，小于119,040,000本批上限。16world全队列模型／相机／机器人／CGR常驻，原动作、规划、物理、完整接触与回位保持。未录制视频，必要单次RGBD和完整原始日志继续保存在ignored outputs。

`FINAL_REVIEW.json`确认54个源／输入／archive精确绑定，111登记owner／进程组退出、6锁可取、4资源预约released、全局registry released。原工具session结束，无本批后台采集。采集阶段结果中的`training_updates:0`保留历史含义，后续训练另立根目录。

## 正式100样本训练

入口`run_author_mjx_fit100.py`，新根`outputs/boya_mjx_fit100_20261119_v1`。以同域merged数据固定前100fit（68正32负）训练，16internal（11正5负）用于原验证／选模；第101fit不进入本cap100。全三体源对象在fit与internal间隔离，无重复trial。seed2026111903，仅一次，无类别重平衡或换seed重试。

复用未改的`author_training.fit_head`及绑定公开源码：canonical480→20、final-ReLU selected-output MSE、Adam lr0.0005／weight_decay0.0005、batch128、100epoch、原PolyLR与0.5／0.7／0.9选模。实际100次optimizer更新，非development缩短。CPU训练主体6.720秒；含组件加载和核验的训练run段41.565秒，不拿训练耗时当采集加速。工具session38200已exit0，0新物理／CGR／视频。

原选模规则仅在epoch56留下`head/best_0.5.pt`，0.7／0.9没有合格保存，不补造权重。发布inference严格加载通过；独立只读检查再次核对12个绑定源码及archive、dataset／manifest、固定first100前缀、100epoch／100更新、权重hash和全部16行推理输出逐值一致。

| 内部验证，0.5门 | 数量 |
| --- | ---: |
| 正例判正 TP | 1 |
| 正例漏判 FN | 10 |
| 负例误判 FP | 0 |
| 负例判负 TN | 5 |

正例召回为1/11，效果仍弱。16条内部验证已经参与选模，不能作为独立外测；本次也没有用新头进行新场景物理抓取，不能报告模型提高抓取成功率。0.5门的召回不直接等于同池候选排序质量，后续先检查分数分布与多候选共享观测下的排序，再做冻结的新布局评测。当前原CGR采集策略未替换，不自动追加物理批次或重训。

## 复核入口

采集终态看`CAMPAIGN_RESULTS.json`、`merged/MANIFEST.json`、`collection/READ_ONLY_AUDIT.json`和`FINAL_REVIEW.json`；不要用终态前的旧PROGRESS误报仍运行。训练看新root的`RESULTS.json`、`head/TRAINING_RESULTS.json`及`FINAL_REVIEW.json`。两个已派发root均不重跑。

轻证据在`evidence/author_mjx_to100_20261119_v1`和`evidence/author_mjx_fit100_20261119_v1`，大型NPZ、权重、完整接触／轨迹不入Git，不删除／改写原始证据。新轻代码、文档与证据提交用户origin，历史冻结源码和CanonDex保持。
