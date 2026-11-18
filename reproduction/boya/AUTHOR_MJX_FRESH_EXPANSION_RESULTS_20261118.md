# 首批100新场景扩采结果

100场已处理完成，正式导出20条（16正、4负），fit16／internal4。7批完整原始数据审核通过，所有20条导出行均有完整动作、原功能标签、物理回位和独立审核依据。未合并旧域／诊断数据，未训练。

| 结果 | 场数 |
|---|---:|
| 有效成功 | 16 |
| 有效失败 | 4 |
| 布局为空 | 37 |
| 碰撞筛选后无可用候选 | 27 |
| 规划准入拒绝 | 15 |
| 原5mm硬停止，无效空标签 | 1 |

这批不是100条有效样本。63个ready场景均重新稳定、RGBD、CGR及规划；实际执行21抓、21次回位通过。fit的16条为14正2负，internal的4条为2正2负，全部对象身份保持原32/8隔离。

27个无候选场景均有CGR原始输出，碰撞检查前有候选，检查后0；未进入机械臂IK。15个规划空中，11个固定端点／非HOME段碰撞，2个首次RRT连接后最终逐帧或导数门拒绝，另2个（75/80）局部IK种子未收敛；后者不能证明几何不可达。此前只核前7个规划空时IK均已解，后半新增2例不改变那次结论。

71号执行到1902/2351步遇原5mm停止，深度5.063880mm；标签为空，不当失败负例。其回位通过。汇总原字段`all_complete_grasps_pass=false`保留，不能把“raw复算PASS”说成全部21抓都过物理门。20条有效导出均通过完整独立审核、canonical480特征和原480→20评分头数据消费／无梯度loss检查；权重未更新。

40种参与抽样、38种在ready场景中出现、29种出现在有效执行场景中。后者含旁观对象；实际机器人接触和手载荷仅涉及9种，成功提起7种：000=1、002=4、014=2、021=1、022=4、057=3、066=1。不声称40种均有实际抓取或正负分布，也不把旁观体标负。详见[逐体统计](evidence/author_mjx_fresh_expansion_20261118_v1/OBJECT_COVERAGE.json)。

采集＋后台审核parent4260.605295秒（71分0.6秒）；主调度含导出至统计异常4285.983127秒。正式元数据文件mtime距派发约4303.899秒（约71分44秒，仅近似全流程区间）。晚查时派发已过279分钟，运行早已结束，不能把这个查询间隔当实际耗时。

冻结入口在最终逐体统计遇已预知的`object_outcomes` KeyError；原collect/audit、staging export均exit0，原调度exit1及PIPELINE_FAILURE保持。独立只读watch exit0，仅对该精确错误重新核对既有formal NPZ所有科学数组／逐record／SHA／原audit，exclusive补缺失覆盖表和MANIFEST；0物理重试、0新标签。最终以`EXPANSION_METADATA_COMPLETION.json`、`formal/MANIFEST.json`和`FINAL_REVIEW.json`为准，原`EXPANSION_RESULTS.json`未生成。

46342个全局控制步、5931776原生环境积分，NativeCall92685行完整begin/end连续，未知0，低于本批16665600预算。48源码及input/archive绑定保持；32登记owner/PGID全退、7锁可取、5预约释放、registry released，无本批后台任务。源模型、完整RGBD／逐点接触／轨迹继续本地ignore，CanonDex和历史数据不改。

本批20%抽样任务产生有效标签。下一批需依据布局与碰撞候选淘汰、实际手接触覆盖决定方案；当前没有自动追加物理或训练。正式数据在`outputs/boya_mjx_fresh_expansion_20261118_v1/formal/gesture_1.npz`；[轻量结果与资源核验](evidence/author_mjx_fresh_expansion_20261118_v1/SUMMARY.json)保留可复核来源。
