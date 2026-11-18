# 首批100个新场景扩采

用户授权先提交推送代码、ignore大文件和无关本地产物、保持工作区干净，再开始扩采。此为有限新场景数据批次，不是旧重复布局测速，也不启动训练。

- root：`outputs/boya_mjx_fresh_expansion_20261118_v1`，本地产物均ignore。
- seed：2026111802；原确定性source sampler，无替换、无失败后换seed。
- 场景抽样：80 fit＋20 internal，每场三体，目录对象32/8严格隔离；按4 fit＋1 internal交错排队，与几何和标签无关。
- 固定100抽样包含全部40物体；63个ready场景包含38种，37个placement_null保留。graspnet_040/046仅出现在null布局，不能称实际覆盖40种。
- 固定g1五指握；16常驻环境，7个wave，至多100次CGR/规划/抓取，最大16,665,600原生环境积分；0重试、不补足有效标签数。
- 沿真实小时已验收native_batch v1、entry_v9和hour collect_wave的原750以上稳定、同世界拍照/CGR/原实测起点规划、24位置＋6前馈、0.5ms×8、逐250Hz完整17接触字段、原5mm停止及功能标签、原物理回位与完整只读独立审核。
- 原1mm穿透门仍保留诊断，其他六门和5mm硬停止保持。此为独立MJX功能数据域，非PhysX等价认证。
- 没有实现新的binary/GPU分块/异步分组物理。仅新场景、去掉一小时准入截止；后台审核沿已验收实现。

`run_author_mjx_expansion.py freeze` 在0物理下冻结输入、源码及archive。`run` 要求工作区干净，记录commit后单次派发collect＋后台审核，再执行原完整staging导出。完整绑定审核及物理回位通过后另建`formal/records`与NPZ/MANIFEST；原staging记录和历史诊断不回写，旧数据不合并。新formal数据可供后续授权的Boya评分头训练，本次training_updates=0。

数据语义：正常完整有效失败是0；无候选、未稳定、未执行、硬停止、数值无效均为空。全局label为任一三体满足原提起/受载规则。逐体OBJECT_COVERAGE单独报告抽样、ready出现、实际机器人接触、手载荷与成功提起；接触但未提起只是诊断，旁观物体不是负抓取标签。legacy target_object_id/mesh_sha256不能当实际抓起对象。

24项CPU检查通过：新增2个Progress检查、1个旁观对象归属检查，加既有21项时钟/Journal/接触字段/功能标签/提起规则检查。新Python文件均AST检查；上述检查0新增积分。63ready场景及扩展物体银行的实际物理状态以本次RUN_STARTED、审计及最终回执为准，不用旧小时数字保证本批速度/成功率。

查看进度：

```bash
cat outputs/boya_mjx_fresh_expansion_20261118_v1/collection/BATCH_COUNTERS.json
cat outputs/boya_mjx_fresh_expansion_20261118_v1/collection/PROGRESS.json
tail -n 20 outputs/boya_mjx_fresh_expansion_20261118_v1/timing/collect_and_audit/process.log
```

`PIPELINE_STARTED.json`是调度启动；`collection/RUN_STARTED.json`是持GPU锁后的采集启动；`BATCH_COUNTERS.json`区分落盘与审核数量。全部结束以`EXPANSION_RESULTS.json`为准，异常保留`PIPELINE_FAILURE.json`。严禁对已派发root重试。运行产生的数据不会弄脏Git工作区。

实际启动绑定提交`c99d211`，调度1164791、采集1164862，16世界／38银行已经进入首wave实际GPU积分。冻结源码和协议保持，不根据新结果调整控制。

只读复查补充：不完整执行按原规则无object_outcomes，当前冻结入口的最终逐体统计可能因此KeyError。新增独立`finish_author_expansion_metadata.py`只读等待原任务：若原成功则不操作；仅该精确统计错误且原完整collect/audit/staging导出均通过时，复核已经生成的formal NPZ/原record/哈希/科学数组，再以exclusive写入补缺失的覆盖表和MANIFEST，保留原失败回执，并生成`EXPANSION_METADATA_COMPLETION.json`。未评估物体结果记不可用，不当负标签；其他错误不重试。新增partial-outcome测试通过，总25项检查，0新增积分。watch运行状态见METADATA_WATCH_STARTED/FINISHED/FAILURE。
