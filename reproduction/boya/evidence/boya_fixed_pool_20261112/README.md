# 固定57候选诊断轻量证据

结论及边界见 [完整报告](../../FIXED_POOL_RESULTS_20261112.md)。本目录只保存轻量源码快照和JSON回执，原始输出在 `outputs/boya_fixed_pool_20261112_v1`，轨迹、模型、视频不提交。

- `PROTOCOL.json`、`CATALOG.json`、`SCORES_PREPHYSICS.json`、`PREPARED.json`、`DISPATCH.json`：冻结57姿态、三路排名、15旧复用与42新一次队列及全部输入/源码哈希。
- `RESULTS.json`、`METRICS.json`：57实际结果与同池统计。5成功，三物体各1/31、2/11、2/15；新82688步、复用30575步不混计。
- `PREEXECUTION_INDEPENDENT_AUDIT.json`、`INDEPENDENT_AUDIT.json`：独立核对队列、物理参数、2100行完整指令、实际源底/承载、历史匹配以及单列A0正对照。
- `METRICS_INDEPENDENT_AUDIT.json`：未调用生产summarize的独立分数、排名、阈值、结果与topK复算。
- `FAILURE_MECHANISMS.json`：接触、失载、源底、参考升高、早停与各门的重叠诊断计数，不代表唯一因果归因。
- `TESTS.json`、`COMMAND_RECONSTRUCTION_MUTATIONS.json`：24测试/5子测试及四类指令篡改拒绝检查。
- 两份 `*_FAILURE.json`：准备类型检查和审计相对路径调用错误，均零物理，原记录保留。
- 两份 `.py`：事后独立指标/失败测量脚本的原字节快照；回执保留原执行路径，不能把重算覆盖旧回执后冒称原产物。

`MEDIA_DELIVERY.json` 绑定完整57段正常速度视频（15历史+42新回放），包含逐帧解码与来源核验；回放0积分、无失败。`FINAL_FREEZE_VERIFICATION.json` 为收尾冻结复核，`FILES.json` 记录本目录各文件字节SHA；回执内部一些 `json_sha` 是规范化对象哈希，与文件字节SHA口径不同。

候选来自两套旧评分筛过的保存池，三个对象是已知难例，使用原粗代理和受控腕。事后任一成功3/3不是部署成功率，原1/12与新六件3/6不改。没有新增训练、CGR前向、标签回写或默认晋级。
