# Boya共享评分器离线对照证据

正式结果与两个独立审计均complete。RESULTS的gate未通过，未新增物理/CGR、未晋级默认。

PROTOCOL_SUMMARY是省略6650项输入哈希映射的摘要，保留完整协议SHA和位置；不能当原协议字节。MODEL_MANIFEST仅引用25个输出权重，权重/NPZ/图像不提交。FAILED_V1_PROGRESS的0计数由RECOVERY_DECLARATION更正为已完成1000更新；旧回执未改。正式14500加失败1000为15500真实数据更新，合成单测另48更新。

独立audit_run重建输入和checkpoint前向；audit_decision重算预测指标、停止条件和失败/恢复首模型tensor字节。脚本保留本机outputs原路径，运行需要原始产物。

结论及表格见[报告](../../SHARED_SCORE_RESULTS_20261111.md)。
