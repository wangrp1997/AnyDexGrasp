# S40预激活训练对照：轻量证据

本目录只含源码、固定指标、来源SHA及独立审计。权重、原始NPZ/轨迹、RGB-D、视频均留在outputs，不提交。SUMMARY.json为训练→同域评分→同预算臂→top1各一次实抓→录像的完整交付摘要。训练死区、BN负对照、稀疏监督和映射负探针分别保存，不能合并冒称原论文复现成功。

权威训练结果：outputs/s40_preactivation_control_20261107_v1/TRAINING_RESULTS.json。
权威实抓结果：outputs/s40_preactivation_execution_20261107_v1/HEAD_CONTROL_EXECUTION_RESULTS.json。
原code100兼容view聚合指标残留见INTERFACE_VIEW_METRICS_NOTE.json；它们未参与排名。

所有路径为当前服务器的来源绑定；复查脚本需要原未提交资产。verify_delivery.py会独立复算控制数组、物理时钟、源网格末态最低点/载荷/标签与视频时钟/来源SHA。audit_training.py只读核对shuffle/训练前向和隔离组。其余两个audit_base*及probe_contact_reference保留诊断计算，不属于部署策略或新的物理结论。
