# S40稳定落位与覆盖补采启动

- launch.json / training_watcher_launch.json：实际PID、argv、启动时间与有限并发/后续训练范围。
- settling_summary.json：40稳定与重建验证、严格v1探针及313250初始化object steps；不代表40抓起。
- launch_progress_snapshot.json：真实阶段快照，非最终结果。
- tests_and_sources.json：40合同测试与新源码SHA。
- coverage builder receipt：真实旧帧零物理8个准入及来源/排除键验证。
- prehead masks与saved candidate diagnostic：原作者基础mask缺口的只读对照；原3200拒绝不覆盖。

完整初始化/图像/CGR/轨迹/标签/模型在本机outputs/s40_stable_supplement_20261107_v2；旧严格探针在v1。EGAD仍未使用。冻结代码运行中，不重复启动或热改。
