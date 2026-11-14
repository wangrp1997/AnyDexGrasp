# 采集进展、真实双路执行与现成方案复用（2026-11-15）

当前新流程实际有 **14 条 fit（10 正 4 负）＋1 条 internal（正）**；正式训练仍为 0。两个固定开发预算均已结束，尚未达到每型 1000 fit＋200 internal。旧 435 条及八个评分头不变，不自动混入新数据。整体论文复现尚未验收。

用户要求参考已有实现。本阶段已把后续主线改为 **kywch fork 的常驻采集循环＋本机 Isaac Lab 官方场景管理/克隆＋现有 Boya 动作与标签**，停止继续开发局部数学优化。来源、版本差异及接点见[复用方案](AUTHOR_RESIDENT_REUSE_PLAN_20261115.md)。常驻和向量环境尚未实现；以下双进程实测不等于官方向量环境已经接通。

## 两个开发预算的实际产出

| 固定预算 | 全部槽位 | 有效 fit | 有效 internal | 其它结果 |
|---|---:|---:|---:|---|
| 原 development/batch_v1，seed2026111401 | 40 | 14＝10正4负 | 0 | 10 null_geometry、12 placement_null、3 no_candidate、1 infrastructure_failure |
| fresh6 waves_v1，seed2026111402 | 24 | 0 | g7 正例 1 | 4 null_geometry、12 placement_null、7 no_candidate，0 infra |

fit 四型 g1/6/7/8 数量为 4/3/3/4，正例 3/2/3/2；internal 只有 g7 一条。累计 15＝11 正 4 负。这是采集标签构成，不能当作重新训练后的成功率。null 和基础设施错误不记负例。

原预算已完成 7 次 capture，5250 初始化 tick/15750 对象步，14 次执行共 44494 tick；7 份初始化独立审计与 14 份执行独立审计通过。一次 index5/g6 的 BACKEND.json 写后变为全 NUL，发生在物理执行前，原因仍未查明；原失败、文件和成本保持，不重试、不声称已修复。原完整墙钟 5295.205s（88.25min），不能把当前慢速解释成作者算法必需耗时。

fresh6 父协议 SHA `c657176d0478391e2907289d0c454c57c05deaf3262ba1e3c30a69b991191c25`。原序保留 3 个 placement_null，另 3 次 capture 共 2250 tick/6750 对象步；只有一个候选形成完整计划，g7 实际 3242 tick，通过原独立数值审计，产生首条 internal 标签。阶段首末跨度 714.650s，0 未知执行成本、0 infra。独立 host 监测只见最多 1 个实际 Kit；不能用此批次声称证明双路并行。

现有资产池仍是 S40 的 GraspNet/DGN 派生几何、32 fit/8 internal，并非作者全部 L144 CAD。15 条远未采足，不能提前启动正式训练或宣称数据比原论文更多。

## 固定两计划的真实并发诊断

为直接回答能否利用机器并行，另冻结两份原已成功的 scene1/g1、g6 计划，各执行一次，不再采样、规划、加载 CGR 或评分模型。新准备目录只逐字复制 READY/协议/准备回执，候选、命令和来源仍绑定原只读文件。外层协议明确 `diagnostic_only`、`exclude_training=true`，两条新记录不进入采集导出或成功率分母。

协议 SHA `080e470ce022322b7b4a7ec73afadb7edba6bc568d3912682b4e3bb148d11d5d`；结果为：

| 比较项 | 实测 |
|---|---:|
| 两条历史串行 execute stage 合计 | 471.457s |
| 本次双路 execute 首末跨度 | 237.844s |
| execute stage 重叠 | 226.284s |
| 历史合计/本次跨度 | **1.982 倍** |
| 本次两执行＋各自完整独立审计 | 302.148s |
| 实际物理 tick | 3139＋3008＝6147 |

两条均成功，原独立审计各 PASS；完整接触、源网格底面、双侧承载及 24→28 列实际指令仍按原门核验。这是两个已知成功计划的性能诊断，不是新模型成绩。比较基线来自历史串行回执，未另做随机配对串行实验；1.982 倍只适用于这两条 execute stage，不能套用到完整采集、4000 条配额或其它场景。

独立 host 观察核对 PID/startticks、真实 Kit 动态库、GPU0/global lease 和不同 kit slot 的继承 FD/独占 flock，并在同一采样区间看到两份原接触 journal 同时增长；nvidia-smi 同时列出两个任务 PID 在 GPU0。物理仍是原 CPU 后端，并非 GPU PhysX 向量模拟；GPU1 未分配给本任务。所有进程组/槽位完成后回收。证据见 `evidence/author_parallel_execution_20261115_v1/`。

## 已完成的小项及验证边界

[证书复用导出](AUTHOR_CERTIFIED_EXPORT_20261115.md)避免对相同已审计接触再次做完整数值解码：逐条核原 PASS、原成功审计 stage 和全部输入 SHA，发布前再完整读一遍。原 14 条导出 16.587s；fresh6 的 1 条 internal 导出 3.354s，733 个唯一输入两遍核验。原数组构建、配额、全三体 fit/internal 隔离不变。不是跳过首次独立审计，也不是新采集获得数十倍加速。

新 benchmark、证书导出、离线接触格式共 16 tests 通过，root 组合运行 6.580s。首条 root 测试命令漏 PYTHONPATH，两个 import error、0 Kit/物理；只补环境后通过。独立实际 NPZ 对照核原 candidate/record/全三体身份；原 435/八权重 13 文件 SHA 复核不变。只读接触 profile 的格式试验未集成、未提速，详[成本分析](AUTHOR_CONTACT_PERFORMANCE_20261115.md)；此前拟议的局部优化不再继续。

本阶段所有实际任务均已结束，无后台物理或训练。新工具 opt-in，冻结旧源码不改，CanonDex 只读。代码、报告、轻证据阶段推用户 origin；数据、NPZ、接触、权重和视频继续放忽略目录/public。

后续实现按上述现成方案先接一个常驻 Boya 三体环境，再用官方接口核验少量环境的隔离和重置，保留原命令、物理和标签；通过后才冻结足量有限采集队列。当前不宣称已具备几小时采足 4800 条的能力。

已有成功演示：[Boya 完整机械臂提起 EGAD A1](../../outputs/boya_fast6_20261111_v1/execution/videos/egad_A1/isaac_full_arm_pickup.mp4)，正常速度约 9 秒、提起约 10cm。这属于旧模型的已有验证，不是本阶段新模型结果。
