# 2026-11-04 失败机制诊断凭据

原12场景仍为8/12；三个碰撞全拒没有执行，香蕉0°原失败保留。只新增一次不含物体的预抬升负载对照（12000步/6.25s），不是抓取成绩。

`COLLISION_SUMMARY`/`SOURCE_MECHANISMS`/`DEPTH_SEMANTICS`保留三全拒场景归因、219候选的点标签、原源码筛选与深度语义摘要；完整候选点位/体素复算和历次版本在outputs诊断目录的collision_attribution_v1/v2/v3。

`STATIC_GEOMETRY_SUMMARY`保留三个代表候选的原执行目标处距离；源表面不等于Bullet凸几何加margin。完整逐link统计、原脚本和SHA在outputs/diagnostic_sources。

`BANANA_MECHANISM_REPORT`/`BANANA_TRACE_METRICS`/`PAPER_EXTRACT`给出已有接触时序和原论文4DoF/2motor事实，但没有确认真实被动拓扑。外部厂家描述不充当机械辨识证据。

`LOAD_ABLATION_PLAN`在积分前冻结；`LOAD_ABLATION_RESULTS`是实际12000步；`LOAD_ABLATION_AUDIT`独立复算源SHA/原控制数组、同模型/约束/bodyID和结果。原始trace和contacts只存outputs，不提交。

`BUNDLE.json`给轻量复制的SHA及成本；有些回执保留最初/tmp执行路径，字节未改，持久副本映射在outputs/BUNDLE.json。它们不是在旧12场景执行前就存在的文件。静态v1的宽度回执列错误、负载对照v1的零步JSON类型核对错误均保留，不混为抓取失败。
