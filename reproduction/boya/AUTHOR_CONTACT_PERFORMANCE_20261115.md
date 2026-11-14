# 提起采集执行与接触存储：只读成本核对和离线实测

本轮只分析已结束 `development/batch_v1` 的 14 条执行，并对第一条 `scene 0000001 / g1` 的固定前 100 tick 做离线 CPU 测量。没有 Kit、规划、物理积分、标签更新或旧输入改写；没有修改任何冻结执行器或 CanonDex 源码。新实验格式尚未接入采集。

## 原 14 条执行能证明什么

| 回执字段 | 14 条合计 | 每条平均 | 实际范围 |
|---|---:|---:|---|
| execute STAGE.wall_s | 3119.462 s | 222.819 s | 子进程完整阶段 |
| LIFT_PIPELINE_EXECUTION.wall_s | 3096.003 s | 221.143 s | native launcher 前到最终 record/receipt 前 |
| native evaluation.wall_s | 3017.454 s | 215.532 s | executor 内部初始化与循环等 |
| native setup_wall_s | 602.647 s | 43.046 s | native 输入核验、建场/导入/碰撞配置/审计等；不是单独 Kit 启动 |
| native loop_wall_s | 2380.080 s | 170.006 s | 循环及其嵌套多体审计，并包含 native trace 最后序列化 |
| 物理模拟时间 | 177.976 s | 12.713 s | 44494 tick × 0.004 s；不是测得的 CPU 求解器时间 |

14 条原独立 runtime audit 已数得 **109128993 条接触**，平均每条约 779.5 万、每 tick 2452.7 条。gzip 文件共 **3791720154 bytes**。不能把模拟时间与执行 wall 的差额全部称为“日志耗时”，也不能从 wall 比值推断纯求解器速度。

`execute_template_arm_positive.py:361–366` 在 `yield(step)` 前后给 `physics_step_update_s` 计时；`author_lift.py:320–327` 先推进 native generator 到下一 tick 的 yield，之后运行上一 tick 的 `runtime.after_step()`。所以原 profile 的“physics”桶会计入外层上一 tick 的接触解码、源网格测量、JSON 编码和同步 gzip。14 条这个桶合计 2223.662 s，**不能作为纯 PhysX 时间**。Kit launcher、native 建场、mesh cooking、host contact transfer、gzip、最终保存均没有现成的独立完整计时，不能给出各自精确占比。

## 固定前 100 tick 的实测

一次预先固定样本，3 次重复，中位 wall；OMP/BLAS 各 1 线程。对应原执行 0.4 s 模拟时间；全部仍是静置/早期 HOME 阶段，不能代表受载闭合/提起的接触分布。

| 离线操作 | 中位 wall |
|---|---:|
| 原 gzip 前缀读取/解压 | 0.127 s |
| 原 JSON 解码 | 0.498 s |
| 与原 journal 相同选项 JSON 编码 | 0.571 s |
| gzip level 1 + buffered 文件写入 | 0.158 s |
| 原逐体承载 + 全源顶点最低/最高点 + 穿透归约 | 0.745 s |
| 实验列存：从现有 dict 校验/打包 | 1.302 s |
| 实验列存：未压缩 NPZ 写入 | 0.006 s |
| 实验列存：gzip level 1 + buffered 写入 | 0.070 s |
| 实验列存：解压 + NPZ 数值数组载入 | 0.037 s |
| 实验列存：恢复原 dict | 0.956 s |

完整样本 **252790 条全部是物体↔地面接触，robot_contacts 为 0**；其中 251731 条为零法向力，252248 条 separation>0。这推翻了“这个样本主要被机器人冗余自碰撞日志占满”的猜测，不能外推全部 1.09 亿条均如此。零力/正 separation 仍属于原生原样接触记录，本实验全部保留。

三份源网格共有 889808 个顶点。单次 cProfile 的 0.753 s 聚合操作中，原 `source_metrics()` 300 次调用累计 0.666 s，原 `object_robot_loads()` 累计 0.073 s。原源底复算与保存 trace 的最大误差为 **0**。这是新增全源网格审计的可测 CPU 成本，不是作者网络推理或新的物理步。

## 无损候选与负结果

新 `profile_author_contacts.py` 的实验格式采用路径字典、tick/group offsets、原次序 body pair ids、binary64 的位置/法向/separation/力，以及可选 normal 掩码；未使用 float32、有损量化、接触删减或时间抽样。原样本 JSON 未压缩 68049134 bytes，gzip 为 **8511391 bytes**；实验 NPZ 未压缩 18458070 bytes，gzip 为 **3580690 bytes**（约小 58%）。

从压缩列存重新载入并重建所有 contact 后，独立 canonical JSON 逐字段比较通过，包含路径、接触次序、可选 normal 的存在性、全部数值和负零。4 个独立单测还覆盖非零承载、反向 body pair、重复接触、空 tick、未知字段拒绝和数值篡改拒绝。

**当前 dict→列存并不提速**：1.302+0.006+0.070 s 比原 JSON 编码/压缩 0.729 s 更慢。不得把文件变小写成端到端已加速。原生 buffer→数值列存、数组上的独立承载/穿透审计尚未实现或测量；反建 dict 的成本也说明只在末端换扩展名不足。

## 下一步可以单独冻结的改动

1. 先实际测已冻结 wave 并行入口，保持原每场景/四型次序、候选、控制和门不变。并发吞吐不能由这份单进程离线结果承诺。
2. 新 opt-in runtime 直接从原 `contact_data_to_host()` 数值 buffer 保存每个有效原始区间/所有 pair、counts/starts、路径和可选 normal，使用 binary64 或保留原生 dtype 的无损列存；全字段离线对照、同一 contact 顺序/overflow/finite 校验和全部门仍需保持。独立 audit 直接读数组，避免 dict 创建、JSON 往返和反建 dict。旧冻结模块不改，新入口须绑定来源和新 schema。第一阶段应只转换旧数据并逐字段核验，再冻结执行小样本。
3. 更小的存储改动是新 journal 把原 JSON byte chunk 送有界压缩队列，保存全部原始行；结束前必须 join、处理后台异常、关闭 gzip、核 data SHA，随后才允许有效 label/receipt 和 Kit.close。本次 gzip 写入只占样本编码+压缩约 22%，不能声称仅异步 gzip 会产生大倍数收益。
4. 源网格固定顶点有限性可在绑定加载后审计一次并设只读；最低/最高点仅依赖世界 Z 投影，可探索只算 Z 而不分配完整 XYZ。此候选可能改变浮点运算顺序，当前未测也未验收；必须核原所有 source bottom、阈值邻域和逐体标签，不能直接替换成粗代理、少顶点或低频检查。
5. Kit/URDF/凸件建场复用可另外评估，但当前 43 s setup 不是纯 launcher 时间，不能承诺可全部消除。需要区分初始 cold recreation 与试抓期间禁止 reset，保留所有初态/原生几何/接触配置审计；本轮没有启动 resident scene 实验。

新计时应在真正调用 `sim.step(render=False)` 周围以及 `after_step` 内分别计时，另外记录原生 buffer host transfer、dict/数组转换、承载/源底、编码、压缩写入、finish/最终 SHA；先修计时归属，不能继续把外层成本归给 PhysX。所有这些属于新 opt-in 协议，未改动本轮旧数据、控制、物理或标签。

轻证据见 `evidence/author_contact_performance_20261115_v1`；大样本/二进制/cProfile 原文件在忽略目录 `outputs/boya_author_contact_profile_20261115_v1`。这是旧数据离线诊断，不是新采集吞吐或新成功率。
