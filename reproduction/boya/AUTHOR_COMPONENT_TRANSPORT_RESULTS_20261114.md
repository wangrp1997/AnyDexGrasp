# 固定三路搬运准备与原三角面诊断（2026-11-14）

三路新搬运纯规划全部 `null_geometry`，0 ready、0 基础设施失败、0 物理抓取、0 新标签。局部 IK 全部返回有效解，拒绝发生在完整命令几何检查。它不是抓取成功率，也不能将无计划记录写成训练负例。

本轮独立协议 SHA `47f65a0736026b98958b6fef8a23f6ae30382a0a637541c88f4001e6e2c5be02`。结果目录为 `outputs/boya_author_pipeline_home_release_20261114_v1/`；轻证据在 [author_component_transport_20261114_v1](evidence/author_component_transport_20261114_v1/EVIDENCE.json)。完整阶段背景由 [集成报告](AUTHOR_HOME_RELEASE_INTEGRATION_20261114.md)维护。

## 固定输入与实际成本

复用旧 v2 的 g1/6/7 原候选、原预抓 IK 和新 retime 后的全场景 HOME；原 raw879、rank73、nonflip、额外深入10mm、seed20261114、4cm接近不变。原 387 次采集碰撞查询直接复用，没有重新采 RGB-D、运行 CGR、评分头或原采集器。g8 的原 HOME 碰撞 null 保留，未追加准备。

显式固定投放 sidecar SHA `43223161bf53e4fa40335960e1b51185320646c8ee38f03793f2bea4fed7522c`。位置 `[-.4,-.25,.35]m`、固定 HOME FK 方向是声明的 Boya 适配，不依赖物体/候选真值。每路只调用一次新 transport prepare，无重试、换候选、移箱或放宽门。

| 手型 | 完整命令步数 | 原搬运 IK 开始/返回 | IK 内部 wall(s) | 准备内部 wall(s) |
|---|---:|---:|---:|---:|
| g1 | 5270 | 96/96 | 4.270 | 26.739 |
| g6 | 5266 | 96/96 | 4.678 | 27.162 |
| g7 | 5259 | 96/96 | 4.436 | 26.760 |

三路各3段×32个原 IK 调用，全部使用原 `max_nfev=300`；共288次开始/288次返回/0异常，原 IK wall 合计13.385s。透明计数器只转发原参数及原返回，不改数值行为。三路并行墙钟27.639s，不能将各内部时间相加当墙钟。0 新 HOME 搜索、0 原接近 IK、0 Kit、0 物理步、0 训练更新。

## 几何拒绝的确切归属

原 retime `seed_target_geometry` 已包括3体全部22个凸部件及5个箱体部件，自碰及 floor 同时检查；样本数等于各自50步 settle＋HOME＋500步 approach。transport 另对25个 extras 部件重复做开手门，并同时保留原 full-scene receipt，三路均通过。因此这里没有漏掉 retime 全场景检查。

post 刚体门排除手指可接触 link，使用3源体的冻结初态代理与5箱体；没有预测抓住后的动态物体轨迹。三路闭合、保持、退4mm、等待及退回预抓均无刚体违规。所有违规均在 `transport_bin/release/release_settle`，全部涉及 `upper_arm_link`。这里的计数是“采样状态×违规对”，不是独立试抓或碰撞事件数。

| 手型 | 箱x正墙 | 箱y正墙 | 自碰 | transport_bin | release | release_settle |
|---|---:|---:|---:|---:|---:|---:|
| g1 | 1121 | 1005 | 4185 | 1811 | 1500 | 3000 |
| g6 | 1273 | 1140 | 4857 | 2770 | 1500 | 3000 |
| g7 | 1195 | 1068 | 4505 | 2268 | 1500 | 3000 |

原3源体代理违规均0，floor违规均0。箱碰索引24/26分别为 x正/y正墙；22是箱底、23/25是负向墙。自碰对象只有 `wrist_1_link/wrist_2_link/wrist_3_link/fabase`。完整分组、首次/末次/最深样本及全命令索引见 [POST_GEOMETRY_BREAKDOWN.json](evidence/author_component_transport_20261114_v1/POST_GEOMETRY_BREAKDOWN.json)。原 post sample 从闭合开始计数，证据显式映射回完整命令，避免错报 HOME 或接近阶段。

首次违规均为上臂碰箱x正墙，g1/g6/g7分别发生在完整命令0基4149/3993/4064，16.600/15.976/16.260s。最深均为上臂与腕2自碰，代理有符号距离分别为−69.001139/−69.005381/−69.001139mm。

三路末态均仍有相同六对违规：上臂对腕1/腕2/腕3/fabase，以及箱x正/y正墙。各对代理深度约14.207/69.001/51.033/5.833/10.139/1.318mm。这是当前局部 IK 释放分支的末端违规，单纯绕过中间路径不能让这组末端状态准入；不据此声称固定掌目标所有分支都不可达。

## 原始三角面核验

只读取保存命令，每型核一个最深自碰状态，及释放末态两面箱墙，共9个代表查询。独立使用原 combined URDF 的 collision 三角面及声明箱体盒面，未使用 whole-link convex hull 替代三角面；没有运行新的规划或物理。

三型上臂2738面与腕2的1782面在各自最深状态均实际相交；三型最终上臂与两面箱墙也全部实际相交，9/9代表查询均为真。两机械臂 mesh 均 watertight；箱墙每个12面。诊断内部 wall 0.119s，源码和输入 SHA 见 [SOURCE_TRIANGLE_DIAGNOSTIC.json](evidence/author_component_transport_20261114_v1/SOURCE_TRIANGLE_DIAGNOSTIC.json)。

这排除了这些代表违规仅由 whole-link 凸包填凹造成的解释。原 planner 代理深度不是原三角面穿深，本审计没有测出三角面穿深，也没有逐个证明所有违规。实际 native 使用分区几何，而规划使用整 link 凸包，二者的保守性差异仍需分别说明；本轮没有改门或几何。

## 代码与完整性

冻结 `author_pipeline.py` SHA `2376c747bd2770e724e3bf4314af688a9a046245d386305c7b71b5373f77d2b0`；测试源码 SHA `353b9946f8e4e247303dc0963c580ec94ab4eb0714a1d2deef402e5299a98b17`（精确值以协议绑定为准）。25项入口测试0.491s通过，覆盖固定 sidecar failclosed、旧无binding计划的合法 continuation、18列/固定段保留、原碰撞修正及37 mesh来源锁，以及执行后完整性失败不得留下有效标签记录。

最终重新检查当前协议所有输入/源码及归档 SHA、JSON有效性和NUL字节，均通过，详见 [FINAL_INTEGRITY.json](evidence/author_component_transport_20261114_v1/FINAL_INTEGRITY.json)。旧流水线和新 continuation 回执分别保留，没有伪装重跑采集器。正式标签及模型仍为0新增；完整搬运小闭环尚未通过，正式重训及外部验收尚未开始。
