# 无箱新首选四型规划 null：只读审计

`outputs/boya_full_data_lift_20261114_v1/g{1,6,7,8}` 均在 source 4 cm pregrasp 的有限 IK 阶段被原 native 拒绝。四型各原 8 个声明 seed，共 32 个 IK 请求，每请求 cap300；实际 nfev、失败终点和残差未被原失败回执保存。0 可执行计划、0 HOME driver、0 lift 准备、0 实际抓取、0 新标签、0 训练。初始化物体步属于此前观测成本，不是试抓。

对照为旧 `outputs/boya_author_pipeline_fourtype_20261114_v2/prepare_g{1,6,7,8}` 的有箱 raw879。新无箱首选均为 raw1538，extra0；旧首选为 raw879，extra10mm。它们是不同冻结候选，不能把旧候选能得到 plan 当成新候选必须能得到 plan 的对照。

| 型 | 新旧掌位置距离 m | 新旧掌方向差 ° | 旧 pregrasp q 对旧目标位置误差 m | 同一旧 q 对新目标位置误差 m |
| --- | ---: | ---: | ---: | ---: |
| g1 | 0.250171 | 170.628495 | 9.436e-12 | 0.291138 |
| g6 | 0.260109 | 170.628495 | 6.361e-12 | 0.298425 |
| g7 | 0.256688 | 170.628495 | 7.153e-12 | 0.296787 |
| g8 | 0.248858 | 170.628495 | 9.262e-12 | 0.290203 |

现存输入核验未发现接线或依赖回归：

- 四型 native 参数语义一致，仅 pool/输出/worldpose 文件路径和进程时钟不同；原 8 seeds、完整 28 关节初态、开闭手目标、base transform 逐值一致，4 cm 和每 seed 300 cap 不变。
- 新旧 camera `T_WC` 及三体观测 pose 最大误差均为 0。compatibility primary 的 source object 与 worldpose 是同一矩阵；独立重建原 placement 后 native palm 与 bridge palm 最大误差为 0。没有 GT 目标变换偷偷移动候选。
- 原 native prepare、IK、ranker、bridge、PlanningModel、URDF FK 六份源码，在旧 scene 源锁、新 scene 源锁、当前文件之间 SHA 全部相同；URDF、physics、diagnostic 与旧/新 frozen input 也一致。
- 只读重算旧 plan 的末 HOME q 和原 8 seeds 共 36 个现存 state FK；用禁止 `least_squares` 的 guard 确保未重新求解。旧 q 能精确复现旧 pregrasp 目标，对新目标有约 29–30 cm 和 170.63° 残差；这说明旧解不能直接代表新目标的解，不证明新目标无解。

失败发生于 `rank_arm_ik_solutions` 原种子列表全部局部不收敛后，尚未进入 HOME/approach 碰撞门。因没有新命令数组，无法声称本轮真实闭合或 lift 命令已执行/验收；已有合成入口测试仅验证绑定逻辑。不能由一个新场景四个相关首选的 null 推断整条采集链不可用，也不能报训练后的成功率。

本审计没有新增 IK、候选、规划、CGR、Kit、物理或标签，没有修改冻结源码、输入或队列。下一项应是独立冻结多场景采集协议和入口，保留无计划成本；本轮不为该姿态增加 seed、调阈值、换候选或派发成功率评测。

轻证据：`evidence/author_lift_null_readonly_20261114_v1/AUDIT.json`、禁止优化的 `audit.py` 和原输出日志。审计脚本自身成本为 36 次纯 FK，实际原四型规划成本单独保留。
