# 关节空间 HOME 绕行与整段时间缩放

g1、g6、g7 已得到原门通过的纯规划 HOME；g8 保留碰撞 null。全部使用原 v2 的同一候选、初态和预抓关节终点，无新 CGR、IK、物理或标签。此结果只允许继续准备搬运段，不能当物理抓取成功。

原六条笛卡尔路线失败后，另立 `author_joint_home_route.py`，运行前冻结四型共用的 RRT-Connect：seed 20261114、六臂未环绕欧氏步长 0.25 rad、每 edge 最大单关节采样间隔 0.05 rad、goal_bias 0.2、最多 1200 iterations、每 connect 最多 32 extend，树从 start/goal 交替。只取首次连接，不 shortcut、不换候选、不追加搜索。

每个 edge 与最终每个 `.004 s` tick 均使用原 ArmPathGeometry/Coal，含三体全部源 CoACD 凸包、五箱体盒、原机器人 self/floor，容差 `1e-7 m` 不变。固定非 HOME 段先检查；原 50 settle、500 approach 与全部 18 个非臂主动坐标严格保留，20 手关节仍由原 16 电机和 mimic 关系控制。

四型首次搜索均在第 28 iteration 连通，分别 14/12/13/15 个路点。实际搜索各约 0.65–0.68 秒，四路准备墙钟分别 18.297/17.088/17.760/18.880 秒（并行执行，求和 72.024 秒是准备成本，不是并行经过时间）。全部首次结果仍是 null：g1/6/7 完整 tick 几何通过，但两点 quintic 分段各自过门后，拼接整段有限差分 jerk 为 32.127/31.902/32.078 rad/s³，超过原 30；g8 除 jerk 32.279 外，还有 15 个 ffdistal–graspnet_000 违规 tick，最深 0.415224 mm。

g8 表明 0.05 rad 离散 edge 检查会漏过细碰撞；最终逐 tick 门确实拒绝，没有宣称连续无碰撞或放宽容差。四个原 null、源码 cc390e0363696c5c65c368672a385c885f5ccb4ba450bdb1323d7766b2a0fcc6、树、首次路径、候选数组与成本全部保留，不回写。

拼接 jerk 是时间参数实现问题，随后使用独立 `author_home_retime.py`，先冻结新协议，只复用 g1/6/7 已几何通过的原 connected_path。统一按 `max(v_ratio, sqrt(a_ratio), cbrt(j_ratio))*1.01` 伸长每条原 edge 的时长，最多含原失败共八次。三型均只需一次新时间伸长；不重新 RRT、不改变任何路径坐标、端点、seed、候选或原几何门。g8 无新准备，不借重采样消除原碰撞。

| 手型 | 新 HOME ticks | 全前缀 ticks | 最大加速度 rad/s² | 最大 jerk rad/s³ | 结果 |
|---|---:|---:|---:|---:|---|
| g1 | 2180 | 2730 | 2.0071 | 28.9407 | ready |
| g6 | 1887 | 2437 | 2.0178 | 28.7985 | ready |
| g7 | 2025 | 2575 | 2.0271 | 28.9906 | ready |
| g8 | 无新准备 | 原 null 保留 | — | — | null |

新完整全场景检查共 7742 tick，三型零违规，最大实际命令速度不超过 0.472 rad/s，原关节限位及速度通过。重定时准备墙钟分别 15.840/14.695/15.510 秒（并行，求和 46.045 秒）；新增 search/IK/Kit/物理步/训练更新/标签全部 0。

独立保存数组审计逐值核对原 50 settle 与 500 approach、HOME 末条 q 与 desired_palm、18 列非臂命令，以及 connected_path NPZ 字节 SHA。还按原每段两点 quintic 公式重建新采样，误差小于 `1e-13 rad`；证明只改变同一 polyline 的时间采样。独立 XML/NumPy FK 检查全部 7742 条，最大 desired-palm 矩阵误差 `1.4292e-10`；从保存数组重算完整前缀的原 XML 限位/速度与 3 rad/s²、30 rad/s³ 均通过。root 另有独立全前缀与 HOME 导数审计，输出 SHA 也绑定。

该规划使用原源凸包并集；002 native collision_refinement 与物理 cooking 保留在原观测/源绑定中，未替代为 native cooked 几何。此处几何门是源代理与采样路径检查，不能代替真实驱动跟踪、动态多体接触、承载、搬运投放和完整七门。

5 个 RRT tests 包含阻挡矩形的实际二维绕行、确定性重复、edge 采样和预算耗尽；3 个 retime tests 真实调用原 timed_segment 复现拼接超 jerk 后验证统一时间伸长修复。原 HOME 合同 7 tests 继续通过。最终 frozen RRT/home/transport 均未为 retime 修改；独立 retime 源码 SHA 为 `0beb0b8aa0640d6ced6c9ed3a7f3ad500fd6f335ccac17eaa8699986507bf884`。

ready 入口为 `outputs/boya_author_home_retime_20261114_v1/g{1,6,7}/native`，新 plan 绑定原 v2 seed SHA、原 RRT 路径与失败、全部 input/source、scene archives 和新 retime 协议。轻证据在 `evidence/author_joint_home_route_20261114_v1`，大树/数组/全日志仅保留在 outputs。后续固定 release 的搬运准备另记，本报告不将其尚未完成的结果计入。
