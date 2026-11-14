# 固定 release 终点：单 HOME seed 诊断

本阶段完成，结果为 `complete/null_geometry`，未进入搬运 RRT 或物理执行。没有修改既有 transport/pipeline、固定 release sidecar、三路 null 结果或任何碰撞门。

先冻结独立协议 `outputs/boya_author_fixed_goal_probe_20261114_v1/PROTOCOL.json`（SHA `96bca7a61bd1a3a753fe52ef269f47bf83edf479909b92043107be065df7d71c`）及源码快照，再调用一次原 `arm_only_palm_ik`，预算 max_nfev=300、bounds margin=1e-6，与现有 transport 相同。目标仍为原 SHA `43223161bf53e4fa40335960e1b51185320646c8ee38f03793f2bea4fed7522c` sidecar 的固定 XYZ `[-.4,-.25,.35]` 和 home-FK 旋转。仅六个 arm seed 换为原 reference HOME；g1 最后 `transport_bin` 行（0 基 index 4519、紧邻开手前）的 18 个非 arm 独立值及展开后 22 个非 arm 值全等。未使用已经开手的 release 行。

一次 IK 在 13 nfev 收敛，位置误差 `2.2123443632707514e-11 m`，旋转误差约 `7.2254e-11 rad`。使用原 10 个 rigid link、自碰排除、地板、三个源物体共 22 源 proxy 及 5 个 bin box，保留原 `1e-7 m` 门。原直线终点有 6 项违规，最深 upper_arm↔wrist_2 为 69.001mm；HOME seed 的新分支消除这些自碰，但仍有上臂与 bin x 正墙 13.358mm、y 正墙 4.887mm 两项相交。因此没有生成 `FIXED_GOAL_ARM_ENDPOINT.json`，不能作为后续 carry RRT 的合法终点。

最深相交另用原上臂 collision mesh 的 2738 个三角面与真实 bin box 的 12 个三角面构建 Coal BVH，表面相交为真。旧终点的上臂与 wrist_2 原源三角面相交也为真。此检查没有凸化源三角面；报告中的毫米深度来自原 convex gate，不冒充精确三角面实体穿透深度。独立 XML/NumPy FK 与求解返回掌矩阵最大差 `1.1102230246251565e-16`，原 SRDF HOME、固定 sidecar、闭合 18 列及单次预算均通过复核，审计新增 IK 为 0。

新独立模块 `author_fixed_goal_probe.py` SHA `61f70d975a8e53de3b803995f2cadb100e3bee32c604fcd86546113d89ffb31a`，CLI 分为 `freeze` 和只允许一次启动的 `run`。三项测试通过：闭合边界、恰好六 arm 坐标替换，以及真实三角 BVH 对相交/分离源面的区分。轻证据在 `evidence/author_fixed_goal_probe_20261114_v1/`，包含冻结协议、原结果、两份 endpoint geometry 和独立数值审计。

成本：冻结 3.237s，单次诊断及几何检查 3.443s；实际 1 IK / 13 nfev，0 额外 seed、重试、Kit、物理、训练或标签。三路历史 null 保持，当前固定终点尚未通过机器人与箱体共同几何验收。单 seed 的 null 不证明全局不可达；后续若开展共同 workspace 标定，须另冻结新协议，并保留本次 sidecar 和负结果。
