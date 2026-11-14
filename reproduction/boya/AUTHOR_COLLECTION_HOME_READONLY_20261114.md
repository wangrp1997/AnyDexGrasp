# 开发首场 HOME 与原公开起始语义只读核验

结论：首场四型被拒绝的原因是固定抓取姿态接近末段的指尖碰桌面，不是 HOME 初态碰撞。当前 native HOME 来自另一份 ROS 具名姿态，并非已验证与作者 ready 等价；但本场保存证据没有显示起始安装遗漏造成拒绝，不能据此更换 HOME、追加搜索或换候选。

原公开 `collect_dh3_grasp_data.py:43` 的 `--global_camera` 默认 false。默认分支在 391 行先 `movel(robot.ready_pose())`，397 行每次观测前再次到 ready，随后等待 0.5s 再读 RGBD。global-camera 分支在 380 行先到 `throwj2`，应与默认 wrist-camera 分支区分。548–550 行调用 `grasp_and_throw(...use_ready_pose=True)`；`ur.py:418–424` 因此用 `ready_pose()` 构造相机到基座映射。

`ur_toolbox/ur_toolbox/robot/ur.py:199–208` 定义的 ready 是硬件 TCP 位姿：默认平移 `[-0.299333229,0.002283554,0.603633973]m`，global 分支为 `[-0.570844481,0.006757302,0.620]m`，并各有明确旋转向量。`readyj` 在 95–97 行只是另一项关节常量定义，collector 没有调用它。`grasp_and_throw` 的 docstring 虽提 ready，其实际运动在 391 行直接执行 `[pregrasp,grasp]` 的 `movels`；起始 ready 由 caller 保证，不能只读 docstring 假定函数内部会回 HOME。`get_robot()` 在 collector 177 行将硬件 TCP offset 设为零；DH3 固定手/TCP及相机外参另在 `ur.py:110–144` 定义。

当前实际 plan 的 `initial_arm_state=null`，六臂初始值为 `[0,-1.12,-2.07,-3.08,-1.57,0]rad`（shoulder_pan/lift/elbow/wrist1/2/3）。原 native `run_template_arm_positive.py:147–152` 直接取 `reference.arm_home`，并加所选候选的开手状态。该 plan 的 `home.source_files` 明确指向 `/home/wangrenpeng/BoyaBlindGrasp/assets/sources/arm_home.srdf`，参数是 `group_state[@name='home'][@group='ur_arm']`；`canondex/robots/reference.py:124–151` 负责读取这一具名姿态。因此其来源是明确的仿真适配，而不是原 DH3 采集 ready 的等价证明。

独立 XML/NumPy FK 没有导入 native planning/model/FK，仅读取现有 URDF、原碰撞 mesh 和已保存命令。四型初始 `tool0` 平移均为 `[-0.269267856,0.174242812,0.475585332]m`，掌平移均为 `[-0.497811359,0.174424774,0.469994975]m`；原食指/中指碰撞三角面最低点在初态均至少 0.440m、在预抓均至少 0.032236m。初始命令与 plan.initial 逐坐标一致。保存的 full-scene 固定段报告对 50 settle、预抓终点均无违规。

硬件 UR 控制器 base/TCP 坐标与当前 URDF world/tool0、Boya 安装链没有通过这一审计建立相同坐标注册。证据 JSON 保留裸矩阵比较并明确标为 `coordinate_comparison_only_no_world_registration`；那些平移/旋转差不能直接当作安装错误。当前候选掌姿由实际 observation.T_WC、CGR 和 Boya reference 映射，`author_collect.py:224,420` / `author_action_semantics.py:208–216` 不读取 HOME 来构造世界候选；改变 HOME 不会消除同一固定末态的指尖碰桌面。

首场保存的固定段拒绝：

| 手型 | 违规数 | 开始 approach 命令索引 | 最深穿桌 | 违规 link |
|---|---:|---:|---:|---|
| g1 | 257 | 1154 | 5.861mm | mfdistal、ffdistal |
| g6 | 86 | 1214 | 1.499mm | ffdistal |
| g7 | 220 | 1166 | 4.801mm | mfdistal、ffdistal |
| g8 | 275 | 1146 | 6.750mm | mfdistal、ffdistal |

四型最深违规均在 sample550、command1299（0-based），即最后一个 approach 命令；没有 self、arm-object 或初始化 floor 违规。独立原碰撞三角面顶点变换的末态 min-Z 与四个保存的最深 floor 数值误差都是精确 0。凸包填充凹部不能解释这种平面最低顶点越界；这里是原 URDF 碰撞三角面本身低于 z=0。末态掌 FK 与 plan.palm 的最大元素误差 3.09e-13。四型均使用同一 raw816、extra0 首选，没有为诊断替换候选。

`author_joint_home_route.py:240–243` 在搜索前检查不可改变的初态/预抓/approach 固定段；一旦固定段违规即返回 `frozen_endpoint_or_non_home_collision`。该笼统 reason 包含 approach，不能缩写为 HOME 不可达。首场四型因此正确跳过 RRT；改变 HOME 或加搜索预算也不能让末态的同一手指几何越过当前门。

本审计 0 CGR、0 IK、0 RRT、0 Kit、0 物理、0 训练，不修改任何冻结源码、候选、协议、碰撞门或运行队列。其 0 成本只指这项静态核验，不覆盖主队列已经运行的真实初始化。后续固定场景队列继续由 root 按原协议执行，首场 null 保留。

轻证据：`evidence/author_collection_home_readonly_20261114_v1/EVIDENCE.json` 与 `STATIC_FK_AUDIT.json`。
