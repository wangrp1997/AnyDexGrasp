# 固定六路线 HOME 纯规划结果

四型均为 `null_geometry / frozen_home_route_budget_exhausted`。这是有限规划失败，不能据此认定目标不可达；没有新抓取或成功率。

输入固定为 `outputs/boya_author_pipeline_fourtype_20261114_v2/prepare_g{1,6,7,8}/arm/native`，不重新采样、CGR、候选选择或求原预抓 IK。三物体观测沿用 v4，历史 scene/transport 源码通过精确 SHA archive 解析。兼容物体、实际候选、HOME 初态、预抓关节终点、18 个非臂主动坐标、50 settle 与 500 approach 行不变；approach 保持作者采集 4 cm。

在搜索前冻结统一策略：抬高到 `max(home_palm_z, pregrasp_palm_z)+{0.15,0.30}` m，世界 Y 偏移 `{0,-0.15,+0.15}` m，按上述顺序最多六路线，每路线两次原 300 nfev 六臂 IK，首次通过即停。原预抓关节终点不重解，原 timed_segment 限位、速度、3 rad/s²、30 rad/s³及 `.004 s` 不放宽。

全场景门包含三体全部源 CoACD 凸包（22 部件）和五个箱体盒，原机器人 convex hull、自碰撞和地面检查，signed-distance 容差保持 `1e-7 m`。四型各 551 个固定状态（50 settle、预抓端点、500 approach）全部通过。原 1300 行全场景审计分别发现 808/763/795/694 条违规，均在 HOME、均为 graspnet_000；因此没有把原单兼容体检查误写成完整场景可执行。

| 手型 | 实际路线 | 实际 IK 调用 | 结果 | 准备墙钟秒 |
|---|---:|---:|---|---:|
| g1 | 6 | 10 | null | 12.093 |
| g6 | 6 | 10 | null | 11.514 |
| g7 | 6 | 10 | null | 11.354 |
| g8 | 6 | 10 | null | 11.643 |

每型两路线首个 IK 未收敛，另四路线第二 IK 未收敛，未生成可定时的新 HOME 数组。合计 40 次调用、46.605 秒；16 次成功返回共报告 128 nfev，24 次失败的原函数异常不含实际 nfev，只能声明各上限 300。未调整参数补跑。该墙钟不含前置独立预检、单测与审计。

独立 XML/NumPy FK 核对原四型共 5200 条命令，最大 desired-palm 矩阵误差 `1.611e-10`，HOME 终点误差不超过 `2.221e-16`；独立按原 XML 关节限位、速度及原有限差分加速度/jerk 复算通过。每条失败、路点目标、已返回 IK 误差/成本、原输入 SHA 与冻结先后均保存。无候选新数组，故不存在已验证的新路线跟踪或动力学表现。

002 的 native `collision_refinement` 保留并绑定，但规划继续使用原 object_asset 的源凸包并集，不替换为运行时细分/修复子凸体。v4 的 1633 native hull 与 20 µm cooking 保留审计是另一层证据；不能声称规划已经使用 native cooked hull 或精确源 CAD，也不能把源凸包近似误当接触真值。

旧运行源码 SHA 为 `d6acc99c7ca1b3baa3959d6c53d2572515adb2737c2564241d0d660ac56bae0c`，每型独立输出保存其原源码快照。结果均 null 后仅完善了 HOME 末条 desired_palm 逐值保留原值与输入数组不被修改的合同，最终源码 SHA `1bf86feaccd7880128b54ace69b1289027d313e7f8a641bc44ef6aa00d0ff0d8`，7 tests 通过；不回写或重跑旧证据。初测试 fixture 期望错误、独立审计初版速度上限读法及其修正均在 ERRATA 和原输出中保留。

源码 `author_home_route.py`；接口 `prepare_home_route(seed_plan_dir, scene_path, output, *, config_path, archive_sources=None)`，config_path 为完整 BACKEND JSON。`ready` 需要所有新 HOME tick 及原非 HOME 段全场景门通过。轻证据在 `evidence/author_home_route_20261114_v1/`，原日志与大输入摘要在 `outputs/boya_author_home_route_20261114_v1/`。本阶段 0 Kit、0 物理步、0 训练、0 新标签；后续关节空间策略使用独立模块、协议和输出，不覆盖本轮失败。
