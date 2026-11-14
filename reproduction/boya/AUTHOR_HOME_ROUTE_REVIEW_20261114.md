# HOME 路由及采集接入独立只读复核

本复核只读检查 author_home_route、author_joint_home_route、author_pipeline 和原 ArmPathGeometry；没有调用规划入口、Kit、物理或训练，也未修改这些路由/集成源码。被复核源码 SHA 另存轻量 REVIEW.json，后续 owner 的变化由其新冻结证据负责。

两类 HOME adapter 均保持原初态和 pregrasp 24独立坐标终点。笛卡尔方案仅解6臂关节，再把原终点直接拼入；joint RRT 零IK，仅搜索6臂空间。18个非臂独立列（2手腕+16电机）由原初态保留，4个 mimic 由原模型派生。replace_home 检查终点逐值相等，保留 settle 和500条 approach；pipeline 另核 shape/dtype/bytes、18列不变、原 desired endpoint、候选元数据和连续4ms时钟。

fixedsegments 先检查全部源体原CoACD convex union、5个箱盒和全部robot collision links的原self/floor门；端点或非HOME段不通过就不启动搜索。新路径按原时序限制生成，最终每条实际命令再用原 ArmPathGeometry 审计；joint RRT 的粗搜索边不是最终准入依据。精确值缓存仅缓存固定18列下的相同6臂状态，不做近似阈值放宽。源planning hull union与native物理partition refinement分别绑定，未把20µm导入门当精确源CAD。

collector新接入仅复用原固定首候选、pool、seed和已准备路由，不重新采集/选择候选或重解终点。它验证路由协议与源依赖，补齐collection metadata时不改变原24命令列；固定release sidecar与scene/bin/SHA均显式绑定。执行前完整核验，EXECUTION_STARTED独占创建，最终标签/费用/异常回执在SimulationApp.close之前保存。未发现新的标签阻断项。

复核中发现旧 transport 的 post rigid 门漏 primary，已按root另行授权在独立补丁中修成全部3体初态proxy；详见 AUTHOR_TRANSPORT_SYMMETRIC_GEOMETRY_20261114.md。本复核不把更长HOME对应的物体动态误称为静态预检保证。物体在HOME期间会自然沉降/漂移；闭合后手指接触、finger-vs-rigid自接触与物体实际轨迹必须由既有逐tick原生接触、非手支撑、穿透、源网格pose及完整投放门判定。当前没有新的物理成功率或完整任务验收结论。
