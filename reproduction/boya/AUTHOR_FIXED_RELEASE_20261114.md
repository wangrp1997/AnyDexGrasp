# 固定投放位姿：去掉兼容物体真值耦合

旧 `_waypoint_targets` 用 `palm.xy + bin.xy - compatibility_object.xy` 计算投放位置，并沿用候选朝向。这会让允许任一源物体被抓取的任务，隐式依赖兼容主物体的真值位置。新接口完全删除这项计算，必须显式传入已冻结的固定 world→palm SE3；缺少配置直接拒绝，不再从物体、候选或文件名猜默认终点。旧 scene、观察、队列和结果未修改。

`prepare_transport(..., release_spec_path=...)` 使用 `load_release_spec` 验证 sidecar 的 scene 绝对路径与 SHA、完整 bin 的规范 JSON SHA、来源文件 SHA、world/palm frame、固定 SE3 规则、proper SE3 和明确的无物体/候选依赖声明。该 sidecar 及其来源被写入 plan.extra_input_sha256，执行前再次核验。`_waypoint_targets(palm, direction, approach_distance, release_palm_pose)` 只接收固定终点；candidate 仅决定原接近方向的两个 retreat 起点。

新工具 `author_release.py` 用固定 world XYZ `[-0.4,-0.25,0.35]` 和原参考 arm home 的 palm FK 旋转生成独立 JSON。它不读取任何 candidate，不用 scene.objects 计算位姿；scene/bin 只负责绑定配置适用的观察与箱体。工具输出必须是新文件。

公开来源经源码 SHA 与 AST 核对：`ur.py:235` 的 throw_pose 只按 global_cam 选择两组固定 XYZ/旋转向量；`throw:252` 移动到该 pose，实际 `grasp_and_throw:406` 调用 throw，407 随后 open。sidecar 保存两组原数值与调用行号。**home FK 旋转和上述固定 XYZ 是明确声明的 Boya 模拟器坐标适配，不是作者原 TCP 数值，也未声称可达或物理验收通过。** 实际配置由 collector 冻结工具来源后，针对执行用 observed_scene 单独生成；root 使用独立 XML/NumPy FK 核验旋转。

本模块 13 tests 通过，包含改变 primary 身份、所有源体 pose、candidate XYZ/旋转后固定 release SE3 完全相同；缺失、错误 scene/bin、非法旋转/非有限值、物体/候选依赖、错误 frame、空来源及来源 SHA 改变都拒绝。工具不变量测试中 home FK 使用显式 mock，仅验证接口依赖与固定规则，不冒充实际 FK 或物理证据。

原 POLICY、退 4mm/等 0.1s/退 pregrasp、原 close/hold、20 手关节与 24 独立命令列、初态、限位、native 碰撞和完整承载/投放标签门保持原设置。该 agent 本轮新增真实侧边配置生成、规划、Kit、物理、CGR、训练、标签均为 0；后续集成结果另存。轻量来源与测试在 `evidence/author_fixed_release_20261114_v1`。
