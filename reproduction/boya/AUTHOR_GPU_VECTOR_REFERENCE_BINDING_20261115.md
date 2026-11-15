# GPU 初始化引用修复：原参数不变

GPU v1 的单环境启动在 `create_scene → audit_hand_actuation → hand_drive_parameters` 处读到了不存在的旧默认 `CanonDex/third_party/tendon-spin/assets/grasp/scene.xml`。v1 原回执保留：1 次 Kit 启动、0 物理步、0 命令环境步、标签为空；这不是 GPU 不支持原物理的证据。

根因是新入口先 `load_physics_config`，其配置类初始化提前导入 `hand_actuation`；模块用 `from canondex.paths import HAND_SCENE` 捕获了原默认。原 generator 中的 `bind_hand` 到此已太晚。v2 只在 AppLauncher/physics 导入前调用既有 `run_template_arm_positive.bind_hand`，输入仍是冻结 plan 的 hand_urdf/hand_scene。文件或 reference 目录缺失、共享 lane 绑定不一致、调用者已经导入不同默认时，均在 Kit 前拒绝。

源码 `author_gpu_vector_execution.py` SHA256 `263cf9fd7b146da66a31bd43c70d43bac01361c9904b627cc03bbd357d092e0d`；测试 SHA256 `2bf41ec9a1f75a35629113a4fb98aea8db7889c51bdcbabcd12ff1a894a06e01`。原命令、物理参数、接触记录、门、标签、CLI/schema 和预算均未更改。CanonDex/旧冻结模块及旧输出只读。

11 项 CPU tests 通过（1.377 秒）；新增 clean subprocess 真实读取原 scene.xml，确认 physics 导入后的 paths/hand_actuation/函数默认引用相同，并逐值比较默认与显式 hand_drive_parameters 完全一致。missing scene 在 Kit 前失败。原 App.close 直接退出前异常回执落盘测试仍通过。本 agent 本轮 0 Kit/物理/IK/CGR/训练。

v2 允许 root 用独立新目录冻结后恢复未消耗的原预算：最多 6398 shared ticks、9422 原命令环境步、9772 实际环境积分步。v1 回执不覆盖，未实际完成的试次不产生失败标签；是否真正通过 GPU 状态/接触/原门仍由实际运行与独立审计回答。这里未声明 v2 物理通过。

证据见 `evidence/author_gpu_vector_reference_binding_20261115_v2/`：保留实际 v1 错误绑定、CPU 日志及精确 v1→v2 补丁；不回写 v1 已冻结证据。
