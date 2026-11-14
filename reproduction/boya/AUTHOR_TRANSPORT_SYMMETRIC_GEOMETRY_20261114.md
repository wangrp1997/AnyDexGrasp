# 搬运刚体预检对全部源物体对称

只读复核发现，旧 transport 闭合后的 rigid 几何门只检查兼容 primary 以外的物体与箱体。primary 原来只在 seed 的开手路径中检查，因此改变兼容对象身份会改变闭合后机械臂的几何准入范围。全体动态对象的运行时接触门虽然仍在，这不能称为完整的全场景 post-arm 规划预检。

新 `scene_collision_obstacles` 明确分开两组：开手保持原 seed target + 其余源体/箱体；闭合后 rigid 使用全部3件源体在冻结初态的 source CoACD convex union + 5个箱体盒。post 的 self/floor 审计仍调用一次原 ArmPathGeometry，原默认1e-7米门、POLICY、手指端点、24独立命令列、闭合/退回/搬运/释放动作与运行时各门均未改。新报告列出每个 proxy object ID，并声明没有排除 primary。

这仍是每条命令的**初态凸体代理预检**，不是物体后续动态轨迹预测、连续扫掠保证或精确源CAD。闭合后 fingers 对物体的接触、finger-vs-rigid 自接触和实际动态源体位置继续由原生运行时逐步检查；没有附着或物体状态重置。

15 tests 通过。新增反例使用原 ArmPathGeometry/coal：将 rigid palm 置于 primary 内部，旧 extras-only 门错误放行，新全体门拒绝；切换3个 primary 身份后，9个逐物体内部 probe 均拒绝，远离物体的 probe 仍通过。

另对实际 v4 observed scene 三源资产只读构建：000/002/005 分别2/17/3个原规划凸部件，加5个箱盒共27部件。切换兼容 primary 后，post 全部 vertices/faces 的 SHA 完全相同；原 coal 门对9个真实源凸部件内部 probe 全部拒绝。该数值核验14.961秒，0 native规划、0 Kit/物理/CGR/训练/标签。原 c54 transport 与首次失败的 9d263 release 工具旧源码已收入新轻证据，原冻结证据不回写。

输出为 `outputs/boya_author_transport_symmetric_20261114_v1`，轻量证据为 `evidence/author_transport_symmetric_20261114_v1`。本次只修适配门的身份依赖，不给出新的物理成功率。
