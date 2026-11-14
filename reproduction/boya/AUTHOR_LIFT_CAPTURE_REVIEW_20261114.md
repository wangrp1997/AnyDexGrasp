# 无箱三物体观测：独立只读复核

复核对象为 `outputs/boya_full_data_lift_20261114_v1/scene/capture`。结论通过；未修改冻结源码、原观测、场景或原回执，未重跑 Kit、物理、IK、CGR、训练或抓取。可复算审计脚本及轻证据在 `evidence/author_lift_capture_review_20261114_v1`，本次审计耗时 18.808s。

场景严格为新 lift schema/task，显式 `bin=null`，完整原物体为 000/002/005。当前冻结代码据此产生零箱体视觉／碰撞／射线几何；实际保存的静态碰撞路径仅 `/World/ground/collisionPlane`，全接触记录也无箱体路径。三体质量均为 float32 的 0.1kg、摩擦为 float32 的 0.6，持续向下 0.98N 的源代码与绑定保持。没有把静态干扰物称为动态多物体。

独立读取 `751×3×7` 全 pose 和 `751×3×6` 全 twist，有限值及单位四元数检查通过，确有 750 ticks／2250 object steps。末 250 帧稳定性各指标独立重算与回执一致：最大平移漂移为 8.691／1.564／2.289µm，最大角漂移为 75.303／23.003／108.589µrad，均通过原门。observed_scene 的末 pose、twist 以及世界 SE3 与最后一帧完全一致。执行仍明确独立 cold recreation／初始零 twist，不能称与本次动态初始化同一进程连续观测。

660MiB 原接触 JSON 以有界内存流式读取：750 帧、1,530,750 条接触全部核验，时间步连续，所有接触身份及数值有效。三体末 250 帧分别都有 250 帧地面支撑，平均支撑力分别为 0.979992／0.979995／0.979998N；全记录最大穿透约 0.710µm。接触总数包含零载荷的候选点，未把所有记录都称为受载接触。

native 审计实际 1633 hulls，三体分别 955／370／308，与 schemas 启用形状数一致，没有禁用形状。全部 4899 个 source／cooked／native-plane 数组维度与有限值通过；独立重算 cooked 顶点相对 native 面的不等式，逐 hull 与回执误差为 0。原 source↔cooked 的 20µm 导入门通过，最大 source-outside-cooked bound 为 19.895µm。此处没有重新运行原 convex-distance solver；20µm 只指原生导入保真，不能把粗凸分解称为精确源 CAD，也不能把它当接触求解器验收。

RGB PNG 与原生 RGB 数组逐值相同；深度 PNG 与原深度按冻结有效域／毫米量化逐值相同，558859 个有效像素；meta 与 native intrinsics/depth factor 一致。独立实现 Moller–Trumbore，对三物体完整源三角面加解析 z=0 平面重算原 96 条射线：37 条首命中物体，期待深度与原诊断最大差 `6.661e-16m`，与 RTX 深度最大差 `6.899µm`。本次 render 积分为 0，初始化物理成本单列；没有新增抓取标签。

53 项输入及 519 项源码，共 572 项 SHA 全通过。初始 scene、observed scene、observation、原生审计、接触检查与 RGB-D 文件绑定通过；全部 16 个 capture／source-scene 文件在审计前后 SHA 不变。新 observed_scene SHA 为 `0a0c971968af02760e411244e62bda12c98658d178d6cd583c96f4b6cce7ed92`；observation SHA 为 `d239736a2322b9ea72ad62f2bd8c6044ff8a2c625b07e9436e9af73abae48f60`。

这是无箱开发初始化与观测的验收，不是新抓取成功率、正式采集完成或重训结果；旧 435 不因此变成新三物体标签。
