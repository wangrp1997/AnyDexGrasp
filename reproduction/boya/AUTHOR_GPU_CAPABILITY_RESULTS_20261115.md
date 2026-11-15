# 本机 GPU 物理与完整接触接口核查（2026-11-15）

本机官方安装提供 GPU PhysX、批量 articulation／rigid body tensor 控制，以及逐接触点、法向、法向力、距离和摩擦数据。**没有发现需要因“GPU 拿不到完整接触”而停止接入的源码依据。** 原 Boya 的两种 contact reporter 已走 tensor API，应优先复用现成路径。这里完成的是源码核查和 CPU 探针合同测试，尚未由本审计启动 Kit 或验证真实 Boya GPU 执行；正式大采集、训练均未启动。

## 官方安装与实际 API

安装元数据为 Isaac Sim `6.1.0.0`、Isaac Lab `3.0.0rc1`、Warp `1.16.0`、Torch `2.11.0+cu130`。六个直接相关官方源码的 SHA、锚点行号与方法表保存在 `STATIC_CAPABILITIES.json`，不是按其它版本记忆判断。

| 核查项 | 本机证据和接入要求 |
|---|---|
| 真正 GPU 物理 | `isaaclab_physx/physics/physx_manager.py` 按 `sim.device` 设置 `physxScene:enableGPUDynamics`、`broadphaseType=GPU`。实际运行还需读回 scene 属性、`physics_sim_view.device_ordinal` 与真实 tensor device；渲染器占用 GPU 不足以证明 GPU 积分。 |
| 前端与材料 | `omni.physics.tensors/api.py` 明确拒绝 GPU pipeline 使用 NumPy frontend；Warp／Torch 可用。PhysX 参数 tensor 的 `param_device_ordinal=-1`，材料和质量读回在 CPU 属于该 API 设计，不能据此断言物理回退 CPU。材料三列是 static friction、dynamic friction、restitution。 |
| 原 filtered 接触 | `get_contact_data(dt)` 返回六数组：法向力 `(M,1)`、点 `(M,3)`、法向 `(M,3)`、距离 `(M,1)`、count／start `(sensor,filter)`。必须按原索引保留间隙，检查总容量、边界和活动槽有限值。`dt=.004` 负责将冲量换为 N，不能再次除以 dt。 |
| 全 actor 接触 | 本版新增 `get_raw_contact_data(dt)` 七数组，不需 filters；count／start 为 `(sensor,)`，另有 `(M,) uint64 other_actor_ids`。`get_other_actor_paths_from_ids()` 要求先复制为同 frontend 的 CPU uint64 tensor，空路径表示无效 ID。该 API 存在尚不等于本轮 GPU 实测通过。 |
| 摩擦 | `get_friction_data(dt)` 返回摩擦力向量、摩擦点、count、start。官方 PhysX sensor 明确 normal／friction 共享 count/start 缓冲区；读取 friction 前须复制 normal 的表，不能保存两份指向同一缓冲区的引用。PhysX 的摩擦报告需要 filters。 |
| 三体外力 | `RigidBodyView.apply_forces_and_torques_at_position`／`apply_forces` 接受各体 `(count,3)` 力及 indices、world/local 标记，示例覆盖 cpu／cuda。仍须沿原执行路径逐体保证永久 0.98 N 向下力，不把“API 接受 CUDA”当作力已实际正确施加。 |
| mimic | 本版 importer／transformer 明确保留 `NewtonMimicAPI`，由 runtime 直接消费，不再重复 author `PhysxMimicJointAPI`。没有本机源码依据说 GPU 必须拆掉 mimic；仍需真实 24 独立列→28 DOF 指令核验和原实际 mimic 误差门。 |
| convex cooking | 官方 asset-validator 提供 `convex_gpu_compatibility_is_valid(stage_id,prim_id)`，并指出薄凸体可能不兼容 GPU。没有用未匹配本安装的“固定 64 顶点限制”替代检查，也不改近似几何来求通过。原 native 20 µm 导入门继续保留。 |

源码给出的明确限制是 GPU 模式自动关闭 CCD、GPU 多种缓冲区不能无限动态增长；官方还警告 stabilization 可能影响 contact force 报告。应先核原冻结配置实际值，显式记录 GPU 迁移的物理差异，不能只把 device 改为 cuda 后宣称全部物理数值等价。

PhysX manager 初始化会设置 `/physics/disableContactProcessing=True`，官方 ContactSensor 随后设置 False。构造前设置会被覆盖，应在正确生命周期启用并读回；GPU 默认 `/physics/suppressReadback=True`，动态状态应取 tensor，不能用可能陈旧的 USD pose 做标签。

## 原仓库的可复用部分

只读核对 `CanonDex/canondex/scene/execute_motion.py::RobotContactReports` 和 `object_contacts.py::ObjectContactReports`，二者都通过 `sim.physics_sim_view.create_rigid_contact_view` 读取 filtered 六数组，`contact_readback.py` 已提供 CUDA Warp `.numpy()` 与保留间隙的完整 prefix 复制。它们不是 CPU-only 事件回调，因此“需要先替换 CPU contact callback 才能 GPU”不是当前主线的实际阻碍。

本机官方 `get_contact_report`／`get_full_contact_report`／订阅 API 文档只说明数据有效期是一物理步，没有在所读源码中声明 GPU 禁用；本阶段没有实际测其 GPU 完整性。当前主线无需依赖这些 callback，不能把这个未测项扩大成 GPU 不可用。

现成 `benchmark_template_gripper_gpu.py` 与 `benchmark_template_arm_parallel.py` 已支持 cuda 和同一物理时钟下的克隆环境；本次仅阅读、绑定 SHA，没有运行或修改 CanonDex。这些源码证明可直接参考已有结构，不代表三体 Boya 的原控制／全部审计已验收。尤其已有 gripper benchmark 明确限制批量模式的 native cooked 审计，不能据此省掉新适配所需的每体／每 lane 几何证据。

不要直接在活跃仿真中执行官方 `CookingApproximationChecker.CheckStage`：其整体流程会改变 settings、释放 cooking cache、attach/reset/detach simulation。本探针完全不调用它；这里只定位官方兼容性接口，实际调用时机仍由受控运行协议决定。

## 已交付的最小探针

新增 opt-in `author_gpu_capability_probe.py`，导入及 `scan` 不导入 Isaac、Torch、Warp，不启动 Kit。已实际运行：

```text
python -B reproduction/boya/author_gpu_capability_probe.py scan --output-dir OUT
```

供统一小预算执行器调用的只读接口为 `capture_contact_view(view, dt, mode='filtered'|'raw'|'friction', require_cuda=True)` 与 `probe_live(sim, output_dir, contact_views=..., rigid_views=..., articulation_views=...)`。它们不 launch、不 step、不 reset、不改变控制或力，只读取已初始化的 caller-owned views，保存实际 scene 属性、tensor device／shape、完整缓冲区 NPZ 和 API 错误。无法读取会记录失败，CPU payload 不能当 CUDA；合法空流会保留为空，不能冒充 reporting 已验证。raw／friction 是可选检查，不强迫主线替换原 filtered 路径。

7 项 meaningful CPU tests 全过，覆盖容量饱和／unsigned 越界、活动槽 NaN 与法向、raw uint64、未引用 NaN、friction 覆盖共享表、CPU 伪 GPU 拒绝、原 backend 报错留证和不增加物理时钟。测试中的 CUDA device 字符串是 CPU fixture，不是实际 GPU 证据。

轻证据见 [`evidence/author_gpu_capability_20261115_v1/EVIDENCE.json`](evidence/author_gpu_capability_20261115_v1/EVIDENCE.json)。本审计新增 Kit／物理／模型／规划／训练／标签均为 0，旧冻结源码不改；真实 GPU 批量动作、全接触完整性、隔离、native cooked 几何、mimic、逐体外力与原门／标签一致性，须由另行冻结的小预算运行和独立审计回答，不能从 API 存在直接置真。
