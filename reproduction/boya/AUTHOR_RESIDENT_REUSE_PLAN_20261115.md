# 常驻采集的现成实现与最小迁移路径

本轮只读核对已检出的 `kywch/pixi`、本机 IsaacLab 源码和已有 Boya 执行接口。按用户最新要求，停止 `source_metrics` 局部优化开发；尚未创建该优化模块、运行基准或注入当前 wave。没有新增模型前向、规划、模拟或训练，也没有修改冻结采集源码或 CanonDex。

可以直接参考现成架构。主线应采用 fork 的“环境与模型在循环外创建，循环内重置和采集”，由本机 IsaacLab 提供常驻场景、批量资产与相机，Boya 保留已有候选、命令、逐体标签和审计。现有多进程 wave 只是当前采集入口的并发封装；常驻/向量环境尚未接入，不能把两者的收益混称为实测。

## fork 已经做了什么

固定来源为 `outputs/reference_repos_20261114/kywch_pixi`，Git 提交 `3888737ef280fb4707601501c593a5a5a2f0bb20`。关键代码可直接阅读：

| 环节 | 本地文件与行号 | 可参考的内容 |
|---|---|---|
| 外层生命周期 | `collect_rs_inspire_grasp_data.py:195–206` | 手云载入、环境创建及 `GraspNetRunner` 构造均在多轮循环之前。 |
| 每条试验 | 同文件 `:217–218,248–269,281–300` | reset → RGBD → 采样 → 选首候选 → 执行 → 保存特征与标签。 |
| 环境重建周期 | 同文件 `:313–316` | 每 10 个外层 round 才显式 close/recreate；一个 round 还包含多个手型与深入值。不能解释成严格每 10 次实际抓取。 |
| 模型常驻 | `AnyDexGrasp/utils/graspnet_utils.py:175–219` | runner 构造时建网、eval、载 checkpoint，随后 `get_grasp` 复用网络。 |
| reset 细节 | `robosuite_env.py:305–325`；`diverse_lift/lift_env.py:117,163,226` | reset 后还执行 50 个移开手的控制步；`hard_reset=True` 默认继续传给 robosuite。源码只能证明 Python 环境对象常驻，不能证明每次 reset 不重建 MuJoCo model。 |

不直接搬入的部分也很明确：Panda + Inspire 的 `OSC_POSE` 控制器在 `robosuite_env.py:275–299`；6→12 手动作变换在 `:51–88`；抓取变换和抓取/退出动作在 `:359–401`。这些依赖 MuJoCo/robosuite 的状态、site、action 和 reward 接口，不能直接运行在当前 UR16e+Boya PhysX 场景中。

fork 还用物体真值分割（`collect_rs_inspire_grasp_data.py:230–252`、`diverse_lift/lift_env.py:303–313`），最多 5 次采样并在后续尝试增加增强（前者 `:95–110`），标签是物体 body 原点高于桌面 11cm（后者 `:290–301`）。这些和当前无 GT 候选筛选、固定一次 identity、固定首选、任一原桌面体源底/升高/双侧承载标签不同。本次参考其组织循环，不替换已冻结的算法和任务规则。没有找到该 fork 在本机可直接对比的采集耗时回执，不能根据代码推报倍数。

## 本机官方入口与版本差异

本机元数据：`isaaclab=3.0.0rc1`、`isaacsim=6.1.0.0`、`newton=1.5.2`。查阅安装源码即可，不需要换仿真器或重新开发环境管理框架。IsaacLab 核心源码根为：

`/home/wangrenpeng/tendon-spin/.venv/lib/python3.12/site-packages/isaaclab/source/isaaclab/isaaclab`

| 已有能力 | 本机入口 | 本任务接法 |
|---|---|---|
| 场景与资产批处理 | `scene/interactive_scene.py:134–204` | `InteractiveSceneCfg(num_envs=..., ...)` 构造时进入 `cloner.ReplicateSession`，复用框架资产管理。 |
| 手动克隆 | 同文件 `:110–123` | 现成 `cloner.clone_plan_from_env_0` + `cloner.replicate`；配置式方式更少手工生命周期代码。 |
| 环境隔离 | 同文件 `:284–316,380–387` | 环境坐标 `env_origins` 与跨环境碰撞过滤。源坐标/相机/接触记录必须明确 env 索引。 |
| 重置与每 tick 数据 | 同文件 `:479–545,551–622` | `reset(env_ids)` 清缓冲；`reset_to` 写物体/机器人 pose、速度和关节状态；`write_data_to_sim/update` 管批量缓冲。仍须重设本任务永久重力、控制目标和全部三体初态。 |
| 常驻直接环境 | `envs/direct_rl_env.py:106–181,318–345,363–449,723–744` | 一个 SimulationContext/InteractiveScene，按 env 索引重置，批量动作。若采用 DirectRLEnv，则 decimation=1、dt=.004，并在默认自动 reset 前完成本任务终态轨迹、SHA 和标签落盘。 |
| 批量 RGBD | `sensors/camera/camera.py:82–118,548–626` | 已有 Camera 的匹配 prim、批量帧缓冲和 depth；保持当前 K、深度语义和工作区。`tiled_camera.py:6–12` 明确旧 TiledCamera 为弃用别名，优先当前 Camera。 |

**不能照搬旧 `InteractiveScene.clone_environments` 调用。** 本机当前 `InteractiveScene` 中已无此方法；本机 `third_party/sharpa_upstream/.../sharpa_wave_env.py:165` 只是旧 API 示例残留。应使用本机已实现的配置构造/`cloner.replicate`，无须为旧示例回退版本。[官方当前 InteractiveScene 教程](https://isaac-sim.github.io/IsaacLab/main/source/how-to/create_scene.html)也展示一次构建、多环境循环及批量状态写回。[官方 Direct workflow 教程](https://isaac-sim.github.io/IsaacLab/main/source/how-to/create_direct_rl_env.html)提供按环境重置与动作钩子。采用这些接口不要求训练 RL 策略。

`replicate_physics=True` 适合相同资产实例；不同三体组合不能假装同一资产的克隆。本机 `InteractiveSceneCfg:104–126` 保留异资产 `False` 模式及相应启动成本，另有 `clone_cfg.clone_combinations`。先对相同三体组的不同初态/四手型复用；后续固定不同资产组仍按预定义 queue 消费，不能按结果改对象或候选。保持当前 PhysX 后端，不能因为安装了 Newton 就更换物理。

## Boya 接入只补缺少的适配层

已有接口足以承载迁移，无须另造大框架：

1. **先把现成循环接成一个常驻环境。** 参考 fork 的外层构造方式，保留一次 launcher、一个已载入的 CGR runner；每次严格消费冻结 scene/job。模型复用须按原 seed 单次 identity 前向，并保存同一候选/特征格式。当前一次 CGR 约 4.14s，不把它当主要瓶颈。
2. **复用原执行器的常驻钩子，补三体状态适配。** `/home/wangrenpeng/CanonDex/scripts/research/execute_template_arm_positive.py:24,150–177,245–266` 已接受 `resident_scene`/`runtime_factory`，具有原生初态、材质、执行和 native20um 审计。`canondex/grasp/arm_resident_scene.py:48–115` 已管理 bind/reset/计数；但其 reset 只处理一个 `objects['object']`，内容锁也只覆盖单源物体。当前 `author_lift.execution_steps:294–327` 尚未传 resident 参数。因此需要独立 Boya 三体适配，重置三个物体、接触流和永久外力，扩展全场景绑定；不能仅传一个 resident 对象就宣称三体常驻完成。CanonDex 保持只读。
3. **再用官方克隆扩到少量向量环境。** `InteractiveScene` 管资产/状态/相机；Boya 当前所有 `[0]` 读回、固定 `/World/Object` 和三体接触路径须改成显式 env 索引/namespace 的新 opt-in 适配。`author_transport.py:560–586` 与 `author_lift.py:214–227` 是具体接点。每 env 播放原已冻结 24 列 commands，保留 4cm approach、250 close、50 hold、原 lift 和全部七门/逐体门。先完成单 env 对照，再做两个 env 的隔离与标签对照，不能直接宣称几千 env 可跑。

输出继续复用 `author_collect.collect_execution`、`author_lift.evaluate_lift/finish`（`author_lift.py:57–116,235–277`）、`audit_author_collection_runtime.audit_trial` 与现有父/子账本和 exporter。每 env 分开的实际轨迹、接触 journal、candidate/commands/source SHA 需在 reset 前写完；null/infra 不记负例，首 quota 仍按原 scene/gesture 顺序接纳。`DirectRLEnv.step` 默认会自动重置结束环境，因此最小初版可只用官方 InteractiveScene 循环并复用已有执行 generator，避免先接入无关的 reward/自动 reset 行为。

本阶段交付是迁移依据与明确接点，尚无常驻运行或向量采集的加速证据。正在运行的冻结 wave 不受影响；新常驻路径须另立版本与小预算对照后再用于扩采。
