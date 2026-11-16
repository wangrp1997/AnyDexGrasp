# 参考现成采集后端的实际进展：紧凑 MuJoCo 原型

用户要求先参考开源采集方案，解决当前异常慢的问题。本轮完成代码对读、独立 MuJoCo 模型/指令适配和两次小前缀测试。**不是完整采集提速验收，尚无新标签或完整抓取倍率。** 原冻结 PhysX 源码、数据、失败和 CanonDex 保持。

## 参考了什么，实际采用了什么

| 现成实现 | 本轮核对与采用 |
| --- | --- |
| [DexGraspBench MjHO](https://github.com/JYChen18/DexGraspBench/blob/main/src/util/hand_util.py) | MjSpec 直接加载源模型/CoACD 网格，0.004 s、implicitfast、独立模型/状态、显式物体力和接触参数。已落到本项目的 MuJoCo 适配代码；未声称跑过其原始 benchmark。 |
| [Get a Grip 验证器](https://github.com/tylerlum/get_a_grip/blob/main/get_a_grip/dataset_generation/utils/isaac_validator.py) | 核对批量建场、推进和标签读取组织方式；源码构造默认 use_cpu=True，不能将其笼统描述成必须全 GPU。它的浮动手/概率标签不替换我们的完整臂及标签规则。 |
| [DexGraspNet 验证器](https://github.com/PKU-EPIC/DexGraspNet/blob/main/grasp_generation/utils/isaac_validator.py) | 核对多环境统一步进；GPU PhysX 与 use_gpu_pipeline=False 分别声明，不能混为纯 GPU 数据链。其100步重力验证不是本项目稳定提起标签。 |
| 本地 kywch/AnyDexGrasp pixi 分支 | 确认 RGB-D→CGR→抓取提起→保存特征/标签闭环。其 Panda+Inspire、GT mask、重试和高度标签不直接移植。 |

本机已有 MuJoCo3.11，Isaac Gym/robosuite 未安装。因此先沿现成 MuJoCo 路线接入原源模型，无依赖安装、环境降级或新调度框架。

## 发现的具体工程差异

现有 PhysX 配置 `collision_max_extent_m=.02`、vertex_limit=64，并有额外精细划分。保存的 scene0 审核中机器人/手部12681凸体，其中上臂3316、前臂1943、肩1719。这是仿真适配选择，不是 AnyDex 学习算法规定。

DexGraspBench 从源网格建场，因而本轮另做直接源模型候选。**不能仅凭数量认定全部慢速的根因，也不能把不同引擎的形状数量当性能倍率。** 新候选碰撞表示及求解器改变，尚未通过旧 PhysX20µm导入门或动力学等价认证。

## 已实际落地

- `author_mujoco_preflight.py`：复用原 combined UR16e+Boya URDF，24主动电机、28机器人关节、4 mimic equality；从现有原生配置/URDF/手XML读取控制参数。
- `author_mujoco_scene_preflight.py`：复用原已观测三物体初态和原 CoACD 部件，质量各0.1kg，原24列指令按名称映射。未重跑CGR、改候选或改旧记录。
- `author_mujoco_source_filters.py`：逐条迁移原手XML的28组 contact/exclude，没有按测试结果另加排除对。
- `author_mujoco_prefix_smoke.py`：固定20条原 settle 指令的独立CPU接口测试；有自己的新协议/目录，不接生产collector，不输出抓取标签。

两场原3389/3749指令全部7138行的电机目标映射逐值一致。36个静态状态/1476 link矩阵与独立原URDF FK核对，最大误差1.3322676e-15。两场均核三个物体原初态和0.1kg质量，模型分别52/53个碰撞几何：其中机器人30个源网格，另为原物体CoACD与地面。模型total nq49/nv46，24执行器/4 mimic；静态核验0积分。

原生控制配置解析时保留了力上限继承URDF、手关节被动阻尼及源armature；MuJoCo3.11的三系数阻尼表示已适配。**这些静态接口检查不证明其PD积分、mimic约束、摩擦或接触力与PhysX等价。** 新候选采用显式DexGraspBench式摩擦参数，不能说标签物理已认证。

## 真实前缀与保留的问题

首个20步数值推进通过，但独立检查发现URDF默认导入缺少原手XML排除，出现12.393mm的结构自碰撞、最大广义速度16.11。它不具备原碰撞语义，结果保留，不能作为可用抓取证明。

仅补回原XML28组排除后的另一个独立20步测试：

| 项目 | 修正后结果 |
| --- | ---: |
| 实际返回步/三体步 | 20 / 60 |
| 仿真时长 | 0.080 s |
| mj_step 合计 | 0.003040 s |
| 当前前缀字段读回合计 | 0.001417 s |
| smoke内部总时钟 | 6.363170 s |
| 全部前缀接触点 | 345 |
| 前缀最大接触穿透 | 0.329270 mm |
| 最大广义速度（混合角/线单位） | 0.441233 |
| 完整抓取、标签、训练更新 | 0 / 0 / 0 |

内部总时钟含该函数开始后的建模、推进、读回、末尾核验和轨迹输出，**不含进程导入、最初源校验和协议落盘**；它不是完整冷启动采集wall。前缀只覆盖settle，不覆盖接近、闭合、提起、完整点流审核，不能据此报百倍加速或采满ETA。

全部345接触点均记录，但当前诊断schema只导出geometry ID、距离、点、frame、接触frame下六维力等列，并非完整原PhysX合同。初始trace中`fields_dropped:false`不能用于认证原合同字段齐全；`CONTACT_SCHEMA_NOTE.json`明确这一限制。完整采集前须补齐兼容观测/逐体状态、原判定和独立审核。

## 成本与接续

7项CPU测试通过。4次静态接口开发错误（配置命名空间、力上限继承、新版阻尼形状、密度浮点质量）均在积分前失败，源码快照及回执保留；修复时改为明确体积分配质量，未放宽质量检查。

两种模型各20步，合 **0 Kit、40 MuJoCo环境步/120对象步、0完整抓取/标签/训练**。原PhysX阶段成本另记，不混成同一后端吞吐。累计有效新数据仍26fit+5internal=31，无正式扩采/训练/新外测，无后台任务。

目前是可运行候选后端，`production_ready=false`。下一项是同一冻结计划的完整动态提起、源网格离桌/持续受载及原七门/逐体判定的对应核验，之后才有依据决定是否接入新有限采集；本轮没有自动派发完整试抓或扩采。

轻证据：`evidence/author_mujoco_reference_adapter_20261116_v1`。原始模型/轨迹留在忽略的`outputs/boya_mujoco_*`，不入Git。
