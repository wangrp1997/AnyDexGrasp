# 关节限位响应修正（2026-11-17）

**独立限位测试通过；尚未重跑完整抓取。** 原URDF上下限和启用标志没有丢失，问题是MuJoCo默认的关节限位响应仍为20ms。此前源配置中3ms的`geom.solref`只影响接触，不改变`joint.solreflimit`。

新增[`author_mujoco_joint_limits.py`](author_mujoco_joint_limits.py)，对全部启用限位的hinge统一设置响应时间为`2*native_dt`、临界阻尼1。当前0.5ms原生步对应1ms限位响应，原限位阻抗保持。采用MuJoCo[官方数值安全边界](https://mujoco.readthedocs.io/en/stable/modeling.html#reference)，不按抓取结局逐候选调参。它仍是有限刚度约束，不能声称绝对零越界或等同实物硬挡。

URDF范围、质量惯量、关节驱动/限矩/阻尼、mimic、物体接触参数、原动作均不变，不裁剪q/qdot或目标。新配置显式改变限位求解模型，保持opt-in；`World(...,rigid_joint_limits=True)`及新acceptance freeze的`--stiff-joint-limits`绑定该选项和源码，默认false。旧失败根/协议/源归档不改、不热续跑。

## 同模型独立对照

[`check_author_mujoco_joint_limits.py`](check_author_mujoco_joint_limits.py)固定读取原验收归档模型与原开手目标。两模型仅限位响应不同，21组范围/质量/惯量/驱动/接触/mimic等数组逐值exact。每模型3个固定场景、各200个4ms控制步，每控制8次0.5ms推进，0重试。外载是显式施加于THJ2自由度的±1Nm诊断力矩，不是抓取接触；三体均停放，六例接触数均0。

| 测试 | 原默认最大越限 | 新配置最大越限 | 原0.002rad门 |
| --- | ---: | ---: | --- |
| 原实测开手初态→原新计划开手目标 | 0.010007629rad | 0.001347146rad | 通过 |
| THJ2下限、向外-1Nm | 0.064423809rad | 0.001031698rad | 通过 |
| THJ2上限、向外+1Nm | 0.064423809rad | 0.001029340rad | 通过 |

开手一行统计所有28关节、每个原生子步，最坏关节由原LFJ3变为新THJ1；THJ2开手最大从0.007467271降至0.000820370rad。没有只挑THJ2读数忽略其他关节。全部有限状态、原生warning0。三个限位检查没有改旧0.002rad容差。

这证明该配置在原开手转换与独立双向外载下抑制越界，不是实际物体接触/受载hold/完整回位的复验。原验收015提起与THJ2越限结果继续保留，新增训练标签0，不把这三例当正负抓取数据。

## 验证和成本

35项`test_author_mujoco_*.py`通过，包括3个新增限位测试；其中微型真实动力学测试验证±1Nm下上下限，不用状态裁剪。单独新测试及后续回归各1600次微型原生积分，合3200；原机器人对照6例共1200控制/9600原生积分、2静态编译模型/6独立初态Data。该阶段总12800原生积分，其中微型单测单列，不混入采集环境步；0Kit/CGR/抓取/标签/训练。

新配置另冻结为`outputs/boya_mujoco_stiff_limits_static_20261117_v1`，只检查绑定和新选项，`STATIC_FIXTURE.json`强制拒绝实跑，0积分。实际机器人诊断结果为`outputs/boya_mujoco_joint_limits_20261117_v1`，[轻证据](evidence/author_mujoco_joint_limits_20261117_v1/RESULTS.json)保存原/新全部结果和绑定；大轨迹继续忽略。无后台测试或采集。
