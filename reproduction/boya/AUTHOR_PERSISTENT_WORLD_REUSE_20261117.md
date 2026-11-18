# 一次初始化、换物体与常驻规划（2026-11-17）

后续更新：[新观测采集诊断](AUTHOR_MUJOCO_FRESH_COLLECTION_20261117.md)已完成两份保存计划连续执行，并接通首轮新RGBD/CGR/规划/抓取/回位；物理准入未通过。下文为此前阶段结果，原失败及计时保持。

用户要求按真实工作台组织：机器人、桌面和环境初始化一次，以后只换物体，机械臂通过控制返回。此前“一个Kit/一个世界”仍每轮写机器人初态、运行750步初始化，不能等同于这个要求。本阶段实现独立MuJoCo候选世界，并分别验证两次换物连续性与一条完整保存计划。**尚未完成该新世界的新拍照→CGR→规划→抓取两轮完整采集。**

轻证据：[SUMMARY.json](evidence/author_persistent_world_reuse_20261117_v1/SUMMARY.json)。旧冻结源码、失败结果和CanonDex均未改写。

## 世界只初始化一次

[`author_mujoco_persistent_world.py`](author_mujoco_persistent_world.py)在启动时建立机器人、地面和六物体银行，只创建一个活动Model/Data，只调用一次`mj_resetData`、一次机器人初态写入。启动内部尚有原适配器的两次中间编译，加最终银行编译共三次，不能声称启动只编译一次。

边界`exchange()`只写物体位置/速度和外力，停放未使用物体。更换物体不会写机器人关节位置、速度或世界时钟，模型和Data句柄保持。对象必须属于启动时预载银行；新几何超出银行会拒绝，当前不支持任意未预载CAD热加载。当前代码另检查重复身份、几何身份、有限刚体位姿，并禁止在活动轮中换物。

实跑两次不同三物体布局，各64个4ms home hold tick，共128步。两次换物前后机器人非零位置/速度、世界时钟、Model/Data句柄均exact；总显式重置1次。两次换物wall分别2.823ms、1.046ms；两个含接触读取和JSON的64步循环分别36.202ms、34.045ms。启动另14.029秒。**短home hold没有动态闭合，不能推广为完整抓取/采集倍率。**

## 一条完整保存计划

`outputs/boya_mujoco_persistent_replay_20261117_v1`固定两份已有准备结果、最多17500步、0重试，实际只执行首份3818步，第二份没有开始。24电机目标及六臂速度前馈逐行exact。首轮完整物理结束后，结果整理调用了不存在的`author_lift.evaluate`，进程失败；原错误、执行源码及56.9MB原始轨迹保留。当前通用脚本改为原`evaluate_lift`，**修正版本未重跑物理**，未为取得正例重抓。

独立[`audit_author_persistent_replay.py`](audit_author_persistent_replay.py)只读原轨迹，0额外积分，复算原七门、源网格极值、接触载荷及独立XML FK，结果如下：

| 检查 | 实测 |
| --- | --- |
| 原七门 | 全部通过 |
| mimic最大误差 | 0.000764700rad，原门0.002rad |
| 最大穿透 | 0.975336mm，原门1mm |
| 最大掌位置/角度误差 | 4.814383mm / 0.007781544rad |
| 臂非法支撑/基座漂移 | 0 |
| 026末源底 / 相对hold末升高 | 102.370365mm / 100.651363mm |
| 末th/ff/mf/rf/lf载荷 | 5.219590 / 5.519123 / 0.650040 / 0 / 0N |
| 源底/顶及接触载荷复算误差 | 0 |
| 掌矩阵对XML FK最大差 | 1.4433e-15 |
| 保存的原生接触点 | 67574 |

末受载1.616秒仅诊断，未新增门。首轮在该候选物理域中为诊断正例，正式训练标签仍0。

失败parent wall24.862611秒，含一次性世界setup12.310721秒，剩余12.551890秒还包括输入/协议、执行、完整观测、JSON和部分后处理；原错误路径没有落盘独立循环计时，**不能把剩余数当纯物理时间或稳定吞吐**。3818步对应15.272秒仿真时钟；这不是新RGBD/CGR完整采集耗时，不能与旧589秒完整采集直接相除。

此为显式新后端：原URDF物理网格由MuJoCo凸表示，接触、求解器和约束域与原PhysX不同，未通过原生20µm凸体认证或PhysX完整filtered字段合同。相比旧URDF端口，补读源手XML的四个mimic equality参数，并采用公开DexGraspBench手接触设置（[hand_util.py](https://github.com/JYChen18/DexGraspBench/blob/main/src/util/hand_util.py)）：solimp前三项0.5/0.99/0.0001、solref前两项0.005/1、elliptic/noslip2/impratio10。机器人驱动、限矩和原24指令保持，手/物摩擦仍显式0.6/0.02/0.0001。控制/积分步长4ms，源手XML的0.5ms步长没有复现，不能称完整照搬源XML或PhysX物理等价。没有针对这条已知正例反复调参、放门或换候选。

## 重复规划开销

[`author_planning_reuse.py`](author_planning_reuse.py)、[`author_compiled_kinematics.py`](author_compiled_kinematics.py)将URDF/网格/参考资源保留在一个CPU拥有者中，固定运动学树只解析一次。局部变换按原精确输入缓存，几何审核缓存键包含全部网格字节、物体位姿、link/排除对、容差和完整float64状态，不量化、不以旧物体布局代替新布局；新状态仍调用原审核。保留原FK乘法次序和数值类型。

同两份保存输入，原串行规划74.360518+86.654464=161.014982秒，常驻编译FK版34.292773+49.850722=84.143496秒，观察减少47.74%。两场六个科学指令数组各自逐字节相同，候选掌姿/行/slot相同，26份完整几何报告相同；带新输出路径的两个SHA数组各对自己的完整内容独立验真，不要求不同路径的SHA互相相等。

中间版本127.906375秒、静态origin版95.108418秒也保留。初版缺libusb在规划前失败；第二版原比较器误将新路径SHA要求相等而报错，原结果不改，独立内容审核通过。后续驱动已采用独立比较器。编译FK版复用原计时基线，并非新随机顺序对照，不能称稳定总体均值或全采集提速。

[`author_reused_fixed_scene_collection.py`](author_reused_fixed_scene_collection.py)已将常驻CPU规划接入原固定世界入口，限定单worker并纳入原资源槽/退出清理。两轮六银行静态freeze通过；mock GPU fixture明确禁止实跑。**该入口仍继承旧PhysX每轮机器人复位/750初始化，不能替代上面的严格一次重置候选；新MuJoCo世界尚未接相机、CGR和该规划服务。**

## 验证与成本

29项不同CPU用例通过（精确FK、缓存失效/负例/源变动、时限、请求绑定、单worker清理和真实小MuJoCo模型换物连续性）；测试没有积分。新物理为128+3818=3946个MuJoCo环境tick、11838个活动三体tick；六银行的时钟机会数不是awake工作量。全资产构建另有一次0步静态检查，两次带物理世界各一次；均无Kit、CGR、RGBD、新训练标签或训练。旧数据、权重和配额保持，未恢复正式采集。所有本阶段启动的CPU进程已返回，无后台采集。

只读复核命令（不启动仿真）：

```bash
PYTHONPATH=reproduction/boya /home/wangrenpeng/tendon-spin/.venv/bin/python -B \
  reproduction/boya/audit_author_persistent_replay.py \
  --root outputs/boya_mujoco_persistent_replay_20261117_v1 \
  --output /tmp/anydex_persistent_audit.json
```

剩余验收是一次启动后完成两轮物理返回/换物/新拍照/CGR/规划/抓取，并测完整外层耗时。当前`production_ready=false`，不以分开的短前缀和单抓取合成“两轮完整采集已经通过”。
