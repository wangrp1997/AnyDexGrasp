# 官方仿真与真实闭合/路径接口核对

更新：本文保留此前原模型/源码审计；后续独立DH3单方块物理烟测已通过，当前结果见[仿真说明](simulation/README.md)。

官方源码本地根目录：`/home/wangrenpeng/AnyDexGrasp/`。本报告所有官方相对源码路径均以此目录为基准。静态探针、论文源/源码SHA等原始证据在 `outputs/paper_code_audit/`；CanonDex原始工作目录保留。


固定官方源码 `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`；2026-11-03 只读搜索。主论文/补充原始 TeX 见独立仓库 `outputs/paper_code_audit/source/sections/{methods,results,supp}.tex`。未运行任何硬件入口、未连接机器人、未运行动态仿真。

**结论：该版本没有官方桌面动态抓取仿真评测环境/任务入口。** 有 PyBullet 依赖和使用，不等于已有可照搬的抓取仿真基准。其作者原验证是 UR5 + 三种真实手 + D415 的真实清桌试验。可先在离线表示/决策模型层验证原方法；在我们服务器做实际动态抓取，需要一个另行声明的 simulator adapter，不能叫作者原仿真协议。

## 搜索范围与 PyBullet 的实际用途

对官方全部 `*.py/*.md/*.sh/*.txt` 搜索 `isaac|mujoco|bullet|pybullet|gym|simulator|simulation|stepSimulation`：

- `generate_mesh_and_pointcloud/recover_inspire_hand_to_stl.py:139–159,190–245`：加载 fixed-base 手 URDF，逐开口 `resetJointState`，读取 FK 合并各 link STL，生成每手势/开度的网格/点云和 JSON 标定。
- `recover_allegro_hand_to_stl.py:87–105,189–212`：相同的手姿几何生成流程，未建立桌面物体/接触抓取任务。
- `recover_dh3_hand_to_stl.py:31–84`：GUI 滑块设置手关节；全仓唯一 `stepSimulation` 是83行这一单步，紧接 `sleep(10000)`。不是闭合/抬升回合。
- 同 DH3 文件137–155为固定手 URDF 的 DIRECT加载/FK接口。
- `generate_mesh_and_pointcloud/angles2stl.py` 导入 PyBullet，也是几何加工配套。
- `ur_toolbox/python-urx/urx/urrobot.py:set_simulation` 属于 vendored 真实 UR 控制接口，不是包含物体动力学/接触的本地机器人仿真环境。
- 未找到 Isaac/Mujoco/Gym 桌面任务、物体接触仿真步循环、仿真抓取成功评测或仿真训练 rollout。

这些是对当前固定公开版本的结论；不据此推断作者从未内部使用其它仿真代码。

## 论文真实流程

`methods.tex:111`：相机点云→CGR→预定义手势候选→决策评分→碰撞→世界坐标→socket发送UR5；UR5 **内置运动规划器**导航，沿接近方向在最终抓姿后方 **10cm** 设 waypoint；设置手势，到位后闭合到预定关节受力上限，再把物体搬到目标箱。人工标搬运成功/失败。

`results.tex:110–117`：腕部 Intel RealSense D415，初始相机垂直桌面、约60cm高，真实随机杂乱清桌，重复两次；逐次成功尝试/总尝试为指标。

`results.tex:176`：作者称仅视觉、无触觉反馈。应区分底层电机力/力矩限制与额外可感知接触分布、据此在线修正掌位/闭合/再抓的策略。

## 官方代码真实路径

`ur_toolbox/ur_toolbox/robot/ur.py:334–416`：

1. 根据相机—TCP、手—夹爪标定转换抓姿；硬编码标定在 `109–144`，ready/throw机械臂关节在 `95–107`。
2. Inspire/Allegro最终TCP还沿接近轴加 `depth+0.014m`（373–374）；DH3按开度作5mm量级修正（375–377）。
3. `pre_pose = final_pose - approach_dist * approach_axis`（379–380）。
4. `movels([pre_pose,final_pose], ..., radius=0.02)`（391）提交真实UR控制器；这不是公开的整臂碰撞优化求解器，也没有我们使用的全轨迹审计接口。
5. 闭合后沿接近方向先退4mm（401–403），再退回pre_pose、throw、开手（405–409）。

三个robot脚本实际普遍 `approach_distance=0.06`，例如 `robot_inspire.py:706,787–789`；与论文10cm不同，需冻结复现采用哪个配置。

## 官方手指闭合

### Allegro

- `robot/Allegro/Allegro.py:61–84`：当前关节位置到该手势 `close_pose` 以10步线性插值，每步 `set_pose` 后等0.1秒；读回位置/effort用于记录。
- 形参名 `tactile` 确实存在，但该函数实现没有使用它，也没有按每指接触结果更新闭合路径。
- `ur.py:395–398` 传手势 `close_pose_matrix`/`close_pose_torque`，但后者落入该未使用的形参，不能据此声称真实执行了传入力矩曲线。底层Allegro controller可能有自身控制机制，其具体硬件表现不可由本函数单独推出。

### DH3

- `robot/DH3/DH3.py:169–188`：硬件初始化固定 `force=70`, `speed=100`，写入驱动寄存器。
- `210–213`：闭合发 `SetTargetPosition(0)` 并等0.2秒；未在上层作学习接触反馈。

### Inspire

- `robot/Inspire/InspireHandR.py:25–44`：串口初始化power/speed数值均1000。
- `1112–1125`：依据手势确定哪些手指闭合，循环19次，每次指令下降60直到该手势目标；读取actangle但没有在此函数以受力停止。
- 文件另有使用 `get_actforce` 的流程（1066等），但不是此 `close_gripper` 的执行链，不能把它当实际默认控制。
- 当前 `ur.py:400` 无参 `close_gripper()` 与 Inspire 函数要求 `InspireHandR_type` 参数存在公开接口不一致；原硬件入口需适配/核对，不能直接假定可运行。

## 独立仓库复现建议

先验证原手离线 checkpoint→CGR→手势/开度/深入评分→碰撞候选，记录全部公开配置差异；此阶段无需改Boya控制。若继续仿真抓取，优先保留原 Allegro/DH3/Inspire 手模型/预抓映射/闭合目标，再建立仿真输入和执行后端。新的后端、接触物理参数和成功标签明确标为本项目实现，不用“作者提供的仿真环境”称谓。等作者原方法可独立运行后再做 Boya 和 CanonDex 对照。
