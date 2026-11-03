# AnyDexGrasp 原方法复现

本目录属于独立仓库 `/home/wangrenpeng/AnyDexGrasp`，与CanonDex的抓取改进分开。官方源码固定为 `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`，原算法/硬件入口没有修改。来源与原论文SHA见 [SOURCES.lock.json](SOURCES.lock.json)。

## 2026-11-03 已实际完成

| 阶段 | 实际结果 | 证据 |
|---|---|---|
| 原论文与代码 | PDF/TeX已下载，核对CGR、手映射、训练、筛选和指标 | [论文—代码核对](PAPER_CODE_AUDIT.md) |
| 官方数据与权重 | 32模型635,102,941B，8份有效真实决策JSON 7,403,397,972B | public `CURRENT_STATUS.json`、[下载记录](data_intake/README.md) |
| 发布决策组件 | 31/31严格加载，作者test阈值P/R/F1均匹配文件名至4位小数；三手22类、1440不重复手/记录ID | [运行与指标](runtime/README.md) |
| CUDA依赖 | 现有Python3.12/Torch2.11/cu130/Blackwell上局部构建ME与原PointNet2，真实GPU算子通过 | [独立构建配方](native/README.md) |
| RGB-D→CGR | 原表示202 tensors严格加载，2048候选及480维CGR全有限，完整首帧forward 1.364s | runtime `evidence/official_rgbd_frame_20261103.json` |
| DH3完整离线推理 | 原identity+10增强，1544候选→原学习评分top400→碰撞后9个→原top10/top1选出最终手姿 | `outputs/official_dh3_eleven_branch_20261103_v1/RESULTS.json`及`selection/RESULTS.json` |

全部31决策文件都保留并评估，没有按test表现选最佳模型。1800条模型—记录计算含重复，不能当1800个独立例子。test可能参与过作者checkpoint选择；指标匹配证明组件与配置复现，不证明独立泛化或物理抓起率。根代理另按全部保存预测独立重算31份指标，见runtime `evidence/root_decision_predictions_verification_20261103.json`。

完整CGR使用原ME/PointNet2 CUDA计算，无替代网络；第一帧runner内部5.184s含输入/模型加载/保存，不含进程启动、下载、编译。DH3续段原学习评分0.417s、内部总2.849s。这是早期单identity组件验收。随后完整11分支已实跑：表示模型只加载一次，11分支15.988s，完整runner内部39.549s（含原评分、资产缓存核验、碰撞；不含进程启动/首次资产编译生成），最后选择另0.075s。DH3采用原深度域/宽度尺度，未混用Allegro默认。这些均非观察到抬升的耗时。

**完整DH3离线检测、原碰撞筛选与最终手姿选择已完成；尚未机械臂执行或仿真抓起。** 按原URDF/13 STL/已发布12d表和原采样函数生成28个碰撞点云槽，4路CPU/13.206s/零物理步，首次缺libusb失败保留。单identity400→59准入，11分支重新评分得到的另一个top400→9准入，原最终top1一致（type2、90mm开口、score1.55894；不是概率）。两批候选池不同，准入数不能直接当质量比较。全部真实碰撞bits、分支合并和最终选姿已经根代理独立重算通过。没有空点云替代。原DH3 loader循环清空字典的问题仍由独立runner显式装载4发布head解决；原评分/门/标定/碰撞/选择表达式保持。

## 数据与训练边界

官方文件专用目录为 `/mnt/public/datasets/AnyDexGrasp/`；**当前以 `CURRENT_STATUS.json` 为准**，`metadata/intake_20261103.json`仅是早期网络阻塞快照。三手开度标定表另列 `source_tables/`，不当作训练数据。

网络已由现有nros代理解决：`source /opt/nros_tools/nros_rc`后运行`nros-proxy-sg`或`nros-proxy-sg2`，只影响该进程。旧localhost:7897未启动，不再重复排查。

三个CGR训练标签ZIP共55.8GB仍为Google Drive按文件下载配额耗尽，已设每6小时一次、最多4轮的后台重试。另一份DH3 obj40 JSON官方源含异常字节，已隔离，未修补当有效数据。GraspNet完整RGB-D尚未下载；单帧来自固定提交的官方GraspNet demo，不能当独立AnyDex测试集。

**无需重训通用表示网络即可继续发布模型推理或新手适配。** 三个大标签包只在从头重训表示时需要。Boya需自己的手势/开口标定与该手成败数据训练决策器，不能把DH3/Inspire评分器当Boya已验证权重。

## 接续

原手动力学/控制的实际零步审计见[runtime/SIMULATION_HAND_READINESS.md](runtime/SIMULATION_HAND_READINESS.md)，场景数据与无需下载的最小合成输入见[场景附录](SIMULATION_SCENE_INPUT_APPENDIX.md)。官方2D→12D公式已提供并在355槽全部精确匹配；缺项是已验证的物理驱动/接触模型及一致的URDF配套，不是重新训练表示网络。

1. 保留已完成原DH3完整离线结果；不要重复从下载、依赖编译或单帧前向开始。原手碰撞只检查可见点云，未证明臂可达或抬升。
2. [官方没有桌面动态仿真](SIMULATION_AND_CONTROL.md)，只有真实UR5+D415/三手和资产准备工具，不直接运行硬件robot/collect脚本。
3. 再做明确标注的原手仿真移植或Boya适配；不能冒充论文真实抓取成功率复现。

复用已有解释器，依赖局部放`outputs/native_deps`，未新建环境、未改共享Isaac/Torch或系统头。模型/数据/PDF/二进制/大日志/视频不提交，只提交轻量工具、来源和凭据。仅有作者`upstream`，不向作者推送。CanonDex由另一服务器维护；本任务已按要求pull其最新代码，后续只处理独立AnyDex仓库，不派发原项目批次。
