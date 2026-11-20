# AnyDexGrasp
Dexterous grasp pose detection network built upon [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).

## 当前主线（2026-11-20）

**仿真端到端闭环已跑通，目前在第一轮评分头效果验收。** 固定 g1 的常驻 MJX 采集已独立审核并合并 **117 条有效标签（101 fit、16 internal，79 正、38 负）**。首个正式评分头用前 100 条 fit 训练完成，权重重载和内部评分逐值核对通过。早期小样本评分头也已在新仿真场景完成拍照、打分、规划、提起、回位和审核。[端到端复核](reproduction/boya/evidence/author_mjx_mainline_20261120_v1/CHECK.json)

**先保留原 CGR 排序，暂缓正式扩采。** 在 14 个未参与训练或选模的归档布局上，正式头的 ROC-AUC 为 **0.556**，原 CGR 为 **0.622**。0.5 阈值只识别出 2/9 个成功抓取。另查 37 个同帧候选池，5 个多候选池的 15 个抓姿全部有原完整规划结果，每帧最多只有一个可执行抓姿。下一轮评估需要用同一帧至少两个可执行候选，成对比较真实提起，再决定是否扩充训练样本。[留出评估](reproduction/boya/AUTHOR_MJX_HEAD_HOLDOUT_20261120.md)

后续用另一批归档帧补做了 10 个完整臂规划，找到 4 个两种策略选择不同且均有计划的场景，24 轴实测起点核对一致。这 4 场的原 CGR 历史动作曾因深碰撞停止，仍待新布局的物理提起对照。[配对规划预检](reproduction/boya/evidence/author_mjx_pair_readiness_20261120_v1/SUMMARY.json)

主线使用已有 g1 和原闭合、原 CGR 排序。完整执行但未提起的样本保留为负标签，摆放失败、无候选、规划失败和无效执行保留为空，个别物体的闭合诊断留到后续。采集与评估全程不录视频。

## 2026-11-18 至 19 的扩采和首次闭环记录

2026-11-19 用户已明确要求[启动累计100条有效fit扩采](reproduction/boya/AUTHOR_MJX_TO100_20261119.md)：现有16fit还差84，沿原CGR／g1，不录视频，16环境整个队列常驻。800新场景为上限（静态480ready），达到累计100有效fit并完成审核后自动收尾，保留在途超额；只合并原正式同域20条，不混诊断／对照标签。进度看ignored `outputs/boya_mjx_cgr_to100_20261119_v1/collection/BATCH_COUNTERS.json` 的累计已审核fit，启动事实看RUN_STARTED及原生积分日志，最终CAMPAIGN_RESULTS／merged/MANIFEST。

2026-11-19：[评分头实抓对照已完成](reproduction/boya/AUTHOR_MJX_POLICY_COMPARISON_RESULTS_20261119.md)。8个新布局／16槽，head组唯一实际执行的一次提起与回位及完整审核通过，训后新场景执行闭环已通。但13个候选池为空、3个仅单候选，评分头没有改变任何top1；两边无合格实抓配对且实际观测不同，**此次无法证明排序收益**。保持原采集排序，不自动追加配额；0录像，进程与资源均已收尾。

用户优先跑通闭环，**后续不录视频**。现有20条真实数据已完成[小样本训练与推理验证](reproduction/boya/AUTHOR_MJX_HEAD_PILOT_20261118.md)：16fit／4internal，原模型与训练函数100epoch、100次更新，CPU训练约2.87秒；权重保存重载及原推理逐值核验通过。此为16条工程pilot，正式100条训练规模与新场景评分实抓尚未完成；当前无新采集任务。上一批视频文件为0，占用主要来自完整轨迹/接触JSONL。

**首批100新场景已完成**：正式导出20条（16正／4负，fit16／internal4），7批完整审核通过，实际采集和收尾约72分钟；其余80场为空（37布局、27碰撞筛选无候选、15规划拒绝、1深碰撞停止）。原统计异常已由独立只读收尾补齐，0物理重试；32登记进程组已退、资源释放，当前无本批后台任务、训练0。正式数据在本地ignored `outputs/boya_mjx_fresh_expansion_20261118_v1/formal/`；[最终报告](reproduction/boya/AUTHOR_MJX_FRESH_EXPANSION_RESULTS_20261118.md)与[轻量结果](reproduction/boya/evidence/author_mjx_fresh_expansion_20261118_v1/SUMMARY.json)。以下启动说明与运行快照保留。

用户已授权在代码推送、工作区清理后，开始首批 **100 个全新场景**：80 fit＋20 internal，目录40物体保持32/8身份隔离，固定g1五指握、16常驻环境。沿用真实一小时已验收的native物理与完整审核，不使用重复旧布局；正常失败保留0，未执行／无候选／无效执行为空，不补抓。固定抽样涵盖40种，几何可建场的63场涉及38种；040和046本批只出现在placement_null，不能声称实际抓取覆盖全部40。

入口为 [`run_author_mjx_expansion.py`](reproduction/boya/run_author_mjx_expansion.py)，本批运行产物在 `outputs/boya_mjx_fresh_expansion_20261118_v1/`（已ignore）。`collection/BATCH_COUNTERS.json` 查看落盘和独立审核数量，`PROGRESS.json` 查看阶段；正式数据仅在完整审核后生成于 `formal/`，训练不自动启动。启动事实以 `PIPELINE_STARTED.json` / `collection/RUN_STARTED.json` 为准；见[扩采协议](reproduction/boya/AUTHOR_MJX_FRESH_EXPANSION_20261118.md)。

实际连续一小时已测得39条有效标签落盘，其中36条也在时限内完成回位与独立审核；整机平均92.3／100秒一条。组件倍率和旧小批次折算不代替这个实测口径。此前报告中的“未完成／未启动”是历史状态。

## MJX接触差异：主要漏检已修复，仍有残差（2026-11-18）

定位原1e-10米CCD精度在float32 GPU中漏掉食/中/无名指接触；GPU副本改用MuJoCo默认1µm后，原三指载荷恢复，128步机器人关节偏差0.314631→0.000029617rad。CPU参数、驱动、几何和穿透验收门保持。修复版1/16/64常驻两轮换物通过，64环境物理吞吐为CPU串行约2.21倍；所有可比实例末态最大差仍0.01284rad，六态回归还留一处0.104N弱接触差异，尚未完成完整抓取/采集验收。见[修复报告](reproduction/boya/AUTHOR_MJX_CCD_FIX_20261118.md)。以下2.33倍是修复前短段历史结果。

## MJX并行短段测试：64环境常驻换物通过（2026-11-18）

完整六轴机械臂＋Boya手＋16物体银行，MJX-Warp实跑1/16/64环境，每环境机器人只初始化一次，两轮换物前后全部机器人q/qvel和时钟exact，0机器人重置。64环境两轮物理块约5479环境步/秒，相对匹配CPU串行约2.33倍；单环境和16环境本组较慢。纯MJX-JAX首次编译/首块240秒超时，原记录保留。

这是关闭原noslip的独立诊断域，且GPU手指末态与CPU有明显差异；仅固定闭合短段，尚无相机/CGR/规划/完整抓取回位和逐帧接触审核，**不能称整个采集已提速2.33倍或直接正式采集**。本轮0新标签/训练，资源已释放。见[测试报告](reproduction/boya/AUTHOR_MJX_PERSISTENT_BENCHMARK_20261118.md)。正常完整抓取失败可保留为负标签，不以每条抓起或力闭合作为采集前提；保持固定协议，按整批正例数量与覆盖评估数据。

## Boya采集管线：有限功能验收通过（2026-11-17）

用户目视诊断视频已导出：[物体014单段](outputs/boya_mujoco_failure_videos_20261117_v2/trial_01_slow.mp4) · [三次合集](outputs/boya_mujoco_failure_videos_20261117_v2/three_grasps_slow.mp4) · [逐帧播放器](outputs/boya_mujoco_failure_videos_20261117_v2/index.html)。双视角、五倍慢放，按实际q/物体姿态与载荷重渲染，0新积分；前臂隐藏/掌壳透明仅为观察。详见[视频说明](reproduction/boya/AUTHOR_MUJOCO_FAILURE_VIDEOS_20261117.md)。原控制保持，本轮只做上面的并行后端测试。

正式扩采仍未启动，队列和规模尚未冻结。同版本3次全负不足估计总体成功率；正常失败有负样本价值，不要求逐条修到成功。只读复查的闭合/保持接触丢失诊断仍保留，见[阶段统计](reproduction/boya/evidence/author_mujoco_failure_diagnosis_20261117_v1/SUMMARY.json)。

修正版同一世界/机器人/相机处理固定6场，3次完整抓取及控制回位通过，导出 **2 fit＋1 internal，共3条负标签**；原始逐帧审核与原版480→20评分头的实际数据读取/loss检查通过，权重未更新。另两场无候选、一场未稳定均保留空标签，不补抓。修正限位响应、迁回源网格质心/惯量，并在独立版本中启用已有桌面滚动摩擦；不能称原PhysX严格物理等价。39项检查通过，无后台任务。

**当前完成的是采集流程验收，正式扩采和训练未启动。** 本批无正例，前版1正例独立保存不混入，不能用3条数据直接训练模型。下一阶段是冻结正式单型队列和规模。详见[收尾报告](reproduction/boya/AUTHOR_MUJOCO_COLLECTION_CLOSURE_20261117.md)与[全部验收检查](reproduction/boya/evidence/author_mujoco_collection_closure_20261117_v1/CLOSURE.json)。下面未通过的段落保留为历史。

## Boya采集验收：实际提起，但尚未通过（2026-11-17）

关节限位修正已实现：此前关节仍用20ms默认响应，现提供独立1ms限位配置。原开手转换及THJ2双向±1Nm独立测试均在原0.002rad容差内，35项检查通过；范围、驱动、动作保持，不裁剪状态。**完整抓取尚未重验，正式采集状态不变。** 见[限位修正](reproduction/boya/AUTHOR_MUJOCO_JOINT_LIMIT_FIX_20261117.md)。

固定4场验收中，首场无可执行候选；第二场新拍照/原CGR/规划/3591步抓取完整，将015提起约10.26cm，拇指11.62N及其余手指受载。但THJ2最大越限0.0367rad，控制回位也未通过限位门，因此保留空标签、后两场未派发，**不能启动大批量采集或训练**。完整轨迹独立复核通过，实际训练导出0行；32项CPU检查通过。1世界/相机/初态写入、8782控制步/70256原生积分，0重试，原结果及旧数据保持，无后台任务。见[验收报告](reproduction/boya/AUTHOR_MUJOCO_COLLECTION_ACCEPTANCE_20261117.md)及[证据](reproduction/boya/evidence/author_mujoco_collection_acceptance_20261117_v1/ACCEPTANCE_SUMMARY.json)。

## Boya新观测全流程：已接通，物理准入未过（2026-11-17）

最新判定调整：按用户要求，**物体之间的小幅穿透只记录，不再单独否决**；成功仍看真实提起离桌并由手承载。规则已独立接入，原记录保持。对既有首轮0积分重判仍未成功，因为没有提起任何物体。见[判定说明](reproduction/boya/AUTHOR_MUJOCO_LIFT_POLICY_20261117.md)。

MuJoCo常驻世界的首轮“新拍照→原CGR→常驻规划→抓取→控制回位”已实跑，outer **62.92秒**，其中规划31.65秒、执行与后处理14.72秒、回位6.05秒。1世界/1相机/1显式重置，照片后状态逐值保持；这不再是只重放保存动作。但本轮物体—物体穿透1.538mm超过原1mm门，第二轮换物后未过原稳定窗口，未派发第二抓。**不能当有效采样吞吐或正式训练数据。**

此前两份保存计划也已在同一世界通过控制回位连续跑完，parent45.47秒，首抓有效、次抓两项数值门失败。相机零积分深度审核和38项CPU检查通过；全部失败、初始化和0.5ms源积分成本均保留，无后台采集。详见[新观测采集诊断](reproduction/boya/AUTHOR_MUJOCO_FRESH_COLLECTION_20261117.md)与[轻证据](reproduction/boya/evidence/author_mujoco_fresh_collection_20261117_v1/SUMMARY.json)。

## Boya一次初始化、只换物体候选（2026-11-17）

新MuJoCo候选已验证：一个世界/一次显式重置/一次机器人初态写入，跨两次换物保持机器人位置、速度、时钟和模型句柄exact，换物约1–3ms。一条保存的3818步计划完整执行，原七门及提起受载判定经独立只读复算通过；parent24.86秒含一次性建场12.31秒。运行在首轮结果接口报错，原失败保留，第二轮未开始；修正未重跑。它仍是新后端诊断，尚未接新拍照/CGR，不能当完整采集24.86秒或PhysX等价。

常驻规划另在同两份输入上从合计161.01降至84.14秒，指令及26份完整几何报告相同；29项CPU检查通过。新规划入口仍继承旧PhysX每轮复位，严格一次重置的MuJoCo世界需另完成两轮全流程集成。详见[一次初始化与换物报告](reproduction/boya/AUTHOR_PERSISTENT_WORLD_REUSE_20261117.md)及[轻证据](reproduction/boya/evidence/author_persistent_world_reuse_20261117_v1/SUMMARY.json)。正式采集/训练未启动。

## Boya提速进展（2026-11-17）

新入口[`author_accelerated_fixed_scene_collection.py`](reproduction/boya/author_accelerated_fixed_scene_collection.py)保留固定场景、原物理和全部检查，编译全源几何计算以减少临时数组。完整同一份保存数据的审核从69.2降到30.6秒；新一轮同图像、同3818步命令的完整提起与审核通过，parent耗时9分49秒，执行器从旧对应310.6降到250.7秒。对应阶段总和约缩短27%，属历史匹配观察，不能当全量采集倍率；旧36分12秒是两轮，不能与新一轮直接相除。

同时纠正此前计时解释：旧“物理推进＋更新”桶包含外层观测。新直接测得物理82.2秒、完整观测96.0秒；仍需改善接触处理与存储，尚未达到作者实机节奏。原几何、250Hz、64迭代、质量摩擦和完整接触保持，原生轨迹与旧对应场景exact。25项检查通过。仅1条排除训练的诊断正例，正式扩采/训练未启动，新映射fit仍2条。运行命令与完整限制见[提速报告](reproduction/boya/AUTHOR_COLLECTION_SPEEDUP_20261117.md)，来源与结果见[轻量证据](reproduction/boya/evidence/author_collection_speedup_20261117_v1/SUMMARY.json)。

## Boya当前入口：固定场景连续采集（2026-11-16）

已实际跑通同一个机器人、桌面、相机和物理世界中的两轮“换物体→稳定→拍照→CGR→抓取→记标签”，拍照到抓取不清场、不重建、不重置物体。1Kit/1建场/1相机，CGR载入1次、前向2次；两轮分别提起成功和提起失败，初始化及完整执行审核均通过。两轮完整耗时36分11.808秒，**尚未证明整体加速**，主要耗时仍在物理执行和审核；这两条为排除训练的流程诊断，正式扩采和训练未启动。此前新映射2fit保持不变。

新入口为[`reproduction/boya/author_fixed_scene_collection.py`](reproduction/boya/author_fixed_scene_collection.py)，支持冻结有限场景列表后在一个物体银行中循环；不依赖调试失败记录。可复制的`freeze`/`run`命令、成本及限制见[固定场景报告](reproduction/boya/AUTHOR_FIXED_SCENE_COLLECTION_20261116.md)。原作者实机入口见下文；Boya硬件驱动和标定仍需另接。

[查看两轮实际RGB对比](outputs/boya_fixed_scene_rounds_20261116_v1/same_world_two_rounds.png) · [首轮实际提起图](outputs/boya_fixed_scene_first_execution_20261116_v1/actual_execution.png) · [轻量结果与来源证据](reproduction/boya/evidence/author_fixed_scene_collection_20261116_v1/SUMMARY.json)。原始大数据保存在public，不进Git。

## Requirements
- [Anaconda](https://www.anaconda.com/) with Python 3.8
- PyTorch 1.13 with CUDA 11.7
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) v0.5

## Installation
1. Follow MinkowskiEngine [instructions](https://github.com/NVIDIA/MinkowskiEngine#anaconda) to install [Anaconda](https://www.anaconda.com/), cudatoolkit, Pytorch and MinkowskiEngine.**Note that you need ``export MAX_JOBS=2;`` before ``pip install ``due to [this issue](https://github.com/NVIDIA/MinkowskiEngine/issues/228)**

2. Install other requirements from Pip.
```bash
    pip install -r requirements.txt
```

3. Install ``knn`` module.
```bash
    cd knn
    python setup.py install
```

4. Install ``pointnet2`` module.
```bash
    cd pointnet2
    python setup.py install
```

5. Install ur toolbox.
```bash
    cd ur_toolbox
    pip install .
    cd python-urx
    pip install .
    pip install -r requirements.txt
```
6. Install Allegro Hand.

   Install Allegro Hand upon [Allegro-Hand-Controller-DIME](https://github.com/NYU-robot-learning/Allegro-Hand-Controller-DIME).

7. Download model weights and data at [GoogleDrive](https://drive.google.com/drive/folders/1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo) and put it under ``logs/`` and extract zip files in ``logs/data/representation_model/graspnet_v1_newformat/`` to ``logs/data/representation_model/graspnet_v1_newformat/``.   
   Download the data from the [Graspnet](https://graspnet.net/datasets.html) web page and extract it to ``logs/data/representation_model/graspnet_v1_newformat/``

## Generating the STL file for dexterous hand.
```bash
    python command_generate_mesh_file.sh
```

## Training
```bash
    sh command_train_representation.sh # Representation model
    sh command_train_decision.sh # Decision model
```


## Collecting data
```bash
    python realsense.py
    sh command_collect_multifinger_grasp_data.sh
```


## Robot grasp
```bash
    python realsense.py
    sh command_robot_multifinger_grasp.sh
```


## License
The code is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) for non-commercial purposes.


## Bibtex
If you find the code useful, please consider citing our paper
```
        @article{fang2025anydexgrasp,
          title={AnyDexGrasp: General Dexterous Grasping for Different Hands with Human-level Learning Efficiency},
          author={Fang, Hao-Shu and Yan, Hengxu and Tang, Zhenyu and Fang, Hongjie and Wang, Chenxi and Lu, Cewu},
          journal={arXiv preprint arXiv:2502.16420},
          year={2025}
        }
```

