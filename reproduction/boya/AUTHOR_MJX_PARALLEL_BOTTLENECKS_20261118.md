# 并行吞吐问题定位：物理核并行，采集流水仍被串行屏障阻塞

2026-11-18 用户要求检索现成实现并核查并行为何没有发挥。结论是**当前16个物理世界没有形成16路持续采集流水**。问题在采集调度、逐控制主机同步、CPU逐环境观测/解码、过大的缓冲分配和串行审核，不能将现有速度解释成仿真的固有下限。

## 已有完整实测

新32槽/16环境/两轮均完成，20次抓取/20次回位；13正7负、12规划空结果。collect内部1917.017888秒，外部1925.387044秒；独立audit981.679487秒，export5.675468秒，完整2912.789502秒（48分32.790秒），合格诊断标签24.718573条/小时。两轮重复全部七个既有布局，非32独立新场景，不作为正式训练集。

推进＋GPU观察/读回389.467255秒，占collect内部20.316%；这是主机墙钟范围占比，**不是GPU利用率或CUDA kernel占用率**。照片/源面射线200.234470秒，CGR45.371332秒，两个规划wave各取三worker最长合519.820050秒，接触解码154.563342秒，setup80.769334秒，未进一步分区外层526.792105秒。不能把该余量全部归给JSON。

原始JSONL合13,061,330,009字节。独立审核158874分阶段行对完整18702控制周期/全部3511341点接触一致，掌姿对独立FK最大差5.176980e-7，所有完整抓取按原functional政策有效、20回位原七门通过。源/命令/标签/导出/head eval通过、0训练。完整费用2393856＋微对照196608＋换物检查32＝2590496原生环境步，未知0，本阶段上限5000000；历史阶段费用另列。

## 源码中确实存在的阻塞

| 问题 | 当前代码和实际行为 | 后果 |
|---|---|---|
| 整轮规划屏障 | [entry v9](author_mjx_batch_collection_v9.py)先逐环境photograph/CGR，再等待planning_batch全部返回，最后motion；拍照/规划期断言整个batch状态不变 | 一个环境准备好也不能提前执行，整个GPU物理batch暂停 |
| 每4ms主机同步 | [NativeControlGraph.advance](author_mjx_native_graph.py)每控制拷全部观察数组至host，并synchronize_stream | 16路物理并行仍被250Hz CPU处理节奏阻塞 |
| 逐环境CPU工作 | [NativeBatch.step/observe](author_mjx_native_batch.py)逐点转Python字典、按环境做CPU运动学/source极值/载荷和JSON；全局与分阶段重复存接触 | 环境增加会增加主线程工作，GPU需等待 |
| 等最慢任务再补充 | motion等所有抓取及物理回位完成，才进入下一wave；无候选环境整轮空闲 | 16槽每轮仅10个执行，不是持续补充的任务队列 |
| 审核串行挂在末尾 | [完整控制入口](run_author_mjx_queue_control.py)collect结束才执行完整原始audit | 另加16分21.679秒，不能当在线积分时间；持续生产还没有审核流水重叠 |

各环境机器人确实只初始化一次、1model/18bank/1renderer，跨轮换物保护机器人/其他环境/时钟逐值不变；**当前瓶颈不是每次抓取重新建场**。CGR已常驻一次加载32 forward，也不能重新宣称每帧加载模型是主因。

## 检索对照

1. [MuJoCo Warp官方性能与overflow说明](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#overflow-detection)明确指出CPU读取设备数组会触发主机设备同步，吞吐流水应避免逐步同步，或在设备端检查。这里应保留每控制overflow/原停止检查，在GPU计算它们；不能简单降低检查频率而漏掉原失败。
2. [官方性能调优说明](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#performance-tuning)说明contacts/constraints容量会影响内存和计算量。当前硬编码容量没有按实测需求调优。
3. [mjlab架构说明](https://github.com/mujocolab/mjlab/blob/main/docs/source/architecture_overview.rst)展示共享模型、批量Data、一次捕获图、批量动作/观测/停止处理的生命周期。新native graph已经做了一次捕获，但外围仍是串行wave。
4. [DexGraspNet原始validator](https://github.com/PKU-EPIC/DexGraspNet/blob/main/grasp_generation/utils/isaac_validator.py#L263)是先统一simulate一段，再遍历环境读接触。它没有本项目完整臂/RGBD/逐250Hz全字段合同，不能直接套速度；其PhysX GPU打开但use_gpu_pipeline=False，也不能误称整条管线都在GPU。

## 缓冲配置遗漏

对完整GPU_JOURNAL作零积分统计，并核SHA/总点数与独立审核相同：全局接触峰614，单世界约束峰316；实际分配分别4352和每世界8192，约为峰值7.09倍和25.92倍。

候选诊断容量按两倍实测峰向上取二次幂为naconmax2048/njmax1024；**尚未运行该容量的物理对照，不宣称提速或未来需求上界**。原overflow检查必须保留，任何容量溢出直接失败，不能截掉接触或约束。统计在`outputs/boya_mjx_allocation_review_20261118_v1/RESULTS.json`。

## 已完成的小范围代码修复及其限制

- 原全scratch staging每控制553649468字节/9FFI，已换同内核单图；三个同初态同指令短窗热段19.373629→7.516407秒，2.577512倍，仅推进/观察/读回组件。
- [无损记录器](author_lossless_journal.py)4项单元检查通过，176条实际全状态/接触记录三次轮换顺序对照：JSON中位0.455370秒，同步binary0.058659秒、异步binary0.063399秒（7.763/7.183倍）；全部字段/浮点/顺序exact，关闭包含drain和fsync，不把入队耗时当全部存储时间。
- [批量接触解码](author_mjx_bulk_contacts.py)12个保存的真实GPU包/1162点全部原生17字段及派生字段canonical exact；同样工作量中位0.801634→0.353823秒，2.266倍，0物理。
- 新native_batch_v3/entry_v10接上述两项、保留同一物理核/控制/判定；v3显式补回原JSON的有限值拒绝，不能因pickle支持NaN而放过无效数据。audit_v2兼容JSON和binary。完整旧轨迹35282记录迁移逐字段/浮点/顺序SHA exact，1,153,480,147→714,398,632字节；二进制独立审核18016分阶段行/232280接触、全部原门/标签与原JSON审核指标exact，0新增积分/标签。native step实录包fixture及NaN/Inf拒绝通过，同样0积分。尚无新入口完整物理吞吐结果。

组件修复没有取消整批规划屏障，也没有取消250Hz主机同步，因此不能声称“并行问题已解决”。真正需要的是GPU上保留逐控制判定、成块推进与无损状态缓冲，CPU异步处理观测/记录/审核，并让独立环境组在另一组规划时继续执行；不能直接让已拍照等待规划的世界继续积分而破坏观测绑定。

报告结果与代码均在本地工作区，原冻结source/input/archive、CanonDex和旧数据不改，0正式训练/新模型，未启动正式扩采。
