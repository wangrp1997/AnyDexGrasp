新的采集准备入口和完整臂接口已接通；当前验收范围是**已有真实 RGBD/CGR 缓存的离线准备与纯规划**，不是正式采集、多物体搬运验收或新成功率。新增标签、CGR 前向、物理步和正式训练更新均为 0。

| 环节 | 本阶段实际行为 |
| --- | --- |
| 输入 | 原始未筛选 Nx15 候选及成对 Nx2261 特征、原 CGR 稀疏输入场景云、绑定的观察/校准/手表面库 |
| 前门 | 仅明确的传感器/工作区/原始宽度/接近方向适配；不经过旧 Boya 手、桌面、深入可行性或学习评分门 |
| 作者采集 | 原 DH3 top500、固定请求类型、原单次翻转、独立均匀 0/10/20/30mm 额外深入 |
| 手几何 | Boya 计算参考表的 ff 对手指、1.7 开度展开及稀疏表夹限/2mm 最近槽；全手原表面云 |
| 碰撞 | 完整原 DH3 collection detector：3mm voxel、approach 参数 4cm、允许包含场景点数≤2、无中心修正 |
| 选择 | 几何准入后按原 CGR 分数降序保留 top10，固定 index0；不按臂可规划性重排，也不尝试其它 top10 |
| 动作绑定 | 原始行、实际翻转行、canonical480、原始/额外深入、槽、相机与世界掌姿、动作合同和来源 SHA |
| 完整臂 | 单个原首选转为 `candidates.npz`/`generation.json`，调用原 `native_arm` prepare；不启动执行 |
| 标签 | 命令与执行回执双向绑定；仅完整且合格的同任务搬运/释放结果可为 0/1；无计划、基础设施和未完成为 null |

代码为 [author_collect.py](author_collect.py)、[author_collection_bridge.py](author_collection_bridge.py)，均为独立 opt-in 入口，原算法和旧冻结工具未改。`prepare_collection(...)` 返回 JSON 候选和可审计 NPZ trace；`run_protocol(...)`/CLI 读取冻结 JSON。协议 schema 为 `anydex_boya_author_collection_protocol_v1`，`inputs` 要求 `raw_candidates`、`representation_outputs`、`representation_receipt`、`observation`、`calibration`、`hand_cloud_manifest` 各自的 path/SHA。CLI 拒绝旧筛选池、11 分支增强缓存、输出哈希错配、观察错配和虚报的输入深度范围。源 single-forward 回执证明输入位于任何手几何门之前。

`domain.mode=source_optical` 精确复用作者光学候选门；`declared_world_rig` 显式给定世界 AABB、世界接近方向锥、宽度区间及 sensor_adapter。既有 A0 图像为本机 fx=fy640、深度 .2–.65m，作者源输入为 .15–.72m；相机/rig 差异保留，不冒称相机参数相同。这里固定世界 box `[-1,-.04,-.02]` 至 `[-.57,.24,.30]m`、朝世界 -Z 的 dot>.9、宽度 .01<w<.099m。没有使用真值物体分割或物体身份挑候选。

源 DH3 **采集翻第 0/1 列，部署翻第 1/2 列**；代码和候选均保留这个差异。采集实际执行姿态来自前者，canonical480 则按源保存的 if_flip 和发布训练/推理的特征半周交换、角度循环规则生成一次，不调用旧 `canonical_candidate(... forcedflip ...)` 产生另一种姿态。原递归扫掠也直接保留：approach=.04 的这条源循环实际包含端点和后退 2cm 的云，不改成旧 21 个均匀采样。

已有 EGAD A0 开发缓存的四型结果相同：2048 原候选，原始宽度通过 1072、世界工作区 568、接近方向 496，联合 8；每型 8 次完整原碰撞、3 准入，固定首选均原行 7、该首选随机额外深入 0。首选相同是这张旧观察上的结果，不是四种手型等价或成功。精确作者光学门在同缓存联合 6 个，与世界适配门 8 个明确区分。该观察是已使用的孤立物体，不称新多物体数据或未见外部测试。

独立 NumPy 重建工作区门、排序/MT19937、源第 0/1 列翻转、480 特征交换/循环和原 depth 平移；四型共 12 个保存准入候选全部通过。原归一化 1cm 探针方向所得 float32 深入平移与保存值最大差 **0m**，深入只加一次。手动作合同由另一并行模块独立重建，候选只记录数值通过、`physical_semantics_validated=false`。

完整臂桥接只写单行原首选池，不再次映射、不再次加深入、不迁移物体。为满足原臂 API 的严格 SE3，仅对 float32 旋转舍入作 SVD 投影，掌平移不变，差值逐候选记录。兼容物体由场景显式传入；不根据候选距离、物体身份或已知结果选择物体。原臂规划仍只检查这个兼容物体，`additional_scene_collision_checked=false`，因此此阶段 ready **不足以**证明三体场景或箱体路径安全。

A0/g6 的实际原首选 native 准备通过。源命令为 2100×24 列：6 臂、2 腕、16 独立手电机；原执行器才扩展 4 mimic。桥接审计独立展开这四条源公式，验证 20 手关节、原闭合路径和实际候选掌姿；保持原 24 列计划文件不变，不把规划展开称实际观测到 28 关节物理命令。原计划添加 `author_collection_binding`，供后续多物体搬运桥接。派生 backend 可显式绑定 primary 物体 `collision_refinement`，路径及 SHA 传给 native prepare；其它参数由原 backend 配置保持。

10 项测试及独立缓存审计通过。测试包括 520 条合成输入的 top500 截断/源单翻转/特征配对/后几何选择、源光学严格边界、世界方向坐标变换、五种错误缓存拒绝、六种标签状态、真实 native 24→28 规划展开、修改掌姿/闭合合同拒绝及原首选 null 不尝试后续候选。没有优化器更新。

开发过程完整保留：四批×四型共 16 个离线准备、128 个原碰撞查询，最后 `final_v4_g*` 为最终绑定；旧结果不回写。最早边界测试误改了旋转单元素导致非 SO3，修正为合法边界旋转后通过。第一次 native 计划本身通过，但后置审计误要求源 24 列计划显式包含 mimic，0 物理失败已留档；改为审计独立展开。另一次最终准备验证接口后，为传递源绑定 collision_refinement 再做独立 final_v2 桥接；各次仍为 0 物理，不能冒充新试抓。时间仅作本阶段成本，见轻量 summary，不作加速倍数结论。

本地入口：`outputs/boya_author_collector_20261114_v1/`；最终桥接 `bridge_final_v2_g6_arm/ARM_PREPARATION.json`。轻量证据见 [evidence/author_collector_20261114](evidence/author_collector_20261114)。大型 trace、候选池和原 628MiB 表面库均留忽略输出。CanonDex 只读，原 435、八权重、外部成绩及旧冻结证据不改。

后续由总入口连接新的动态多物体单帧观察/CGR、全场景路径检查、无状态重置的接近—闭合—撤回—搬运—释放—释放后检查；通过后才采兼容任务数据、正式重训和冻结外部验收。此文件描述本子阶段边界，不将接口存在或离线 ready 写成完整复现通过。
