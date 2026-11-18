# MJX执行开销修复与连续队列验收（2026-11-18）

用户指出低吞吐来自代码。本轮已定位并修复执行路径中不必要的完整scratch往返复制和多次FFI调用；原来的八槽测试也只有一轮，不能用总标签率除以16来判断单环境持续产能。新16环境连续两轮已完整采集、审核和导出，20抓取/20回位、13正7负。**在线采集31分57秒，含独立审核共48分32.790秒，仅24.719标签/小时，仍然慢；推进＋GPU观察/读回只有6分29秒，不能把慢主要归咎于物理求解。** 最新检索与架构问题见[并行瓶颈定位](AUTHOR_MJX_PARALLEL_BOTTLENECKS_20261118.md)。

## 已确认的实现问题

旧 `author_mjx_graph_staging.py` 为恢复完整私有接触字段，将每次step/forward的全部in-out数组加入stage_in和stage_out，其中包含大块求解器scratch。实际16环境记录：每个0.5ms step输入30,824,852字节、输出30,709,640字节；一个4ms控制周期执行八次step及一次forward，合553,649,468字节GPU暂存拷贝、九次FFI调用。这是该暂存路径的字节数，不是全部GPU流量，也不能把全部旧耗时都归因于拷贝。

新 [author_mjx_native_graph.py](author_mjx_native_graph.py) 使用已安装同一MJX-Warp内核，将八次step、一次完整live→独立observer复制及observer forward捕获为一个CUDA图。图地址固定，每个控制周期提交一次，再以固定pinned host缓冲批量读回。16环境独立observer复制30,752,072字节/控制，全部原生接触17字段与EFC载荷保留；observer与live逐数组拒绝别名，live warmstart在观察前后逐值核验。捕获前后q/qvel/time/ctrl/warmstart相同，0隐藏积分。

新 [author_mjx_native_batch.py](author_mjx_native_batch.py) 只在某环境实际被观测时更新其CPU运动学镜像。CPU没有积分或接触求解。所有环境（含空闲环境）的原生状态及全部接触仍逐控制保存。native费用日志逐行写入，在每个控制块前后分别记录pending/completed，避免旧1MB默认缓冲导致中断费用边界不清楚。

## 同初态、同指令的组件对照

根目录 `outputs/boya_mjx_execution_path_control_20261118_v1`。取旧完整16环境轨迹三个无换物事件窗口（接近前、受载闭合、回位），每个128条原float32控制，四种模式都从相同q/qvel/time/ctrl/xfrc和零warmstart开始；合196,608原生环境步，0新抓取标签。

| 原窗口 | 旧完整暂存热段秒 | 新单CUDA图热段秒 | 观察倍率 |
|---|---:|---:|---:|
| 接近前 | 5.514778 | 1.539942 | 3.581 |
| 受载闭合 | 8.695606 | 4.480493 | 1.941 |
| 回位 | 5.163245 | 1.495972 | 3.451 |
| 合计 | 19.373629 | 7.516407 | **2.577512** |

每段热计时仅后127条advance，不含首条JIT、建场/捕获、拍照/CGR/规划、JSON和独立审核，不能报告为完整采集2.58倍。旧模式原样重复热段19.598秒，重复本身未消除主要开销。

备选 `author_mjx_graph_io.py` 的最小stage名单通过两项CPU合同检查，但同进程微对照中的JAX primitive lowering复用了旧FFI，未证明实际替换，`observer_copy`模式计时全部排除。完整队列使用的是已实际执行的native单图方案。

新旧不是逐值相等：受载窗口机器人q最大差0.001475573rad、qvel最大差0.512053rad/s，源body位置最大差0.118315mm；旧自身重复也有轨迹差，但不足以据此宣称完全等价。三窗口新方案限位最大0.000604600rad、mimic最大0.000280231rad，低于原0.002rad门，只是这三个短窗结果；完整抓取仍按原功能判定与独立原始日志审核，不放宽门。

## 常驻与换物检查

`outputs/boya_mjx_native_bank_check_20261118_v1` 已PASS：两环境、两个控制周期共32原生环境步，一次共享模型/18对象银行/相机，各机械臂只初始化一次。第二次给环境0换物时，实际GPU机器人q/qvel、其他环境q/qvel、所有time/ctrl/warmstart逐值不变；CUDA图地址不变。独立CPU FK与原生palm一致。建场78.096秒只发生一次；两次advance合0.025433秒，不能拿两步时钟当完整采集速度。

## 两轮连续队列

根目录 `outputs/boya_mjx_continuous_queue_control_20261118_v1`，[entry v9](author_mjx_batch_collection_v9.py)、[准备入口](prepare_author_mjx_queue_control.py)、[完整验收入口](run_author_mjx_queue_control.py)。固定32槽、16环境、两轮；循环纳入全部七个旧placement-ready布局，不按旧抓取结果筛选。每次重新稳定、实际拍照、CGR、原实测起点RRT/几何/retiming、执行、提起判断和物理回位；不复用旧照片、指令或标签。

这是布局重复的有限队列利用率诊断，**不是32个独立新场景，也不是正式扩采数据**。无候选/不稳定为空、完整失败为0，不补抓。六臂与Boya手在同一世界持续运行，轮界限只换对象，不写robot q。GPU沿既有独立测试域noslip0、CCD1µm、float32；250Hz控制/原生0.5ms、原24位置及六臂速度前馈、全部接触保留。既有functional政策保留原1mm穿透诊断、原5mm停止及其余六门，没有为性能测试放宽物理接受规则。

两轮collect已complete：32次新RGB-D/CGR/规划，20次完整抓取（13正7负）与20次物理回位，回位原七门全过；另8个no_candidate、4个null_geometry保留，不补例。机器人各初始化一次/0reset，1共享模型/18银行体/1renderer/1次CUDA图捕获，18702控制周期/2393856原生环境步。collect内部parent1917.017888秒，外部派发1925.387044秒。仍使用相同CPU MODEL.xml SHA，与旧16基线完全一致。

| collect内的实测范围 | 秒 |
|---|---:|
| 建场/模型上传/捕获等setup | 80.769334 |
| 推进＋独立GPU观察/读回 | 389.467255 |
| 全量原生接触解码 | 154.563342 |
| 32次拍照及原射线检查合计 | 200.234470 |
| 32次CGR帧处理合计 | 45.371332 |
| 两轮三个CPU规划worker各取最长后相加 | 519.820050 |
| 未进一步分区的外层余量 | 526.792105 |

推进＋GPU观察/读回占内部parent20.316%。新receipt沿用旧字段名`contact_decode_and_kinematics_s`，在native实现中其154.563秒**仅计接触解码**；按需CPU运动学计在外层，不拿该旧字段名误认成总观察耗时。规划worker内部计时不含进程导入/等待，余量包含按需CPU运动学、source极值/载荷、JSON/IO、计划绑定、原数值评价和其他外层开销，不能把526.792秒全部归给JSON或某个kernel。

完整JSONL记录合13,061,330,009字节，含全局与各阶段的接触/状态重复记录。后续性能工作应针对这些已量到的代码路径，而非直接外推仿真速度下限。新32槽与旧八槽的任务量/有效候选/状态不同，没有完整同32槽旧执行对照，不给完整代码因果倍率。

完整collect→只读audit→诊断export均已exit0：外部collect1925.387044秒、audit981.679487秒、export5.675468秒，完整2912.789502秒，20合格诊断标签/24.718573条每小时（13fit/7internal、13正7负，另12null）。158874分阶段行/3511341接触对完整原生全局日志一致，source极值、载荷、原控制/七门/lift/返回、RGBD/CGR输入和导出/head eval通过；palm对独立FK最大差5.176980e-7。独立CGR常驻1load/32forward、0评分头选择/0训练，权重不变。该同布局重复cohort独立，不自动合并旧数据，不推正式采满ETA。

v9的`comparison_underfilled`布尔沿用旧八槽比较描述，不反映本轮队列；本轮实际两wave各16个输入槽，候选空结果另计，旧冻结元数据不回写。v10新freeze按实际rows计该字段。

## 成本与范围

本修复阶段预算5,000,000原生环境步，实际组件196608＋换物32＋连续采集2393856＝2590496，未知0、0重试，所有空闲环境积分计入。12登记owner/PGID全退、3预约槽released、5锁可取/registryreleased，原物理批次已收尾。此前阶段的费用和中断未知边界独立保留。0Kit/正式训练/权重更新；CanonDex、历史冻结source/input/archive、旧数据和失败结果保持。

后续只做零积分CPU修复验证：无损binary记录4测试通过、176实录记录写盘中位7.2倍组件收益；bulk解码12真包/1162点logical exact、2.266倍函数收益。新native_batch_v3/entry_v10接异步无损trace与bulk，显式保留原JSON有限值拒绝和全部writer失败清理；auditor_v2读两种格式。旧完整轨迹35282记录迁移后字段/浮点/顺序logical SHA exact，二进制独立审核的18016分阶段行/232280接触/原门/标签与原JSON指标exact，0新增积分/标签。Native step包fixture和NaN拒绝通过，不当实际GPU运行。

`outputs/boya_mjx_binary_entry_static_20261118_v1`仅freeze16槽/16环境，0Kit/积分/派发；最多2380800，与前已花合上限4971296<5000000，未自动启动。新完整IO入口吞吐未测；整批规划等待和250Hz主机同步仍在，不能称并行问题已解决。最新候选容量2048/1024也仅基于完整日志统计，尚无其物理对照。

微对照只读复核：[READ_ONLY_REVIEW](../../outputs/boya_mjx_execution_path_control_20261118_v1/READ_ONLY_REVIEW.json)；换物检查：[RESULTS](../../outputs/boya_mjx_native_bank_check_20261118_v1/RESULTS.json)；连续协议：[CONTROL_PROTOCOL](../../outputs/boya_mjx_continuous_queue_control_20261118_v1/CONTROL_PROTOCOL.json)。
