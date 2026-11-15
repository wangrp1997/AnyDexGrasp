# GPU 数据通路与中断收尾只读核查

本次 GPU PhysX 诊断已由主执行者确认进入真实 GPU dynamics、250Hz、CUDA 完整接触读取及 native20µm 检查。这里核查的是其数据通路及取消行为，**不是本轮运行的最终物理验收或速度归因**。未启动 Kit、积分、附加进程、发送信号或修改冻结运行源码。

当前实现复用逐条原执行器，并让各条共享一个物理时钟。现成代码中确实存在更完整的批量资产与读取通路；两者有可直接核对的结构差异。不能仅据 GPU 利用率或这些差异，把本轮约 0.57 秒/步归因于某一个函数。

## 已核实的数据通路差异

| 路径 | 当前逐条适配 | 已有实现与官方接口 |
| --- | --- | --- |
| 关节命令 | 原执行器每步从 CPU 构造 position、velocity 两个 CUDA tensor | `template_gripper_batch.py` 在循环外构造 `command_gpu`，循环内取整批切片 |
| 状态读取 | 每条原 native 路径分别读取关节位置、速度、全机器人 body pose、主物体 pose；全三体包装又分别读取三体 pose/velocity | 同一批量参考把所有环境 q/v/body/object/force 拼成一个 `torch.cat(...).cpu().numpy()` |
| 接触读取 | 每条保留原 native object、native robot、全三体 object、全三体 robot 四套 filtered6 读取；两套 robot 查询重复覆盖同一机器人/静态环境 | 现成批量参考创建一套跨环境机器人查询与一套机器人至物体查询，再按环境分派记录 |
| 资产 view | 每个环境单独一个 Articulation、三个 RigidObject | 现成参考用正则路径匹配批量 Articulation；官方 `RigidObjectCollection` 以一个 rigid-body view 提供 `(env, body)` 数据和永久外力接口 |

正常活动条目每步约有 24 次接触 `Warp.numpy()` 调用，加上 10 次状态 `.cpu().numpy()` 转换；这是静态调用计数，不是实测耗时、GPU kernel 数或传输字节数。原 `contact_readback.py` 已只复制最大有效 endpoint 之前的前缀并保留间隙，**不能称它每步都把 262144 个槽完整传回**。官方 tensor API 的接触输出 buffer 也在首次读取时分配后复用，不能声称每步重新分配整个容量。

Warp 1.16 `types.py::array.numpy()` 明确说明：CUDA 数组在 default stream 上同步复制到主机。源码使用 `ScopedStream(null_stream)` 后 `to("cpu")`。因此许多小块主机读取是确定存在的同步边界；但相邻读取不等于每次都重新完成全部物理工作，不能用调用数直接估算加速倍数。

`ArticulationData`、`RigidObjectData` 按时间戳缓存状态；`update(dt)` 及重复访问 property 不自动等于再进行一次 native fetch。这里区分了 native 查询、CPU 转换及 Python 解码，避免把所有重复访问都算成同一种瓶颈。

## 可直接复用的范围与限制

优先参考 CanonDex 已有 `template_gripper_batch.py` 的整批命令预载、批量 Articulation 和整批状态读取；三体可采用本机官方 `RigidObjectCollection`。后者已有统一 native view、环境/物体索引和 `permanent_wrench_composer`，无需另建 IPC 或常驻服务。该参考是受控手部 fixture，不能原样代替完整 UR16e＋Boya 执行器；原 24 独立坐标、28 实际 DOF、mimic、初态、逐 tick 外力、原门和全场景审计仍须逐项保留。

官方 ContactSensor 的另一项可参考做法是：native buffer 就地更新，Warp 后处理留在设备上并复用 CUDA graph。但其源码明确禁止把 PhysX tensor 读取本身包含进 graph，避免重放陈旧接触。其平均接触点/聚合力输出也不能直接替换现有完整逐点接触日志。任何批量化版本须保留所有被引用的接触点、法向、法向力、分离量、原索引和每 tick 审计；不能靠少记接触、降低采样频率、改 `source_metrics` 或放宽门宣称等价提速。

上述只是来源明确的复用方向。本轮长诊断源码与参数保持冻结；没有实施这些更改，也没有测得它们在 Boya 三体上的净收益。

## 尚未定位的实际耗时

官方 `PhysxManager.step()` 自身调用 `simulate()` 和 `fetch_results()`。当前汇总 progress 不能把这部分等待与状态/接触读取、主机分析、gzip 日志开销分开。

项目已有一次相关历史证据：`evidence/collection_20261105/backend_benchmark_audit.json` 保存旧 GPU 双条目的 431 个完成时钟步，记录 `sim.step` 累计约 351.46 秒、跨环境接触保护约 0.719 秒，中断堆栈位于 `fetch_results`。该阶段存在共享 GPU0 负载，且任务与本轮不同；它不能直接解释本轮，但足以反驳“已经证明全部慢在 readback”的结论。原失败与在途第 432 步未确认范围继续保留。

现成 `physx_step_profile.py::profile_physx_steps` 对原 `simulate/fetch_results` 加 host wall 计时，不额外调用 synchronize、simulate 或 fetch。原 native 的 `profile_execution` 及 `benchmark_template_arm_parallel.py` 也已有分区计时。主执行者正准备独立 20 tick 分区入口；这里未启动或修改它。至少应分别报告：物理 simulate/fetch、命令写入、状态读取、四套接触读取、全三体分析和完整日志写入。host wall 包含等待，仍不能冒称独立 GPU kernel 时间。

## 原冻结超时与取消行为

`author_collection_batch.py::_stage` 在 `process.wait(timeout=...)` 发生超时或其他异常后，先向子进程组发送 SIGTERM，等待 5 秒，必要时再 SIGKILL。它随后保存基础设施失败，旧 started 不重启。自然超时会终止该阶段；顶层 control 的首组失败分支会停止后续组。

本机 AppLauncher 官方实现的 SIGTERM/SIGABRT 处理器直接执行 `app.close(exit_code=128+signum)`，之后按默认动作重新发出信号。fast shutdown 可直接退出进程，**不会先展开用户执行栈**。因此不能指望现有 1800 秒自然超时一定执行 vector 的 `finally`、关闭 gzip footer 并保存完整 partial receipt。SIGKILL 更没有用户收尾保证。

同一官方实现把 SIGINT 恢复为 Python `default_int_handler`，允许 `KeyboardInterrupt` 展开用户栈。若主执行者决定提前取消，来源支持的较好收尾路径是：先复核子阶段 PID、PGID、start identity 和继承 lease，再仅对该子阶段进程组发一次 SIGINT，由原 vector `finally` 关闭各 generator/journal、保存失败回执，最后关闭 App；父阶段等非零返回并保留原失败，不重跑原尝试。**不要对父 driver 发 SIGINT**，因为父 `_stage` 的异常处理会转而向子组发送 SIGTERM。这里仅核查方案，未发送任何信号。

SIGINT 仍不是可保证“精确最后积分数”的边界取消：它可能到达原生 `simulate/fetch_results` 内部。官方 `SimulationContext.step()` 在调用 physics manager 前就增加 `_physics_step_count`，所以这个数在中断时也可能仅表示已尝试的步数。vector 的完成计数则在 `sim.step()` 返回后增加；原 trace/接触观察在随后恢复 generator 时完成。必须分别保留：请求/尝试数、正常返回的时钟步数、已完成观察行数，以及最多一个在途步的不确定性。不能把它们强制写成相同的“精确物理步数”。

SIGINT 成功展开时，现有 vector `finally` 能保存 partial rows、journal 指针及 `recorded_physical_steps`/`observations_completed_steps`；不足整段的数据只作失败成本证据，不生成负例或正式标签。若只剩周期 progress 或未闭合 gzip，步数只能按实际可验证的完成前缀给下界并明确未知范围。进程组全部退出、slot/全局锁释放仍须独立确认。

证据目录：`evidence/author_gpu_data_path_review_20261115_v1`。其中 `SOURCE_FACTS.json` 给出逐项源码路径、SHA、精确行号和短定位片段；`STATIC_REVIEW.json` 保留最初只读复核；`EVIDENCE.json` 绑定报告与轻量材料。所有实际 GPU 结论以本轮独立运行回执和数值审计为准。本报告不批准正式大采集或训练。
