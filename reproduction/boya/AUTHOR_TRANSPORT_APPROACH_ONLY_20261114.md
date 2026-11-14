# 搬运入口 v4：移除无用的旧 world-Z lift 准入

本补丁修复上一版需要先规划一个随后会丢弃的竖直 lift 的限制。统一入口现在只请求原生 `--approach-only`，保留原 settle/home/approach；然后用原 `TwoSidedClosure` 补足 250 步 close、50 步 hold，再规划真实需要执行的退 4mm、等待 0.1s、退回 pregrasp、固定 bin 搬运和开手。没有改变原手指端点、24 列独立关节格式、初态、驱动、限位、碰撞门或候选选择预算。

`author_pipeline.prepare(..., archive_sources={原源码绝对路径: 历史快照绝对路径})` 及 CLI 的 `--archive-sources mapping.json` 支持复用已完成的相机观察。只允许 source_sha256 中显式列出的源码按原 SHA 解析归档；物理资产及其它输入仍逐项检查现文件。当前执行源码另由新协议绑定，不执行归档源码，不回写 scene/observation，也不把解析旧源码写成重新渲染。v4 真实三物体观察的 518 项来源锁已通过这一解析。

`execute` 现在持有模拟器 launcher，在 `COLLECTION_RECORD.json`、最终 `PIPELINE_EXECUTION.json` 或异常回执保存后才调用 `SimulationApp.close(exit_code=...)`。这是实际需要的顺序：本机 close 可能直接结束进程。新子进程测试把 close 替换成 `os._exit`，检查成功返回及执行异常两路都先留下回执。失败启动仍有独占 `EXECUTION_STARTED`，不自动重试。

原生只检查兼容主物体的接近推物保护依然保留，新包装同时对每个动态源物体应用相同的闭合前 XY 位移 >3cm 保护。它是保守的提前碰撞停止条件，不能据此确定候选目标；最终搬运标签仍允许任一原桌面源物体满足全部双侧承载、离桌、保持到 release 前及箱内静止条件。

新增检查共 19 tests / 23 子测试通过，包含只准 approach-only、原 null 保留、精确归档与资产不可替换、任一物体提前推物、以及进程退出前回执。独立只读审查核对了闭合公式、原列序、初态和退出顺序。

实际纯规划另存于 `outputs/boya_author_transport_20261114_v1/approach_only_v4`。它复用已用过的 A0/g6 固定候选，额外源体只是独立开发规划夹具，不是三物体正式采集或外部验收。原 native approach-only 一次 ready，1300 步、9.811951s；添加原 close/hold 后的前缀 1600 步 × 24 列，与此前原 native 完整计划的前 1600 步逐值比较，**全部 6 数组**完全相同：initial_position、joint_names、time_s、joint_target、phase、desired_palm。没有把实际命令展开成 28 列；4 mimic 仍只用于审计展开。

固定 bin 后的真实 transport 纯规划一次返回 `null_local_planning`，4.878690s，原因是需要执行的新路径在原 IK 预算内不收敛；没有更换候选、bin、初态或重试。原 `transport/TRANSPORT_PREPARATION.json` 随后读回为 2291 个 NUL 字节，另存的调用者返回值 `TRANSPORT_RESULT.json` 是有效 JSON，保留该 null、错误和成本。原异常文件未覆盖；只读复查确认 inode 3485211、blocks=1、无可见打开句柄，并非正在写入的暂态。原因尚未确定，不能把原损坏文件当有效回执。轻量 `RECEIPT_INTEGRITY.json` 同时记录两者 SHA；此开发规划的返回值和落盘基础设施异常均保留，不用重跑掩盖。

本 agent 本次新增 Kit/物理/CGR/训练/标签均为 **0**，原 A0 smoke 的 null 不是物理失败率。旧 v3 报告、证据和负结果保持原样，本文件及 `evidence/author_transport_approach_only_20261114_v4` 独立描述本补丁。原 source SHA、成本和检查见该目录的 `EVIDENCE.json`、`PREFIX_AUDIT.json`、`TESTS.log`。
