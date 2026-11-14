# 常驻完整采集诊断入口（2026-11-15）

新增 `author_resident_collection.py`，复用已有采集阶段，不修改冻结 batch、waves、capture、prepare、执行器或 CanonDex。本文说明实现合同；另立小预算的实际验证及耗时见 [完整吞吐报告](AUTHOR_RESIDENT_THROUGHPUT_RESULTS_20261115.md)。实现测试与实际物理验证分别留证，不将离线测试当作吞吐结果。

`freeze_from_batch(source_batch, scene_indices, output_dir, resident_workers=1)` 从旧 batch 的完整冻结协议取原 scene spec、原 queue seed、传感器/工作区、模型与准备预算。输入索引必须按原顺序且不重复，预算在运行前固定。输出是新 schema `anydex_boya_author_resident_collection_v1`，没有能让旧 `B.run` 接受的父协议兼容视图。旧 RGBD、候选、ready 和执行产物均不复制。

实际运行顺序为：新 build → 原 compact capture → 原 `run_collection_representation.py` 单进程、单模型、固定多帧 manifest → 原四型首选 prepare（总并行上限 3）→ 每 scene 一个 `author_resident_execution.py execute-many` 子进程 → 原独立 runtime audit → 再次核验 SHA、record、全三体身份和 split → 诊断账本。常驻执行组内仍按 g1/6/7/8 原序，只执行 ready 一次；不同 scene 可以按 `resident_workers=1..3` 并行。原 `PersistentOfficialCGR.forward` 为每帧恢复相同初始化后的 RNG 状态，保留原 queue seed 语义。

父进程不启动 Kit。执行子进程沿用原 ResourcePool 的 kit 槽、GPU0 全局 lease、跨 run registry、线程局部 pass_fds、全 PGID 退出判定与 quarantine。capture 仍是逐 scene 子进程；这版没有相机常驻、官方向量环境或批量物理积分。独立 audit 仍串行，原每阶段前后完整 SHA 核验仍保留。因此完整吞吐必须实际测量，不能用已完成的执行段 1.982 倍替代。

新分支使用的既有 `arm_resident_scene.py` 与 IsaacLab `simulation_context.py` 在 collection freeze 时即锁 SHA，随源绑定传入 resident worker；不能等到执行时才采当前版本。

所有结果 `exclude_training=true`，父账本 `label=null`、`formal_training_eligible=false`，独立审计通过的科学结果另存 `raw_label`/`diagnostic_label`。不得把本轮旧场景吞吐对照计入累计 15 条新标签或正式训练集。未准备、未执行和基础设施错误均不补负；显式 `not_started` 与未知执行成本分开。组内后续失败不抹掉前面已有完整回执、通过独立审计的诊断结果。物理成本包括已完成 capture 的 750 ticks/三体 2250 object-steps，以及全部已知执行回执；完整性失败无法入账时另留 `unadmitted_execution_costs`，不丢成本。

`RUN_STARTED.json` 是不可重启的一次性标记；本薄入口不宣称可断点重派发。结果、完整账本、阶段计时和开始/结束 Unix 时间写至 `RESIDENT_COLLECTION_RESULTS.json`、`LEDGER.json`、`PROGRESS.json`。父循环时间包含 resident 子进程实际退出和审计；外层进程启动/退出仍由 benchmark 单独测量。`prepare_sum`、`execution_groups_sum` 是各任务壁钟之和，`execution_parallel_wall` 才是执行波段时间。

示例（只作命令交付，实际派发由主代理统一执行）：

```bash
python -B reproduction/boya/author_resident_collection.py freeze-from-batch \
  --source-batch /mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development/batch_v1 \
  --scene-indices 1 4 --resident-workers 2 --output-dir /ABS/FRESH/fit_diagnostic
python -B reproduction/boya/author_resident_collection.py run --run-dir /ABS/FRESH/fit_diagnostic
```

另一来源的 internal child batch 可单独 freeze local index 0；不会把两个源队列静默拼接。没有自动训练、扩预算、按正负补场景、换候选或重试。

离线测试覆盖一次模型处理两 scene、prepare 上限、两组调度、ready-only、无计划不补负、独立审计失败、部分组失败、未知成本、一次运行守卫、输入/执行后完整性篡改、全初始化 contact 文件绑定、新 resident helper 的 prefreeze SHA，以及原 kit 槽分类。测试使用合成阶段与普通 Python 线程，0 Kit、0 模型前向、0 规划、0 物理；它们不替代真实多 job reset/完整吞吐验收。测试日志位于 `outputs/boya_author_resident_collection_20261115_v1/`，轻量源绑定回执位于 `evidence/author_resident_collection_20261115_v1/`。
