# CPU/GPU 扩展吞吐的独立审计口径（2026-11-15）

新增 `author_scaling_audit.py` 已实现，仅读产物、不启动模拟器、模型、规划或训练。当前完成的是静态接口审核与 11 项纯 CPU 合成回归；真实六组结果到齐后运行一次，不能把这些回归当成实际并发或提速证据。

固定对照为同一个只读 g6 preparation：GPU 单进程 1/4/8 lane 共享物理时钟，CPU 1/4/8 个独立单 lane worker；每 lane 原命令前 100 tick、原 250Hz，每组仅一次、不重试。共 16 worker attempts、2600 environment steps、1600 worker physical clock steps。GPU 三组共享时钟合计 300 tick；CPU 三组独立时钟合计 1300 tick。不能把 lane 数直接当成加速倍率。

| 比较量 | 独立计算 | 范围 |
| --- | --- | --- |
| 前缀吞吐 | `100 × lane 数 / prefix wall` | GPU 使用单 worker 的 monotonic 区间；CPU 使用最早 prefix 开始至最晚结束的包络，包含空隙及原 generator 结束序列化 |
| 含启动吞吐 | `100 × lane 数 / group wall` | 控制器启动组至全部子进程真实退出，包含加载、建场、探针、日志、收尾与关闭 |
| 实际重叠 | prefix 区间 sweep 的 span、union、峰值、全 worker 共同重叠 | CPU 不相加各 worker wall；GPU lane 是一个共享时钟，单独核全部 lane 每 tick 产物 |
| 主机与 GPU 占用 | 绑定的每 2 秒 `/proc` 与指定 GPU UUID 的 NVML 样本 | 报告观测峰值、RSS、截取区间 CPU seconds、GPU 利用率/显存/功耗；采样可能漏短重叠，NVML 不独占归因给本任务 |
| 内层成本 | 原 initialization / execution_prefix / finalization 计时分别保存 | inclusive 项相互包含；只可累加 disjoint exclusive 项。`sim.step` 已含 simulate/fetch，没有新增 CUDA 同步或 kernel 计时 |

静态审核读取新 `author_scaling_execution.py` 与 `author_scaling_control.py`，确认结果结构为 `SCALING_EXECUTION.json`、`SCALING_CONTROL_PROTOCOL.json`、`SCALING_CONTROL_RESULTS.json`，原 `GPU_VECTOR_EXECUTION.json` / `GPU_VECTOR_PROFILE.json` 及逐 lane 原生输出完整保留。新 full-CPU 入口不是本次 100-tick 协议组成部分，本审计也不把完整执行与前缀混合。

每 lane 独立核验：同一 READY/candidate/canonical480/commands/scene 与来源 SHA；前 100 条原 24 独立坐标展开至 28 实际坐标、六臂速度前馈、零手速度目标、desired palm/phase/time；全三体原源网格顶点最低点、完整逐点接触重新累计的五指/总载荷/非法承载；材料、质量、重力、每体向下外力与冷初态；origin 只对物理世界 entry/path 投影，已在原 plan frame 的轨迹不重复减 origin。原接触原样流式读，不删零力点，跨环境路径拒绝。初态与首 tick 两套接触探针逐数组核有限值、设备、count/start、引用区间和容量，CPU 必须实际 CPU tensor，GPU 必须实际 CUDA tensor。

精确积分成本要求 native attempted counter、已返回步数、各 lane 完整观测步数均为 100，且上下界相同、无未确认在途步。中断或基础设施异常不能冒充精确完整前缀，首次失败审计会保存为 `failed`，不会产生二元标签。

所有 worker、lane、外层 record 的 `label/raw_label` 必须为空，`exclude_training=True`、`formal_training_eligible=False`。这些 settle/HOME 前缀没有执行完整闭合和提起，审计明确 `full_gates_certified=False`、`full_lift_certified=False`、`full_collection_ETA=None`。同一已知模板、每尺寸一次，只能说明本工作负载的实际吞吐与占用，不能推完整采满时间、总体有效率或轨迹物理等价。

11 项回归覆盖 CPU/GPU probe、原 24→28 和速度目标、非 primary 底面/载荷篡改、跨 env、二元前缀标签、在途成本、区间死区/相邻端点、实际重叠与资源设备身份、六组总账及末尾 SHA。全部为临时合成文件，0 Kit / 0 积分。测试日志和源码绑定见 `evidence/author_scaling_audit_20261115_v1/EVIDENCE.json`；开发日志保留在 ignored `outputs/boya_scaling_audit_20261115_v1/cpu/`。

真实完整六组结束后唯一命令（此处尚未执行）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD/reproduction/boya:$PWD" \
/home/wangrenpeng/tendon-spin/.venv/bin/python -B reproduction/boya/author_scaling_audit.py \
  --control-results /ABSOLUTE/FROZEN/RUN/SCALING_CONTROL_RESULTS.json \
  --output outputs/boya_scaling_audit_20261115_v1/actual/AUDIT.json
```

输出拒绝覆盖；所有唯一输入先后各流式 SHA 核验一次，共享大文件不按 lane 重复哈希。原冻结审计、采集器、执行器与 CanonDex 均未改动。
