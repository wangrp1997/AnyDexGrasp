本入口只比较固定旧 g6 计划的执行前 100 tick，不采新候选、不生成训练标签、不训练。它复用原完整状态与逐点接触日志、控制指令、物理参数及门限，使用新的诊断目录。相同模板重复不代表随机场景成功率，早期 settle 前缀也不能给出完整采集 ETA。

固定顺序为 GPU 单 Kit 的 1、4、8 环境，再 CPU 的 1、4、8 个单环境进程。组间串行、CPU 组内并发；总计 16 个 worker、1,600 次 worker 物理时钟步、2,600 个环境步。每 worker 900 秒，0 重试，任一失败停止后续全部组。所有 `label/raw_label` 保持空，`exclude_training=true`。

```bash
python -B reproduction/boya/author_scaling_control.py freeze --output-dir /absolute/fresh/control
python -B reproduction/boya/author_scaling_control.py run --output-dir /absolute/fresh/control
```

`freeze` 只读核验原 g6 READY、计划、源输入和依赖，绑定物理 GPU0 UUID，归档全部唯一运行源码至 `frozen_sources/`，并写 `SOURCE_ARCHIVE.json`、`SCALING_CONTROL_PROTOCOL.json`。数据输入及 nvidia-smi 二进制只绑定 SHA，不复制。实际 Kit 启动仅发生于显式 `run`。默认模板来自原 `gpu_vector_control_20261115_v2` 的 `single_g6`。

控制器复用原全局 flock、run registry、资源 FD 继承与 PGID 回收；独立 `ScalingPool` 的 Kit 上限为 8，原资源模块的上限 3 不变。`RUN_STARTED.json` 一经存在便拒绝重启。取消或超时先对核实 PID/start_ticks/PGID 的自有子进程组发送 SIGINT，等待 30 秒，再按需 SIGTERM 5 秒、SIGKILL 5 秒；不能核实身份则隔离，不向其它进程发信号。

`SCALING_CONTROL_RESULTS.json` 绑定每组及每 worker 回执、原底层 group/profile、资源样本。worker 外层 wall 包含进程启动与实际退出/回收；setup 和前缀执行时间分别保留。CPU 前缀吞吐以最早开始至最晚结束的跨度计算，另外保存区间并集、峰值并发和所有 worker 共同重叠时长，不把各 worker 时长相加当作组耗时。每 2 秒只采声明 GPU0 的 NVML 指标（经 nvidia-smi）和自有 worker 的 `/proc/stat` CPU/RSS，不读取其它进程命令。

成功前缀要求 native 尝试计数、已返回步数、每 lane 完成观测数和上下界全部等于 100，且无在途不确定步；失败保留已确认下界、尝试上界和 unknown 字段，不能把中断回执中的整数直接称为精确积分成本。完成前缀仍不等于完整原生物理验收。

CPU 单测覆盖计数/标签/backend/probe 拒绝、八线程同步进入组、独立八槽锁、失败全局停派、防重启、registry/锁释放、仅自有身份的 SIGINT 优先取消、失败 37 返回/38 尝试的成本保留，以及源码归档去重和字节保持。测试未启动 Kit、子进程、模型或物理，也未实际发出信号；真实 GPU/CPU 吞吐由新的固定预算实测确认。
