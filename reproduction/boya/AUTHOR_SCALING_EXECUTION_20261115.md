# CPU/GPU 扩展规模诊断入口

新增 `author_scaling_execution.py`（SHA256 `4037bebc5296977848cc2413d833d396eb2669df470b9c74a53863c91dbf0375`），仅复用冻结 vector/profile/probe/native 模块，未改这些源及 CanonDex。GPU 支持同一冻结 g6 计划的 1/4/8 个官方 USD 克隆环境，共一个 Kit/物理 clock；CPU 每进程严格单环境，由外层现成资源调度器并发。所有重复模板仅性能诊断，不是新采集标签。

```bash
python -B reproduction/boya/author_scaling_execution.py \
 --prepared-dir /ABS/FROZEN/SCENE4/g6 --output-dir /ABS/FRESH \
 --mode gpu --lanes 8 --prefix-ticks 100 --protocol /ABS/PROTOCOL.json
```

CPU 改 `--mode cpu --lanes 1`；GPU lane 数为 1/4/8。布局显式固定 `/World/envs/env_i`，origin `[4*i,0,0]`，准入逐值核对而不猜坐标。prefix 默认 100、范围 1–200，且必须小于原完整长度；完整模式另立入口/协议。

旧 g6 真实只读预校验通过，八份独立 lane manifest 均绑定原 3024 行 commands SHA `2bd97148919c3f10be5c2be315de1328446b824b1bcd9880926bf3000d7c20ea` 与 geometry `d9e6636e7520dda1406d4f39c051b88e40c875ebc4d41661ccf2a2142bdc5999`。此预校验在最初 `aad39785...` 版本完成，仅 source 写入性能/成本元数据前；原 JSON 保留该 source SHA，不重写为最终版。

原 clone 方法本身已可接多个环境；新层仅扩 layout/同模板重复准入、每 lane 到其余所有 lane 的完整跨环境接触视图，以及显式 CPU device/probe 适配。严格 source pin 后，原 run_group 只内存改一个设备准入表达式，原 profile.run 只改 device 参数转发。CPU 保留原 probe 全部六接触数组与状态读取，仅将 CUDA 特有准入替换为 `enableGPUDynamics=false / broadphase=MBP / ordinal=-1 / 实际 CPU tensors`，不能把 CPU 误报为 GPU。GPU 仍核原 GPU 属性；两者均保持 250Hz/原质量摩擦/手驱动/native20µm/全部碰撞/原控制与逐点日志，未为吞吐放门或删报告。

结果 `SCALING_EXECUTION.json` 绑定原 `GPU_VECTOR_EXECUTION.json`、`GPU_VECTOR_PROFILE.json`、source、READY/commands/scene 和完整 lane 产物。成功前缀须 `partial_scaling / complete=false`、returned counter=实际 native counter=每 lane observed=100、标签均为空；记录单 clock100步和实际100*N环境步。setup/execution 时长及 UNIX/monotonic 区间可用于独立核并发。prefix 内原 generator 关闭/最终保存的成本包含在 execution interval，simulate/fetch/读回/JSON/gzip 分项另在 timing，不把全部 wall 冒称纯物理时间。

异常时记录 returned、observed、native step counter、inflight/unknown 与环境步上下界。native counter 在 simulate/fetch 前增加，若高于 returned，仅代表尝试积分的上界，不能冒称已完成。读取 native counter 失败保留未知。此前 v2 的 fetch 中断 937–938 边界因此不会被新层误报为完全已知 937 步。

8 项 CPU tests 0.158 秒通过：动态原函数接线、CPU/GPU 后端硬拒绝、原 probe 六数组与混设备拒绝、同模板独立 manifest、4m 布局、在途上下界以及原 profile→新 sidecar 在 os._exit 前持久保存。本 agent 0 Kit/物理/IK/CGR/训练；没有从测试声称实际 4/8 env 已通过。完整 6 组 100tick 协议由 root 独立冻结启动，不是正式 bulk。
