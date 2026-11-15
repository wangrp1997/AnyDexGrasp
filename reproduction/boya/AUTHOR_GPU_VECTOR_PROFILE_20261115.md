# GPU 短前缀分项计时入口

新增 opt-in `author_gpu_vector_profile.py` 直接复用已冻结 `author_gpu_vector_execution.run_group`，只在当前进程包裹计时函数与原 barrier 的 `step_limit`。原 3 个 GPU 模块、原 native/CPU 模块及 CanonDex 均未改。本报告只验收 CPU 接线，本 agent 0 Kit/积分/规划/训练；实际 GPU 耗时需 root 冻结后的短预算回答。

入口源码 SHA256 `ea3c7ef466ac1978d49a63a7035e9ba483cdb9f24693f06b5b278c249c6b4f1b`；CPU tests SHA256 `a8eb62ee862c49492c3bd5fcb7f7b3a65f45e0c192fb3a02ecf98f5d82aca53c`。原 vector SHA 固定为 `263cf9fd7b146da66a31bd43c70d43bac01361c9904b627cc03bbd357d092e0d`，任一变化需独立新版本，不能绕过此 source pin。

## 可运行命令

```bash
python -B reproduction/boya/author_gpu_vector_profile.py \
 --prepared-dirs /ABS/OLD/SCENE4/g6 \
 --output-dir /ABS/FRESH_SINGLE \
 --env-roots /World/envs/env_0 \
 --origins-json '[[0,0,0]]' --prefix-ticks 20 --protocol /ABS/FROZEN_PROTOCOL.json
```

两环境传 `--prepared-dirs .../g1 .../g6 --env-roots /World/envs/env_0 /World/envs/env_1 --origins-json '[[0,0,0],[4,0,0]]'`，其余相同并使用另一个 fresh 输出。原环境依赖/资源 lease 由 root 的原调度器提供，本入口不增加调度器，也不自行派发。

`--prefix-ticks` 显式限定 1–200，且必须小于每条原命令长度；本轮协议建议固定两组各 20，不依结果追加。现成 `advance_together(step_limit=20)` 保留全部逐步原控制、接触、源底/承载检查和 gzip 明细；没有抽样、丢接触或聚合替代。原 barrier 在读完第 20 步后会预写第 21 行控制，再结束积分；这一行未积分，源码与回执明确说明，没有截断或修改 commands 文件。

## 实际可核查输出

`GPU_VECTOR_EXECUTION.json` 必须为 `status=partial_profile`、`complete=false`、`clock.prefix_limited=true`、`clock.all_streams_complete=false`；clock 的 `physics_ticks=20`、`lane_steps=[20]` 或 `[20,20]`，recorded/actual 环境步为 20 或 40。每 lane 原完成时点字段是此部分记录的 20，`raw_label`/`label` 均为空，原 lift `label_eligible=false`。原 `trace.npz`、`MULTI_OBJECT_TRACE.json` 和 CONTACTS gzip/pointer 均保留 20 步，不能按原 native 部分评价里的 task_success=false 记负样本。

`GPU_VECTOR_PROFILE.json` 的 schema 为 `anydex_boya_author_gpu_vector_profile_v1`，含全部 source SHA、真实物理成本、group 回执 SHA、原 native `profile_execution` 结果及计时。此文件与 group 回执均在真实 App.close 前落盘；初始化异常也保存失败回执，不伪造部分成功。

计时分 `initialization`、`execution_prefix`、`finalization`。每项含 calls / inclusive_wall_s / exclusive_wall_s：`sim.step`、原 PhysX `simulate`/`fetch_results`、Warp numpy 与 Torch cpu、robot.snapshot、filtered6 全读回、两层接触 decode/load、全源顶点底面、robot/object update/write、JSON 编码、gzip 写入、完整 journal append 和最终保存。`prefix_clock_including_generator_close_wall_s` 明确包含 generator 收尾，不能当纯积分耗时。原 native timing 的 physics_step_update 项跨越 generator yield，也包含外部 on_step/probe，不能再与新分项相加。

所有计时为 host wall，未额外调用 CUDA synchronize。GPU 异步工作可能在后续阻塞读回记账；inclusive 有包含关系，不能相加。exclusive 仅在已包裹的作用域之间互斥，不等于覆盖所有 Python 工作。第 1 tick 的原完整 probe、初始化、最终压缩保存和短样本计时开销必须单列，不能用 20 tick 吞吐直接外推正式采集。

## 可复用路径与下一步判别

原 native `execute_template_arm_positive.py` 每步读 robot snapshot/主物体状态，再读 primary-object 与 robot 两套 filtered6。新三体 `author_lift.py:208` 每步又读 all-object 与 robot 两套 filtered6、三体 pose/velocity 并完整解码。原 `contact_readback.py:10` 对每套数据分别将 count/start 与四个 payload 读回；按静态调用可有 24 个接触 host 数组读回及约 10 个状态读回/lane/tick，另有双环境隔离视图和首步 probe。此为源码调用数，不是已经测得的时间比例。

本机现成 `canondex/grasp/template_gripper_batch.py:148` 已有统一多环境 contact views，`:280` 已有 q/v/body/object 等状态打包后一次 CPU 读回，并预载 command_gpu。若状态/读回实际占主导，可把这些已存在模式移植到三体，逐字段核等价；三体统一句柄可用官方 RigidObjectCollection。若 simulate/fetch 占主导，单改 JSON 或打包读回不会解决主要耗时，应先检查实际 PhysX scene/report/cooking 路径。当前不作未经测量的比例或倍数承诺，也不先删重复报告或改变审计门。

6 项 CPU tests 0.097 秒通过：真实原 barrier 的每 tick 完整观测/成本/close、异长未完成边界、native simulate/fetch 参数/结果/异常仅转发一次、嵌套分区不重复相加、任一部分标签拒绝、source pin 与还原，以及 subprocess 直接 os._exit 前 profile 已保存。CPU tests 不能替代真实 GPU 路径验收。
