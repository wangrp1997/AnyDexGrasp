# GPU PhysX 单组入口：CPU 接线验收

新增 opt-in `author_gpu_vector_execution.py` 已定版，支持显式 1/2 个环境命名空间、同三体资产克隆、每环境原命令与读回、单 Kit 共享物理时钟。此报告只验收代码接线与 CPU 测试；本 agent 未启动 Kit、物理、规划、CGR、训练或正式 bulk。真实 GPU 是否通过原门由 root 独立冻结的小预算运行回答。

源码 SHA256：`bd5e916ca7635a5cc2c805c54852bdd0db199dafe411898184eacf11403e896d`。测试 SHA256：`58256f4bc6cd9ecdc0d696c6c158ae7acb6cb7c3dd45fe70c689dd9617075625`。旧冻结源码和 CanonDex 均未修改。

## 现成接口及保留边界

本机实际版本为 IsaacLab `3.0.0rc1`、Isaac Sim `6.1.0.0`、Torch `2.11.0+cu130`、Warp `1.16.0`。入口沿 CanonDex 已有 `benchmark_template_arm_parallel.py` 的 generator/runtime_factory 接线，直接调用 `canondex.grasp.arm_parallel.advance_together`，克隆调用本机官方 `isaaclab.cloner.usd_replicate` 与 `filter_collisions`。没有新仿真器、IPC 或外层任务调度器。

原 native executor 保留完整 UR16e+Boya 的 24 独立命令列、28 实际关节展开、速度目标、原步长 4 ms、原质量/摩擦/驱动/初态/native20µm 与七门。三体保留每步永久外力写入、source 底面测量、双侧载荷及全物体非法臂支撑检查；完整 filtered6 接触明细仍无损写入既有 gzip journal，不改为净力聚合。GPU 接触/状态实时读回失败会拒绝继续，没有 CPU/aggregate 兜底。

GPU 是显式更换 PhysX 计算后端，不能从源码可用推断轨迹、成功率或 CPU/GPU 数值等价。GPU broadphase 与 Fabric 路径是官方后端差异。原 profile 的 `enable_ccd=False`，入口也会拒绝静默关闭原本要求 CCD 的配置。实际原生 GPU 属性、设备与接触数组通过初态及第 1 tick 的独立 probe 保存；probe 不增加积分。

## 最小坐标适配

固定 SHA 的原 lift 方法仅作三处 AST 接线：`ready` 的机器人报告器增加当前 lane root；实际初态 desired 加 lane XY origin；`after_step` 的对象 pose 减 origin。其余循环、载荷、碰撞/推物停止条件及原 finish/evaluate AST 保留。原 native executor 已支持 `environment_origin_m`，自身将轨迹 pose 减回原 plan frame。

因此 `trace.npz` 和 `MULTI_OBJECT_TRACE.json` 为原 plan frame；接触位置仍是物理平移世界，刚体路径是实际 lane namespace。`GPU_VECTOR_EXECUTION.json.jobs` 明确保存两者差值，独立审计须按此转换，不能对新路径硬套 `/World/Robot/` 前缀。所有源 prepared 目录只读；新输出不得位于其内部。

克隆前后逐关系核对结构性碰撞 exclusions，拒绝悬空或跨 lane 引用。官方碰撞组负责环境隔离，并另建逐 tick 的跨环境接触视图验证零接触。新代码没有修改源凸体以求 GPU 通过。

## 异长命令与成本

原 barrier 要求 ready 的调度长度相同；薄封装只把 ready 的 barrier 上界设为最长长度，不改变任何原 generator、命令数组或迭代次数。每条原轨迹在自己的完成 tick 保存原评价，短 lane 所在物理世界继续随共享 clock 积分。

短 lane 完成后的每个 tail tick 复写原缓存驱动和三体 force、更新全部状态，并另存 `TAIL_TRACE.json`/`TAIL_CONTACTS.json[l.gz]`，不调用新目标设置、不写状态 reset、不覆盖完成时评价。尾部碰撞/载荷诊断单列，不能当另一个标签或新增抓取。

root 选择的旧 scene4 g1/g6，实际只读核验分别为 3374/3024 行，相同三体 geometry SHA `d9e6636e7520dda1406d4f39c051b88e40c875ebc4d41661ccf2a2142bdc5999`。若先 single g6，再 batch g1+g6 完整执行，预算为 6398 shared ticks、9422 原命令环境步、9772 实际积分环境步，差值是 batch g6 的 350 tail ticks。

## CLI 与产物

```bash
python -B reproduction/boya/author_gpu_vector_execution.py run \
  --prepared-dirs /ABS/OLD/g1 /ABS/OLD/g6 \
  --output-dir /ABS/FRESH_OUTPUT \
  --device cuda:0 \
  --env-roots /World/envs/env_0 /World/envs/env_1 \
  --origins-json '[[0,0,0],[4,0,0]]' \
  --protocol /ABS/FROZEN_TOP_PROTOCOL.json
```

单环境传一个 prepared、一个 env root、`[[0,0,0]]`。root 控制顶层 lease/资源与单次预算，本入口仅负责一个已冻结执行组。

组回执为 `GPU_VECTOR_EXECUTION.json`，schema `anydex_boya_author_gpu_vector_execution_v1`；含输入/源码 SHA、每 lane 原长度/完成时点/尾部成本、真实时钟、评价/诊断记录与完整 journal 绑定。`env_i/execution/` 保存原 native 和 lift 全部产物；`env_i/DIAGNOSTIC_RECORD.json` 保留 raw label，正式 `label=None`、`formal_training_eligible=False`。不写旧 COLLECTION_RECORD，不进入正式导出或训练。

所有组回执、异常物理成本和日志 pointer 先保存，再调用可能直接 `os._exit` 的真实 App.close。未派发/未观察完成的 lane 不伪造 0 标签。

## 验证

10 项 CPU tests 通过；裸 venv `unittest discover` 同样通过（1.182 秒）。覆盖原 barrier 异长完成/尾部精算、不同 sim/错步拒绝、后续失败不改已完成 lane、原 `after_step` 的真实 source 底面/双侧力与非法支撑门、命名空间/XY 原点约束、源码篡改拒绝、tail cached write/update 不改原记录，以及 subprocess 中 App.close 直接 `os._exit` 前回执已落盘。

实际旧 g1/g6 的原 ready/协议/完整依赖/候选/三体 geometry 与 commands 只读预校验通过，0 Kit/物理/IK。首次 root 裸 discover 曾因测试子进程未继承父进程追加的 `sys.path`，无法导入新模块；仅测试 fixture 改为显式 child PYTHONPATH，正式执行入口没有该问题，也没有物理重试。

轻证据目录：`evidence/author_gpu_vector_execution_20261115_v1/`。真实小预算运行及独立数值审计另由 root 汇总，不以这 10 项 CPU tests 宣称完整 GPU 标签流程已经验收。
