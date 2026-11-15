# Fabric 同步与凸体兼容性实测（2026-11-15）

本次小对照已完成，**没有测出新的采集加速**。关键发现是：原默认执行在 App 创建后就已将 `fabricUpdateTransformations=false`、`fabricUpdateVelocities=false`，并已设置 `fabricUseGPUInterop=true`。新入口显式双关闭没有改变实际配置，因此这不是两种不同 Fabric 设置的有效 A/B，只能作为同一配置的两次短前缀重复。不能把耗时差归因给关闭 Fabric，也不能套用上游约 7 倍收益。

用户在查阅网络与原论文后授权开始小规模验证。本轮仅执行冻结 g6 旧计划的前 100 tick 两次，保持原 250 Hz、24 独立坐标至 28 实际 DOF、三体几何、质量/摩擦、逐体向下力、完整逐点接触及原门。正式扩采、训练和外部评测均未启动。

## 官方依据与本机接入

[NVIDIA 性能指南](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html)说明，无相机且仅使用 GPU tensor 的工作负载可以同时关闭 Fabric 位置和速度发布；只关闭其中一项仍会触发同步。需要 Fabric 输出时可测试 GPU interop，小场景不保证获益。[官方仓库问题 #740](https://github.com/isaac-sim/IsaacSim/issues/740)有相关确认和大型简单环境基准。该建议是本次测试依据，不是本项目已证实的瓶颈解释。

本机机器人快照通过 `robot.data` 的 PhysX tensor 读取关节和 body pose，接触通过原 `get_contact_data` 读取。原初始化 `forward_fabric` 一次性发布及设置恢复逻辑保留，没有改 CanonDex。新入口只在 App 创建后、建场前执行指定设置；整个测量阶段无任务相机/visualizer，逐 tick 保存实际开关。没有改 RGB-D capture、渲染或其数据依赖。

## 固定预算与计时

协议位于 `/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/fabric_control_20261115_v1/FABRIC_CONTROL_PROTOCOL.json`。运行前冻结源码、全部原输入、物理 GPU0 UUID、baseline→both_disabled 顺序、各 100 tick、每 worker 600 秒上限、2 次 Kit 和 0 重试。复用既有租约、资源槽、子进程身份检查与 SIGINT 优先的限时回收；原启动标记禁止重启。

| 指标 | baseline，保留默认 | both_disabled，显式双关闭 |
|---|---:|---:|
| 实际位置/速度发布 | false / false | false / false |
| 实际 GPU interop / suppressReadback | true / true | true / true |
| 完整前缀 tick / 环境观测步 | 100 / 100 | 100 / 100 |
| 前缀 wall，含 generator 收尾 | 58.417396 s | 58.662117 s |
| `fetch_results` | 54.553399 s | 54.575174 s |
| fetch 占前缀 | 93.39% | 93.03% |
| 全 filtered6 接触读回，inclusive | 0.233291 s | 0.275146 s |
| 组 wall，含启动和真实退出 | 124.023485 s | 120.059069 s |

控制器总 wall 为 244.123901 s；组 wall 包含建场、原生检查、探针、前缀、后置兼容查询和进程关闭，不含事后独立审计。前缀速度约 1.71/1.70 环境步每秒。只有单次固定顺序重复，未增加独立统计重复；外层启动差异不是物理加速。inclusive 计时相互包含，不能相加，没有新增 CUDA synchronize，也没有进行 kernel 归因。

## 完整性与兼容性

独立审计用时 6.879916 s，验证 1286 个唯一绑定、200 条原轨迹、200 条无损接触行、431400 个接触点。每组 native 尝试计数、返回步数及全观测均为 100，成本精确为 **2 Kit / 200 shared tick / 200 环境步 / 600 三体对象步**，无在途未知步；探针、兼容查询和事后审计额外积分均为 0。两次真实 GPU dynamics、GPU broadphase、ordinal 0 和 CUDA filtered6 探针通过，原三体 1135 hull 的 native20µm 门保持。

两组原 24→28 目标、六臂速度前馈、desired palm、phase/time 均逐值一致。保存的所有 14 个 native trace 数组逐值一致，全三体 entry 与逐 tick 多体状态也逐值一致。这只覆盖前 100 tick，不能外推完整接触闭合或轨迹物理等价。

原始接触列表从第 1 tick 起顺序不同，独立审计如实保留 `full_contact_rows_exact=false`。随后单独只读复核每 tick、每类完整点记录的精确多重集合：100/100 行相同、各 215700 点；没有删除零力点、字段、重复项或数值取整。该复核用时 2.643654 s，说明本前缀差异可由记录顺序解释，不能写成原始逐行相同。

兼容查询仅在测量结束后启用官方 validator 接口，调用每 prim 的 `convex_gpu_compatibility_is_valid`；没有调用会 attach/reset/detach 或释放 cooking cache 的完整 `CheckStage`。两组各查询 **13846 个凸网格 prim**：机器人 12711、primary 328、另两体 807，均返回 GPU compatible；查询前后原 stage layer SHA 和物理计数不变。prim 数不是唯一 cooked hull 数，也不能据此证明全部接触/约束 kernel 都运行于 GPU。没有发现不兼容 prim，不为性能修改几何。

## 当前结论与边界

此官方 Fabric 建议在当前执行配置中已满足，不能靠再次设置双关闭解决慢问题。`fetch_results` 仍承担主要等待；内部 CPU/GPU 工作、驱动或物理图的具体成本尚未定位。更细的 Tracy/Nsight 分区或 CPU/GPU 扩展规模对照仍需真实验证，本轮没有自动追加批次、开启 1/4/8 路、调物理、降几何/控制精度或重抓。

所有二元 label/raw label 均为空，`exclude_training=true`。本轮只通过前缀完整性检查，不认证完整抓取 gates、提起或总体采集 ETA。累计有效新数据仍为 **14 fit＋1 internal**，旧 435 条和八个权重的 13 个文件 SHA 未变；正式训练、新模型和训后评测为 0。旧 fit 约 15.02 天仅为原条件情景，4000＋800 总耗时仍未知。

8 项新 CPU 检查及 30 项复用组件检查通过，均无真实物理。最终 host 检查确认两个自有 worker 进程组已退、当前资源槽 released、3 个 driver/全局/slot 锁可取、registry released；原输入、冻结运行源及旧 GPU 协议保持。轻量证据为 `evidence/author_fabric_control_20261115_v1`，原始输出继续留 public/ignored outputs，CanonDex 只读。
