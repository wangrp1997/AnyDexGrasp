# GPU 单/双环境 20 步性能前缀只读审核（2026-11-15）

`gpu_partition_profile_20261115_v1` 已按冻结预算完成单环境 g6 与双环境 g1/g6，各 20 个 shared tick。唯一一次独立审核状态为 `prefix_integrity_pass`，耗时 1.794024 秒，未启动 Kit、积分、模型或规划，也没有重读旧 937 步接触日志。完整输出为 ignored `outputs/boya_gpu_prefix_audit_20261115_v1/AUDIT.json`，轻证据为 `evidence/author_gpu_prefix_20261115_v1`。

这是 settle 阶段的性能前缀，不是完整提起或七门验收。三个 lane 的 label/raw_label 均为空，formal_training_eligible=False、exclude_training=True；没有训练标签或模型更新，不能外推完整采集产率与 ETA。下一条未积分命令可能已被准备，但原有边界明确在 20 次 step 返回后结束，不存在旧 SIGINT 路径的在途积分不确定性。

核验两组 source/input/artifact 的前后 SHA，保留同一原 READY、场景、候选与完整命令。三个 lane 各自前 20 条 24 独立命令经源 XML 的四个 mimic 展开后，与实际 28 目标逐值相同；desired palm、phase、time 逐值相同。完整读取 **60 条三体轨迹、60 条 lossless gzip 接触记录，共 129,420 接触点**，行号/身份/finite、完整 pointer/SHA/bytes 均通过。合计 **40 shared tick、60 environment step、180 三体对象步**，无 tail、新命令或重试。

四次真实 GPU probe 的 CUDA payload、count/start、原引用间隙、容量及 snapshot SHA 通过；原 native 20µm receipt 和固定基座结构保留。第二环境 entry 的物理位置减去 [4,0,0] 后与冻结场景一致，原 trace 仍在 plan frame；接触路径属于相应环境或共享 ground，运行时跨环境完整 view 记录 cross=0。这里只核前缀完整性与原导入证据，不重新认证完整掌跟踪/提起/最终承载门。

| 原始 host 计时（秒） | 单环境，20 envstep | 双环境，40 envstep |
| --- | ---: | ---: |
| 前缀 wall（含 generator close） | 12.038500 | 13.624220 |
| PhysX simulate 调用 | 0.002443 | 0.003318 |
| PhysX fetch_results 等待 | 11.020907 | 12.434994 |
| sim.step inclusive | 11.027216 | 12.442459 |
| filtered6 readback inclusive | 0.045093 | 0.108594 |
| allbody contact decode | 0.478093 | 0.127600 |
| journal encode/gzip inclusive | 0.161455 | 0.324112 |
| fetch / 前缀 wall | 91.547% | 91.271% |

**主要已测时间记在 fetch_results 阻塞等待。** 不能继续笼统归因为多次 host readback。这里没有新增 GPU synchronization，异步工作可归到后续阻塞调用；计时不能进一步证明等待究竟来自哪个 GPU kernel、调度或同步原因。sim.step 已包含 simulate/fetch，inclusive wrapper 与子项不得相加，两个 lane 的 native physics 桶也重叠。AUDIT 保存各段原 timing 和不重叠 exclusive 汇总，初始化、execution_prefix、finalization 分开保留。

root 整个两组 outer wall 为 169.312958 秒；group close 前 wall 为单路 68.678840 秒、双路 95.851791 秒，profile 最后写入前分别 69.123009/96.641735 秒。这些包含建场、原生审核、快照和收尾，不能用前缀 wall 代替整个运行成本。短前缀不能证明完成抓取的吞吐倍率；本次没有新增完整 GPU 抓取成绩或有限的正式采集 ETA。

性能 profile 源 SHA 为 `ea3c7ef466ac1978d49a63a7035e9ba483cdb9f24693f06b5b278c249c6b4f1b`，原 GPU adapter 仍为 `263cf9fd7b146da66a31bd43c70d43bac01361c9904b627cc03bbd357d092e0d`。冻结原 full audit `7c508445…` 没有修改或调用，其纯 snapshot/path helper 被复用作只读前缀核验。父调度已另核清场和旧标签/模型不变，本审核不重复该主机检查。
