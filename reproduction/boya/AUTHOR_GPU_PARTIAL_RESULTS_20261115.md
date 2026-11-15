# GPU 单环境部分执行只读审核（2026-11-15）

真实 GPU v2 单环境 g6 诊断在确认路径过慢后由 root 发送 SIGINT，保存了部分回执；双环境 batch 未派发。本报告只核已保存证据的完整性、命令前缀和成本，不认证完整抓取、七门或二元标签。没有调用完整 `author_gpu_vector_audit.audit_execution`，也没有修改其已冻结 `7c508445…` 源码。

`gpu_vector_control_20261115_v2` 的原协议 SHA 为 `b3bb4d9bd01f8fb5fc8410b6c65a0d191ab2dc5f6e16e018e91c6f64c397389b`，真实执行源为 `263cf9fd7b146da66a31bd43c70d43bac01361c9904b627cc03bbd357d092e0d`。v2 保留同一冻结旧场景、候选和 3024 条命令；该已知控制诊断始终 exclude_training，不增补已有 15 条可用新标签或旧 435。v1 的导入顺序零步失败另由 root 留档，不在本次部分数值审核中重跑。

## 实际结果和成本边界

原 group 为 `infrastructure_failure`、`complete=False`，LIFT 评价为 `label_eligible=False`、`complete=False`，仅经过 settle 与 home_to_pregrasp。顶层和 job 的 `label/raw_label` 都为空，没有 DIAGNOSTIC_RECORD 或 COLLECTION_RECORD。原 native 最终完整 gates 不存在；LIFT 中占位 False 不能当已测量的关节或跟踪违规，也不能把原始 task_success=False 填成物理负例。

已确认返回的 step/callback/allbody observation 都为 937。SIGINT 的真实 traceback 位于下一次 `physx_sim.fetch_results()`，因此第 938 次在途物理推进是否完成不能确认。实际积分成本必须报 **[937,938] shared/environment tick**，三体对象步 **[2811,2814]**，不能把已返回计数 937 当精确总积分数。没有重试、补候选或补负例。group 最后 close 前 wall 为 718.325745 秒，root outer wall 为 719.738294 秒；这是未完成诊断的消耗，不是完整采集吞吐或成功率。

## 唯一一次独立只读审核

Ignored 输出：`outputs/boya_gpu_vector_audit_20261115_v1/single_g6/AUDIT.json`，状态为 `partial_integrity_pass`，wall **13.863646 秒**。该时间单列，不混入原执行阶段。

- 696 个原来源、输入和实际产物 SHA 前后核对通过，包括当前执行源、旧 source READY/协议/命令/模型输入、全部三体原视觉和执行 mesh、命令绑定、两 GPU probe 与 lossless gzip。
- 完整读取 937 条 allbody trace 和一次流式解压 937 条接触记录，合 **2,021,298 个接触点**；原行号、phase、时钟和三体身份齐全，所有已读数值 finite，接触路径只属于本环境或共享 ground。
- 前 937 条原 24 个独立命令经原 XML 的 4 个 mimic 展开，与实际 28 目标逐值相同；desired palm、phase、time 也与冻结完整命令的前缀逐值相同。没有新增 IK、积分或坐标修补。
- 两次真实 GPU snapshot 都为 CUDA 原生 contact view，count/start/引用区间/容量及 NPZ SHA 通过；初态共 5369、首 tick 共 2157 个活跃接触引用。scene 为实际 GPU dynamics/broadphase；这证明接触 API 已真实运行，不证明整段等价或加速。
- 原 native receipt 保存 1135 个 hull、within_20um=True，actual robot 为固定基座、28 DOF，执行中状态重置计数为 0。只绑定并核原 receipt，没有重新运行 cooking 优化器，也没有认证未完成阶段的 gates。
- root outer 仅有 single_g6 stage，batch 没有 STAGE/STARTED/最终文件，明确未派发。没有全流程新标签、正式采集预算扩展或训练更新。

轻量来源、审计脚本和摘要：`evidence/author_gpu_partial_20261115_v1`。完整接触、NPZ、轨迹仍留在 public 数据根；审计没有修改任何实际产物。部分路径已真实接通，但本轮既不是 GPU 不支持的结论，也不是完整 GPU batch 验收；后续仅可按独立新协议定位性能，不能从该部分执行外推正式采集 ETA。
