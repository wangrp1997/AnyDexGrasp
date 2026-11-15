# 采集冷启动／常驻实际对照：尚无有效加速比

用户“比较下”授权原固定小对照。冷启动组完整完成，**1098.021376 秒＝18 分 18 秒**；常驻组完成两场观测，在第一条抓取执行建场时发生基础设施失败，**不能将其提前结束时间作为加速结果**。本轮没有证明常驻入口可用于完整采集。

| 项目 | 冷启动组 | 常驻组 v1 |
| --- | --- | --- |
| 组 wall，含建场至审核结束 | 1098.021376 s，完整 | 351.079361 s，失败停止 |
| 实际 Kit 启动 | 4 | 1 |
| 750 tick 完整观测 | 2 | 2 |
| 独立初始化审核 | 2 PASS | 2 PASS |
| CGR | 1 load / 2 forward | 1 load / 2 forward |
| g1 计划 | 2 ready | 2 ready |
| 抓取执行步 | 3389 + 3749 | 首条 0；第二条未启动 |
| 原抓取数值审核 | 2 PASS | 0 完整执行，无可认证标签 |
| 诊断标签 | 1 正、1 负，排除训练 | null，不能计负例 |
| 可报告加速比 | — | 未成立 |

实际主控制器 wall 为 1449.215703 s。冷组、新组均使用同样的规划/审核流水安排与常驻 CGR，本来只比较 Kit 跨阶段复用的成对效果；不代表整个流水安排相对旧八路生产入口的收益。

## 已通过的部分与具体失败

两组两场 RGB/深度、原 CGR 全输出、原始候选/特征、科学指令数组共 8 个组件全部逐值相同。指令数组中的两个路径相关绑定 SHA 分别由原准备／审核验证，不作跨目录字符串等同要求。冷组两条科学指令还与原尾批对应结果逐值相同。

常驻组在同一个 Kit 中完成 capture0 → capture1，随后切换到 execute0 时失败。导入观测后端时，`canondex.robots.hand_actuation` 等模块已捕获默认路径 `/home/wangrenpeng/CanonDex/third_party/tendon-spin/assets/grasp/scene.xml`；后续原 `bind_hand` 调用未能改变已捕获的默认值，创建机械臂时 `FileNotFoundError`。原实际应绑定 `/home/wangrenpeng/tendon-spin/assets/grasp/scene.xml`。这是服务接线的导入顺序错误，不是抓取失败或物理速度结果。

原 `LIFT_PIPELINE_EXECUTION.json` 报 infrastructure_failure、physics_steps=0、label=null，`RESIDENT_EXECUTION_ENTRY.json` 报 execution_steps_unknown=false。原 failed job、空接触流及存储回执全部保留，没有重跑或回写。

## 独立修复与离线核验

新增 `author_collection_service_bound.py`，在首次 capture/Kit 之前从冻结 backend 读取真实 hand_urdf/hand_scene/reference_root，调用原 native `bind_hand`。如果已导入模块仍捕获旧路径，则在启动前拒绝；每条执行还核对计划中的三项引用与预绑定一致。仅替换旧服务构造 runtime 的一个 AST 位置，原 mailbox 顺序、存储、物理命令和判定不变；旧服务源码保持。

4 项新回归检查加既有 17 项，共 **21 项通过**。另在两个真实、无 Kit 的新 Python 进程中核验：先导入默认模块会被修复版明确拒绝；先绑定再导入可从实际 XML 读到 22 源关节＝18 主动（2 腕＋16 手）＋4 被动。初次前检误将源关节总数断言为 16，断言失败及更正单独留档，0 Kit/物理。修复版尚未进行新真实物理运行，不能声称已实际修复完整链路。

## 成本、收尾与后续边界

本轮 **5 Kit、3000 初始化 tick＋7138 抓取步＝10138 环境步／30414 三体步**，初始化接触点 10020812；执行未知成本 0。2 个完整诊断标签全部 exclude_training，累计有效新数据仍 26 fit＋5 internal＝31；正式训练、新模型与训后外测均 0。

host 独立核验 19 个阶段子进程组＋主控制器，共 20 组全部退出；4 当前资源槽 released、6 把锁可取、全局 registry released。无后台物理、规划或审核。固定 GPU0；未向 GPU1 派发任务。旧冻源和 CanonDex 不改。

原协议上限 5 Kit 已用完，retry_budget=0。本轮未自动扩预算。已请求追加 **1 Kit／最多 17500 环境步**，仅对修复版再做两场 g1 常驻组，复用已完成冷组作为历史基线；依旧排除训练、0 重试。若获准，需独立冻结新协议并保留本轮失败和成本；若未获准，停在本次未完成比较，不推断倍率或采满 ETA。

公共运行根：`/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/stream_collection_control_20261116_v1`。

轻量证据：`evidence/author_collection_stream_comparison_20261116_v1/`；启动前状态另见 `AUTHOR_COLLECTION_SERVICE_STATUS_20261116.md`，它是本轮运行前快照。
