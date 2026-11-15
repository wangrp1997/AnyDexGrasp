# 八路采集与两路首次审计：完整固定队列结果（2026-11-15）

独立 `author_pipeline_collection.py` 已接通新建场 → RGB-D capture → 官方 CGR → 四型 prepare → 最多八路 CPU 执行 → 最多两路原首次数值审计 → 固定顺序 ledger。固定旧开发 scene1、scene4 的八条首选重新走完整流程，外层 **905.730148 秒＝15.095502 分钟**。旧同组 resident 完整外层为 **1765.610592 秒＝29.426843 分钟**，历史 wall 比 **1.949378**，减少约 48.7%。这是已知两场景的历史计时比较，未做新配对或随机完整采集对照，不能外推全部采集倍率或用它除旧 15.02 天 ETA。

这次八条实际执行共 **25,656 环境步**，八份原数值审计全部 PASS，6 正 2 负与原开发记录相同；这是诊断标签，父记录 `label=null / exclude_training=true / formal_training_eligible=false`，新增可训练标签为 0。累计有效新采集仍 **14 fit＋1 internal**，正式 4000 fit＋800 internal、训练、新模型和训后外部评测均未启动。

协议 SHA256：`1020cf12313ad87abf668d2eb469bf9ffb39276eecc4ee280574efaa0c2f6fd3`。大输出位于 `/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/pipeline_collection_20261115_v1`；轻证据见 [EVIDENCE.json](evidence/author_pipeline_collection_20261115_v1/EVIDENCE.json)。本轮只允许两场景、八次执行、零重试，完成后停止，无自动补例或恢复生产。

原冻结 resident collector 与资源模块保持原字节。新适配器对其固定 SHA 源码做私有 AST 编译，逐一断言调度替换位置：Kit 上限 3→8，首次审计完成后交给两路队列，主线程沿原 scene/gesture 顺序验收原证书后写 ledger。场景生成、CGR、候选选择、碰撞与路径准备、实际执行和标签判定函数保留。资源独立登记，子进程持有所属 flock，进程组全退后才释放槽；取消时先保存意图再对已核身份的所属组 SIGINT 收尾，本轮无超时或取消。

执行分组显式改为**一条原 ready 计划一个冷 session**，两场景四型共八 session，每个只建一次资产、无边界 reset。旧 resident 对照为两个热 session、各顺序执行四型。本次证明的是八路冷执行与审计集成的完整流程吞吐，不是八路热复用，更不能说与旧热轨迹物理等价。每条仍用原 CPU 物理、250 Hz、0.1 kg、摩擦 0.6、三体向下力、native20µm、24→28 指令和原全部判定门、完整逐点接触日志；只选择固定 GPU0 UUID 提供场景/RTX/CGR 服务，不使用 GPU1。

所有八条在新 capture 后重新准备，没有复制历史观测、候选或计划。只读比较确认两场景初始化 trace、RGB-D，以及各 17 个 CGR 表示数组与原开发数组 exact；八候选各 36 个科学语义字段一致，候选池和实际命令的数值数组一致。候选/spec SHA 元数据因新路径与来源绑定变化，保留新值，不冒称所有 NPZ 字节或物理轨迹一致。完整输入和父 ledger 审核核对 745 个绑定文件，通过原三体身份、split、canonical480 特征、target index、原证书标签、成本和训练排除检查，见 [INPUTS_VERIFIED.json](evidence/author_pipeline_collection_20261115_v1/INPUTS_VERIFIED.json)。

| 原 queue index | 手型 | 实际步数 | 诊断标签 | 原首次审计 |
|---|---:|---:|---:|---|
| 1 | 1 | 3139 | 1 | PASS |
| 1 | 6 | 3008 | 1 | PASS |
| 1 | 7 | 3021 | 1 | PASS |
| 1 | 8 | 3312 | 1 | PASS |
| 4 | 1 | 3374 | 0 | PASS |
| 4 | 6 | 3024 | 1 | PASS |
| 4 | 7 | 3196 | 1 | PASS |
| 4 | 8 | 3582 | 0 | PASS |

29 个外部 stage 全部正常退出、重试 0。下表由保存的 STAGE 起点和单调 wall 重建；worker 合计包含并发，不可逐行相加冒充总 wall。父循环包络含额外协调/证书/绑定检查，与单 stage 的时间范围也不同。

| 阶段 | 次数 | worker wall 合计（秒） | 阶段活动并集（秒） |
|---|---:|---:|---:|
| build | 2 | 14.284016 | 14.284016 |
| capture | 2 | 121.277630 | 121.277630 |
| CGR | 1 load / 2 forward | 6.112350 | 6.112350 |
| prepare | 8 | 522.776236 | 262.887856 |
| execute | 8 | 2012.894970 | 275.472510 |
| 原首次 numeric audit | 8 | 484.660081 | 259.444753 |

执行与审计 stage 实际重叠 **56.941966 秒**。独立 host 观察 447 次，实见 **Kit 并发峰值 8、audit 并发峰值 2**、同一采样区间八份接触 journal 增长，27 次采样同时有执行和审计；自有 worker 峰值 RSS 合计约 24.17 GiB。仅查询固定 GPU0 UUID，采样错误 0。此处 RSS 是所观察 worker 的和，不是机器总内存；采样不能代替连续资源轨迹。详见 [SUMMARY.json](evidence/author_pipeline_collection_20261115_v1/SUMMARY.json)。

八个首次原审计共核 **62,194,441 个执行接触点**，没有第二遍执行点流审计。两个 capture 各 750 初始化 tick、2250 对象步，共 1500 tick／4500 对象步；独立只读 capture 审核各一次，22.066790／19.402881 秒，检查 1,892,396／1,595,112 个初始化接触点、1111／1135 原生 hull、全源顶点底面、稳定性、末 250 tick 地面承载、每场 96 射线及 RGB-D。两次额外 capture 审核在运行期间执行，竞争开销自然落在本次实测 wall 中，未从总 wall 扣除；它们不是 collector 内置 stage。审核均不启动 Kit、不积分、不调用模型。

本轮精确成本 **10 Kit＝2 capture＋8 execution**，27,156 独立环境 tick、81,468 三体步，其中初始化 4500 对象步、执行 76,968 对象步；执行在途未知为 0。两个 capture 和八次提起各执行一次，零重试，没有用 null 代替负例。CGR 载入一次、前向两次，评分头载入和训练 0。

新适配器 10 项 CPU/mock 测试及原 resident/wave 组合共 **39 项通过**，mock 不计物理。首次组合命令误写不存在的测试模块，23 项运行中有 1 个导入错误，原日志 [INTEGRATION_TESTS.log](evidence/author_pipeline_collection_20261115_v1/INTEGRATION_TESTS.log) 保留；更正模块名后的 39 项全部通过，之后才冻结和运行物理，未隐藏初失败或重试物理。

host 最终检查确认 **29 个所属 stage 进程组退出**、父进程与独立观察进程退出，14 个资源槽 released、16 个锁可取、registry released、无本轮隔离事件或后台任务。555 项运行源绑定和 24 组输入及完整源归档验证通过；旧 13 数据/权重文件与原 15 条新标签绑定不变，CanonDex 只读。见 [FINAL_STATE.json](evidence/author_pipeline_collection_20261115_v1/FINAL_STATE.json)。

当前默认入口未改变；新入口仍是最多两场景、排除训练的 opt-in 诊断。完整已知组约 31.80 条诊断标签/小时只描述本组八条全 ready 条件，不能迁移到随机场景产率，internal 三型零产出仍未解决。下一生产阶段仍需独立冻结对象、场景/尝试预算、训练准入和正式协议，再由用户决定是否启动扩采。
