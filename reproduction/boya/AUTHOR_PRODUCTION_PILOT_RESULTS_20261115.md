# 有限生产试采：6 条合格新标签，日志完整性故障后停止（2026-11-15）

用户授权的 16 fit＋16 internal／32 场景／128 手型槽已实际启动；**本批未跑完**。首个零 tick 依赖失败保留后，只接续原未启动场景；接续在第二窗口因两份接触日志损坏停止。最终新增 **6 条通过原数值审计且已导出的 fit 正标签**，四型 g1/g6/g7/g8 为 **2/1/2/1**，新增 internal 为 0。其余 15 场景／60 槽未派发；没有重抓、补场景、改阈值或训练。

累计有效新采集标签为 **20 fit＋1 internal＝21 条**，其中此前 14 fit＋1 internal 保持原绑定，本次 6 条单独导出，不自动混入旧数据。旧 435 与八权重的 13 文件 SHA 保持不变；全量 4000 fit＋800 internal、训练、新模型及训后外部评测均未启动。上述标签数不是模型成功率。

| 手型 | 本批新 fit | 正／负 | 本批新 internal |
|---|---:|---:|---:|
| g1 | 2 | 2／0 | 0 |
| g6 | 1 | 1／0 | 0 |
| g7 | 2 | 2／0 | 0 |
| g8 | 1 | 1／0 | 0 |

初协议 SHA `acc3f76398eece862006e2e91d46f3745257c2f43f90735e7cf74fef5cca81e8`，接续协议 SHA `d1c9dbb38f4b9bea9554774c7768b6a24ab5f4dc3dd26e7e543648bf8aa71d8c`。大输出分别在 public 的 `production_pilot_20261115_v1` 与 `production_pilot_20261115_continuation_v1` 下，均属于 `/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/`。合格导出为后者 `run/export/MANIFEST.json` 和四份 NPZ；原 OUTER/PRODUCTION_RESULTS 保持 stopped，另有独立 `PARTIAL_EXPORT_RECEIPT.json`，不把导出前缀冒称完整批次。

原 sampler 在抽样前固定 seed2026111501，fit/internal 交错。32 场景中 19 ready／13 placement_null，失败不补抽；原 S40 32 fit／8 internal 物体池及全三体隔离不变，未扩大为论文 144 物体。生产入口复用固定 SHA 的八路原运行时，最多 8 CPU 冷执行、2 原首次审核、3 prepare、1 observation；本批实际峰值为 **4 Kit／2 audit**，不是八路始终占满。保持原 250 Hz、0.1 kg、摩擦 0.6、三体向下力、24→28 指令、native20µm、完整接触记录与原全部判定规则，GPU0 服务场景／RTX／CGR，GPU1 不选。

首次运行因新 freeze 未继承原已验证的 clarabel 覆盖 PYTHONPATH 和 LD_LIBRARY_PATH，在第一个 capture 的原生审计导入阶段报错。ERROR/PROGRESS 记录初始化 0 tick／0 对象步，1 Kit、0 CGR／规划／抓取；初父表保守 unknown_capture=1 原样保留。独立检查确认失败发生在原初始化循环之前，无完整初始化证书。原失败、源码和协议不回写。恢复原环境后，用精确子进程预检成功导入 clarabel、Torch 和原 cooking audit，额外绑定依赖包文件；只取原 queue index1..31，31 场景／124 槽，窗口 8/8/8/7，首个失败场景从未重跑。

接续前两窗的 64 个终态为：

| split | 合格标签 | 无候选 | 几何拒绝 | placement_null | 审核基础设施失败 |
|---|---:|---:|---:|---:|---:|
| fit | 6 | 8 | 8 | 8 | 2 |
| internal | 0 | 6 | 2 | 24 | 0 |

这里是手型槽计数。加上初失败场景的 4 槽和未启动的 60 槽，合原 128 槽；初轮对未启动场景写出的停止元数据保留，但不重复加总成额外尝试。只读拒绝复核发现 10 个 geometry 均记录 `frozen_endpoint_or_non_home_collision`，14 个 no_candidate 的 collision_admitted 均为 0；只反映保存的直接路径与原拒绝原因，不是全局不可达证明。见 [NULL_PREFIX_REVIEW.json](evidence/author_production_pilot_20261115_v1/NULL_PREFIX_REVIEW.json)。

本轮实际有 8 个新完整 capture，各 750 tick，独立初始化审核全部 PASS；共 **6000 tick／18000 对象步／15,080,831 初始化接触点**。官方 CGR 共 2 次载入／8 前向，0 手评分头载入。8 次实际执行报告 **25,462 步／76,386 对象步**，成本包括两条被排除的记录：g6 3509 步、g8 3679 步；每次完整步 trace 与最终回执对齐。六份通过的首次原数值审计核 33,994,219 个执行接触点，原全三体源底面、受载、全部实际命令与判定通过。

两份失败均来自新 fit scene6 的 g6/g8：g6 在第 742 行首先报 JSONDecodeError，后续纯 deflate 结构检查也报 invalid literal/length code；g8 读取至第 719 行时报 invalid distance code。两份 gzip 的字节大小、当前 SHA 与原 pointer 已记录的 SHA 都一致，复制到 `/tmp` 的文件逐字节相同并在相同位置失败。因此现存字节流不可用，不能用回执内的原始标签替代完整数值证书；父标签均保持 null。**具体损坏根因仍未证明。** 原文件未修补、未删除，也未重做抓取或将其记成负例。纯 gzip/JSON 结构诊断不认证物理标签，不是重跑完整 force/source-bottom/command 数值审核。见 [CONTACT_SYNTAX_DIAGNOSTIC.json](evidence/author_production_pilot_20261115_v1/CONTACT_SYNTAX_DIAGNOSTIC.json) 与 [G6_GZIP_CRC_DIAGNOSTIC.json](evidence/author_production_pilot_20261115_v1/G6_GZIP_CRC_DIAGNOSTIC.json)。

接续控制器停止后，仅对已有六份 PASS 证书进行绑定核验和原数组 writer 导出，没有再次解码其执行点流。独立前缀核验检查 **1556 个文件绑定**，逐条重核 trial、480 特征、标签、全三体身份、split、NPZ 形状／类型及完整执行步成本，输出均为非 pickle 数组；两条坏日志不进入数据集。见 [PREFIX_AUDIT.json](evidence/author_production_pilot_20261115_v1/PREFIX_AUDIT.json)。

接续两窗外层 **1792.661695 秒＝29.877695 分钟**，首失败外层 53.460996 秒，两段 wall 合 1846.122691 秒＝30.768712 分钟。首段启动到接续停止实际跨度 2356.866509 秒＝39.281108 分钟，保留中间 510.743818 秒依赖修复／交接间隔；不包含之前的实现测试或之后的只读诊断。该批因故障截断，不能据此给出完整随机产率、加速倍率或采满 ETA，不能用旧两场景 1.95 倍缩放。

总 Kit **17＝首失败1＋新 capture8＋执行8**。接续已报告 **31,462 环境 tick／94,386 三体步**，接续执行及 capture 未知成本均 0；首失败父表的保守 unknown_capture=1 仍单列保留，错误回执报告 0 tick。八次执行成本均纳入，接触坏流的两条未获得完整 gate／标签认证。独立 host 观察 880 次，峰值 4 Kit／2 audit、自有 worker RSS 约 14.57 GiB。首轮 2 个 stage 组和接续 74 个 stage 组均已退出，两轮父／观察进程均退出；接续 10 槽 released／12 锁可取，初轮 2 槽 released／4 锁可取，registry released，无本轮后台任务。

原审计 CLI 会在保存 `status=failed` 后退出 0；原父准入已正确拒绝坏证书，但提前停止不能只看进程退出码。新增独立 `author_production_control_v2.py` 在失败证书返回时即停止后续非审核派发，已有任务与审核仍收尾，原数值判定不变；**该 v2 未进行新物理执行，也不解决压缩流损坏的根因。** 后续新队列的 bootstrap 使用 `freeze_production_pilot_v2.py`，显式新 seed，冻结原环境并在任何 Kit 前做真实依赖预检；历史 v1 冻结脚本保留。当前采集保持停止，这些新入口没有追加本批预算或执行。

原 39 项复用检查、新生产 13 项、接续 4 项及后续修复 3 项不同 CPU/mock 覆盖通过，共 59 项不同覆盖；重叠运行不累计为额外覆盖，mock 不作物理成绩。原运行源和归档保持，CanonDex 只读；轻量代码／证据入 Git，大输出和六条 NPZ 留 public。

下一准入项是执行接触日志写入到持久字节的完整性验证，例如写入前逻辑 SHA、关闭后 readback 校验和可复核的发布证据，随后才考虑剩余 15 场景。当前 internal 三型零产出问题仍在，完整目标的产率／ETA 未知；不启动全量、重抓、补比例、训练或外部评测。
