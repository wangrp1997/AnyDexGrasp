# 新随机场景有限生产试采（2026-11-15）

用户批准后，已启动 **16 fit＋16 internal、共 32 场景／128 手型槽**的有限生产试采。首次运行已在 **53.460996 秒**按规则停止：新冻结入口未继承此前已验证的 clarabel 覆盖路径和共享库路径，第一 capture 在原生审计导入时报 `ModuleNotFoundError: clarabel`，ERROR/PROGRESS 均记录初始化 0 tick；原父成本表仍保留保守的 1 次未知 capture 标志，不回写。该轮 1 Kit、0 CGR／规划／抓取／标签／训练，无物理重试。首个失败场景保留，剩余 31 个未启动场景拟在独立接续协议恢复原依赖环境后继续；原预算不增加、不重抽随机布局。

新协议为 `anydex_boya_author_production_pilot_v1`，SHA256 `acc3f76398eece862006e2e91d46f3745257c2f43f90735e7cf74fef5cca81e8`。真实运行根为 `/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/production_pilot_20261115_v1/run`。本文件为滚动进度，不是完成证明；完成必须同时有 `PRODUCTION_RESULTS.json`、`OUTER_WALL.json`、独立审核和已释放的资源证据。

首轮所属 build/capture 两进程组、父和观察进程均已退出，2 个槽 released／4 锁可取。错误是冻结入口遗漏原 `inherited_env`，并非随机物体不可抓或新抓取负例；原运行的源和回执均保留不改。原 source SHA `99f73043...` 的生产适配器也保持不变，独立接续适配器只私有 AST 编译剩余预算和尾窗大小。

接续协议 `anydex_boya_author_production_continuation_v1` SHA256 `d1c9dbb38f4b9bea9554774c7768b6a24ab5f4dc3dd26e7e543648bf8aa71d8c` 已冻结并启动，根为 `/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/production_pilot_20261115_continuation_v1/run`。它严格取原 queue 的 index1..31，不重抽、不换布局，31 场景＝15 fit＋16 internal、124 槽，窗口大小 8/8/8/7。原 index0 失败消耗的四槽单列保留，合并原始预算仍 32 场景／128 槽。接续 4 项 CPU/mock 检查通过；原依赖环境的精确子进程导入预检 returncode0、stderr空，并额外绑定 clarabel 包文件，0 Kit／积分／模型载入。

随机种子在抽样前固定为 **2026111501**。原 sampler、三体几何、物理、候选与判定规则保持；抽样后按 fit0/internal0/.../fit15/internal15 的预定顺序交错，分四个各八场景的窗口。原 S40 32 fit／8 internal 物体池与全三体隔离不变，未扩大为论文的 144 物体。32 场景中 19 ready＝12 fit＋7 internal，另 13 placement_null＝4 fit＋9 internal，全部保留并消耗预算；不是 19 条标签或成功率。

独立 `author_production_collection.py` 复用固定 SHA 的八路运行时，通过新 schema 的终态与准入函数写生产记录，旧诊断入口及所有冻结源不改。每窗重新 build/capture/CGR/prepare，最多 8 CPU 冷 session、2 原首次数值审计、3 prepare、1 observation；GPU0 UUID 固定，GPU1 不选择。parent 持有整个试采的全局 lease，跨窗资源槽不能绕过锁；一个实际计划只执行一次。

每个完成 capture 都先运行原独立初始化 auditor（每份仅一次，零 Kit／积分），核全三体源网格、完整初始化接触、原生 hull、RGB-D 和射线。实际抓取只有原运行证书、原全数值审计和初始化独立审计全部通过，才能在窗口与父 ledger 准入为新 0/1 标签。无候选、碰撞拒绝、placement_null、基础设施故障保持空标签。导出复用原数组 writer，以已绑定的首次原证书进行核验，不再解码第二遍执行接触流。旧 435／原 15 标签不自动混入本批数组。

首个基础设施／完整性错误停止新的非审核 stage 和后续窗口；已启动进程与原首次审计安全收尾，成本即使不能晋级标签也保留。整个控制器有 4 小时后停止新派发的上限，单 stage 1200 秒；不因此杀掉无关任务。外部停止通过独立持久 `STOP_REQUEST.json` 请求，先写 `STOP_INTENT.json` 再停止新派发；不会将等待或未知成本记成零。所有运行拒绝在 `RUN_STARTED.json` 后重启。

离线验证先有 50 项组合检查通过（包含 11 项新生产测试＋原 39 项），补强子记录训练排除检查后 12 项生产测试通过，再新增 1 项持久停止测试通过，即 13 项新覆盖／39 项复用覆盖；重复执行的测试不累计为新覆盖。均为 CPU/mock，不是物理成绩。日志在 `evidence/author_production_pilot_20261115_v1`。启动前只读核验旧 29 个所属 stage 组全退、registry released、128 logical CPU、public 约 5.15 TiB 可用，原 13 数据权重和 15 标签绑定未变。

父 PROGRESS 仅在窗口结束刷新；窗口内实时状态应读各窗 `ledger/*.json`、`stages/*/STAGE.json`、`AUDIT_PIPELINE_RESULTS.json` 与独立 host 观察，不把父初始快照当作未启动。当前结果、完整 wall、每型新有效标签数和 internal 覆盖均待实际结束核定，不以旧两场景 1.95 倍外推本批 ETA。
