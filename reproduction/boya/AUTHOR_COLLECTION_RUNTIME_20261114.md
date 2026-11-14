# 新采集运行时的无损接触流与独立审计

本阶段交付两个独立入口：`author_collection_execute.py` 执行既有一次性 lift 入口，`audit_author_collection_runtime.py` 审计已经落盘的结果。未修改旧 lift、transport、原生执行器或 CanonDex；本 agent 未启动 Kit、规划、物理、CGR 或训练，未生成实际标签。

执行入口将每个原生 tick 的完整 `object_contacts` 和 `robot_contacts` 写入 gzip level 1 JSONL，保留零载荷点及全部原字段。原类的控制、每步重量、更新、全部动态物体门保持继承。原 `finish` 函数 AST 原样编译，仅通过 `common.save` 代理将 `dict(rows=journal)` 保存为显式指针。指针 `MULTI_OBJECT_CONTACTS.json` 不含伪造的 `rows` 数组，schema 为 `anydex_boya_contact_journal_pointer_v1`；原 `LIFT_EVALUATION.contacts_sha256` 继续绑定该 JSON，JSON 的 `data` 再绑定压缩文件的绝对路径、SHA 和字节数，另列总 tick 数。异常也关闭完整流并保存指针。

新启动器锁原 `_launch_native` AST，只替换 `kit_args`，显式设置 renderer GPU 0 和 `multiGpu/enabled=false`。原 headless、CPU 物理、EULA 和 visualizer 参数保留。所有流与 wrapper 最终回执先落盘，再调用真实 `SimulationApp.close`，覆盖其可能直接 `os._exit` 的行为；物理成功与否由所绑定的原 pipeline 回执判定，wrapper 的 `finalized` 只表示自身收尾完成。

审计使用原始 PLY SHA 识别数据划分，使用转换后 OBJ SHA 核验实际源几何，两者分别报告。三个物体必须全部属于同一冻结划分，不能仅检查兼容 primary。审计核验 READY、协议、实际命令、record、评价、源代码与物理输入的 SHA，并在读完后再次核验。完整解压逐 tick 接触，独立重算每体五指载荷、非法机械臂支撑、穿透、源网格最低顶点和最终提起标签；从 XML 独立展开 24 命令列到原生 28 列，核对实际目标、测得限位与 mimic。保持时长只作为诊断，不添加新的 0.5 秒门。

只有真实完整、所有门通过且原 record 已明确 eligible 的结果，审计才能返回 `status=pass` 和整数标签。完整物理失败可保留 0；无执行、缺 record、不完整、基础设施错误、无效门或任何数值/来源不一致均不能变成 0。审计不生成新的 collection record，也不修改原标签；batch 在入账前检查此独立报告，export 再检查一次。

```bash
python -B reproduction/boya/author_collection_execute.py --prepared-dir PREPARED_DIR
python -B reproduction/boya/audit_author_collection_runtime.py \
  --prepared-dir PREPARED_DIR --split-manifest SPLIT_PROTOCOL.json --output NEW_AUDIT.json
```

两个入口均需要既有项目运行环境。执行入口继续使用原 native 依赖配置；审计只依赖已有 NumPy、SciPy、trimesh 和标准库，不加载模型或 simulator。审计 CLI 正常完成时，无论 pass/fail 均保存完整 JSON 并 exit 0，由 batch 按 `status` 和 `formal_training_eligible` 决定是否入账；输出路径拒绝覆盖。

14 项测试通过，其中 7 项执行适配测试、7 项独立审计测试。审计夹具包含 1351 个连续 tick、完整六阶段、24/28 列与三物体，不启动模拟器。反例覆盖非 primary 成功、完整失败 0、缺回执/不完整/基础设施错误、假源底、伪造手指力、前臂托物、mimic 命令改变、非 primary 跨划分、原身份/转换几何混淆、压缩文件篡改、压缩 tick 数和 flat source SHA。执行测试包含真实子进程 `os._exit` 前落盘、原 AST 评价等价与 GPU 0 启动参数。

另对已用 EGAD A0 旧真实执行作只读核验：独立 XML 展开的 2100×28 原生目标与原 24 列计划逐值完全相同，最大误差 0，实际 desired palm 与时钟也完全相同。它验证审计器能解析真实生产资产/列序，不把旧成功、观测或标签计作新采集。

旧 capture 的 `initialization_contacts.json` 为 691,903,579 字节，750 tick 共 1,530,750 个接触点。对原字节只读压缩且丢弃输出，gzip 1 得到 59,541,998 字节（8.6055%），gzip 6 得到 47,923,112 字节（6.9263%）。这是一份本机存储测量，不能证明新 JSONL 格式具有相同比例或实际采集速度加快。若 4000 个 fit 标签在每 scene 四型均有效时至少需要 1000 个 scene，旧初始化接触约 644.4 GiB；按本次 gzip 1 的比例估算约 55.5 GiB，尚未包括 internal、实际执行、trace 和图像。有效计划比例下降会增加 scene 数量，不承诺队列完成时间。原 runtime 一次执行还会累积百万级接触，故其内存列表也必须改为逐 tick 流式存储。

旧单物体 positive 只可作为明确标注的开发来源，其 10 cm 接近、观测、任务、完整场景均不同；不能复用旧有箱计划、指定旧成功候选、增加种子或把旧成功当作新标签。合成测试与来源审计不证明真实物理可抓性或新增成功率，实际队列由 root 另行冻结执行。

轻量证据：`evidence/author_collection_runtime_20261114_v1/EVIDENCE.json`。运行源已交 root 与 collector 冻结，后续报告更新不回写冻结实验结果。
