# index5 / g6 BACKEND.json 全零只读诊断

2026-11-14，原固定批次 `s40_lift_fit_00005_s2026111401` 的 g6 在准备阶段发生 `JSONDecodeError`。现有 `BACKEND.json` 为 **2150 字节、全部为 NUL**，并非空文件或不完整 JSON 尾部；普通 sandbox 只读和与 driver 相同的 unsandboxed `O_RDONLY + pread` 读到相同字节与 SHA。

原 `LIFT_PIPELINE_PROTOCOL.inputs.derived_backend` 绑定 SHA 为 `fd92caccfb0bbbf8c2e42bd77bc18b1427e0a88212afc64beaae35d4d447ef9d`；现全零文件 SHA 为 `dae81d94ed0b0769cb7037c135ccb70d39e26c882b5cc542671ea5492e3323e8`。原绑定在本 trial 保存配置后，直接通过未缓存的 `SHA256(Path.read_bytes())` 获得。g1/g7/g8 的独立文件仍是有效 JSON，SHA 均等于原绑定；四文件无 symlink 或硬链接共用，inode 各异。

源码中此路径由 `author_lift_pipeline.prepare` 唯一一次写入，使用本 g6 目录中的 `.tmp` 加 replace。随后 collector 只写 `collection/*`，bridge 只创建 `pool/*`；`common.prepare_approach` 在读取配置时就失败，尚未形成或启动 native 准备命令。已检查的执行链中没有发现第二处对这个文件的写入、清零或截断。检查旧 prefix0–5 的 797 个不超过2MiB的 JSON，仅此文件首字节为 NUL；读取时无仍打开此文件或其 `.tmp` 的进程句柄，也无残留 `.tmp`。

**可确认的是：正确 SHA 读取后、后续 JSON 读取前，文件内容发生改变。第二写者及原因尚未查明。现有证据不能将其归因于并行 prepare 冲突，也不能将其归因于存储故障。** 已停止扩大搜索和存储测试。

该 slot 的原准备回执与 ledger 保持 `infrastructure_failure`、实际执行0步、`label=null`，没有 `EXECUTION_STARTED.json`；不能当作负标签。未修复或覆盖原文件，未重试规划或抓取。本诊断自身为0 Kit /0物理/0规划/0CGR/0训练。新 wave 开发协议只有冻结、0派发，真实并发继续暂停；原固定队列保留此失败并按原协议继续。

轻证据：`evidence/author_backend_nul_20261114_v1/DIAGNOSIS.json` 保存原与现 SHA、两种读取一致性、文件身份/字节、兄弟配置、失败回执与日志/账本绑定；`EVIDENCE.json` 绑定本报告和诊断文件。未修改任何冻结源码、原输出或 CanonDex。
