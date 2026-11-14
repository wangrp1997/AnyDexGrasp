# 独立三场景并行采集入口（2026-11-14）

新增 `author_collection_waves.py`、`author_collection_wave_resources.py`，复用固定 SHA 的旧 `author_collection_batch.py`，不修改正在运行的开发队列、旧采集/训练/执行实现。父协议采用独立 schema `anydex_boya_author_collection_waves_v1`，旧 `B.run` 无法直接运行父目录。正式每型 1000 fit / 200 internal 是配额目标，正式总场景预算尚未决定；此实现没有自行扩采或训练。

## 冻结与接纳

先写 `WAVE_PLAN.json`，其后依次写每个单场景子 QUEUE/BATCH_PROTOCOL、`CHILD_FREEZE_INDEX.json`、最终 `WAVE_PROTOCOL.json`，避免父子 SHA 环。每个 worker 的额外 permit 绑定最终父协议、原场景 index、子协议和父 PID/start_ticks。保留原 `queue.seed`，CGR/候选随机策略和布局 spec 均不变，不引入 SeedSequence。

固定每波最多 3 个场景，波内全部完成后按原场景 index、g1/g6/g7/g8 顺序接纳首次合格标签，与正负类别无关。波开始时已满配额的 split 不再启动新子任务，写完整三体身份的 `quota_reached` 行。在途波可能产生超额标签，父行置 `label=None/status=quota_excess_not_admitted`，保留 raw_label、原子账本 SHA、实际执行成本和所有证据。无计划、基础设施故障、未执行均不生成负例。

父 ledger 保留原 index，子 ledger 的本地 index 固定 0。子协议、每条原始 ledger、stage 输出及执行/独立审计 SHA 均重新核验。子进程死于执行之后、写 gesture ledger 之前时，已有终回执成本照记，缺失标签仍为 null；没有终回执的已启动执行明确记 unknown。

## 资源及恢复

父进程持有原 GPU0 全局 lease。子批只在严格源码 pin、permit 和真实继承 flock 核验通过后绕过重复取得这一个锁，原子批 DRIVER 锁保留。资源为跨进程 3 Kit / 3 prepare / 1 capture-or-CGR observation / 2 audit；capture 同时取得 observation 和 Kit，execute 只取 Kit，原 `sim.step(render=False)` 可与 capture 共存。资源等待时间单独记录，不消耗原 stage 派发后的超时预算。

每个 stage 的 FD 集合为 thread-local，通过实际 Popen `pass_fds` 继承父 lease 和本 stage 槽 FD，不继承兄弟 prepare 的槽。所有 lease 只 close，不使用会解除共享文件描述锁的 `LOCK_UN`。完成回执不足以释放资源，必须确认整个 stage PGID 消失。reservation→Popen、Popen→PID 落盘两窗口的不确定状态，以及活着的孙进程，均进入 quarantine；不自动重启或补候选。

全局 lease 旁的 `author_collection_gpu0.wave_active.json` 登记当前 wave run。新的 wave run 取得全局锁后，先审前一个未释放 run 的所有 started worker、slot 和 PGID，阻止另换 run-dir 绕过无继承 FD 的原 native prepare 孙进程。未知 PID/未释放 reservation 需要人工核验，不自动清除。此登记由新入口执行，历史冻结入口没有增加这一新机制。

## 用法

冻结不启动 capture、CGR、规划或物理；输出允许为任意有权限的绝对目录。

```bash
PY=/home/wangrenpeng/tendon-spin/.venv/bin/python
$PY -B reproduction/boya/author_collection_waves.py freeze \
  --queue /ABS/QUEUE.json --run-dir /ABS/FRESH_WAVES \
  --calibration outputs/boya_adapter_preparation_20261105_v2/calibration.json \
  --cloud-manifest outputs/boya_full_alignment_20261114_v2/hand_clouds/MANIFEST.json \
  --backend reproduction/boya/backend.collection.json --python "$PY" \
  --fit-per-gesture 1000 --internal-per-gesture 200
$PY -B reproduction/boya/author_collection_waves.py run \
  --run-dir /ABS/FRESH_WAVES --max-waves 1
```

真实 run 由 root 统一派发；此子任务没有执行上述 run。父数据导出使用独立 `author_collection_wave_export.py`，只导出父接纳标签，独立重算全部 raw 标签的前缀配额、全场景三体隔离并再次审计实际 trial。没有旧兼容父 view，旧 435 不自动混入。

## 验证与边界

`test_author_collection_waves.py` 最终 17 tests / 9.401s 通过，包括纯 Python 真实子进程的继承 lease、无槽 build 活 PGID、两个 launch 故障窗口、跨 run 孤儿阻断，以及 thread-local FD、各资源上限、原种子、配额超额成本、部分父账本/部分子账本恢复。0 Kit、0 CGR、0 IK/RRT、0 新物理/标签/训练。

日志全保留在 `evidence/author_collection_waves_20261114_v1`。第二轮一次恢复测试漏 mock 原 legacy 锁，被实际运行的旧队列正常拒绝，0 stage 派发；仅修合成 fixture 后通过。后续最后两项源码修复对应 v4 全过，未覆盖旧失败日志。

此版本没有真实并发吞吐或加速证据。冻结仍逐子调用原 `B.freeze`，重复验证大 checkpoint/资产，尚不适合声称上千场景可低成本冻结；运行侧新 envelope 验证对共享依赖按 path/SHA 去重，worker 只核自身子批，原采集 stage 的前后 SHA 仍保留。最终导出逐 trial 重新独立审计同样有实际成本，本阶段未优化。

源码 SHA 和完整测试记录见同目录 `EVIDENCE.json`；若将来修改源码，需要新的协议和来源快照，不修改已冻结队列。
