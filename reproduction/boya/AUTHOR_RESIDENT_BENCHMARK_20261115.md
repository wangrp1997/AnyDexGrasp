# 常驻完整链路基准与采满时间估算

本文记录基准工具的实现范围；工具本身只冻结已有 resident 协议的对照关系、读取回执、比较和估算。实际小预算由 root 另立协议统一运行，实际结果见 [完整吞吐报告](AUTHOR_RESIDENT_THROUGHPUT_RESULTS_20261115.md) 与 [时间估算](AUTHOR_RESIDENT_ETA_RESULTS_20261115.md)。没有启动 4000 fit＋800 internal 大采集或训练，也未修改冻结执行器。

## 固定对照

沿 `author_resident_collection.freeze_from_batch(source_batch, scene_indices, output_dir, resident_workers=1..3)` 复用原场景描述和原 seed；重新运行 capture、CGR、prepare、execution、首次独立数值 audit。root 固定旧 serial scene1＋scene4 共八槽，两组三体各一个 resident 进程并行；scene4 旧结果 g1/g8 为有效负例、g6/g7 为正例。另固定 fresh internal 一个本地 index0 场景，共三场景/十二槽，硬预算最多十二次执行，历史预期九次 ready。选择在运行前冻结，不补候选、不换场景、不因结果追加预算，全部排除训练。本工具最多绑定 6 场景/24 槽，真正运行预算及结果以 root 独立协议为准。

该组已知结果适合检查四型正负例、不同指令的常驻 reset、候选和命令一致性与完整成本，不能代表随机场景的产出率。随后 2–3 个 resident session 并发也必须实测；不把 worker 数量或旧两执行的 1.98× 乘进全流程。

`compare_controls` 逐槽报告原三体 identity、candidate SHA、commands SHA 和 label 是否一致；缺失比较项明确为 null。新 scene 绝对路径和 SHA 可自然改变 candidate 的元数据摘要，SHA 不同不等于数值动作不同。root 在实际运行后独立比较 actual camera 行、canonical480、raw/extra、掌变换与 commands.npz 数组。wall 比值仅表示固定控制组观察值。

阶段耗时之和可能重叠，工具用阶段首尾跨度或新 driver 的 `timing_seconds.total_wall`；阶段跨度是漏外层边界工作的下界，不伪装为精确全程序时长。旧 serial 只选非连续 scene1＋4 时，采用两个场景各自跨度之和，排除未选 scene2/3 的工作；不对新并行队列加总场景跨度。新 driver total_wall 包含关闭执行子进程和首次 audit，进程最外层启动/退出仍由 root 独立测量。

## 现有数据只读核验

读取既有 serial 10 场景与 fresh wave 6 场景的 ledger、STAGE 和已经实际 PASS 的首次 `RUNTIME_AUDIT` 证书；仅核证书 SHA/status/label，不重复解码接触数据，不重新生成标签。

| 来源 | 场景数 | 有效记录 | 阶段首尾跨度 |
|---|---:|---:|---:|
| serial batch_v1 | 10 | 14 | 5295.205 s |
| fresh wave waves_v1 | 6 | 1 | 714.650 s |
| serial 固定 scene4 | 1 | 4 | 1270.318 s |

两批场景不同，714.650/5295.205 不能作为并行加速比。null 成本和放置失败均保留在分母。有效 label=0 计入采集完成；null、基础设施失败、未知成本不当负例。

| split | 各型尝试数 | g1 有效 | g6 有效 | g7 有效 | g8 有效 |
|---|---:|---:|---:|---:|---:|
| fit | 11 | 4 | 3 | 3 | 4 |
| internal | 5 | 0 | 0 | 1 | 0 |

按当前经验率，fit 最慢的 g6/g7 需要约 3667 个场景，才在期望工作量意义上填到各 1000 条；这是小样本情景推算，不能保证配额完成。internal 的 g1/g6/g8 五槽均零产出，既没有有限经验点估计，也没有可靠采满上界，不能借用 fit 产出率补齐。工具始终分别计算四型与 fit/internal，取同 split 中最慢型的场景需求。

仅作旧流程成本情景：旧 serial 平均 529.521 s/尝试场景，乘经验 fit 需求约 22.47 天，internal 仍不可估。这不是新 resident 预计耗时，也不是大采集启动建议；新流程必须填入实际完整 wall，不能仅替换执行倍率。已知控制场景的成本迁移必须显式 `control_timing_scenario=True`，报告保留选择偏差和场景工作量构成假设。

scene4 已有成本：build 6.804 s、capture 54.009 s、CGR 4.530 s、四 prepare 合计 258.879 s、四 execute 合计 843.043 s、四首次 audit 合计 225.700 s。prepare 存在重叠，不能直接加总作为 elapsed。

## 资源与边界

只读主机快照确认双 EPYC 9355：64 物理核、128 线程；总 RAM 503 GiB、可用约 422 GiB。快照 CPU/内存 PSI 约 0；历史 swap 占用不等于当前内存抖动。GPU0 由 root 在运行前核验，GPU1 保留其它任务，工具不探测 GPU 或自动增加 worker。仍沿已有 3 Kit / 2 audit 资源门，不改旧 pins。

现有官方 USD cloning / collision filtering 路径可作为后续多环境实现参考；当前 benchmark 不假称已接通多环境共享物理时钟。新 resident adapter 采用同组三体复用、各次原初态完整 reset，完整循环仍使用现有 capture 子进程、PersistentOfficialCGR 和原准备/独立 audit。

## 使用

```bash
python -B reproduction/boya/author_resident_benchmark.py freeze \
  --reference-batch OLD_BATCH \
  --resident-protocol NEW/RESIDENT_COLLECTION_PROTOCOL.json \
  --scene-indices 4 --output-dir FRESH_BENCHMARK

python -B reproduction/boya/author_resident_benchmark.py analyze \
  --run-dir NEW --selection-basis known_scene_control --output NEW_ANALYSIS.json
```

Python API：`analyze_run(run_dir, scene_indices=None, selection_basis=...)`；`combine_yields([observational_summaries])`；`compare_controls(reference, resident)`；`estimate_eta(yields, timing, control_timing_scenario=False)`。`combine_yields` 拒绝重复 trial 和已知控制组。`estimate_eta` 不接执行单项时长，不给零产出型填伪造速率，不启动采集或训练。

11 项离线测试通过，覆盖有效负例/null/infra/not-started、四型完整性、诊断排除训练、分层最慢配额、零产出、重叠计时、首次 audit 证书篡改、固定源 seed/spec/预算、候选/命令比较、重复统计、resident receipt 去重和非连续旧场景计时。`RESIDENT_EXECUTION_ENTRY` 与 session receipt 单独保留实际步数、Kit 启动、建场与 reset 数；共享 session 四槽只计一次。真实执行进程区间给出最多同时存活数，不能将该数称为 solver 同时运算证明。

最终只读结果保存到独立 `outputs/boya_author_resident_benchmark_20261115_v3`，原 v1/v2 开发分析保留。轻证据更新于 `evidence/author_resident_benchmark_20261115_v2`，含上一版源码 snapshot；冻结的 v1 证据不回写。
