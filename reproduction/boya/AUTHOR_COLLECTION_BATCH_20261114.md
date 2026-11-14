# 固定多场景真实标签采集入口

本阶段新增 `author_collection_batch.py`、`author_collection_export.py` 和入口测试，连接现有无箱三体、单 identity CGR、四型作者首选、完整臂 HOME 与原闭合提起流程。入口不调用旧评分头，不自动训练，不合并旧 435 条。正式目标是每型 1000 个 fit 有效标签，internal 每型另外 200 个；目标数量不是已经获得的样本。

开发队列固定 seed=2026111401、8 个 fit 场景与 2 个 internal 场景，每场景 3 个同组物体。队列中 7 个 ready、3 个 placement_null，全部保留原顺序和预算；placement_null 不创建物理场景、不补场景、不生成负例。开发 10 场景只用于实际贯通和成本检查，无法满足正式容量要求。正式总场景预算须在开发结果之后另行冻结，不能按正例比例延长这份队列。

`freeze()` 只写协议；`run()` 才启动外部进程。每个 ready 场景最多一次原 750 tick 初始化、一次 identity CGR，四型 g1/6/7/8 分别映射 DH3 reference 1/2/3/4；每型只准备原 top500 固定型、原 source flip 和随机 extra 所选的首个候选。原 4cm approach、HOME 一次 RRT 与允许的一次同路径 retime、原 250 close/50 hold/10cm lift 保持现有模块语义。最多 3 个 prepare 并行；capture、CGR 与实际执行顺序串行，冻结 GPU0 可见性，原执行 wrapper 显式指定 GPU0。

每一阶段在启动外部进程前先持久化 `STAGE.json`，已经启动的阶段不重试。完成阶段按输出 SHA 复用；中断阶段记基础设施 null，孤儿进程仍存活或进程身份未知时禁止继续派发。每个 scene/gesture 对应一个独立不可覆盖的 ledger 行。无候选、无计划、初始化/基础设施失败以及未执行均不能计为负例。各型按场景顺序保留前 quota 个有效标签，不按成败平衡或更换候选。

实际执行结束后，入口先运行独立 `audit_author_collection_runtime.py`，保存完整审计 JSON 与 SHA，再复核实际候选、canonical480、原 selected output、命令、场景、评价和 record。只有独立审计 pass、三个原始物体身份与原视觉 mesh digest、全场景 split 和实际标签一致时，才将该标签计入进度。初始化接触 journal、完整 gzip 数据、轨迹、RGBD、PNG/meta、contact sensor/render 检查和 overlay 一并绑定；恢复与导出会实际复核 stage 输出。SHA 使用流式读取。

成本分开记录：`actual_physics_steps` 仅累计已持久化执行回执中的已知执行步，不包括初始化；`captured_initialization_ticks` 和 `captured_initialization_object_steps` 分别记录完整 capture 的 750 ticks 与 2250 object-steps。失败或中断初始化成本不含于这两个下界。已启动但无最终执行步数的情况通过 `execution_starts_with_unknown_steps` 显式记录，不能解读为实际零步。

导出必须重新验证每条 record 并重新运行独立全体物体审计。每型一个非 pickle NPZ，包含实际 `features(N,480)`、`labels`、`target_index=raw_depth_index*4+extra_depth_index`、trial/candidate/command/evaluation/record 绑定，以及对齐排序的 `scene_object_ids(N,3)` 和 `scene_mesh_sha256(N,3)`。后两列保存全部原始视觉 source 身份；`mesh_sha256` 只为兼容保留 primary，不能单独作为隔离依据。导出拒绝任何 fit/internal 的全场景对象或 source digest 交集。未达到冻结容量仍可导出开发数据，但 `size_ready=false`，不会自动启动训练。

运行命令（由 root 在冻结全部源后派发）：

```bash
PY=/home/wangrenpeng/tendon-spin/.venv/bin/python
RUN=/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development
$PY -B reproduction/boya/author_collection_batch.py freeze \
  --queue "$RUN/scene_queue/QUEUE.json" --output-dir "$RUN/batch_v1" \
  --calibration outputs/boya_adapter_preparation_20261105_v2/calibration.json \
  --cloud-manifest outputs/boya_full_alignment_20261114_v2/hand_clouds/MANIFEST.json \
  --backend reproduction/boya/backend.collection.json \
  --fit-per-gesture 1000 --internal-per-gesture 200 --prepare-workers 3
$PY -B reproduction/boya/author_collection_batch.py run \
  --batch-dir "$RUN/batch_v1" --max-scenes 1
# 首场独立核验后，继续同一冻结队列；不会重复已启动阶段。
$PY -B reproduction/boya/author_collection_batch.py run --batch-dir "$RUN/batch_v1"
$PY -B reproduction/boya/author_collection_export.py \
  --batch-dir "$RUN/batch_v1" --output-dir "$RUN/export_v1"
```

freeze 会保存所用 Python 虚拟环境路径及继承的 `PYTHONPATH`、`LD_LIBRARY_PATH`；派发前需使用既有 CGR native-dependency 隔离环境。大输出可使用任意显式绝对目录，仓库只保留轻量源码与证据。

14 个入口/导出测试在 3.976s 内通过，覆盖 once-only 恢复、3 个 prepare 并发上限、串行执行、全场景隔离、quota 前缀、placement null、真实轻量子进程的来源篡改、孤儿阻断、执行未知成本、独立审计拒绝标签，以及无 pickle 导出。第一版 12 tests 有 2 个 export mock 错误：测试对同一模块使用了两个 import 名称，mock 未作用于 exporter 所引用的对象；修正测试 import 后 12 tests 全过，再增加初始化证据/中断成本两项最终 14 tests 全过。所有初失败日志保留。

本报告的开发测试未运行 Kit、CGR、实际机械臂规划或试抓，未产生真实标签或训练更新。root 随后已按冻结源码启动新公共目录中首场开发前缀；其实际结果另由运行协议、ledger 和阶段报告记录，本工具测试通过不代表正式数据容量或完整复现已经完成。

源码与测试证据：`evidence/author_collection_batch_20261114_v1/EVIDENCE.json`。
