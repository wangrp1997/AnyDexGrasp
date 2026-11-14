# 已完成独立审计的证书复用导出

新增 opt-in `author_collection_certified_export.py`，只包装冻结的串行 `author_collection_export.export` 或 `author_collection_wave_export.export`。原准入、原始账本、四型前缀/数量、全三体 fit/internal 隔离、480 特征与全部 NPZ 字段均由原函数处理；原采集、执行、审计、导出和 CanonDex 源码没有修改。没有新增采集、物理、模型前向或训练。

```bash
python -B reproduction/boya/author_collection_certified_export.py \
  --batch-dir /path/to/completed/batch \
  --output-dir /path/to/fresh/export

python -B reproduction/boya/author_collection_certified_export.py \
  --run-dir /path/to/completed/wave \
  --output-dir /path/to/fresh/export
```

应使用项目既有 venv 和 `PYTHONPATH`；本次实际命令及环境存于轻证据。只支持独立 CLI 进程中的主线程，不支持把临时函数替换嵌入多线程服务。通过受控 context 暂时替换 `E._audit_trial`，`finally` 恢复原函数；没有 numeric audit fallback。

## 复用条件

每条数据必须已有一次真实完整的独立数值审计 PASS。原 labelled ledger 必须同时绑定 `RUNTIME_AUDIT.json` 和原成功 audit STAGE；该 STAGE 的 source、prepared-dir、split-manifest、output、returncode 与输出 SHA 必须精确对应。证书的 trial/scene/split/candidate/commands/label/全三体身份和原 record/spec 必须一致。

新入口锁定原五份 exporter/batch/waves/audit 源码 SHA；证书 `audit_source` 必须是当前冻结的原独立审计源码。重新完整流式 SHA256 验证 `audit_inputs_sha256` 的每项输入，必需的 trial、source visual/converted mesh、trace、contact pointer/gzip、split manifest、原 audit 源码不得缺失。不是仅检查小 JSON pointer。

同一次导出只对完全相同 `(canonical absolute path, expected SHA)` 去重，不跨运行缓存、不读取 mtime 决定跳过，也不抽样。最终发布前对全部唯一输入再完整读一遍。原证书、账本和 STAGE 也在复核集合中；任一内容变化、缺失、失败或身份错配均拒绝，没有第二次接触解码。新输出先写独立 staging 目录，最终核验通过后才原子改名为目标目录；失败 staging 留存，目标正式目录不发布。

新 `MANIFEST.json` 的 schema 为 `anydex_boya_author_certified_collection_export_v1`，明确 `independent_audit_mode`、原 exporter schema、每条原证书/账本/STAGE/source SHA、原 numeric audit 成本、本次 hash 验证成本和 `numeric_replays=0`。复制的原 PASS 证书仍保留原时间和内容；不称为重新完成独立数值审计。

## 原 14 条数据的实际验证

源为已结束的 `development/batch_v1`。新目录 `outputs/boya_author_certified_export_20261115_v1` 一次成功导出，完整入口计时 **16.587 s**；1434 个唯一文件完整核验两遍，共 2868 次文件读取、9537915454 bytes，hash 计时 6.019 s。缓存/当前并行负载会影响时间，未承诺冷盘或大队列同比速度。

复用的是 **14 份原 PASS 证书**，其原首次 numeric audit wall 总和 **815.965 s**；本次 numeric replay 为 0。这避免重复解码已独立验过的接触，不代表新采集快了约 49 倍，也不是一次新的数值审计。

独立脚本直接读原 candidate、record、ledger 对照新 NPZ，全部字段/类型和非 pickle 格式通过；14 行 ×480 float32 值、label、`raw_depth*4+extra`、split、全三体身份/源 digest、命令/evaluation/record SHA、复制证书 SHA 均一致：

| 型 | 行数 | 正 | 负 |
|---|---:|---:|---:|
| g1 | 4 | 3 | 1 |
| g6 | 3 | 2 | 1 |
| g7 | 3 | 3 | 0 |
| g8 | 4 | 2 | 2 |
| 合计 | 14 | 10 | 4 |

本批全部 fit，internal 为 0；原 `size_ready=false`、不足预算和旧负结果保持，不能把它当正式数量采集完成。旧 435 未合并，0 训练。

7 tests /13 合约子例通过，5.848 s，覆盖 contact/mesh/source/certificate 篡改、缺证书、错误 trial/审计源码、failed 证书、缺 input、首次核验后数据变化、fit/internal 非 primary 泄漏、串行 NPZ 字段逐值相同，以及 wave 原准入前缀。首次测试仅 wave 合成 fixture 漏 mock 原 `W._admit` 二次真实 validator，报 ready preparation 错误，0 Kit/物理；已只修 fixture 并留失败记录，生产入口源码未因此修改。

轻证据：`evidence/author_certified_export_20261115_v1`。这项优化到此收尾；没有继续开发接触列存 runtime，后续架构沿用户选择的现成批量循环和官方 env cloning 主线。
