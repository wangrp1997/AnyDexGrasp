# 2026-11-14 原增强实抓对照证据

报告：[AUGMENTED_PHYSICS_RESULTS_20261114.md](../../AUGMENTED_PHYSICS_RESULTS_20261114.md)。`SUMMARY.json` 为简明结果，`PROTOCOL.json` 为正式物理协议；`STUDY_PROTOCOL.json` 与 `ALL_CHOICES_FREEZE.json` 保存事前范围和16条件选择冻结。

三件已用物体、受控腕诊断。原MSE主比较1/3→1/3，现用preactivation次比较0/3→2/3；A0另列四条件失败，不是独立外部或完整臂成功率。14姿态=7历史+7新一次，14700新积分步，无训练/默认晋级。

`snapshots/` 仅保存原本在忽略输出目录执行的准备、零步诊断、独立核验和媒体脚本；相对目录语义绑定其原位置，勿直接在证据目录执行。原位置及源SHA见 `FILES.json`。完整数据、CGR、八权重、网格、轨迹、视频、依赖轮包和逐凸体原生审计均不提交。

零步故障、更正和依赖来源均保留。`ZERO_STEP_RECOVERY_PASSED.json` 是本地完整回执的摘要，含原文件SHA；原缺依赖失败回执保留不改，其继承的 `scope` 文案更正见 `DIAGNOSTIC_SCOPE_ERRATUM.json`。`KNOWN_SUCCESS_RETENTION.json` 为执行后描述性检查，未用于选姿或新回退策略。

本地完整入口：`outputs/boya_augmentation_physics_20261114_v1/index.html`，正式运行在其 `physical_runtime_v3/`。媒体14段（7历史+7新），全部25fps/全帧解码/源绑定通过、回放0积分。
