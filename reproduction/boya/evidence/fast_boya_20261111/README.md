# 2026-11-11 Boya 有限诊断与六件验收轻量证据

权威报告：[FAST_BOYA_RESULTS_20261111.md](../../FAST_BOYA_RESULTS_20261111.md)。

- `SUMMARY.json`：三个协议分开汇总，原外部1/12不改。
- `planning/`：旧29候选恢复2计划，零物理；两ready逐指令/全源几何审计。
- `candidate/`：9新候选1成功/8失败，18900步；受控腕诊断，不是top1泛化。
- `external6/`：新六件3成功/2实际失败/1无计划，5实际10500步；初始化另7500步。
- `independent/FINAL_AUDIT_SUMMARY.json`：选择、未泄漏、排名、规划、完整物理、媒体与基线不变的独立核验。
- `independent/TESTS.log`：27项合同测试通过，零物理。
- `SOURCE_ARTIFACTS.json`：所有复制件与原始输出的SHA绑定；不含权重、网格、轨迹、图像或视频。

视频与输入总览在本机 `outputs/boya_fast_delivery_20261111_v1/index.html`。新增规划对新6实际准入没有收益，五ready均为baseline。碰撞代理采样表面差最高33.479mm、单摆放、小样本均保留限制；整体论文复现仍未验收。全部数据435和旧权重不变，0训练，无重抓/扩样。
