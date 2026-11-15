# 完整采集常驻入口：实现与待实测状态

截至 2026-11-16，已交付独立入口和 17 项通过的 CPU 检查，**尚未启动真实对照，不能宣称加速**。本阶段新增 Kit、物理步、采集标签和训练更新均为 0；有效新数据仍为 26 fit + 5 internal = 31。正式 4000 fit + 800 internal 队列未启动。

实现文件：`author_collection_service.py`、`author_collection_service_client.py`、`author_stream_representation.py`、`author_collection_stream_control.py`。全部为 opt-in；历史冻结源码、失败、证据和 CanonDex 不变。

## 已实现的具体变化

- 一个拥有线程中的 Kit 连续接收初始化观测与抓取执行任务。原 750 tick 初始化、250 Hz、完整三体接触日志、原指令和数值判定保留。执行使用已有常驻控制器及已验证的无损接触存储。
- CGR 使用原逐帧处理循环和原 `PersistentOfficialCGR`，同一模式下模型只加载一次，每帧仍按原规则恢复 RNG；不加载手型评分头，不训练。
- 第二场观测可与第一场初始化审核/CGR/规划重叠，后续执行可与前一条数值审核重叠。资源槽覆盖真实工作进程的整个寿命，退出后才能释放。
- 每次观测仍重建 SimulationContext/场景。这里只复用 Kit 和 Python 生命周期；执行之间由已有控制器处理几何重建。**尚未实现跨观测的 USD/原生碰撞资产缓存，也未证明物理轨迹等价。**

## 已冻结、尚未运行的对照

工作区协议：`outputs/boya_stream_collection_control_20261116_v1/CONTROL.json`。计划的公共目录版本尚未创建；两次公共目录 freeze 命令均在执行前因自动权限审核超时而未派发。

| 项目 | 固定范围 |
| --- | --- |
| 场景 | 尾批既有 fit_00009、fit_00013；从原 scene spec 重新构建和观测 |
| 手型 | 每场仅原 g1 首选一次 |
| 顺序 | cold，再 resident；每模式 capture × 2、execute × 2 |
| Kit 上限 | cold 4，resident 1，合计 5 |
| 环境步上限 | 每模式 17500，共 35000；执行计划每条最多 8000，原初始化每场 750 |
| 失败策略 | 0 重试、0 换候选、0 补例；错误停止并收尾 |
| 数据用途 | 全部 exclude_training，不导出新训练集 |
| 硬件 | 固定原 GPU0 UUID；物理仍 CPU；GPU1 不使用 |

两模式均包含相同流水安排和常驻 CGR，因此其比值主要考察跨阶段 Kit 复用，不能归为全部流水安排相对旧入口的净收益。必须核 RGB-D、原 CGR 输出/候选和全部科学指令数组，并通过原独立初始化与抓取数值审核，才报告完整 wall 比值。路径相关的两个指令绑定 SHA 由各自原审核验证，不要求跨目录字符串相同。

仅一组固定、已知场景，cold 先行，存在顺序与缓存影响；即使通过也不是随机产率、训练后成功率或采满 ETA。普通常驻/流水安排有现成先例，不构成论文创新证明。

## 离线验证与阻塞

`test_author_collection_service` 7 项新增检查，加已有 verified execution 2 项与 resident execution 8 项，共 17 项通过。覆盖原观测函数字节码/常量的 launcher 之外保持、单 App 生命周期、失败封锁、原 CGR 循环与单模型、固定预算，以及真实无 Kit 子进程持槽/并行审核/退出释放。mock 和轻量子进程均不是物理性能实测。

系统两次返回：`The automatic permission approval review did not finish before its deadline.` 没有据此认定动作不安全。普通沙箱只读 `nvidia-smi` 返回无法连接 NVIDIA 驱动，因此不能在普通沙箱中运行真实 GPU0/RTX 对照。已提交明确的固定预算权限请求，等待回复；批准前不启动依赖主机权限的动作。

验证命令：

```bash
cd reproduction/boya
/home/wangrenpeng/tendon-spin/.venv/bin/python -m unittest \
  test_author_collection_service test_author_verified_execution test_author_resident_execution -v
```

轻量证据见 `evidence/author_collection_service_20261116_v1/`。后续先完成同预算真实对照，确认收益和原审核通过后再决定是否将入口用于新的有限采集批次。
