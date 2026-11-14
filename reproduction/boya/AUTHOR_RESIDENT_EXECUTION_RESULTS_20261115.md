# 常驻执行与动作一致性独立审计（2026-11-15）

固定三场、十二手型槽已结束：实际九次执行的原独立数值审计全部通过，三个执行进程合计 28,898 命令物理步，诊断标签为七正二负；另两槽 `null_geometry`、一槽 `no_candidate`，没有补负例。九条与旧对应执行的步数、标签和物理命令数组一致。**六次热复用的实际状态轨迹存在差异，不能由标签一致推断轨迹等价或采集统计无偏。** 全部记录为已知场景诊断、排除训练，未启动正式 4000 fit＋800 internal 扩采或正式训练，整体算法复现仍未验收。

## 实现与冻结范围

新增 opt-in `author_resident_execution.py`，沿已有 native `ResidentArmScene`、`execution_steps(resident_scene=...)` 和原 lift 执行入口接线；没有修改原冻结执行器、原标签函数或 CanonDex。`ResidentExecutionSession.execute(prepared_dir)` 保留原 once-only marker、完整命令、三体接触报告和 lift 完成流程；组级 `execute-many` 由原资源槽调度，父进程不启动 Kit。每个 job 的原 close 延后，组收据先落盘，再关闭唯一的 SimulationApp。

三体几何相同时复用真实 handles，并在 job 边界恢复机器人和全部三体的位姿、速度，重新施加原向下力及原材料检查。改变几何时清理 context／stage 的官方 API 分支仅经过 CPU 测试；本轮真实 Kit 证据仅覆盖**每组固定三体、依次换四种手型**，未验证同一 App 跨不同三体几何的重建。

执行源码 SHA 为 `346aa86a4632f88c4c8c3dc046e60e2a2c4954482ef13bec01a928401c3c4dac`，测试源码 SHA 为 `5d52aeb48e4985d8c32da5a941deb846644dc0f8ff66f3f1fc8ad98de0979719`。8 项 CPU 测试覆盖三体 reset、几何身份／别名绑定、同 App 两 job、官方清场、失败停止与 close 前收据、重复开始保护、外传依赖锁先于同名覆盖校验；0 Kit、0 积分。实际运行期间未再修改这些源码。

## 实际执行与边界重置

根目录为 `/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/resident_collection_control_20261115_v1/`。fit 从原队列 scene1、scene4 重新 build/capture/CGR/prepare，internal 从原 fresh-wave child 的 local0 重新运行；本审计读取保存产物，不重新派发任何阶段。

| 场景 | g1 | g6 | g7 | g8 | 命令物理步 | Kit／建场／边界 reset |
|---|---|---|---|---|---:|---|
| fit scene1 | 正／3139 | 正／3008 | 正／3021 | 正／3312 | 12,480 | 1／1／3 |
| fit scene4 | 负／3374 | 正／3024 | 正／3196 | 负／3582 | 13,176 | 1／1／3 |
| internal local0 | null_geometry | null_geometry | 正／3242 | no_candidate | 3,242 | 1／1／0 |

fit 两组各四个 job 的 `EXECUTION_STARTED.pid` 与各自 session PID 相同；首 job 原 native 报告 `reused=false`，其后三次为 true，native 导入审计计数逐 job 递增。internal 只有 g7 ready，该 job 是冷启动，不能算作新的热 reset 样本。九份原 `RUNTIME_AUDIT.json` 均为 PASS；本审计绑定其 SHA 和接触 journal 指针，未第二次解压全量接触流。

每个 job 的三体实际读回初态 pose/velocity、机器人初始 q/qdot 与旧对应记录逐值相同。三体相对冻结观测的最大位置分量误差分别为 scene1 3.73 nm、scene4 7.45 nm；速度读回全零。原 primary 和另两体的质量／摩擦／永久向下力保持 0.1 kg／0.6／0.98 N，额外两体 restore 记为 0 积分，并通过原读回门。九次均通过原 native 20 µm 和全体材料／初态检查；native 20 µm 只证明导入保持，不证明凸代理等于精确 CAD。

两场 fit 首命令前真实物理时钟分别为 `[0,3139,6147,9168]`、`[0,3374,6398,9594]`。相邻时钟差减去上一 job 的实际步数全部为 0，未发现边界附加积分。各 job 执行内状态写入数为 0，所有命令完整执行，无未知步数、无重抓或追加路由。三次 capture 的 2250 初始化 tick／6750 对象步由 capture 单独计数，不混入上述执行步数。

独立 host 样本实见两个执行 PID 同时加载 libcarb／PhysX、GPU0 上对应两个进程，并继承同一个 GPU0 lease FD 和不同 kit 槽 FD；不是仅靠调度 stage 重叠推断两 Kit。最终 host 检查确认三执行 PID 均不存在，GPU0 lease 和本轮三个 kit 槽在 `/proc/locks` 均无持有者。该检查只读，没有发信号或获取锁；完整吞吐与资源时序另见总报告。

## 命令与物理结果分别核对

九个新条件的 `initial_position/joint_names/joint_target/phase/desired_palm/time_s` 与旧对应计划逐数组相同。本审计另从实际使用的 URDF mimic 关系独立展开 24 独立列（6 臂＋2 腕＋16 手）为实际 28 关节，逐值核对保存的全部 position target；原六臂速度前馈、手部零速度目标、phase、time 和 desired palm 全部一致。原 native 七门及额外全三体／phase／运行状态门均通过，原接触与源底标签没有改写。

三个冷启动 job（fit 两个 g1、internal g7）的完整机器人与三体保存状态和旧记录相同。六个热复用 job 的机器人 settle／HOME／approach 关节状态与旧记录相同，关节首个超过 1e-7 rad 的差异均在 close；**三体状态更早出现微差**，close 前最大平移范数差为 43.738 µm。下表对同时间索引比较全三体轨迹：平移为 XYZ 差的欧氏范数，旋转为符号不敏感的相对旋转角；不把四元数分量差当长度。

| 热复用条件 | 首个关节差异步／phase | 最大关节差（rad） | 最大三体平移差（mm） | 最大三体旋转差（度） | 新／旧标签 |
|---|---|---:|---:|---:|---|
| scene1 g6 | 2221／close | 0.003364 | 0.807 | 1.092 | 正／正 |
| scene1 g7 | 2249／close | 0.009177 | 0.441 | 0.811 | 正／正 |
| scene1 g8 | 2552／close | 0.049351 | 1.210 | 2.626 | 正／正 |
| scene4 g6 | 2321／close | 0.011218 | 4.372 | 37.343 | 正／正 |
| scene4 g7 | 2475／close | 0.016138 | 0.748 | 6.936 | 正／正 |
| scene4 g8 | 2854／close | 0.140629 | 149.905 | 179.916 | 负／负 |

scene4 的最大差异主要发生在实际受抓的 `graspnet_015`；兼容 primary `graspnet_063` 轨迹保持相同，只审 primary 会漏掉该问题。scene4 g8 两次都在末态失载、源底贴近桌面，故均为原负标签，但其掉落轨迹明显不同。scene4 g6 虽同为正标签，也有 4.372 mm／37.343° 的状态差异。这里仅证明这九条的门、步数和标签保持，**没有证明热启动原因、没有证明缓存导致差异，也没有证明生产采样无偏**；没有为消除差异调整物理、容差或追加对照。

## 证据与复算

轻证据见 [`evidence/author_resident_execution_20261115_v1/EVIDENCE.json`](evidence/author_resident_execution_20261115_v1/EVIDENCE.json)，包括 fit／internal 独立审计、三体状态按 phase 比较、host 同时 loaded／最终释放样本、CPU 测试回执及只读复算脚本；原数值审计、源输入和大 journal 保留在 public 目录，用路径及 SHA 绑定。`CONTROL_SUMMARY.json` 核十二槽与旧准备状态一致，并核父账本全部 `label=null`、`formal_training_eligible=false`、`exclude_training=true`，null 槽没有执行 marker 或 collection record。

本审计新增物理步、模型载入、规划、训练、标签均为 0；冻结运行源码及旧回执未改。采集吞吐、历史比较和 ETA 的条件范围由 [总报告](AUTHOR_RESIDENT_THROUGHPUT_RESULTS_20261115.md) 单列，不能由本组已知场景的七正二负计算外部成功率或随机场景产率。
