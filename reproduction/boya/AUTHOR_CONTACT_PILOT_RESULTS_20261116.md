# 新预闭合参考接采集：固定两场 g1

本轮把用户指定的“先按目标宽度确定虚拟闭合构型，再调整掌姿，实际继续闭合”接到完整采集入口。只处理手动队列中预先固定的两场新布局，不启动大批量或训练。本报告末尾和轻证据以最终落盘结果为准。

## 方法与保留项

`author_contact_pipeline.py` 是独立 opt-in 层，没有修改冻结的原采集、碰撞、动作、机械臂、物理和标签源码，CanonDex 只读。原 CGR 的输入仍是 RGB-D，原发布权重，每场一次 identity 前向，不加载 Boya 评分头。原 source domain、top500、采集 flip、随机 extra-depth、碰撞后排序/top10/index0 保留。

新的 Boya 映射沿公司 g1 五指包握的原 16 电机闭合路径，用 raw CGR 宽度计算预闭合参考；1.7 只用于原预张开配置请求。四个对向分支按 ff/mf/rf/lf 顺序选择首个通过原碰撞检查的分支，不用物体 GT 或历史标签调掌姿。碰撞检查复用原张开构型的完整手点云，原 3mm voxel、4cm 接近扫描、允许两点规则不变。分支检查单独保存 `COLLISION_BRANCHES.json`。

新参考贯穿候选碰撞、动作校验、桥接、机械臂准备和命令绑定。通过独立标量 XML FK 复算参考宽度、参考矩阵和真实待执行掌姿；新动作仍执行原 250 步 open-to-company-wrap 及 50 步 hold，预闭合状态不是指令终点。原 24 指令列、4 派生 mimic、250Hz、CPU PhysX、质量/摩擦/每体外力和完整逐点接触保留。并未实现接触停止控制。

只要求拇指和其余手指合计真实承载，不强制五指齐触，也不要求最终由预选的对向手指单独承载。原全三体离桌、抬升、承载及物理门独立复算。该新映射属于 Boya 适配，不能称原作者硬件控制。

## 固定预算与准备验证

public 根：`/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/contact_pilot_20261116_v1`。固定源为未启动的 `manual_start_20261116_v2` 中 index3、4，两场 fit，seed2026111601，每场仅选首个候选准备/执行一次，无候选回退、无执行重试。最多两次 750-tick 初始化、两次各不超过 8000 步执行，共最多 4 Kit / 17500 环境步，GPU0 UUID 固定，不使用 GPU1。

原协议继承的 `total_scene_budget=16` / `total_gesture_attempt_budget=64` 和四型配额描述源手动队列的容量；本适配控制器实际派发范围由 `rows`、`source_queue_indices=[3,4]`、`gestures=[1]` 和上述更严格 max 字段决定，不运行继承队列其余槽位。原手动队列的 batch 仍没有 STAGE。

先复用旧 fit00013 的观测作纯准备检查：新 raw1322/ff 得到 5210 行完整提起指令，raw 宽度 83.867549896mm，独立 XML FK 为 83.867549896mm，待执行掌矩阵误差 8.174e-9。它只是 0 物理的集成检查，不纳入新标签。开发准备 v1 因缺 libusb 搜索路径停止；v2 因准备途中本轮新适配源发生编辑，被 SHA 门拒绝；两项原失败保留，均 0 物理。冻结适配源后的 v3 完整准备通过。13 项映射/合同测试与 3 项中断接续检查通过。

## 总控中断与接续

第一场已经执行并审核通过，第二场 capture 和独立审核已经完成后，原总控退出码 143；具体外部终止原因未证明，不归因于物理算法。第二场 CGR 只有资源预约和 RESOURCE_WAIT，没有 STAGE 或 process.log。原冻结 `B._stage` 在唯一 Popen 前先持久化 STAGE、打开 process.log，故该 CGR 尚未派发。主机逐一核对原总控与全部已登记子 PGID 均已退出，保存不可变 `INTERRUPTION_REVIEW.json` 和资源原状态，才释放该预约。

独立 `continue_contact_pilot.py` 复用已审核初始化与首标签，仅完成剩余第二场，不重跑任何已开始阶段。剩余最多 1 Kit / 8000 步，仍受原两场总预算约束。原 RUN_STARTED、原阶段/轨迹/协议未回写；接续协议独立绑定源码。最终汇总必须保留 `controller_was_interrupted=true`，完整 outer 包含中断与接续间隔，不能当正常吞吐 benchmark。

## 数据、图与限制

`export_contact_pilot.py` 只导出已经完成的两场结果，复核原数值审核证书 SHA、逐条记录、特征及新映射/指令，不再积分或重复完整数值审核。数据字段为原 canonical CGR480、selected depth/extra 的 0–19 输出索引、实际 0/1 标签及全三体物体身份/split/SHA。新数据单独存放，不与旧映射 26fit+5internal 或旧435自动合并，不启动模型训练。

第一场保存图：[`actual_execution.png`](../../outputs/boya_contact_pilot_visual_20261116_v2/actual_execution.png)。它使用实际 native link pose 和三物体保存位姿绘制源网格，不使用计划目标冒充实测。fixed palm 由实际 pmbase 和源 URDF 固定变换恢复；不重跑物理，不是相机照片或精确 cooked 接触面。初版因 palm 是合并 fixed link、未单独出现在 native body_names 中停止，0 物理，失败目录保留；修正版完成 PNG 全解码验证。

此前单候选四分支约 1.06ms 只表示映射数值计算，不包括完整手点云、机械臂准备、Kit 建场、PhysX 或逐点接触存储。没有同场旧映射的配对物理对照，不能据这两条报告映射净收益、训后成功率、总体采集倍率或采满 ETA。本轮仅验收新映射接实际采集，不自动晋级全量。

## 最终实测结果

| 固定场景 | 新首选 | 参考对向 | 实际步数 | 被提起物体 | 末态源底离桌 | 末态参考升高 | 独立审核 | 标签 |
|---|---|---|---:|---|---:|---:|---|---:|
| fit00003 | raw34 / extra10mm | ff | 3817 | 026 | 101.796mm | 100.203mm | PASS | 1 |
| fit00004 | raw1929 / extra10mm | rf | 2681 | 014 | 95.825mm | 94.099mm | PASS | 1 |

两场原物理门全部通过。第一场末态 th/ff/mf/rf/lf 载荷为 5.69679/5.93571/0.02753/0/0 N；第二场为 11.77012/5.15991/4.41343/4.78929/0 N。两次拇指均真实承载，不强制五指齐触。末尾持续满足拾起条件分别1.608s/1.472s，仅作诊断，没有添加持续时间门。第二场实际选择rf，说明没有锁死ff；实际承载仍按拇指+其余合计原规则判断。

本轮4 Kit、1500初始化tick、6498执行步，共7998环境步/23994三体对象步，0物理重试/0未知成本；2 RGB-D/2原identity CGR、0评分头。完整outer1614.294s=26.905min，包含总控中断和恢复间隔；两个执行stage273.880s/464.457s、准备73.432s/91.661s、原数值审核69.281s/98.641s。不能拿本轮outer当正常吞吐，也未证明整条采集加速。

导出 `export/gesture_1.npz`：2fit、0internal、2正0负，features=(2,480)、全三体身份=(2,3)，全部NPZ数组无pickle并逐值重载核对。正式训练仍0。新单型正式配额/完整队列尚未冻结，旧四型4000fit+800internal不自动改写为新单型目标。旧映射31条单独保留，不计入新映射2条。

主机最终核对原/接续全部已登记PGID退出、两套资源槽released、三个锁可取、全局registry released，无本任务后台采集/训练。原手动batch仍零STAGE。最终轻证据为 `evidence/author_contact_pilot_20261116_v1/EVIDENCE.json`，完整raw/轨迹/几何/NPZ留public，不入Git。
