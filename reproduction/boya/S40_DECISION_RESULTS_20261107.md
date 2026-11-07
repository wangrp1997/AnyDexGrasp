# S40四评分头已训练，原始候选部署未通过内部检查

**后续更正：本报告原始候选检查遗漏作者head前的接近方向/宽度/工作区mask，因此3200几何全拒不能归因为训练不足，也不代表完整AnyDex部署。旧数值保留；详见[mask核对](S40_PREHEAD_MASK_REVIEW_20261107.md)。**

2026-11-07。完整运行 `outputs/s40_decision_20261107_v1/`；[轻量结果](evidence/s40_decision_20261107/decision_summary.json)。这是原S40派生仿真数据上的阶段适配，整体论文复现仍未通过验收。

## 实际结果

原全40采集已收尾：640槽=240合格标签（165正/75负）+50认证早停null+350准备null；290实际、509766物理步，包括继承pilot的5次10500步。未知/未执行均0，原每型100、总400有效标签目标欠160，不能补null为0。实际有标签31/40件、77种不同canonical480。28/40至少一次成功是采集覆盖，不是部署或未见物体成功率。[完成诊断](S40_COLLECTION_COMPLETION_20261107.md)。

新专用导出和训练入口保留预冻32-fit/8-internal（原source visual mesh SHA），全部40仍属外层train，EGAD未使用。196条fit、44条内部检查；只有25/32和6/8件提供有效记录。先训内部四头，再将全部240条refit四头。最终全40模型含内部44条，不能用来宣称对这44条的留出成绩。

复用旧适配器的 `official_components` 和 `train_head`，原Inference、选定单输出MSE、PolyLR直接抽取官方定义；未改原算法源码。固定20epoch、batch128、lr1e-4、Adam WD5e-4、Poly power.99；这是论文epoch/LR/batch加发布代码WD/scheduler的明确混合配方，非发布脚本默认100epoch/5e-4。每头不足一batch，内部4+全量4共160次optimizer更新，trainer内部8.529s，不含导出/启动；8个state_dict全部strict reload。CGR未重训。

| 手型 | 全量有效标签（正/负） | 内部检查条数 | 内部MSE | 仅fit正例比例的常数MSE | 全量零监督输出数 |
| --- | --- | --- | --- | --- | --- |
| 1 | 58（44/14） | 10 | 0.529382 | 0.276736 | 6 |
| 6 | 61（42/19） | 12 | 0.438981 | 0.260204 | 6 |
| 7 | 63（47/16） | 12 | 0.571205 | 0.187524 | 6 |
| 8 | 58（32/26） | 10 | 0.218808 | 0.302517 | 5 |

44条加权网络MSE **0.445549**，常数参考 **0.253756**。这仅是已准入、完整执行记录上的误差；不等于完整8件的抓取成功率。未按内部结果选epoch、阈值或手型。

## 真正先评分、再几何的检查

`score_s40_raw_candidates.py`仅载入32-fit四头，复用缓存的8帧原CGR；不读取原ready队列或试抓标签，排名冻结后才进入几何。冻结发布DH3配方：CGR top1000→四手型/四额外深入全局top3500→严格float32分数>.85→top400几何、原单flip。没有降阈值、mask通道、改回filter-first或用全40模型。

8件共32000次head query、128000种展开组合、3200次几何检查，**几何准入0**。3199个table拒绝，1个visible scene point in hand hull拒绝；零臂调用、零新物理/零新CGR前向。诊断内部29.616s，不含进程启动。

3200个入选候选中1423个对应的手型×目标输出在32-fit中无直接监督；原BN+ReLU输出不是概率，本轮原始候选最大分数2.178–2.375。观察到监督稀疏和准备条件分布与原始部署分布不同，尚未通过消融隔离各自因果贡献。不能因评分头训练程序跑通就称算法可部署，也不能把这8件零准入说成8次实际抓取失败。

## 下一步

1. 所有物体统一先在真实桌面上稳定落位，再以最终实测姿态渲染、CGR和规划；不能延长等待却沿用旧观测。原50次早停全部发生在真正接近前，手与物体保守间隙至少177.762mm，是初始化/观测失效问题，保持原null与5752步。
2. 按预声明的抓姿、摆放与raw-depth×extra覆盖补采；优先补缺失监督和真实负例分布，不能仅重复成功候选凑100。原640槽、全部失败/早停与成本保持，新协议另记。
3. 重训后先复查原始候选排序与几何准入，再做冻结top1实际测试；通过后才启用EGAD外部测试。目前无新的物理成功率或泛化改善结论。

本轮没有自动补采/新物理/EGAD评测。旧DGN4/10对6/10只保留旧有限池结论，不与本轮合并。CanonDex只读。

## 入口

- `prepare_s40_training_data.py --collection-run outputs/s40_collection_20261107_v1 --output-dir <fresh>/dataset`
- `train_s40_decision.py --dataset-manifest <fresh>/dataset/manifest.json --output-dir <fresh>/training --device cuda:0`
- `score_s40_raw_candidates.py --collection-run outputs/s40_collection_20261107_v1 --training-results <fresh>/training/TRAINING_RESULTS.json --output-dir <fresh>/raw_internal_diagnostic --calibration outputs/boya_adapter_preparation_20261105_v2/calibration.json`
- `summarize_s40_decision.py --run-dir <fresh>`

使用现有Boya venv及本地依赖路径，不新建环境。导出/训练均拒绝静默覆盖；旧冻结采集与原算法源码保持。14合同测试通过，实际8帧完整排序/几何集成运行通过（准入结果为0）。源/输入SHA、记录、checkpoint、完整JSON和日志留outputs，大数据/权重/视频不提交。
