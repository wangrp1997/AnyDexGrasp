# AnyDexGrasp 论文—官方代码核对

官方源码本地根目录：`/home/wangrenpeng/AnyDexGrasp/`。本报告所有官方相对源码路径均以此目录为基准。静态探针、论文源/源码SHA等原始证据在 `outputs/paper_code_audit/`；CanonDex原始工作目录保留。


核对时间：2026-11-03。只读官方 commit `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`，论文为 arXiv `2502.16420v1`，PDF 和原始 TeX 已下载。本报告未改官方源码、未安装依赖、未执行机器人/物理仿真、未训练或验证发布权重。静态类接口探针有实际 CPU 前向，详见独立仓库 `outputs/paper_code_audit/interface_probe.json`，不能当任务效果复现。

官方来源：[论文](https://arxiv.org/abs/2502.16420v1)、[仓库固定版本](https://github.com/graspnet/AnyDexGrasp/tree/c9c4a43df33e40860417c7e2dd02f5122d3b2da2)。下文源码行号均以该版本为准。原始论文 TeX 解包在独立仓库 `outputs/paper_code_audit/source/`，便于精确对照。

## 1. 算法究竟是什么

这是两阶段监督学习的抓姿检测与成功评分器，既不是 RL 策略，也不是在每次抓取时求解整手连续优化。

1. **通用局部几何表示**：单视角点云输入 MinkowskiEngine 稀疏卷积网络；预测点抓取性和 300 个接近方向的抓取性，选高分点/方向，沿方向做圆柱分组，再预测局部接触几何。论文的 CGR 为 `r={(d_(a,j),theta_(a,j))}`，48 个圆周角 × 5 个深入截面 × 2 个量 = 480 维。截面深入为 `[5,10,20,30,40] mm`。40 个训练物体在 GraspNet 100 场景/25600 RGB-D 图上重标注训练；不是仅靠 40 张物体照片。
2. **确定性的手势/开口标定映射**：每只手预定义有限手势，手势不等于开口固定；给每种手势人工指定主闭合轴，并预计算每个开口的手腕—夹爪相对变换和关节姿态。由 CGR 解码的二指中心/方向/宽度映射到手的预抓姿。
3. **手相关评分学习**：固定表示网络，用新手实际试抓成功/失败，分别训练各手势的轻量 MLP，给位置/手势/附加深入组合评分；高分候选再做手模型—单视点云碰撞，交给 UR5 接近、闭合和搬运。

论文主干映射：`P -> Phi(P)={r_i} -> K(r_i,h)={g_iq} -> Psi_hq(r_i)`。架构上与我们“二指/局部候选—手势适配—碰撞—执行”明显接近；关键区别是作者用**候选周围的完整局部截面接触表示 + 实际成败监督**学习手势/抓姿质量，而我们当前仍主要靠几何代理和调度。这种接近不等于我们的效果等同，也不证明二指映射这一路线本身失败。

## 2. 论文段落与源码定位

| 内容 | 论文 TeX | 官方源码 |
|---|---|---|
| 两阶段问题定义、CGR 极坐标/截面规范化 | `sections/results.tex:21–76` | `models/minkowski_graspnet_single_point.py:165–249` |
| 手势只是预接触 anchor，接触后继续闭合 | `results.tex:96–100` | `ur_toolbox/ur_toolbox/robot/ur.py:391–407`；`robot/Allegro/Allegro.py:61–84` |
| 基线为 AnyGrasp 二指抓姿 + 人工主闭合轴对齐 + 手模型碰撞 | `methods.tex:5–8` | 三个 `robot_*.py` 的随机分支属于相关启发式路线；不能把任一默认脚本直接等同论文精确基线 |
| 点/视向抓取性、48×5 输出 | `methods.tex:12–24` | `single_point.py:54–134,136–163,189–247` |
| CGR 手映射 | `methods.tex:27–30`；`supp.tex:3–30` | `robot_inspire.py:106–139`；`robot/Inspire/InspireHandR_grasp.py:307–351,579–644` |
| 每手势单独评分网络 | `methods.tex:36–46`；`supp.tex:32–39` | `models/minkowski_graspnet.py:8–62`；`single_point.py:321–356`；`train_multifinger.py:52–120` |
| 手势/深度评分查询 | 同上 | `robot_inspire.py:435–481`（每候选固定二指深入，再查询 4 个附加深入/每手势） |
| GraspNet 重标注和桌面/邻物圆柱准入 | `methods.tex:90–101` | `dataset/graspnet.py:98–114,358–408,426–438` 加载发布的 `fric_rep/fric_collision_label/pre_generated_label_combinescore` |
| 表示网络实际损失 | 论文未给此完整展开 | `models/loss.py:272–346`：score/距离 Huber，加 objectness 和 graspness 项 |
| 真实试抓监督、最终搬运标签 | `methods.tex:99–111` | `dataset/multifinger_hand.py:121–163`；`robot_inspire.py:233–320,785–802` |
| 碰撞和接近检查 | `methods.tex:48–50,111` | `utils/collision_detector.py:35–63,101–194`；`robot_inspire.py:706–712` |

## 3. 不能忽略的论文/公开代码差异

### A. 决策输入不是字面上的米/弧度 480 向量

`single_point.py:161,243` 保存原始输出 `grasp_preds`，其前 240 通道为接触分数 logits，后 240 通道为距离 logits，随后才分别 sigmoid（246–247）。`robot_inspire.py:171–174,185–189` 抽取前 480 通道送决策，不是把已解码的物理距离/法向角直接输入。训练集也读取 `grasp_preds_features[:480]`（`multifinger_hand.py:141–161`），并按手侧翻转与二指平面角做半圈交换/循环移位。因此自己用网格直接算 `(d,theta)` 再塞给发布决策权重，不等价，输入分布会变。

表示标签 `fric_rep` 的 `data[...,0]` 是距离通道，`data[...,1]` 是摩擦/法向相容性标量（`graspnet.py:366–369`）。损失把有效 `mu` 转成 `log(1/mu)/log(10)`，距离按 `0.08 m` 归一化（`loss.py:15–18,289–318`）。应据发布标注数值核对确切物理含义，不能只凭变量名 `width` 判断。

### B. 原样训练脚本与部署网络输出头不兼容：已实际验证

`train_multifinger.py:52–55` 固定 `type=1, depth=4, two_finger_angle=12, two_finger_depth=5`，训练类 `minkowski_graspnet.py:21` 输出 **240**。部署类 `single_point.py:322–329` 默认 `angle=1`，输出 **20**；机器人 `robot_inspire.py:427–429` 直接严格加载。CPU 实际前向分别 `[4,1,240]` 和 `[4,1,20]`，直接加载报 `conv7/bn7` shape mismatch。见独立仓库 `outputs/paper_code_audit/interface_probe.json`。

这只证明当前公开训练/部署接口有不一致，不证明发布权重错误。必须先查看实际 checkpoint 的头尺寸/来源，再决定忠实恢复哪个配置；不能静默切片权重或随意换头后称原样复现。训练 loss 本身还只选择 `depth2f*4+depth_hand` 那 20 个头中的目标（`loss.py:406–415`），其余角度块没有该样本的直接监督。

### C. 决策损失/层宽/训练参数不同

- 论文 `supp.tex:35–39` 描述中间层 1024、交叉熵；公开 MLP 实际是 `480→512→1024→512→1024→512→512→output`，末层 BN+ReLU，非限制在 `[0,1]` 的 sigmoid（`minkowski_graspnet.py:23–35,52–58`）。
- 默认 `add_collision=False` 时实际损失 **MSE**（`loss.py:415`），虽声明 BCE 实例却没使用（370）；不能把该实现称交叉熵概率分类器。回归二元成败仍可产生质量排序，但没有概率校准保证。
- 论文决策训练 20 epoch、lr1e-4、分段下降（`methods.tex:107`）；代码默认 100 epoch、lr5e-4、weight_decay5e-4、PolyLR power0.99（`train_multifinger.py:37–40,120–126`）。
- 数据集硬截每手势训练前 100 个有效条目（`multifinger_hand.py:70–78`），这对应少样本设置，但不能拿它重现每手势1000次的完整数据设置。

### D. 部署默认与论文最高分选择不同

- **Inspire `RANDOM_GRASP=True`**（`robot_inspire.py:60`），`435–442` 直接随机手势/附加深入，跳过学到的决策分支。DH3/Allegro `RANDOM=False`（各 `:60`）。必须显式记录/选择 learned 分支，不能把默认 Inspire 脚本运行结果称学习版 AnyDex。
- 论文方法段 top500 CGR（`methods.tex:30`）、实验段 top100 CGR→top200碰撞（111），同文已有数量差异；代码 Inspire先top1000（636–637）、决策后top3000（470）。
- 论文通常阈值0.9且最终取最高（`methods.tex:50,111`）；代码 Inspire/DH3阈值0.85，Allegro0.7；Inspire `744–750` 和 Allegro `783–789` 从碰撞后 top10 **随机取一个**，DH3 `717` 才选最高。
- `POINTCLOUD_AUGMENT_NUM=10` 再加原始点云共11次表示网络前向（Inspire `511–539`），须计真实计算量/随机种子；不能只报一次前向。
- 代码单视相机工作区和近竖直接近过滤明显：Inspire `390–392` 限 `R[2,0]>0.92`、XY固定范围；另外 `flip_ggarray` 后直接丢一侧（632–633）。这不等于证明手能从任意方向抓任何高物体；Boya侧抓测试若放开过滤，属于明示适配条件。

### E. 二指宽度映射和碰撞也有主动几何修正

论文给二指宽度 `2 max(d_a,d_(a+pi))`（`supp.tex:13`），代码先用两侧半径之和（`robot_inspire.py:135`），碰撞检测时又依据可见内点重新居中，并扩开口（`collision_detector.py:35–63,130–142`）。其中 `_adjust_gripper_centers` 已乘1.7，外层 `134` 再乘1.7；这可能是作者保守开手启发式，必须保留/记录，不能概括成“预测后原样放手”。训练距离尺度0.08而部署按每手 `MAX_GRASP_WIDTH` 0.099/0.10/0.11还原，也须跟实际 checkpoint/标注确认，不能擅改后报忠实复现。

## 4. 数据/权重需求与是否要重新训练

| 目标 | 需要 | 能否直接用于 Boya |
|---|---|---|
| 发布通用表示网络离线推理 | representation checkpoint、MinkowskiEngine/knn/pointnet2编译依赖、单视点云及正确单位/相机坐标 | 表示本来是 hand-agnostic，有复用依据；尚未在本机验证效果 |
| 发布三只手的决策推理 | 对应手/手势20头checkpoint、匹配原表示输出与canonicalization、开口标定JSON/手网格点云 | 不能把Allegro/Inspire权重直接当Boya已验证模型 |
| 从头重训表示模型 | GraspNet原RGB-D/scene/meta+作者新 `fric_rep`、`fric_collision_label`、`pre_generated_label_combinescore` | 通常先复用发布checkpoint；原始GraspNet标签不等于CGR重标注 |
| 重训决策模型 | `obj140.json` / `test_single_point.json`等作者真实试抓条目，含480原始特征、二指角/深入、手势、附加深入、成败 | 新Boya需重新定义手势/宽度映射并采该手成败；可用我们仿真采集，但这叫仿真适配版，不是原真实试验复现 |

官方仓库仅 `process_pregen_label.py` 聚合既有标注，仍硬编码路径与71..129场景；本次全仓检索未找到从网格完整生成论文全部 `fric_rep` 和圆柱碰撞重标注的入口。下载作者发布标签优先，不能认为原GraspNet下载完就够重训。论文144物体CAD也有“available upon request”（`supp.tex:78`），不能先假设所有扫描模型公开已下载。

## 5. 原论文指标的精确口径

- 通用表示训练：40物体，100场景，各256RGB-D图（`methods.tex:93–105`）。决策完整设置：144物体，DH3/Inspire约1000条/手势，Allegro约200条/手势（`results.tex:124`）；小样本设置另算。
- 完整设置日常物体 **DH3 97%、Allegro 78%、Inspire 83%**；启发式baseline **66%、51%、58%**（135）。对抗物体 **99%、82%、79%**，baseline **72%、54%、59%**（145）。不要把三指97–99%说成五指也有同样比例。
- 40训练物体、100试抓/手势：DH3共400条→**94.5%**（167），Allegro共1000条→**75%**、Inspire共800条→**77%**（172–174）。不是“总100条就训练完全部手势”。
- 约150测试物体，随机杂乱摆放清桌，流程重复两次；成功率定义为**成功尝试次数 / 总尝试次数**（`results.tex:115–117`），成功为搬到目标处人工记录（`methods.tex:111`）。这与我们“每个物体最多8次后最终成功 / 物体数”分母不同。
- 论文测试尺寸叙述为约 `2.5×2.5×2.5 cm` 至 `8×8×5 cm`（115），不能直接据此证明13cm宽包装或更大Boya目标同样已解决。
- 报告每场景生成200抓姿平均0.5s，另CPU Open3D碰撞约20s（139），不包含完整机器人搬运；不能把0.5s说端到端抓起。
- 作者明确视觉输入，无触觉反馈（176）。底层闭合使用既定力/力矩上限，不能将其等同我们接触法向反馈后的闭环重抓。

## 6. 可复现顺序建议

1. 保存官方源码/论文/checkpoint/标签SHA，先核对发布checkpoint的480输入、20/240输出、表示网络5-depth/48-angle/half_views配置；先跑原手离线推理/held-out标签，不急于重训或实机。
2. 各自保留“官方代码配置”与“论文协议配置”清单：随机开关、topK、最终top1/top10随机、输入canonicalization、碰撞开口扩张、训练epoch/loss。修代码必须是独立透明适配层，原始版本只读。
3. 先复现发布决策模型在相同输入上的前向/排序，再在同一仿真手/场景或Boya显式适配版执行。Boya版本需要静态开口映射和该手训练数据，不能称跨手零样本抓取已验证。
4. 比较我们与AnyDex时，至少同时报首个实际抓取、最终有限重抓、准入/失败/总耗时；相同partial point cloud或明确分列已知全网格oracle输入，不将作者真实成功率和我们的仿真最终物体成功率直接相减。

当前结论：它是必须认真复现的强基线；结构上确与我们相近，成功评分学习和真实验证规模是已明确差距。公开代码具备主要网络、数据管线和机器人接口，但存在多个论文/脚本配置与训练/部署接口差异，不能在未拿到发布权重并完成前向验证前宣称一键完整复现。
