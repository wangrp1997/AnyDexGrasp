# Boya 映射与公开原手闭合对齐（2026-11-08）

本轮仅只读源代码、原冻结三件轨迹和 FK，零物理、零训练、零模型前向。没有把一项不确定假设改成默认，也没有重写已有标签。

**结论：未发现这三件已保存抓姿的 flip、raw/extra 深度、掌帧逆矩阵或 nominal 闭合端点接线错误。先前把“作者有力/触觉自适应闭合而我们没有”概括到全部原手，是不准确的：公开 Allegro 分支本身是预定义位置插值，不含使用中的触觉/力反馈。** 当前 Boya 校准与实机仍非物理等价；无证据把剩余失败归给原 AnyDex 算法。

| 环节 | 作者公开代码 | 当前 Boya | 判断 |
|---|---|---|---|
| 掌位姿 | `DH3_grasp.py:288–322`、`Allegro_grasp.py:313–346` 用 `T_CG @ inv(T_HG)`；标定表来自手的开口状态 | `asset_adapter.py:41–70` 用相同逆标定组合，明确 CGR X→接近、Y→闭合的轴置换 | 三件逐一重建保存掌姿/方向至 1e-12，未发现矩阵错误 |
| 标定 datum | `angles2stl.py:266–320` 从预选网格顶点构造开手参考，含每手型专门指定表面和轴 | `asset_adapter.py:131–175` 用源 distal pads/FK 指定参考；固定 thumb/index 分支 | 同类几何 LUT 思路，Boya 数据未经过实机标定；不能直接复制 Allegro 14mm 偏置 |
| 宽度 | `collision_detector.py:134–144` 乘 1.7 后按每手标定范围限制 | 同样 1.7，再按 Boya 表范围限幅和离散匹配 | 原开口 LUT 中点随闭合漂移，不能仅凭漂移认定程序错误 |
| 深度 | `ur.py:372–377` 沿二指原 +X 加深，Allegro/Inspire 另有14mm，DH3减开度相关 back-distance | raw + extra 沿原 +X；Boya distal datum 独立定义 | 三件重建与元数据一致；14mm 是原手具体标定补偿，不是普适常量 |
| Allegro 闭合 | `Allegro.py:61–84` 读取实际起始 q，固定 close_pose，10次等量位置增量，每次等待0.1s，再等待0.2s | 250次同步等量增量/1s，再保持0.2s；基于保存开手端点 | 总时长/名义路径一致，离散时序和起始 q 读取不同 |
| Allegro 力/触觉 | `ur.py:398` 的 close_torque 传入第三形参 tactile；`close_gripper` 内没有读取该形参，也未调用 set_torque/get_angle_contact | 无触觉反馈，位置驱动受显式仿真力矩上限限制 | 不存在可直接恢复的原 Allegro 触觉控制器 |
| DH3/Inspire 闭合 | DH3 `force=70/speed=100` 后寄存器 position=0；Inspire 分型预定义角度递减 | 16独立位置电机；源仿真上限0.3Nm、THJ3/4明确0.6Nm覆盖 | 没有 force70→Boya Nm 的公开通用换算，不能凭失败调大扭矩称对齐 |
| 退出路径 | `ur.py:401–406` 先沿接近方向退4mm，再退到pregrasp、throw | 当前诊断垂直抬100mm | 明确任务差异，本轮不改，以免同时混入路径变量 |

源函数已用 AST 隔离（没有 import 真实硬件模块）在 recording fake controller 上实际调用，返回的10个位置目标、10×0.1s+0.2s等待与下述新入口一致。`tactile`、`set_torque`、`get_angle_contact` 在闭合函数体未参与控制，经 AST 检查。

三件实际轨迹独立复核结果：

| 固定原候选 | 闭合开始实测 q 与保存开手最大差(rad) | 闭合末态与 nominal 最大差(rad) | 名义开→闭参考中点漂移(mm) |
|---|---:|---:|---:|
| A6 / gesture6 | 0.00157444 | 0.00209396 | 16.084 |
| A3 / gesture8 | 0.000001015 | 0.828815 | 33.880 |
| E0 / gesture7 | 0.000000165 | 0.626655 | 50.742 |

所有2100步独立电机位置目标与原保存 float32 指令逐值一致，掌帧/接近/开闭端点一致。A3/E0 末态误差说明受接触阻碍后未跟到目标；**不能由此证明力矩不足**，因为抓姿、接触面和物体运动尚会同时影响约束。保存轨迹无逐步实际电机扭矩，因此没有直接“电机持续饱和”证据。闭合开始差异极小，读取实测起点差异不支持作为这两件的主要解释。

新独立 opt-in 入口 `run_boya_allegro_step_control.py` 只将 close 250行的电机目标改为十级阶梯：0.1…1，每级25步×4ms；240行变化，10个共同端点一致。所有非close电机目标、desired_palm、phase、pregrasp均保持，原 gantry 执行器没有修改。新job必须声明 `closure_control=published_allegro_ten_position_steps_v1` 并用新输出目录；CLI `--job <newjob.json>`。仍用 Boya nominal 开闭端点，起点仍冻结开手目标，**不是完全相同的 Allegro 硬件控制器，也没有物理救回结论**。

4项静态测试通过：调用原方法目标/等待复核、仅close修改、原方法没有力/触觉控制、十级端点边界。复核脚本 `audit_boya_mapping_closure.py`；轻量证据 `evidence/mapping_closure_alignment_20261108/SUMMARY.json`。该实验控制是否值得执行由独立协议决定；本审计没有试抓。
