# AnyDexGrasp 公开 fork 与跨手适配参考

2026-11-14。接续 [当前复现完整性审查](FIDELITY_REVIEW_20261114.md)。本轮读取 GitHub 公共 API、固定提交的源码和本项目适配代码；没有执行第三方程序、模型前向、训练或物理试验。

**可以参考，而且找到了一份真正替换自研手几何的实现，以及仿真采数、完整增强推理和机械臂接口的参考。它们提供的是可检查的工程方法；本轮未找到已公开验证、可以直接移植到 Boya 的完整新手模型与成功率基线。**

## 检索范围

上游固定为 `graspnet/AnyDexGrasp@c9c4a43df33e40860417c7e2dd02f5122d3b2da2`。

- [GitHub forks API](https://api.github.com/repos/graspnet/AnyDexGrasp/forks?per_page=100&sort=newest) 返回 23 个 fork；排除我们自己的 `wangrp1997`，检查其余 **22 个仓库的全部 25 个公开分支**。
- 按提交比较：12 个仓库所有分支与基线相同，2 个仅落后，8 个存在新增提交。不能只看默认分支：`kywch/pixi`、`SevenFo/dev` 的主要改动均不在 `main`。
- 另用仓库名搜索找到 3 个未标记为 fork 的相关移植仓库，纳入检查；其中 `Jonas-Z-up/AnyDexGrasp_ws` 最贴近“另一种手怎样适配”。
- 对重点实现取得完整递归目录树，下载 36 份源码/文档，共 415,409 B，静态阅读关键模块。`kywch` 的 compare API 返回文件数达到 300 上限，不能据此宣称只有 300 个改动；另查其完整 1,852 项目录树，未截断。

完整分支 SHA、比较链接、来源文件 SHA256 见 [来源清单](evidence/fork_adaptation_survey_20261114/SOURCES.json)。这是公开可见范围的检索，不是所有私有移植或所有 GitHub 仓库的穷举，也不是所有文件的逐行正确性审计。

## 本地参考仓库已检出

用户随后要求拉到本地参考，已将三个重点仓库检出至 `outputs/reference_repos_20261114/`，固定为下表版本。共检出 887 项源码、文档与配置，8,113,345 B；三个 Git 工作区均干净，先前阅读的 26 份关键文件 SHA256 全部匹配。采用浅克隆和稀疏检出，不下载大模型/网格/录像，不初始化子模块，不运行第三方程序。

本地入口：[源码索引](../outputs/reference_repos_20261114/README.md)、[自研手抓取映射](../outputs/reference_repos_20261114/jonas_own_hand/own_hand/own_hand_grasp.py)、[仿真执行适配](../outputs/reference_repos_20261114/kywch_pixi/robosuite_env.py)、[11 分支候选链](../outputs/reference_repos_20261114/kitt_server/adg_server/planner.py)。轻量检出凭据见 [LOCAL_CHECKOUTS.json](evidence/fork_adaptation_survey_20261114/LOCAL_CHECKOUTS.json)，完整逐文件清单保留在忽略目录。

首次网络出口下一个仓库完成、两个超时；切换既有备用出口及 HTTP/1.1 后全部到齐。首次逐文件核验将两个外部 catkin 符号链接按普通文件读取而失败，随后按链接文本核验通过；没有更改参考仓库文件。两份失败回执保留于参考根目录。

## 最值得参考的实现

| 实现与固定版本 | 实际适配内容 | 对 Boya 的价值 | 公开证据的边界 |
|---|---|---|---|
| [Jonas-Z-up/AnyDexGrasp_ws](https://github.com/Jonas-Z-up/AnyDexGrasp_ws/tree/444199bb10bb715578f014068ab18dbff49a890b)，`main@444199b` | Unitree Z1、D435i、自研五指欠驱手；替换手部几何、接触组、抓取锚点，定义预塑形/逐指闭合动作 | **新手几何和坐标语义的首要参考** | 仍迁移 Inspire 评分；最新版管线文档明确未验证实机成功率 |
| [kywch/AnyDexGrasp 的 pixi 分支](https://github.com/kywch/AnyDexGrasp/tree/3888737ef280fb4707601501c593a5a5a2f0bb20)，`3888737` | Panda＋Inspire 的 MuJoCo/robosuite 移植，采数、训练、评估 | **仿真数据闭环的首要参考** | 原 Inspire 手的仿真移植；关节转换、特征规范化和评估协议均有改动 |
| [kittvnebluda/anydexgrasp](https://github.com/kittvnebluda/anydexgrasp/tree/0b59bfb02bdf59c1a21cae2522edc71bc4ca8f2b)，`0b59bfb` | 原库作为 submodule，外部服务封装 Inspire 推理、映射、碰撞 | **完整 11 分支候选接口与部署拆分参考** | 仍是 Inspire；额外方向/目标区域筛选和碰撞实现有差异，不能视为逐值等价 |
| [RDChen0501/AnyDexGrasp](https://github.com/RDChen0501/AnyDexGrasp/tree/124c4f4ca6a5a7b7a60d5f39cce4220829485c0f)，`124c4f4` | UR 接口替换为 Franka，Allegro 控制与相机转 ROS2 | 机械臂接口和相机外参拆分 | 不是新手形态适配；代码仍留待标定项 |
| [SevenFo/AnyDexGrasp 的 dev 分支](https://github.com/SevenFo/AnyDexGrasp/tree/34323059f21d181ceb107a5bf6e42258c3b5ae58)，`3432305` | Inspire 推理模块化、ROS 服务、Galaxea 接口/方向约定 | 推理与设备接口分离 | 配置仍使用原 Inspire 权重/标定资产；本轮未验证抓取效果 |

其余有新增提交的 fork，主要是依赖兼容、Docker、导入/标签生成修复或 ONNX/QNN 导出，没有发现可供 Boya 直接套用的另一种手的完整标定与采数训练链。`Xuanyuan-jiang/anydexgrasp_with_ME` 主要是环境和扩展打包，不是新手适配参考。

## 自研手参考：几何标定与动作定义

`Jonas-Z-up` 最有价值的文件：

- [`own_hand/own_hand_grasp.py`](https://github.com/Jonas-Z-up/AnyDexGrasp_ws/blob/444199bb10bb715578f014068ab18dbff49a890b/own_hand/own_hand_grasp.py)：逐手型指定拇指与对向接触组，由 FK 后的接触组构造抓取参考系，再计算手基座目标。
- [`scripts/hand_bridge.py`](https://github.com/Jonas-Z-up/AnyDexGrasp_ws/blob/444199bb10bb715578f014068ab18dbff49a890b/scripts/hand_bridge.py)：8 个手型的预塑形和逐指闭合动作，明确电机方向与执行阶段。
- [`scripts/tools/compute_hand_geometry.py`](https://github.com/Jonas-Z-up/AnyDexGrasp_ws/blob/444199bb10bb715578f014068ab18dbff49a890b/scripts/tools/compute_hand_geometry.py)：电机—关节映射、名义闭合端点、指面几何计算。
- [`docs/own_hand_pipeline.md`](https://github.com/Jonas-Z-up/AnyDexGrasp_ws/blob/444199bb10bb715578f014068ab18dbff49a890b/docs/own_hand_pipeline.md)：统一最终执行位姿与碰撞位姿，说明采集/计划绑定和实际验证范围。

其主要几何思路是：用拇指接触组与两组对向接触组的质心形成三角形，以三角形法向定义接近轴，以拇指方向定义开合轴。按各类型接触组计算 `T_hand_grasp`，再用 `T_world_hand = T_world_grasp @ inv(T_hand_grasp)` 得到手基座位姿。这个“每种手独立定义抓取参考”的分层值得借鉴；接触组选择和法向符号仍需根据目标手验证。

一个有用的反例：该仓库曾将手指延伸轴直接当作抓取接近轴，造成预览姿态错误，而当时的代数闭环测试仍通过。其[问题分析记录](https://github.com/Jonas-Z-up/AnyDexGrasp_ws/blob/444199bb10bb715578f014068ab18dbff49a890b/own_hand/problems/%E9%97%AE%E9%A2%98%E5%88%86%E6%9E%90%E5%9B%9E%E6%89%A7.md)说明，**矩阵连乘正确，只能保证所选定义的自洽，不能保证定义符合接触几何。** 本轮没有运行其测试；`test_own_chain_closure.py` 中的 closure 指坐标链闭环，也不是物理闭合抓取测试。

不能直接照搬的部分：其锚点按闭合姿态和类型计算，`base_pose_from_grasp` 没有按输入抓取宽度查询锚点；电机位置到关节角是根据名义端点的线性映射，并非完整硬件辨识。动作类型与 Inspire 的语义也并非完全一致，但评分仍来自 Inspire。最新文档承认评分尚未对自研手重新标定，`sequence_completed` 不能自动成为 `physical_success`。因此它能指导几何/接口设计，不能提供已验收的新手评分器或成功率。

## 仿真参考：可借鉴采数流程，需保留协议差异

`kywch/pixi` 有真实的采数、训练与评估代码入口：

- [`collect_rs_inspire_grasp_data.py`](https://github.com/kywch/AnyDexGrasp/blob/3888737ef280fb4707601501c593a5a5a2f0bb20/collect_rs_inspire_grasp_data.py)：按手型和附加深入组织试验，保存 CGR 特征、原二指索引、翻面、手型/深入及执行结果。
- [`robosuite_env.py`](https://github.com/kywch/AnyDexGrasp/blob/3888737ef280fb4707601501c593a5a5a2f0bb20/robosuite_env.py)：手动作适配、抓取系到腕系变换，以及接近、闭合、退出执行。
- [`train_multifinger.py`](https://github.com/kywch/AnyDexGrasp/blob/3888737ef280fb4707601501c593a5a5a2f0bb20/train_multifinger.py) 与 [`robosuite_inspire.py`](https://github.com/kywch/AnyDexGrasp/blob/3888737ef280fb4707601501c593a5a5a2f0bb20/robosuite_inspire.py)：分别提供训练和学习评分后的评估。

静态阅读发现以下重要差异，不能把这个 fork 直接当“更忠实且更高成功率”的替代基线：

1. **关节与标定近似。** 6 维手寄存器按 actuator 范围线性展开到 12 维；作者注释明确提醒它可能非线性。抓取到腕的变换使用固定宽度索引，还包含手工约 20° 旋转和 7.8mm 平移补偿。这些是该模型的适配假设，不是 Boya 通用常数。
2. **480 维特征变换被修改。** [`dataset/multifinger_hand.py`](https://github.com/kywch/AnyDexGrasp/blob/3888737ef280fb4707601501c593a5a5a2f0bb20/AnyDexGrasp/dataset/multifinger_hand.py) 将原来的 `12 + angle_type * 2` 改为直接使用 `angle_type`。注释依据是怀疑原式有问题，未提供原记录/发布权重的等价性证明。本项目已经核验的官方 canonical480 不能因此改写。
3. **训练划分与损失不提供现成解药。** 未提供独立测试文件时，训练入口按记录索引随机切分，不能自动保证物体隔离；决策网络仍有最终 ReLU，选中输出仍用 MSE。它不构成本项目评分泛化问题已解决的证据。
4. **评估会改变候选和分母。** 无候选时重试并增加增强次数；选择阶段 `randint(0, min(N-1, 5))` 实际最多在前 6 个中随机挑选，虽注释写 top5。汇总成功率排除无候选的 `None`，不是全场景一次 top1 成功率。
5. **成功与视觉条件不同。** 使用仿真物体分割掩码；[`diverse_lift/lift_env.py`](https://github.com/kywch/AnyDexGrasp/blob/3888737ef280fb4707601501c593a5a5a2f0bb20/diverse_lift/lift_env.py) 的成功检查比较物体 body 原点与桌面高度加 0.11m。它不等同于本项目源网格最低点离桌并持续受载的判据。

可借鉴的是清楚的模拟器适配层和特征/动作/结果绑定方式。本项目已有相应绑定和更严格的结果审计，下一步应对照采样与覆盖差异，而不是重建一套重复的采数框架。

## 完整推理与机械臂接口参考

[`kittvnebluda/adg_server/planner.py`](https://github.com/kittvnebluda/anydexgrasp/blob/0b59bfb02bdf59c1a21cae2522edc71bc4ca8f2b/adg_server/planner.py) 明确实现原点云与 10 个增强分支，恢复候选坐标并合并，再做 Inspire 决策评分和映射。可参考其模型服务接口和候选来源组织。它也使用方向/目标区域过滤、改过碰撞实现；README 自述新旧碰撞仍有约 4% 候选分歧，不能视作严格无差异优化。本轮没有核实其速度或成功率。本项目已有原 DH3 完整 11 分支实现，接回 Boya 时应首先复用已核验代码，以此服务作为辅助对照。

[`RDChen0501/robot/franka.py`](https://github.com/RDChen0501/AnyDexGrasp/blob/124c4f4ca6a5a7b7a60d5f39cce4220829485c0f/ur_toolbox/ur_toolbox/robot/franka.py) 将机械臂接口独立出来，但 `search_pose()` 仍抛出 `NotImplementedError`，相机与夹手安装矩阵的注释要求重新测量。因此它是换臂接口参考，不能当作完整实机验收案例。

`SevenFo/dev` 的 [`config.py`](https://github.com/SevenFo/AnyDexGrasp/blob/34323059f21d181ceb107a5bf6e42258c3b5ae58/inference_v3/config.py) 仍绑定原 Inspire 资产和权重，配置 10 次增强；[`robot_utils.py`](https://github.com/SevenFo/AnyDexGrasp/blob/34323059f21d181ceb107a5bf6e42258c3b5ae58/inference_v3/utils/robot_utils.py) 主要提供 Galaxea ROS 目标位姿发布。不能将其归类为新增灵巧手训练方法。

另有[原仓库 issue #6](https://github.com/graspnet/AnyDexGrasp/issues/6) 中 MinamiJogen 的自研手适配说明：准备 URDF、定义手型、逐手型测量电机角度与开口宽度关系，再运行采集脚本。该评论没有给出可核查的实现链接；检查其公开仓库列表也未找到相应 AnyDex 实现，不能用一句“已适配”作为性能证据。

## 与当前 Boya 的具体对应

| 适配环节 | 当前已有内容 | 这次参考能补充什么 |
|---|---|---|
| 手型/开度/抓取坐标 | [`asset_adapter.py`](boya/asset_adapter.py) 已有各槽 FK、开闭端点、指腹对侧参考和逆标定变换 | 参考 Jonas 按类型显式检查接触组、法向、拇指滚转和进深方向，核验“物理语义”而不只核验矩阵 |
| 可变开度与动作一致性 | Boya 已有按开度离散的参考表；已有三件映射/闭合审计 | Jonas 按闭合态固定锚点的简化不应覆盖我们的宽度表；应比较各开度的参考系是否仍符合预期接触面 |
| CGR 候选生成 | Boya 入口目前仅 identity；原 DH3 已有验过的 11 分支 | 以官方已验实现为主，参考 kitt 的封装补齐来源、增强逆变换与合并；先离线比较候选变化 |
| 新手评分 | 当前使用 Boya 自采标签训练，固定 57 池已经证实排序/阈值漏选 | 新手几何改变后必须保持标签语义和 canonical480 一致；不能借用 Jonas 的 Inspire 头，也不能直接抄 kywch 改过的特征变换 |
| 仿真与执行 | 已有完整臂/受控腕、失败保留、源底面和承载审计 | 参考采样组织与覆盖，继续分开报告无候选、无计划、实抓失败；不照搬排除无候选的成功率 |

**建议优先做“每手型几何语义对照＋原 11 分支候选对齐”，同时保留评分有效性作为独立验收项。** 当前 Boya 抓取系的接近方向由源固定方向投影构造；Jonas 则由接触组三角形法向构造。这是值得定量比较的假设差异，尚不是现有映射错误的证据。已有三件映射审计没有发现接线错误；三件在固定池中均有成功候选，也说明不能把所有失败都归给坐标系。

这些建议不覆盖旧负结果，也不授权自动重训、调阈值、改变闭合/动力学或扩测。新物理比较仍应绑定独立协议；本轮只新增此报告与轻量来源清单，原模型、435 条数据、冻结实验结果和 CanonDex 均未修改。
