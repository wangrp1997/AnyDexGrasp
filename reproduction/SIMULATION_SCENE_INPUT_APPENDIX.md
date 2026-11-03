# 原手独立仿真：物体与场景输入附录

2026-11-03；仅输入可行性审计。本轮没有新仿真、渲染、前向、训练或下载。
手控制/动力学另见 `SIMULATION_HAND_READINESS` 主审计；本附录不替它决定控制器。

## 已核事实

原 AnyDexGrasp 的论文评测是真实机器人清桌，不是发布的动态仿真协议。
补充材料 `outputs/paper_code_audit/source/sections/supp.tex:47,78` 说明训练集为
144个商店物体、从40个GraspNet训练物体扩充；图注明确CAD为 **available upon request**。
当前官方README、Drive根/完整data索引中没有这144件扫描CAD、真实试验场景的物体姿态，
或能直接启动的含接触动力学场景。没有联系作者，也未把“可申请”写成“已公开拿到”。

须纠正“官方demo没有物体pose”的说法。已下载的 GraspNet demo `meta.mat` 确实包含：

- `poses`：`3×4×9`，9件物体相对于相机的姿态；
- `cls_indexes`：`[1,3,6,9,23,27,52,62,39]`；
- `intrinsic_matrix` 和 `factor_depth=1000`。

AnyDex的 `dataset/graspnet.py:426–440` 将模型序号加1与标签对齐，因此上述ID对应
GraspNet模型目录 `000,002,005,008,022,026,051,061,038`。这些pose是真实可用信息，
但目前缺对应CAD/纹理、相机到桌面/世界的外参以及质量/摩擦/惯量等动力学声明。
单张可见深度不能补齐背面/被遮挡的碰撞几何，也不能由一次离线候选推理得到实际抬升结果。

GraspNet官方另有 **Object 3D Models / models.zip [4.3 GB]** 公开入口及Google/Baidu/Jbox链接，
格式为88个 `models/000..087`。依据为已保存的官方 `metadata/graspnet_datasets.html`，
原页 <https://graspnet.net/datasets.html>。本次未下载该包，未宣称这些模型已经在本机，
也未假设CAD包附带可复用的动态仿真质量/惯量/控制协议。

## 不新增下载的最小可执行 synthetic smoke 配方

独立仓库的原 PyBullet 依赖已经包含这些标准资产：

`outputs/native_deps/python_overlay/pybullet_data/cube.obj`、`cube_small.urdf`、
`plane.urdf`、`plane100.obj`。`cube.obj` 的顶点范围为每轴 `[-0.5,+0.5]`，
`cube_small.urdf` 的视觉mesh缩放和碰撞box均为 **0.05 m**，质量为 **0.1 kg**。
因此可以固定这一件50mm立方体；它属于新建合成smoke，不能冒称作者144/150件物体。

建议下轮实施时先冻结以下独立场景声明，而不是修改现有demo：

1. 单一固定 `cube.obj`，视觉scale `[.05,.05,.05]`、碰撞box `[.05,.05,.05]`，
   初始物体中心世界坐标 `[0,.08,.025]` m、单位旋转，桌面顶面世界 `Z=0`。
   固定mesh/URDF及依赖的SHA；不引入CanonDex物体或手势/控制代码。
2. 明确声明新adapter的质量、惯量、摩擦与求解器参数。源 `cube_small.urdf` 的惯量
   对角为1且含 `inertia_scaling=3`，不能直接把这些值称作50mm实心方块的物理测量。
   若使用均匀0.1kg立方体假设，解析惯量是每轴 `4.1666667e-5 kg m²`，应在新adapter
   声明中显式采用并记录实际引擎读回值；不能悄悄改原URDF或声称复现作者动力学。
3. 真实引擎渲染partial RGB-D，尺寸 `1280×720`；可声明 `fx=fy=640`，
   `cx=640,cy=360`，near `.1` m、far `1.0` m、深度因子1000。相机世界位置
   `[0,0,.5]` m，`R_WC=diag(1,-1,-1)`（相机Z向下）。物体中心在相机系为
   `[0,-.08,.475]` m，处于原DH3相机workspace/depth窗口内；这不保证候选分数通过。
4. RGB、depth、intrinsics都由该渲染声明产生/校验。保存渲染器及投影矩阵、
   `T_WC/T_WO`、深度转换公式、图像SHA和物理状态。不能把完整网格点云当partial深度，
   也不能把原照片与另造的碰撞世界拼接后报告为同一现场。
5. 新 `meta.mat` 只提供原推理实际需要的内参/深度尺度；物体真值放独立scene manifest，
   不伪造GraspNet的 `cls_indexes`，不把真值pose/完整CAD送入原CGR候选评分。
   原11分支推理、宽度/方向/workspace/learned-score/碰撞筛选保持，真实拒绝就记录拒绝。

几何/相机方案已经足以构造可渲染的单物体输入；是否能实际抓起仍取决于手控制与动态适配。
这些步骤目前只是明确的下轮实施配方，尚未执行。

## 统计边界与待备项

最小待备项是一个单物体scene manifest、渲染/相机读回、原手动态adapter和闭合/抬升
控制声明，均无需作者训练包。推理无候选、原碰撞拒绝、进入实际闭合但未抬起、实际抬升
分别记；一件/一姿态的synthetic smoke不与31个决策checkpoint文件名指标、官方照片离线
碰撞准入、作者真实成功率或CanonDex批量统计合并。若以后要恢复原GraspNet九物体场景，
应另取对应九CAD并核对每个相机pose/单位/遮挡、桌面外参和动力学，不能先称场景已复原。

机器可读审计与已有资产SHA见
`reproduction/data_intake/evidence/simulation_scene_input_audit_20261103.json`。
