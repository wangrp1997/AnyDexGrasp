# DH3 原首选抓姿：三个新几何体实际执行

2026-11-03。三个预先指定的合成几何体，每个只执行官方最终排序第一的抓姿一次，**3/3 抬升且通过沿用的数值/出力/腕跟踪检查**。另做一次原方块接口回归，完整轨迹与之前通过的1920Hz结果逐字节一致。

这是官方发布模型接入独立物理后端的小规模验证。不是作者UR5/D415真实实验成功率、未知物体总体成功率，也没有接入Boya或机械臂路径规划。三个新物体均为100g、每种一个姿态；三者独立选中同一DH3 type4，尚未动态覆盖其他三种类型。

## 实际结果与视频

| 新物体 | 尺寸 | 原碰撞准入/400 | 一次执行末态升高 | 最大传动残差rad | 峰值关节速度rad/s | 结果 |
|---|---|---:|---:|---:|---:|---|
| 直立圆柱 | 直径50mm，高90mm | 251 | 100.329mm | 0.006033 | 1.2904 | 通过 |
| 球 | 直径60mm | 103 | 92.466mm | 0.001393 | 1.1247 | 通过 |
| 长盒 | 100×45×35mm | 50 | 99.455mm | 0.001404 | 0.9669 | 通过 |

三个末尾半秒的960个实际积分步，均逐步保持正法向手物接触。没有选过其他抓姿再保留成功视频，没有调整摆放、放大出力、附着物体、关重力或积分后reset。

全部视频集中在同一目录：

- [圆柱正常速度视频](../../outputs/dh3_primitive_generalization_20261103_v1/videos/cylinder.mp4)
- [球正常速度视频](../../outputs/dh3_primitive_generalization_20261103_v1/videos/sphere.mp4)
- [长盒正常速度视频](../../outputs/dh3_primitive_generalization_20261103_v1/videos/long_box.mp4)
- [三物体过程截图](../../outputs/dh3_primitive_generalization_20261103_v1/contact_sheet.jpg)
- [本地汇总页](../../outputs/dh3_primitive_generalization_20261103_v1/index.html)

每段205帧、20fps、10.25s。帧是在实际物理执行时同步渲染的；编码后完整解码核验帧数，编码不重放仿真。视频/原始帧/轨迹留在忽略目录，不提交Git。

## 完整链路及可复用接口

真实渲染RGB-D → 原identity+10增强 → 原四个发布决策头 → 原评分/标定/可见点云碰撞 → 原argsort/top10/top1 → 原2D到12D映射 → DH3实际闭合抬升。物体类别、分割标签、真值位姿和完整CAD点云没有送入网络；清单只绑定输入来源及后续世界坐标执行。

新增三个独立入口，原算法和已验收冻结执行器保持：

- [`primitive_scene.py`](primitive_scene.py)：空client中创建解析惯性的原生动态几何体，输出同目录color/depth/meta及清单。`create_scene(p,cid,spec=manifest['spec'])`和`assert_scene_matches_manifest(...)`供执行器使用；渲染阶段禁止物理积分。
- [`../runtime/run_official_dh3_batch.py`](../runtime/run_official_dh3_batch.py)：调用已有完整单帧入口，进程内共享一个表示网络和四个评分头，各帧仍重新执行全部11分支、碰撞和排序。没有缓存候选或成功抓姿。
- [`run_dh3_scene.py`](run_dh3_scene.py)：输入场景清单与官方selection回执，执行其绑定的原top1，保存每一步状态/接触和同步帧。直接导入原通过的CoupledDH3，不复制改写机构模型；1920Hz/inner20固定。

沿用主动电机0.6/0.3Nm、理想结构行cap100Nm、腕每行30N/Nm、重力、接触模型、内部邻接排除及判据。100Nm是此前声明的结构数值尺度，并非主动抓力或已识别硬件额定值。物理控制仍是有限世界腕约束的脚本接近/闭合/世界Z抬升，详见[模型与限制](KINEMATIC_CONSTRAINT_MODEL.md)。

原方块接口回归新增一次19680步，18个轨迹数组、trace.npz、contacts.npy和frames.json与旧1920Hz结果全字节一致。回归后的runner仅补源依赖元数据；旧冻结源码保留，补录明确标注执行后增加，未声称这些补充字段在执行前已存在。新物体执行前冻结最终runner/场景/原控制源码。

## 输入渲染问题和保留的零步失败

首轮`scenes/cylinder/FAILED_RENDER.json`拒绝，零物理步。原生TinyRenderer显示圆柱自带1mm外观凸包余量，球外观也有几何偏差；首轮圆柱解析表面最大误差1.803mm、球预检2.761mm，尽管物理碰撞AABB尺寸正确。

修复只把圆柱/球的显示替换为精确尺寸的解析三角网格，原生动态碰撞形状、尺寸、质量、惯性、摆放、相机和物理判据未变。生产输入另存`scenes_v2`，解析表面最大误差分别0.824/0.959/0.806mm，低于1.5mm输入量化/栅格容差；图像像素回投一致，物理状态在渲染前后不变。

辅助native ray点差完整保留。球射线点与图像点最大相差1.690mm，射线点自身离解析面的最大误差0.976mm；它是Bullet近似射线的诊断，未用作抓取准入门。解析表面、实际碰撞类型/尺度/AABB和惯性检查仍通过，**没有因此修改任何物理抓起或稳定性判据**。

## 时间、缓存与完整成本

| 项目 | 圆柱 | 球 | 长盒 |
|---|---:|---:|---:|
| 原11分支合并候选 | 251 | 169 | 640 |
| 11分支实际耗时 | 17.967s | 16.583s | 18.411s |
| 单帧含加载/评分/资产/碰撞/选择 | 41.625s | 36.838s | 42.541s |
| 物理进程内部含同步渲染 | 31.018s | 27.081s | 28.389s |

三帧推理进程内部总墙钟123.051s，包含依赖导入、权重哈希、一次加载、新槽点云准备和所有帧阶段，不含外部解释器启动。执行worker在各自selection完成后启动，与后续帧推理重叠；不能相加表中时长当完整端到端时间。本轮没有单独记录包含所有输入诊断/调度/媒体处理的统一墙钟。

真实缓存计数：表示权重读1次、head权重读4次，模型构建1+4，首次strict赋值5次，重复相同state对象strict加载跳过10次。首次构造保存CPU RNG前/后状态，warm命中重放10次、前态不匹配0；GPU RNG未被构造改变，实际透传计数multinomial/randperm均0。仍逐帧执行原网络/筛选，GPU选择为物理GPU1。缓存生效不等于已测到固定提速倍数：此处不同帧有不同候选/资产成本，没有相同帧冷/热配对计时。详见[批量接口说明](../runtime/BATCH_INFERENCE.md)。

本轮三新物体59040实际步/30.75s物理时间，加一次方块接口回归共78720步/41s。计入全部历史失败、隔离诊断、空手及旧方块，累计177180个实际积分步/160s混合步长物理时间。输入渲染失败、审计和视频编码均零物理步。

## 证据与复用命令

- [`evidence/dh3_primitive_protocol_20261103_v1.json`](evidence/dh3_primitive_protocol_20261103_v1.json)：在新物体运行前冻结的三几何/原首选/固定控制/单次协议。
- [`evidence/dh3_primitive_generalization_20261103_v1.json`](evidence/dh3_primitive_generalization_20261103_v1.json)：逐物体结果、视频SHA、真实缓存计数、全部成本与边界。
- [`evidence/dh3_primitive_inference_audit_20261103_v1.json`](evidence/dh3_primitive_inference_audit_20261103_v1.json)：场景/模型/阶段SHA、像素/体素/11分支合并、原碰撞bits、top10/top1逐位核验。
- [`evidence/dh3_primitive_execution_audit_20261103_v1.json`](evidence/dh3_primitive_execution_audit_20261103_v1.json)：四次完整执行逐步独立复算、原选姿的世界映射、有限力矩/腕/残差/速度、最终接触和全部帧。
- [`evidence/dh3_generic_scene_regression_20261103_v1.json`](evidence/dh3_generic_scene_regression_20261103_v1.json)：旧方块新接口逐位回归。
- 原始完整目录：`outputs/dh3_primitive_generalization_20261103_v1/`。

复用已有环境，不安装/修改共享依赖。重跑必须用新目录，不覆盖本轮；下列`_example`只展示接口，未另跑一轮：

```bash
source outputs/native_deps/env.sh
export PYTHONPATH="$PWD/outputs/native_deps/python_overlay:$PYTHONPATH"
export LD_LIBRARY_PATH="$PWD/outputs/native_deps/os_libs:$LD_LIBRARY_PATH"
export ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python

"$ANYDEX_PY" -B reproduction/simulation/primitive_scene.py \
  --output-root outputs/primitive_example/scenes

CUDA_VISIBLE_DEVICES=1 "$ANYDEX_PY" -B reproduction/runtime/run_official_dh3_batch.py \
  --scene-manifest outputs/primitive_example/scenes/cylinder/scene_manifest.json \
  --scene-manifest outputs/primitive_example/scenes/sphere/scene_manifest.json \
  --scene-manifest outputs/primitive_example/scenes/long_box/scene_manifest.json \
  --output-dir outputs/primitive_example/inference

"$ANYDEX_PY" -B reproduction/simulation/run_dh3_scene.py \
  --scene-manifest outputs/primitive_example/scenes/cylinder/scene_manifest.json \
  --selection-results outputs/primitive_example/inference/cylinder/selection/RESULTS.json \
  --output-dir outputs/primitive_example/physics/cylinder
```

独立物体可以分别派发CPU进程；每个抓取内部保持同一现场并禁止动态reset。本轮是三个进程与推理的流水重叠，不是GPU批量物理或跨物体常驻世界。继续验证时应增加真实网格/摆放与手型覆盖；不要重做已通过的时间步搜索，也不要据3/3宣称未知物体100%。
