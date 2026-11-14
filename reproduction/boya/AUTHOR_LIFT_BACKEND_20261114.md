# 三动态物体、无箱提起执行与评价后端

用户已把当前任务改为“只要能提起”。本次新增独立 `author_lift.py`，任务为 `declared_boya_clutter_lift_v1`，场景为 `anydex_boya_author_lift_scene_v1`，必须显式 `bin: null`。旧有箱场景、投放任务和历史结果保持原语义。本文件只记录本 agent 的代码、单测与离线审计；本 agent 新 Kit、物理积分、CGR、训练、正式标签均为 0。实际小闭环由 root 在统一冻结后另行记录。

## 场景和执行接口

`author_scene.py build --task lift` 仍使用原 S40 000/002/005 的三套源资产、原 settled 方向与高度、原尺度 1、质量 0.1kg、摩擦 0.6，以及原相机；三体 XY 分别为 `[-.75,.10]`、`[-.63,.10]`、`[-.89,.10]`。这是已经使用的开发物体和固定摆放，不是外部泛化或随机场景覆盖。无箱模式不会生成箱体视觉、碰撞或射线检查几何，静态接触路径只纳入实际 ground。

初始化仍是共同场景 750 ticks × 0.004s = 3s，共 2250 object steps。每个动态体逐 tick 写入、更新、接触回读，并独立施加持续世界向下 0.98N；全局重力零的原生方案保持。原生导入 20µm、材质/质量回读、末窗稳定与接触流验收保留。观测阶段只 render，不额外积分；执行明确为 measured pose 的独立 cold recreation、初始零 twist，不冒称同一进程持续观测，执行期间不 reset、附着或改写对象状态。

`target_object_id` 仅保留原 native target body 接口，完整 `original_object_ids` 另列。它不参与候选身份筛选，也不能单独决定新成功标签。

新 backend 提供 `validate_scene(path, archive_sources=None)`、`args_for_plan(plan_dir, output_dir)`、`execution_steps(args, scene_path)` 与 `run(args, scene_path)`。plan 必须含 `author_lift` 的任务、scene SHA 与无箱协议绑定；原 native 24 独立列／28 实际关节、20 手 DOF、位置驱动和限位不改。动作由独立 `author_lift_plan.py` 生成：原 approach-only prefix、250 close、50 hold，接原 10cm world-Z lift。backend 无投放、release、containment 或新增末尾 hold。

`run` 不创建或关闭 SimulationApp，也不写正式 COLLECTION_RECORD。调用方负责执行前后绑定校验、collection label/最终回执落盘后才 close。backend 在返回前写 `LIFT_EVALUATION.json`、`MULTI_OBJECT_SETUP.json`、`MULTI_OBJECT_TRACE.json`、`MULTI_OBJECT_CONTACTS.json`，异常/早停保留实际成本与逐物体结果；零步初始化失败不会引用未初始化 dt。native target 每 tick 更新一次，额外两体各一次，共享时钟只允许推进一次。

## 提起验收

每体使用完整源可视网格顶点与 native root-link XYZW pose 计算世界最低点，不用 COM 或粗凸最低点替代。每体参考高度取最后一个 `hold` 帧的本体 reference Z。完整原阶段序列后，末帧必须为 `lift`，同时满足：源网格最低点 ≥5mm；reference Z 相对最后 hold 升高 ≥5cm；拇指接触力 >1mN 且其余四指接触力之和 >1mN。

原七门全部显式必需：关节限位 ≤2mrad、mimic ≤2mrad、穿透 ≤1mm、掌跟踪 ≤3cm/15°、非手支撑 ≤1mN、固定基座漂移 ≤20µm/20µrad、完整执行。任一门缺失也拒绝；全三体接触深度和非法机器人支撑门额外逐体保留，深穿透 >5mm 与闭合前任一体 XY 推移 >3cm 保守早停也保留。没有降低原门来取得成功。

pickup 发生时采用原观察条件（最低点和 reference rise 均 ≥5mm、双侧同期受载），记录首次、最长和连续末尾 pickup 时长；**时长只诊断，不新增 0.5s 门**。这也不新增“其它物体／环境完整支撑链”默认门：当前明确检查全体非手机器人支撑，完整外部支撑链另作为边界，不能把未实现的门写成通过。

结果含完整且唯一的 `object_outcomes[{object_id,task_success,...}]` 和 `original_object_ids`，顶层为逐体成功的 any。原 primary 的旧 `task_success` 完全不读。完整且有效执行但未提起是可采负例；阶段不完整、基础设施错误、原门或全体门失败均不生成有效标签。连续持握后滑落、只抬 reference 而源底贴桌、只单侧受载都会失败。

## 离线核验与范围

26 tests 通过：15 旧 transport 回归、3 场景、8 新 lift。覆盖任意其它源体成功、缺失／重复阶段、瞬间提起后掉落、源底未离桌、升高不足、单侧／阈值边界、全体身份缺失、非法支撑、原七门缺失、完整负例与零步失败、单时钟和每体单次更新。首次单测 25/26：新增 fixture 误以为每个原资产都有可选 repair，`Path(None)` 报错；只修 fixture 的条件检查后 26/26。没有修改原资产或修复其结果。

另对真实三物体完整源顶点，逐一做四个旋转 × 三个高度共 36 状态，与原 `tabletop_pickup_state` 独立最低点公式对比，最大误差 `1.3877787807814457e-17m`。`ready` 的原生质量、材质、接触、合法 hand owner、entry pose/twist、20µm 审计代码经 AST 比较，仅 read/save 名称限定与最终去 containment 测量作显式适配，其余结构一致。这些是离线数值与接口核验，未验收新三体的实际物理提起。

轻证据位于 `evidence/author_lift_backend_20261114_v1`。旧 435 是旧单物体 lift 数据，不能直接称新三体采集完成，也不能自动转成新任务标签；正式扩采和正式重训仍须新流程完成后另计。

## root 的新场景命令

在项目根目录使用既有 venv 和成功的原生依赖环境；固定 GPU0 由 root 设置。先 build，输出目录必须尚不存在：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/reproduction/boya" \
 /home/wangrenpeng/tendon-spin/.venv/bin/python -B reproduction/boya/author_scene.py build \
 --task lift \
 --source-manifest outputs/s40_stable_supplement_20261107_v2/settling/objects.json \
 --native-summary outputs/s40_stable_supplement_20261107_v2/collection/native_preprocessing/SUMMARY.json \
 --backend reproduction/boya/backend.collection.json \
 --output-dir outputs/boya_author_lift_scene_20261114_v1
```

再用已验收 LD_LIBRARY_PATH、PYTHONPATH（含现有 clarabel overlay）环境：

```bash
/home/wangrenpeng/tendon-spin/.venv/bin/python -B reproduction/boya/author_scene.py capture \
 --scene outputs/boya_author_lift_scene_20261114_v1/scene.json \
 --output-dir outputs/boya_author_lift_scene_20261114_v1/capture
```

capture 在 close 前保存结果／错误／进度并保留正确 exit code，末尾复核全 source/input SHA。后续表征和新 pipeline 必须使用这个新 capture 的 `observation.json` 与 `observed_scene.json`；旧有箱 RGBD 不冒充无箱观测。
