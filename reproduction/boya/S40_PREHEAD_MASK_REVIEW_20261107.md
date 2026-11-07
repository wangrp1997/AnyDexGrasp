# S40 score-first 评分前筛选核对（2026-11-07，只读）

结论：缺口成立。`deploy_score_first.make_queries:64–89` 对未经过 rig 筛选的原 parser 行直接按 CGR 分取前1000，随后才评分；它没有原 `get_grasp` 的方向、原始宽度、工作区联合 mask。`score_s40_raw_candidates.py` 复用该函数，因此本轮3200条几何全拒不支持“训练不足导致”的因果结论。现有结果只说明这套未对齐输入域、评分和Boya几何串联没有准入；补齐筛选也尚不保证抓取成功。

本报告不改仓库、不执行物理、不新增CGR/评分/训练。今天已冻结的无head覆盖采数可独立继续，不应为了本问题中途更改其候选规则。新稳定落位v2已经完成，无需再做稳定性审计。

## 1. 原源码全部入口

用AST枚举仓库根目录所有 `get_grasp` / `get_ggarray_features` 定义，只有以下6个脚本。所有边界比较都是严格不等式；长度单位m。R是 parser raw列3:12的行优先3×3旋转，`preds[:,9] = R[2,0]`，是**相机光学坐标中接近轴第一列的z分量**，不是置信度。raw列12/13是相机x/y。

| 路径 | RGBD z输入范围 | 方向条件 R[2,0]> | raw宽度范围 | 相机x范围 | 相机y范围 |
|---|---|---:|---|---|---|
| robot_dh3.py:327,389–393 | (.15,.72) | .90 | (.045,.099) | (-.25,.25) | (-.205,.03) |
| robot_allegro.py:337,399–403 | (.20,.65) | .93 | (.040,.110) | (-.20,.20) | (-.200,.07) |
| robot_inspire.py:328,390–393 | (.35,.68) | .92 | (.010,.100) | (-.25,.25) | (-.205,.03) |
| collect_dh3_grasp_data.py:293,340–344 | (.15,.72) | .90 | (.010,.099) | (-.25,.25) | (-.205,.03) |
| collect_allegro_grasp_data.py:304,352–356 | (.35,.60) | .90 | (.050,.119) | (-.25,.25) | (-.205,.03) |
| collect_inspire_grasp_data.py:293,339–343 | (.35,.68) | .85 | (.020,.100) | (-.25,.25) | (-.200,.05) |

三种robot入口还各有 `get_ggarray_features`（DH3:504，Allegro:540，Inspire:511）：identity与10个增强分支分别进入 `get_grasp`，**先恢复增强/反射到原输入相机坐标，再逐分支筛选，再合并**。采数三个入口只有单帧 `get_grasp`，也含评分前基本筛选；采数与部署的参数本来就不同，不能混称同一协议。

DH3部署顺序：原mask→合并→`flip_ggarray`及成对特征规范化→按原CGR分取1000→四hand heads / extra-depth组合→学习分threshold .85与全局预算→手映射→top400碰撞。**score-first的“first”在基础候选域筛选之后、完整手碰撞之前。**

注意 parser 的逐seed objectness索引在原作者代码中被注释，只做“存在任一正seed”的判断；因此raw不能假定全是物体点。不能将新foreground mask伪称官方已经做过，也不能本轮偷偷加入而不声明端口策略。

## 2. 已跑原DH3离线链没有漏mask

- `reproduction/runtime/run_official_dh3_frame.py:70–102` 从原 `get_grasp` AST取出两条 `mask/workspace_mask`赋值，执行原方向/宽度/工作区联合筛选，并同时筛features，然后才top1000。
- `run_official_dh3_augmented_frame.py:36–70,218–302` AST执行原 `get_grasp/get_ggarray_features`，只替换fx/fy/cx/cy/s。11分支恢复、方向、宽度、作者工作区均保持；合并后的文件虽然也叫 `raw_camera_candidates.npz`，这里**已经筛选过**。后接单帧runner重复同mask并断言幂等。
- 已存证据 `reproduction/runtime/evidence/dh3_eleven_branch_pipeline_20261103.json`：1544条经过原get_grasp gate，重复筛选仍1544→top1000→学习门2952组合→碰撞top400→9准入。这是此前原DH3离线链的实际边界。
- 对比 `run_official_rgbd_frame.py:239–242` 与Boya `run_pilot_representation.py:182` 明确把产物标注为 **before rig-specific width/approach/workspace and collision filtering**。文件同名不等于语义相同。当前Boya `make_queries`没有消费该边界差异。

## 3. 本轮3200条的实际保存证据

轻量复读8个内部诊断的 `geometry.json` 和冻结raw数组，无重评分：

- 8×400 = 3200条；**3199条table拒绝、1条visible_scene_point_in_hand_hull拒绝**，0 arm检查、0物理。
- 假设机械地套作者DH3相机方向条件，3200中只有10条满足；作者完整mask仅1条满足。这是输入域不一致的有力提示，**不能据此把作者相机阈值当Boya最终策略**。
- 即便只作世界竖直向下 `a_W.z < -.9` 的诊断，原shortlist也只有138条满足（不是已验证可行解）。方向检查是必要的端口问题，但不能据此宣布其为3199条table拒绝的唯一原因。
- 过早top400会让不合适方向、桌面seed、极端width及其extra-depth组合占预算；对这批全拒直接多训并不能验证或排除坐标映射、接近方向域、候选覆盖和台面几何的原因。

独立数值产物：`evidence/s40_stable_supplement_20261107/s40_prehead_saved_candidate_diagnostic_20261107.json`；原mask机器清单：`evidence/s40_stable_supplement_20261107/anydex_prehead_official_masks_20261107.json`。各对象及真实拒绝原因均保留。

## 4. Boya必须显式适配的评分前契约

保持的应是**基础候选域先筛选、再topK、再canonical/head、最后完整几何/臂检查的顺序与数据对齐**。适配数值要来自Boya相机/台面/手标定，不能复制作者相机矩形。

1. 输入观测：冻结相机K、深度单位、有效深度范围、`T_WC`；今天新settled-pose产生的新图、新CGR必须绑定同一观察。旧parser与网格source index保持，不重排成另一个raw身份。
2. 接近方向：`a_C = raw[3:12].reshape(3,3)[:,0]`，`a_W = R_WC @ a_C`；用预先声明的世界/台面允许方向cone或cone集合筛选。若研究入口选择上抓，则例如以台面向下法向为cone中心；若允许侧抓，要显式声明侧向cone，不能把作者camera+z硬当world-down。cone阈值/方向来源必须预冻，不用8件诊断或EGAD结果调参。`asset_adapter`已说明CGR+X approach / Boya参考+Z approach，勿直接比较Boya palm某列与raw列9。
3. 宽度：作者raw宽度门与后续1.7扩展/标定表clamp是两件事。Boya应冻结raw宽度可用域，以及每手型的标定开度域和扩展/clamp策略；不能用“无论raw多大/小都夹到表端点”替代输入域说明。若保留作者clamp，明确记录，不能把DH3的45–99mm硬当Boya硬件值。所有手型共用门还是按手型门要声明；按手型门时每个query的有效性/排名预算需明确。
4. 工作区：先把raw平移变换到world/table frame，再用台面平面、可达区域中心和边界约束；或将声明的世界区域正确变换为相机域。不能把作者 `x∈(-.25,.25), y∈(-.205,.03)` 原样套在当前Boya光学坐标。不得使用目标mesh包围盒真值做“官方foreground”，除非明确声明额外仿真近似。
5. 联合mask在CGR top1000之前应用到raw/features/source_indices同一行；TTA若使用则恢复原相机帧后再mask。记录每一门单独通过数、联合通过数、pre/post-topK数及策略SHA。mask后没有候选就输出有来源绑定的no-candidate，不能放宽碰撞或补成失败标签。
6. 后续完整手/桌/接近/机械臂检查保持，不把简单cone或width门称为几何可行性证明。验证修复可先用固定已有raw做零物理对照，报告候选组成与准入变化；不要重跑当前冻结补采或改其旧结果。

当前建议：将此作为独立 **Boya pre-head domain adaptation v1** 修复任务；今日无head覆盖采数照冻结协议完成。新的部署诊断使用新目录和预冻mask参数，旧3200全拒及其成本完整保留，不再单凭此结果断言训练样本不足或算法失效。
