# 首个随机三物体场景：四型 null 的只读定位

四条 `null_geometry` 都是 **原 4 cm 接近末段的手指碰地板**。固定预抓端点没有违规，也没有 object/self/arm 碰撞；因此 HOME 的 RRT 搜索尚未开始，不能归因为 RRT 预算不足或机械臂全局不可达。

场景 `s40_lift_fit_00000_s2026111401` 包含 graspnet_017/046/011，无箱。capture/CGR 已由 batch 完成；本诊断不重做观测或前向。四型固定首选均来自原 raw row 816，使用各自冻结的 Boya slot。每份原命令有 50 settle、750 HOME、500 approach；HOME 末端为零基索引 799，approach 为 800–1299。

| 型 | 接近阶段的碰地板部位 | 首次 approach 步，1 基 | 违规条数 | 最深低于地板 |
|---|---|---:|---:|---:|
| g1 | 中指末节 / 食指末节 | 355 / 390 | 146 / 111 | 5.861 / 2.936 mm |
| g6 | 食指末节 | 415 | 86 | 1.499 mm |
| g7 | 中指末节 / 食指末节 | 367 / 415 | 134 / 86 | 4.801 / 1.473 mm |
| g8 | 中指末节 / 食指末节 | 347 / 380 | 154 / 121 | 6.750 / 3.677 mm |

每一组违规都持续至最终抓取姿态，末端碰地板仍在。预抓时上述指节最低点均为正，约 32.2–44.3 mm；原掌位移的模长约为 0.04 m。换 HOME 中间路线无法消除原样保留的 approach/抓取末端碰地板。

`arm/native/path_geometry.json` 的 257/86/220/275 条违规，与 `home/rrt/fixed_segments_geometry.json` 按原命令索引还原后 **逐条完全相同**。后者检查 50 个 settle、1 个固定 HOME 端点和全部 500 个 approach，共 551 个状态；838 条合计违规全部是 approach/floor，没有 settle 或固定 HOME 端点违规。四条 `frozen_endpoint_or_non_home_collision` 在本场景的具体含义就是其中的 non-HOME approach 碰撞。

另外使用独立 XML/NumPy FK、原 mimic 关系和原指节 STL 全顶点，重算四型预抓端点及全部 500 个 approach 状态的食指/中指最低点。所有违规步与冻结回执一致，最大最低点误差 4.025e-16 m。平面对顶点的最低值与其凸包相同，这里没有重跑碰撞规划或把粗代理称为精确物体 CAD。

原 collector 的可见点云准入与完整机器人源几何地板门是不同检验；本结果只说明这四个固定首选不能通过后者，不能代表所有候选都无解。没有补候选、改映射、改变初态、追加 seed、放宽碰撞门或消耗 RRT 搜索预算。四次 HOME preparation 均 `actual_search_runs=0`、`retime_preparations=0`，没有实际试抓或新增标签；null 不能当负标签 0。

本诊断新增 Kit、物理、IK、规划、CGR、训练、标签均为 0。36 个直接输入文件在诊断前后 SHA 不变，所有运行源和冻结输出保持原样。轻量证据与可重算只读脚本见 `evidence/author_collection_null_prefix_20261114_v1/NULL_DIAGNOSIS.json` 与 `diagnose.py`。原 8+2 scene 队列继续原预算，不据此临时更换策略。
