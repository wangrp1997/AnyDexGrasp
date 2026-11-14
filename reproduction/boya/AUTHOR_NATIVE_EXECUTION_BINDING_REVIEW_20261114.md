# 原生执行依赖与修复配置只读复核

本轮只读复核真实资产和三路新 transport 准备结果，未启动 Kit、IK、物理、训练或标签生成。最终复核的 pipeline SHA 为 `2376c747bd2770e724e3bf4314af688a9a046245d386305c7b71b5373f77d2b0`。证据在 `evidence/author_native_execution_binding_review_20261114_v1/`。

原 canonical arm repair 与当前 combined URDF、physics 配置的 SHA 一致。直接使用真实 PlanningModel 源凸包、现有缓存分区数组和原 `refine_collision_parts` 核验：wrist_1_link 493→500、upper_arm_link 3309→3316；原父体/所有子体身份、64 顶点限额、体积保存检查通过。本轮未重新计算分区、写缓存或运行 native cooking；native 20µm 仍须实际运行验收，不能据此声称精确 CAD。

独立按原 loader 规则展开 combined URDF 的全部 visual/collision mesh，37 个去重文件与真实 `STAGING_RECEIPT.json` 完全一致。37 mesh 加源 repair、副本 repair、URDF、physics 共 41 个绑定，实际源/副本字节及全部 SHA 一致，副本位于原 native 查找的 `plan_dir.parent/arm_collision_refinements.json`。此前有 16 个 mesh 未被外层场景绑定覆盖；新 pipeline 已将全部 37 个加入 protocol inputs，并在创建 bound plan 时加入 extra inputs。三个物体的真实源资产与各自 repair 绑定也通过。

原生机器人路径仍为原分区、可选逐 link repair、`author_convex` 和后续 cooking 审计；每个动态物体仍走原 `spawn_object` 的分区与各自 repair，未替换为静态干扰体。原 `create_scene(self_collisions=True)` 会启用 contact processing，`RobotContactReports` 另有拒绝检查，无需改物理开关。

复核还发现执行后完整性检查原本位于 `COLLECTION_RECORD` 之后，可能留下可训练标签。collector 已修复：全部 inputs、sources、capture、ready 和 plan dependencies 的后置核验先于 `collect_execution`，所有标签、成本及错误回执先于可能直接退出进程的 `SimulationApp.close`。新测试覆盖执行后 repair 漂移：保留 23 步成本而不调用标签生成、不生成 collection record。此测试由组件作者运行；本轮独立读取源码顺序并记录 AST 行号，未重复执行测试。

三路真实 g1/g6/g7 准备均为 `complete/null_geometry`。逐路核验 protocol 的 41 项原生修复/mesh 绑定，以及全部 inputs、sources、capture、raw plan dependencies 和 commands/两份 geometry 文件 SHA，均一致。三路均未生成 READY、bound transport plan 或执行标记。因此，bound plan 的 41 项 extra-input 插入只完成源码及组件测试复核，不能写成真实 ready 执行已通过。原 native 会在场景创建/积分前核验 path_geometry 与 lift_arm_geometry；这两项独立派生文件不属于通用 source/extra-input 映射，本轮亦单独检查实际文件。

普通 venv 找不到 Clarabel；现有隔离 overlay 可成功导入 `clarabel==0.11.1` 与真实 `cooking_audit`。该检查只在独立进程临时增加搜索路径，无安装或父环境修改。未来一次执行须在写启动标记前将该现有 overlay 加入同一实际进程环境；本轮未消耗执行额度。

本轮只读数值修复检查 0.457s，最终三路依赖检查 1.193s；其它文件读取和源码复核成本另计。新增 Kit/IK/物理/训练/标签均为 0。结果证明当前资产与依赖绑定一致，未产生新成功率，也未验收动态搬运。
