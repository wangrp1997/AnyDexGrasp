# 独立提起计划组件

新增 `author_lift_plan.py`，把一个已准入的开手 HOME＋4cm approach 原生计划扩展成原闭合、原保持和 world-Z 向上提起10cm。仅接受显式无箱场景 `anydex_boya_author_lift_scene_v1`、任务 `declared_boya_clutter_lift_v1`、`bin=null`。原 transport 计划、已有闭合/提起阶段和非24独立列输入均拒绝。

```python
receipt = author_lift_plan.prepare_lift(
    seed_plan_dir, scene_path, output_dir, archive_sources=None)
```

输出 `LIFT_PROTOCOL.json`、`LIFT_PREPARATION.json`；状态为 ready/null_geometry/infrastructure_failure。只有所有门通过才生成 `native/plan.json`，ready 回执带 `plan_dir`、plan/commands/path_geometry/lift_arm_geometry SHA。`plan.task` 与 `plan.author_lift.task` 均为新 lift 任务；`author_lift` 记录原 seed、scene、历史场景源码归档、冻结协议和策略。

原前缀的 joint_target、desired_palm、phase、time_s 保持 dtype 和逐 bytes 相同，初态及列名原样复制。追加原 TwoSidedClosure 的250步闭合、50步提起前保持；随后使用原32个顺序 IK 路点，每点最多300 nfev、原 1e-6rad bounds margin，再用原 timed_segment 的最短2秒、速度限制、加速度3rad/s²及jerk30rad/s³生成10cm上抬。没有末尾额外保持时长、替代 seed、换候选或重试。执行时仍是6 arm＋2 wrist＋16手电机共24独立列，4个派生关节使实际状态共28关节／20手DOF。

完整开手前缀对所有源物体初态凸代理及全机器人 self/floor 使用原 1e-7m 几何门。闭合、保持和提起阶段对 rigid arm/wrists/palm 与所有物体初态代理、self/floor 使用同门；不会遗漏兼容 primary。手指接触与物体实际运动不作静态预测，仍留给后端实际接触门。另对拼接后的完整6arm轨迹复核原关节、速度、加速度及jerk限制，不因失败延长预算。

原6 tests通过，日志及实际工具回执位于 `outputs/boya_author_lift_plan_20261114_v1`。验证了：旧 g1 的250close＋50hold全部24列及目标掌姿与原 helper 逐值一致；原24列开手输入不变；修改前缀、已闭合或 transport 输入被拒；解析合成机器人通过真实32次原生 IK 与原插值到达10cm、18非arm列不变；局部IK失败不换seed；坏输入保持基础设施错误类别。

测试前后因代码修改共运行两轮合成32点 IK，即64次合成 IK；不是64次真实场景规划。新场景候选 prepare、物理、采集、标签、正式训练均0。按主线要求等待全新无箱 capture→CGR→原采集首选→新HOME产物再做受控开发闭环，没有重新封装旧有箱计划冒充新任务。该组件完成不代表完整算法、稳定提起任务或训练验收已经完成。

最终源码 SHA256：`ee5d2257c3c0e589897d96d545479185317207a5a0693237832f3552c5a4224f`。测试 SHA256：`05499528c43cb071cdb9996b4301253e71afa55949e24ae31ffb3161bd09860b`。轻证据为 `evidence/author_lift_plan_20261114_v1`。

补充数值合法性检查：原 native approach 末目标通过 `(palm−.04axis)+.04axis` 计算，允许1e-12矩阵舍入差；实际原前缀数组仍逐 bytes 不变。增加1ULP回归测试，并重跑两项相关输入检查，3/3通过，0额外IK。原数值实现与门保持不变；旧6test源码72f7/测试255b已按准确SHA归档在 outputs同目录，旧测试回执不改写，新回归另存 ROUNDING_REGRESSION 文件。
