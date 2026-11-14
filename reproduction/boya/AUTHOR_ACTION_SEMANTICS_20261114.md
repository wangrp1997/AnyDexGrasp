# Boya动作语义与旧实际指令核验（2026-11-14）

**四型动作的计算合同已实际核验，供新采集入口绑定使用；物理接触、同场多物体搬运及投放仍须新流程验证。** 本轮没有新物理、训练、CGR前向或标签；不改变旧标定、闭合端点和成功率。

新增[`author_action_semantics.py`](author_action_semantics.py)直接读取原URDF、公司姿态与pad源三角面，用独立XML/NumPy FK验证。没有导入CanonDex控制器或仿真器。`ActionContract.for_slot(slot_id)`给出可哈希JSON合同；`closure_targets`生成全部20手DOF；`validate_command_arrays`核验实际指令，拒绝端点、非参与手指、mimic和参考坐标被改写。源DH3采集/部署的翻面函数仅AST提取执行，明确区分采集翻第0/1列、部署翻第1/2列；采集翻面改变接近轴，不能以旧部署翻面替代。

## 已实际核验的内容

- 四型全部363个pad配对槽：16独立电机＋4原生mimic；公司闭合端点、33档开手状态、腕坐标、参考宽度/中点/接近轴均独立重建。132个ff槽输出完整合同。非参与手指保持公司姿态，所有21碰撞link保留。
- 五个pad落在各自原STL三角面上，最近距离最大`3.5766e-18m`，标注法向与对应最近三角面法向一致。这证明源几何对应，不代表人体标注、物体受载接触或硬件标定已验收；源标注状态仍为`candidate_pending_human_review`。
- 四型101个同步自由闭合命令采样下，拇指与食指pad距离均单调减小，两侧均向最初对侧方向运动。Boya是非对称手指路径；不能将其当平行夹爪或强行固定接触锚点。
- 固定57池全部历史物理执行：`2,394,000`个20手DOF目标标量与合同逐值一致；实际记录的`113,263`步轨迹与下发命令对应。57条抓取掌位绑定误差0，参考接近方向分量误差最大`3.33e-16`；实际关节mimic残差最大`6.28829e-5rad`。这是读取历史执行，没有重跑物理，也没有重算成功标签。

| 手型 | 参与手指 | 开→闭pad间距(mm) | 自由闭合中点变化(mm) | 参考接近轴变化(度) |
|---|---|---:|---:|---:|
| g1 | 拇/食/中/无名/小 | 135.840→29.214 | 42.403 | 19.864 |
| g6 | 拇/食 | 130.518→10.024 | 46.318 | 42.684 |
| g7 | 拇/食/中 | 132.063→13.337 | 50.742 | 19.406 |
| g8 | 拇/食/中/无名/小 | 140.858→37.431 | 33.880 | 110.066 |

这些是同一固定手掌坐标中的自由FK参考变化，不是物体/实际手掌漂移，也没有证明g8接线错误或是失败原因。当前合同保留原计算映射，未针对结果添加闭合锚点、平移或筛选阈值。

## 新发现与未通过的门

把57个已执行掌位固定，对原始全手三角面在11个自由闭合waypoint计算桌面最低点，6条出现负高度，最低范围`−1.571～−14.083mm`。其中包含E0的**两条旧成功**。这说明只检查开手并不能说明自由闭合全过程都离桌；物体和桌面接触会改变实际手指状态，离散自由FK更不能替代动态碰撞或证明失败原因。它不是连续扫掠证书，不是新增筛选规则，不自动排除这6条或更改原作者的开手碰撞算法。

机器回执分别记录`numerical_action_contract_pass=true`、`physical_semantics_validated=false`、`hardware_semantics_validated=false`。剩余证据须来自新入口：实际采集flip与canonical特征/原始候选绑定、观测到执行的全部物体初态一致、动态闭合/非参与手指和桌面接触、接近轴后退及搬运松手后的投放结果。已核验计算语义可以用于实现这一路径，不能据此报告搬运已成功或启动外部成绩测试。

## 调用与证据

```python
contract = ActionContract(calibration_path)
bound = contract.for_slot(slot_id)  # JSON安全，含contract_sha256和未通过的物理门
targets = contract.closure_targets(slot_id, thumb_progress, fingers_progress)
checked = contract.validate_command_arrays(arrays, slot_id=slot_id)
```

省略`fingers_progress`时两侧同步；返回的20列顺序由`joint_names`明确列出。`motor_targets=N×16`允许新执行器自定义阶段，但仍须在原两侧闭合路径内，不能用传入任意期望数组绕过动作合同。候选/掌位核验可显式传入实际翻面后的`candidate_row15`、`world_T_camera`及独立`extra_depth_m`；容差覆盖原碰撞器float32深入位移，不重复加深入。

正式输出：`outputs/boya_author_action_semantics_20261114_v1/run_v3/`，审计内部`5.857s`，7tests/13显式子测试通过。首次准备漏掉package中的palm组过滤，在几何准备时`KeyError`，零新物理/训练，失败回执保留；修复后run_v2完成，再补参考变异和任意显式目标绕过的拒绝检查，run_v3完成。run_v2绑定当时实现的SHA，不回写其报告；run_v3为当前正式版本。两个完成审计分别6.478s、5.857s，不能把末次耗时当全部开发成本。

轻量结果见[`AUDIT_SUMMARY.json`](evidence/author_action_semantics_20261114/AUDIT_SUMMARY.json)、[`INPUT_BINDINGS.json`](evidence/author_action_semantics_20261114/INPUT_BINDINGS.json)、[`TEST_RECEIPT.json`](evidence/author_action_semantics_20261114/TEST_RECEIPT.json)。完整合同和逐条旧轨迹仍在忽略输出中。原435/八权重/历史结果及CanonDex均保持；整体复现尚未验收。
