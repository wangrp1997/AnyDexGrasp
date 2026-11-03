十二场景最终只读分析：8次有效举升，1次实际未提起，3场景原碰撞后无候选。没有“已举升但仅数值无效”的例子。

| 场景 | 原手型/宽度 | 原碰撞准入 | 最终COM升高mm | gear rad / peak速度rad/s | 最终分类 |
|---|---|---:|---:|---|---|
|apple_yaw000|pose4 / 63.000mm|169/400|96.089|0.001909 / 1.079006|valid_lift|
|apple_yaw090|pose4 / 63.000mm|154/400|93.126|0.003768 / 0.952414|valid_lift|
|banana_yaw000|pose3 / 82.698mm|11/221|-0.210|0.006599 / 1.743813|not_lifted|
|banana_yaw090|pose4 / 63.000mm|53/400|85.832|0.001708 / 1.133196|valid_lift|
|bottle_yaw000|pose4 / 63.000mm|270/346|91.987|0.004864 / 1.307127|valid_lift|
|bottle_yaw090|pose4 / 63.000mm|222/400|98.912|0.007049 / 1.385123|valid_lift|
|bowl_yaw000|—|0/83|—|—|no_candidate_after_original_collision|
|bowl_yaw090|—|0/88|—|—|no_candidate_after_original_collision|
|foam_brick_yaw000|pose4 / 63.000mm|100/400|89.115|0.004591 / 1.232235|valid_lift|
|foam_brick_yaw090|pose4 / 63.000mm|92/400|77.294|0.009354 / 1.091680|valid_lift|
|mug_yaw000|pose3 / 99.000mm|1/29|100.261|0.002591 / 1.031608|valid_lift|
|mug_yaw090|—|0/48|—|—|no_candidate_after_original_collision|

- banana_yaw000 is a physical non-lift AND a separate velocity-gate violation: maximum COM rise is only5.143mm, final-0.210mm, not a lifted trial rejected only by a numerical label. No hand-table contacts; the recorded dominant positive contact links/timing describe acquisition and release, but do not prove force closure or isolate controller versus grasp geometry causality.
- banana_yaw000 close/hold has positive object contact only on finger2_tip; common flexion master stops near0.394rad versus1.333rad command. Finger3_tip joins only6.596s and finger1_tip7.026s during lift, all positive object contacts end7.304s. This is consistent with unilateral first contact blocking this declared rigid shared-actuator model before multi-finger acquisition; it does not isolate original-network, collision geometry, firmware or compliance causality.
- banana_yaw090 uses a different original selected top1 (pose4,width63mm) than yaw000(pose3,width82.698mm); it is not an equivariant rotation-only replay. Both approaches are mainly downward; successful yaw090 does not by itself prove a pose-mode intervention would fix yaw000.
- banana_yaw090 has no positive hand-object contact before lift; its finger-table contact precedes three fingertip object contacts at6.888/6.893/6.960s, which persist to10.25s. Its accepted lift is dynamic acquisition during scripted lift, not evidence that it had a stable prelift enclosure.
- mug_yaw090,bowl_yaw000,bowl_yaw090 fail admission after48/83/88 hand candidates and the original zero-overlap hand-approach voxel gate. No physical action occurred, so hand contact/torque/lift diagnostics are not available for them. Final collision receipts override the earlier hand-stage informational collision_assets_missing status.
- Negative native world AABB minimumZ for banana is not a mesh penetration measurement: compound broadphase boxes are conservative after principal-frame rotation; the earlier zero-step diagnosis established this. COM lift and actual hand contacts are reported separately.
- Source triangle visuals are retained; five objects use128-part CoACD compounds, cup/bowl sparse probes preserve cavities. Apple uses one convex VHACD component; sampled source/compound disagreement reaches about9.25mm, so its two valid simulation lifts have a coarse collision-model limitation.
- No repeated physical attempts, candidate reselection, force changes, timestep refinement, learning or new network inference were performed for this analysis.

9次真实试抓合计177120步/92.25s物理时间；本次分析零新增物理/前向/渲染。详细phase/contact/峰值时刻与源SHA见同名JSON。
