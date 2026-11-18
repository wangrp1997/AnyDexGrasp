# 物体间穿透只记录的抓取判定（2026-11-17）

用户最新要求：“物体穿透应该没事吧只要能抓取起来就行？”据此新增独立`anydex_boya_object_overlap_diagnostic_lift_v1`，物体—物体的小幅穿透改为诊断项，不再由原1mm门单独否决。旧协议、原七门、原失败、完整接触及标签全部保留，未来协议明确绑定新规则，不对旧根热替换。

实现：[author_mujoco_lift_policy.py](author_mujoco_lift_policy.py)。当前MuJoCo采集入口已在冻结协议中保存该规则，执行结果分别保存`gates`（原门）、`acceptance_gates`（新门）、`observed_lift_task_success`和新判定；独立审核也复算这两套结果。

成功的核心仍是原真实提起/受载规则：任一原物体在lift末源底离桌至少5mm，相对hold末参考升高至少50mm，拇指及其他手指合计均真实受载超过0.001N。不要求五指齐触，不新增持续受载秒数。单独升高但没有手部承载不会变为成功。

仅两端均为场景原物体且身份不同的接触适用豁免；手—物、物—桌及未知接触仍按原规则记录和审核，其他数值门保持。原5mm深碰撞停止保持，分类不改变碰撞参数、控制或原始点。此改动只改变指定接触类别的准入，不代表物理后端等价认证或正式扩采授权。

0积分只读重判已保存fresh_v3/raw87/ff：物体间最大1.538419mm已不单独否决；其他接触最大1.224830mm，原剩余穿透门仍未过。更关键的是原轨迹没有提起任何物体，故`observed_lift_task_success=false`，新规则下仍未成功。没有因改规则将旧失败改写为成功，正式新增训练标签0。

6项新增CPU测试通过：仅物体间接触豁免、手/地面/未知接触不误分、无真实拇指载荷不判成功、未提起不判成功、其他门/原深碰撞停止保留、非有限接触拒绝；累计不同CPU用例44。新版两轮静态freeze通过并标记禁止实跑，0新物理/CGR/训练，不重抓。

轻证据：[SUMMARY.json](evidence/author_mujoco_object_overlap_policy_20261117_v1/SUMMARY.json)。只读重判入口：[rejudge_author_mujoco_lift.py](rejudge_author_mujoco_lift.py)。之前[全流程报告](AUTHOR_MUJOCO_FRESH_COLLECTION_20261117.md)保留其原判定语义。
