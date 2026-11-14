# 固定投放工具完整坐标修复

首次真实生成在 `outputs/boya_author_pipeline_home_release_20261114_v1` 失败：PlanningModel.full_state 要求全部24个独立坐标，旧工具仅提供6臂关节和2手腕，漏16个手指电机。该 exit1、源码快照、失败日志及未生成 sidecar 的事实全部保留；0物理步。此前13 tests 的 home FK 使用 mock，未覆盖这项真实接口要求。

修复仅限 `author_release.py`：先为24个 active 坐标置零，再覆原参考6臂 home 和两手腕零，经原 full_state 校验并派生4个 distal mimic，得到完整28坐标。sidecar 明确记录独立/展开坐标数与语义：**手指零只是完整 FK 输入，不代表实际开手或执行命令。** 这不改变固定 world XYZ `[-.4,-.25,.35]` 或 home palm 旋转适配规则；transport 源码、旧 sidecar/scene/队列和证据均未改。

14 tests 通过，新增严格检查全部24独立输入及28坐标派生的回归。另对真实 `backend.collection.json` 和 v4 `capture/observed_scene.json` 完成一次 fresh dry sidecar 生成，耗时0.392秒。独立 XML/NumPy 实现从 robot URDF 根逐 joint 计算 palm，和生成工具的 home FK 最大矩阵误差0；改变全部手指坐标后 palm 仍完全相同，因为其祖先链仅含6臂关节和2手腕。release 旋转与此 home FK 相同，固定XYZ逐值相等。

实际 dry 输出在 `outputs/boya_author_release_complete_fk_20261114_v2`，轻量证据在 `evidence/author_fixed_release_fk_20261114_v2`。本次新增1次真实 sidecar 生成、0 native规划、0 Kit/物理/CGR/训练/标签，不表示可达或抓取验收通过。collector 使用新工具快照另生成正式 sidecar，首次失败结果不覆盖。
