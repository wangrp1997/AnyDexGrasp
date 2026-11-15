# GPU 多环境提起：独立只读审计接口（2026-11-15）

本阶段新增 opt-in `author_gpu_vector_audit.py` 与 CPU 测试，没有修改旧 collector、原 numeric audit 或 CanonDex，也没有由该审计启动 Kit、积分、模型、规划或训练。GPU 执行适配由独立 `author_gpu_vector_execution.py` 负责，实际派发由 root 负责。本报告中的 CPU 通过不能证明真实 GPU 接触或物理标签已经通过。

## 实际接口及标签边界

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD/reproduction/boya:$PWD" \
/home/wangrenpeng/tendon-spin/.venv/bin/python -B \
  reproduction/boya/author_gpu_vector_audit.py \
  --execution /ABS/NEW_GPU_RUN/GPU_VECTOR_EXECUTION.json \
  --output /ABS/FRESH_AUDIT.json \
  --split-manifest /ABS/ORIGINAL_SPLIT.json
```

API 为 `audit_execution(group_path, split_manifest=None)`。`--output` 拒绝覆盖。split 参数可省略，此时仍核全部三体原 PLY/执行 OBJ 身份与源 SHA，但报告不会声称已核训练/内部划分。该诊断永远返回 `label=None`、`formal_training_eligible=False`、`exclude_training=True`；每个完整有效 lane 另有 `diagnostic_label=0|1`。不完整、基础设施异常、缺字段、SHA 变化或审计不变量失败均不能得到已认证的二元标签。CLI 保存失败及 traceback 后退出非零；没有将未执行或无计划填为负例的分支。

输入为执行器实际 schema `anydex_boya_author_gpu_vector_execution_v1`，输出为 `anydex_boya_gpu_vector_audit_v1`。只接已冻结原 READY/candidate/commands，核其 preparation/protocol/scene 及输入和源依赖；保持原 CGR480、固定首选、4cm、六动作阶段以及完整三体身份。GPU 后端变化显式保留为诊断，不能称 CPU/GPU 物理数值等价。

## 坐标、接触与数值审核

复用未修改的 `audit_author_collection_runtime.audit_runtime`，通过私有 Python module 实例在内存中做三项投影：该 lane 的 `/World/envs/env_i/...` 转回原 `/World/...`；SETUP.entry 的 XYZ 减去显式 origin；contact 的 XYZ 减去 origin。原 native/all-body trace 已由执行器还原为 plan frame，不能再次减 origin。所有其它字段保持，原文件不写回。临时符号链接仅适配原 audit 的固定文件名；SHA 表解析到真实输入路径，不保留失效临时路径。

完整 filtered6 接触日志逐 lane 只解压一次，重算每一三体的五指/双侧承载、非法臂支撑与穿透，使用原 mesh 的全部源顶点重算底面、5mm 离桌和最后 hold 参考下的 5cm 上升，原双侧各严格大于 1mN。保留完整正负和任一原桌面体成功规则，没有加入额外保持时间或读取旧 primary success。每条接触路径必须属于本 lane 或共享 ground；另一环境或遗留无 namespace 的 robot 路径硬拒绝。

24 条独立命令经原 XML 的 4 个 mimic 独立扩展，与保存的实际 28 目标逐值比较；并复核原六臂速度前馈及零手速度目标、实际 limits/mimic、finite、掌跟踪、基座稳定、时间与 phase。掌固定 attachment 由源 XML 独立重建。三体质量/摩擦/向下 0.98N 依据原 native readback/外力命令和 additional-materials 记录核验；这不是额外独立力传感。基座独立复算从第一记录 tick 的漂移，冷初态绝对偏差仍由已绑定原 native gate 提供。

初态和首 tick 原生 GPU snapshot 的完整 NPZ、CUDA 元数据、count/start、capacity、全部引用区间与 finite 都核对，允许未引用的 native 间隙。共享 PhysX scene、GPU dynamics/broadphase、250Hz、原 CCD=False、全部环境积分步/自身命令完成步/异长 tail 成本一致核对。TAIL_TRACE/TAIL_CONTACTS 的字节、完整 pointer/data SHA 与 size 绑定，tail 不改原 endpoint 标签；本审计不把 tail 另算新抓取。

边界：本轮主线使用完整原 filtered6 日志，并绑定运行时逐 tick 跨环境 contact view 的零接触结果；不声称每 tick raw7 未过滤原生 contact buffers 均已保存或独立解码。`decode_raw_contacts` 作为独立能力函数测试了 7-buffer 原 count/start 间隙、actor ID、容量与跨环境拒绝；不要求执行器为首轮接通另写一套 raw7 全程日志。官方接口没有承诺不同 sensor 区间不重叠，因此保留引用次数，不凭重叠断言不支持 GPU，也不去重。

## CPU 验证及证据

8 项生产形状 CPU 测试通过：双 lane 4m 平移完整正例及所有原文件 SHA 不变；完整负例仍仅诊断；跨环境/无 namespace 拒绝；非 primary 底面/外力篡改拒绝；28 目标和掌姿漂移拒绝；不完整/raw label/tail 成本拒绝；raw7 间隙、ID、容量和重复引用；保存 snapshot 的 CPU device 或容量篡改拒绝。所有 fixture 仅临时合成文件，没有模拟器积分。

轻量证据位于 `evidence/author_gpu_vector_audit_20261115_v1`，较早与最终开发日志留在 ignored `outputs/boya_author_gpu_vector_audit_20261115_v1`。源码与最终测试 SHA 由 EVIDENCE.json 冻结。本报告先交接口和 CPU 证据；实际 GPU 结果须另基于 root 派发后的只读产物记录，不补造通过或吞吐倍率。

## 本机官方复用接点

现有适配复用本机官方 `isaaclab.cloner.usd_replicate` 和 `filter_collisions`；`interactive_scene.py` 第 177 行附近已有 grid origins/ReplicateSession，第 573 行附近 reset_to 将相对 root pose 加 env_origins，第 687 行附近 get_state 做反变换。原 CanonDex `execute_template_arm_positive.py` 第 179 行附近也已提供 environment_origin_m 的机器人/物体观察还原，故审计沿原坐标定义接通。

本机官方普通 `Camera` 已支持多 prim 路径和批量 view（camera.py 第 548 行附近）；`tiled_camera.py` 开头已声明弃用并由普通 Camera 吸收优化，不能照旧接口另造相机框架。这次 GPU 执行诊断复用冻结观测，不新增相机采集，也尚未证明完整 capture→CGR→prepare→GPU 批量采集吞吐。
