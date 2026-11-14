# 提起任务新入口：实现与独立接口核验

本阶段完成 opt-in `author_lift_pipeline.py` 和 collector 的显式 lift task 支持，尚未以真实新场景跑通闭环。本报告仅计入口开发/测试/只读审阅：0 新 CGR、0 实际规划、0 Kit、0 物理步、0 新标签、0 训练。root 另行调度的新无箱观测和后续实验不能计作这些合成测试的结果。

任务固定为 `declared_boya_clutter_lift_v1`，scene schema 为 `anydex_boya_author_lift_scene_v1` 且 `bin=None`。成功按每个原桌面体分别计算，任一合格即成功：完整原生 lift、源网格底面末态离桌至少 5 mm、相对最后 hold 参考高度至少升高 5 cm、拇指和其它手指合计载荷各大于 1 mN，并通过原七项及全物体有效性门。连续承载时间仅作诊断，无新增末尾 0.5 s hold。不读取旧 primary-only `task_success` 代替逐体结果。

## 固定流程与边界

`prepare` 对单个冻结 RGB-D/CGR 缓存执行原 top500、固定 DH3 reference type、原 source 单 flip、随机额外深入、完整原 collection 碰撞类，再只消费原规则的第一候选。Boya 型映射为 g1/g6/g7/g8→1/2/3/4。保留未经过旧手/桌面预门的 raw17/2261 特征、canonical480、实际翻转姿态与额外深入一次绑定。候选选择不读物体 GT 分割，scene compatibility primary 只用于原 native 数据格式。

随后原 native 只准备 4 cm approach；完整计划即使旧直线 HOME 被拒，也必须交给现有 HOME driver。driver 只运行一次固定 RRT；仅允许已知导数失败且全场景几何清时，沿同一条路径确定性 retime 一次。其余 null/infra 不追加候选、搜索或预算。HOME 之后调用 `author_lift_plan.prepare_lift`，原 250 close＋50 hold＋10 cm world-Z lift，无箱、投放或 release。前缀四个数组、原始坐标名称/初态逐 shape/dtype/bytes 核验，元数据封装也不改实际 24 列命令。

旧 `author_pipeline.py` 不改；旧 transport task、协议、失败和标签定义保留。`author_collect.py` 默认仍为 transport，lift 必须显式指定新 task。修改前 collector 归档 SHA 为 `9d93f72307b80ef9ea4202dd8eb95ccbfc99b400dac11ce79e599c3318ed4d41`，本地归档在 `outputs/boya_author_lift_entry_development_20261114_v1/AUTHOR_COLLECT_TRANSPORT_V1.py`；对应修改前源码也在 git 提交 `26eddf68e1d00e51ab461d1b48bead925454e67b` 中。

## 执行和标签绑定

`execute` 只消费该入口的 `READY.json`，独立 `EXECUTION_STARTED.json` 排他落盘后即消耗唯一一次启动机会，基础设施启动失败也不能悄悄重试。输入、当前源码、显式 capture 源归档、scene、候选、命令、command binding、plan dependencies，以及 path/lift geometry 本体均在执行前后复核；所有后验 SHA 检查先于 `collect_execution` 和 `COLLECTION_RECORD.json`。

command binding 的 `execution_spec.original_object_ids` 必须与 scene 全体 ID 精确一致。collector 拒绝缺失/重复/额外物体、非 bool 结果、任务错配或 top-level any 与逐体结果不一致。无计划/不完整/infra 都保留 null，不填负例；完整、有效且所有物体均失败才是负例。标签和成本回执在 `SimulationApp.close` 前落盘，避免 close 的进程退出吞掉结果。

canonical arm repair 按原配置精确复制到真实 bound native 的父目录，绑定原文件/副本、combined URDF、physics 和全部 37 个源 robot mesh。此步骤只交付原物理配置，不更改几何或参数。

## 独立后端审阅

已只读核 `author_lift.py`、其继承的 `author_transport` 与 CanonDex 原 native executor。原 generator 在每个 step 前 yield；caller 只积分一次，恢复 generator 完成原 robot/compatibility body 更新后，再读取其余 bodies 和全接触。`after_step` 的停机使用 `PhysicalStop` 异常，接口不依赖返回值。finish 同时要求完整原七门、全物体穿透/非法臂支撑、phase 顺序、实际步数一致、无错误和无物理早停；只在这些有效性门后标记逐体成功和 label eligibility。

无箱 scene 在 capture 和 runtime 都通过 `bin_boxes(None)` 生成 0 箱体几何。执行沿现有 `_launch_native` 明确使用 CPU physics；入口未新增 renderer active-GPU/cuda0 选项。CGR 表征的 `--device cuda:0` 是单独配置。该审阅没有实际积分，不证明接触动力学成功率。

## 验证与源码冻结

V1 合并入口/collector/旧 pipeline 回归共 41 tests，1.427 s 全过。新增 geometry 本体前后 SHA 检查后，仅重跑受影响的 lift 入口 11 tests，0.124 s 全过；其中新增 path/lift geometry 各执行前/后共 4 个篡改子例。mock 的 23 steps 仅用于成本保留测试，不是真实物理步。CLI help 和 owned diff whitespace 检查通过。

首次只读 hash shell 使用不存在的裸 `python`，exit127、0 source reads/规划/物理；改用声明的绝对虚拟环境解释器完成。该失败另存开发目录，不覆盖测试日志。完整日志及轻证据见 `evidence/author_lift_pipeline_20261114_v1`。

| 文件 | 冻结 SHA256 |
| --- | --- |
| author_lift_pipeline.py | 7ce962e8b18d821fc8871d7721d79b3f3ac0d5ddafc302337e7fce015df7254a |
| author_collect.py | cd978c77007a537e29571e65f8f3fd0c8574c61b44fab62975732d39daa23b8d |
| test_author_lift_pipeline.py | bd386e1dc4452ad7768975b1187e70e8b131a6610e0a529b07bd77231ab1ca75 |
| author_lift.py | 40e519d035c45bd25289fe5dcb154808baef1153e82c62e0cf69b5110b47b20d |
| author_lift_plan.py | ee5d2257c3c0e589897d96d545479185317207a5a0693237832f3552c5a4224f |
| author_scene.py | c5a7cb962bb3d38961d31cc6a70a4a2c748a905d41b9bcc06e6df345387e7fe5 |
| author_transport.py | d5c5c186aa5a49b784e0d224b03ce42d1490b9ee5f7d5f10a6df224c6d9cb6e2 |

## 调用方式

API：`prepare(scene, representation_dir, observation, gesture_id, source_type, seed, calibration, cloud_manifest, backend, output, *, archive_sources=None, domain=None)`；`execute(prepared_dir)`。API 和 CLI 均不会自动派发正式扩采队列或训练。

以下是 root 调度开发闭环的最短命令示例，使用独立新目录和已就绪的 native/CGR 运行环境。本报告未执行这些命令。`prepare` 不启动物理；最后一条 `execute` 仅在新完整几何准备 ready 后按 root 的独立冻结协议运行一次。

```bash
export BOYA_PY=/home/wangrenpeng/tendon-spin/.venv/bin/python
export BOYA_LIFT_ROOT="$PWD/outputs/boya_full_data_lift_20261114_v1"
export PYTHONPATH="$PWD/reproduction/boya:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/outputs/native_deps/os_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$BOYA_PY" -B reproduction/boya/author_scene.py build \
  --source-manifest outputs/s40_stable_supplement_20261107_v2/settling/objects.json \
  --native-summary outputs/s40_stable_supplement_20261107_v2/collection/native_preprocessing/SUMMARY.json \
  --backend reproduction/boya/backend.collection.json \
  --task lift --output-dir "$BOYA_LIFT_ROOT/scene"
"$BOYA_PY" -B reproduction/boya/author_scene.py capture \
  --scene "$BOYA_LIFT_ROOT/scene/scene.json" --output-dir "$BOYA_LIFT_ROOT/scene/capture"

"$BOYA_PY" -B - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['BOYA_LIFT_ROOT'])
obs = root/'scene/capture/observation.json'
manifest = {'objects': [{'object_id': 'development_three_lift_20261114',
                         'observation_json': str(obs),
                         'observation_sha256': hashlib.sha256(obs.read_bytes()).hexdigest()}]}
with (root/'REPRESENTATION_MANIFEST.json').open('x') as stream:
    json.dump(manifest, stream, indent=2)
    stream.write('\n')
PY
"$BOYA_PY" -B reproduction/boya/run_collection_representation.py \
  --manifest "$BOYA_LIFT_ROOT/REPRESENTATION_MANIFEST.json" \
  --run-dir "$BOYA_LIFT_ROOT/representation" --device cuda:0 \
  --seed 20261114 --depth-min .2 --depth-max .65

"$BOYA_PY" -B reproduction/boya/author_lift_pipeline.py prepare \
  --scene "$BOYA_LIFT_ROOT/scene/capture/observed_scene.json" \
  --observation "$BOYA_LIFT_ROOT/scene/capture/observation.json" \
  --representation-dir "$BOYA_LIFT_ROOT/representation/representations/development_three_lift_20261114" \
  --gesture-id 6 --source-type 2 --seed 20261114 \
  --calibration outputs/boya_adapter_preparation_20261105_v2/calibration.json \
  --cloud-manifest outputs/boya_full_alignment_20261114_v2/hand_clouds/MANIFEST.json \
  --backend reproduction/boya/backend.collection.json --output-dir "$BOYA_LIFT_ROOT/prepare_g6"

"$BOYA_PY" -B reproduction/boya/author_lift_pipeline.py execute \
  --prepared-dir "$BOYA_LIFT_ROOT/prepare_g6"
```

其它三型只替换为 `(gesture-id,source-type)=(1,1)/(7,3)/(8,4)` 并用独立输出目录。同一观测只需一遍 CGR，不能把四型候选数或点云采样数计成训练样本。每型 1000 fit、四型 4000 fit 以及额外 internal 仍为扩采目标，本入口实现不代表该数据已采得。
