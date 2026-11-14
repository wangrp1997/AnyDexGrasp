# S40 同组杂乱场景队列：纯构建交付

已新增 `author_collection_scenes.py`，旧场景构建器不改。开发队列按 seed `2026111401` 固定为前8个 fit、前2个 internal 场景，共10个槽；7个初始布局可构建，3个为 `placement_null`。这不是抓取结果，本组件实际新增物理步、观测、CGR前向、规划、标签和训练均为0。

公共队列为 `/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development/scene_queue/QUEUE.json`，SHA256 `971e661ca0b2be4d06495fc3e27596353f330c1e3736c77abda19cbc56fe6eb3`。运行源码冻结 SHA256 `66994aa03dcf22658a0eeb74e58a593973bd32f3eae6a7002590593911787815`。后续消费队列须保持源码和全部输入绑定；不覆盖旧场景、标签或回执。

每个场景在原32 fit／8 internal 物体组内无放回均匀抽3个不同物体，独立均匀世界Z轴 yaw 乘原静置方向。完整旋转后源视觉与碰撞代理的并集 XY AABB 位于 x `[-.95,-.58]`、y `[-.02,.23]` 米，任意两物体在至少一个XY轴上分离不少于3mm。实际渲染源网格最低点初始离桌5mm；代理不得穿桌。每物体最多256次XY采样，固定物体与 yaw，不回溯已有布局。失败场景留在原位置，不换物体、不额外采样、不生成负标签。独立场景使用独立 SeedSequence，前序失败不改变后续物体或随机流。

场景仍复用原750步／0.004秒／0.1kg／0.6摩擦的共同初始化及原相机。无箱、投放、释放或额外提起保持。实际静置、接触流、native20µm、观测及执行仍必须在后续逐场景通过，初始布局通过不代表这些门已通过。源域保持 x `[-1,-.57]`、y `[-.04,.24]`、z `[-.02,.30]`，由采集入口独立绑定。

CATALOG 校验全部40个S40身份、418个源凸体、40个已有 native 预检、原32＋8物体划分、785个输入文件和520个源码。原视觉身份 `source_visual_mesh_sha256`／`mesh_group_sha256` 与转换后的实际几何 `source_mesh_sha256` 分开保存；整场景保存 `dataset_split`、3体原身份列表及 `all_source_mesh_sha256`。不读取历史抓取标签，不混旧435或EGAD，不声称等于作者L144 CAD。

可运行接口：

```sh
/home/wangrenpeng/tendon-spin/.venv/bin/python -B reproduction/boya/author_collection_scenes.py generate \
  --source-manifest outputs/s40_stable_supplement_20261107_v2/settling/objects.json \
  --native-summary outputs/s40_stable_supplement_20261107_v2/collection/native_preprocessing/SUMMARY.json \
  --backend reproduction/boya/backend.collection.json \
  --split-manifest outputs/s40_collection_20261107_v1/protocol.json \
  --output-dir ABSOLUTE_NEW_DIRECTORY --seed 2026111401 \
  --fit-scene-count 8 --internal-scene-count 2

/home/wangrenpeng/tendon-spin/.venv/bin/python -B reproduction/boya/author_collection_scenes.py build \
  --scene-spec FROZEN_SPEC.json --output-dir ABSOLUTE_NEW_SCENE_DIRECTORY
```

`generate_queue(...)` 先冻结协议和有限场景预算，再生成场景规格；`build_scene(...)` 全部重新核SHA及确定性布局，输出兼容现有 capture 的 `scene.json`，拒绝 terminal null 和任何布局篡改。任意绝对输出目录均可，但写入权限由运行环境负责。正式队列通过显式 counts 复用相同实现；正式 scene 预算尚待开发成本／产出核定，不按实际正例数量追加预算。每型1000 fit＋200 internal 是容量目标，不是已有数据；4型共4000 fit 尚未产生。

6个必要单测通过。独立审核重建7个可构建场景的8,337,148个源与代理顶点，核对完整AABB、源底、yaw、3体间隔、分组和SHA，并实际构建全部7个输入；未启动模拟器。轻证据在 `evidence/author_collection_scenes_20261114/`，完整构建结果在 `outputs/boya_author_collection_scenes_20261114_v2/independent_public_queue_audit/`。首个开发版本在生成场景前因误比较文件SHA与原生canonical物理配置SHA退出，0随机场景／物理／标签，原失败回执保留于 `outputs/boya_author_collection_scenes_20261114_v1/FAILURE.json`；修复后冻结新v2，没有改采样策略或选择结果。

## 首场景实际初始化的独立只读核验

主进程随后运行固定前缀第1场景；独立审计未启动或推进模拟器。`CAPTURE_PREFIX1_AUDIT.json` 核全750帧、2,274,278条接触与gzip／完整解压JSONL双SHA，三个物体最后250／250步都有正地面支撑。1648个native凸体的最大源外差19.831µm，原门通过；原相机96条first-hit射线最大深度差0.328µm，观测和RGBD文件绑定通过，渲染0物理步。实际Vulkan日志只有GPU0为 Active。

原源网格全751帧最低点重建，017／046／011初态均为5mm，末态分别0.482／1.543／0.344mm，全部过程源最低点仍在桌面上方；静置指标独立复算一致。本场景实际750初始化ticks／2250对象步与0试抓分开计数。四型原首选均停在纯准备，不能据此报告抓取成功率或产生负例。

初版审计直接沿用NPZ float32复算静置，原函数会先转float64，导致过严数值对照失败；失败源码与回执保留，修正审计精度后对同一已冻结产物重读通过，0物理重跑。轻量更正记录为 `CAPTURE_AUDIT_PRECISION_ERRATUM.json`，未改运行源码、原日志或场景结果。
