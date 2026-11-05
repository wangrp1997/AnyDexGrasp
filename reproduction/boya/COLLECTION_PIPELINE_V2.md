# Boya四型正式采集流水线v2：运行交接

更新：2026-11-06，Asia/Shanghai。运行根为 `outputs/collection40_20261106_v2/run_v2/`；本轮正式`all`入口已于北京时间2026-11-06 01:02:53启动，启动PID3322110。PID是启动快照，当前状态以`pipeline_progress.json`及各阶段收据为准。交接快照为50份RGB-D完成、50个原生资产结果已采用、共享CGR已完成50/50 forward，现处于geometry_running阶段；**尚未训练Boya评分头，也没有最终采集标签或成功率结论。**

## 固定范围与入口

协议为 `COLLECTION_PROTOCOL_20261106.md`；冻结选择仍为40train+10validation、50个唯一源网格组，名单SHA `487017083e79eb1323f243946ddf2b931dac6bd87d86f52632a254ba5226f60d`，与旧8个开发物体精确网格不重合。四型`[1,6,7,8]`、额外深入0/10/20/30mm，**800个独立slot：640训练、160验证**。有效正负标签目标400训练、80验证；null不补0，欠额及所有额外有效记录保留。原pilot/九型比较标签不抵扣本轮目标。

冻结运行协议：`run_v2/protocol.json`，SHA `20414374d11c0ed4df68b339e51a38318901bb3b34119cb7b127adcb3e820703`。入口为 `reproduction/boya/run_collection_pipeline_v2.py`，冻结源码SHA `82cb5deadbe82d6cbcded446a241d3244f1b3114ed11694a830d163ad3aab6d9`。旧v1源码、旧三型600槽dry-run及失败目录不覆盖。

每物体一次观测、一次原冻结CGR前向，供四型四深入使用；表示阶段已共享一次模型加载并完成50个forward。几何与臂准入后，每slot只执行首个完整准入候选一次，不在失败后重抓凑数据。执行为**两个对象CPU worker并发、对象内serial常驻、batch16**，不是GPU向量化物理。四个独立`480→20`评分头尚未训练；当前排序不是已学习的Boya手型选择器。

## 真实启动命令与动态库环境

以下是本轮已执行的启动命令；当前进程活跃时不重复启动同一run。`LAUNCH.json`记录准确argv、环境及协议/源码SHA。完整`LD_LIBRARY_PATH`沿用既有pilot环境，不能只加载Isaac/Python环境而漏掉native库目录：

```bash
cd /home/wangrenpeng/AnyDexGrasp
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/home/wangrenpeng/AnyDexGrasp/outputs/native_deps/os_libs:/home/wangrenpeng/AnyDexGrasp/outputs/native_deps/openblas/lib:/home/wangrenpeng/AnyDexGrasp/outputs/native_deps/cuda-13.0/lib64:/home/wangrenpeng/tendon-spin/.venv/lib/python3.12/site-packages/torch/lib"
/home/wangrenpeng/boya-inhand-wm/.venv/bin/python -B \
  /home/wangrenpeng/AnyDexGrasp/reproduction/boya/run_collection_pipeline_v2.py \
  --run-dir /home/wangrenpeng/AnyDexGrasp/outputs/collection40_20261106_v2/run_v2 \
  --phase all
```

上述短命令继承该run已冻结的配置；原生预处理引用 `outputs/collection40_20261106_v2/assets_preflight_v2`，不重新构建50个原生场景。新目录必须显式冻结自己的配置和来源，不能把现有`protocol.json`当可热改配置。源码、标定、物理配置和源资产不可在后台运行时修改。

## 进度与结果路径

以下路径均相对本轮`run_v2/`；尚未进入的阶段文件可能还不存在，不据此补标签。

| 内容 | 路径/含义 |
|---|---|
| 总进度与终止原因 | `pipeline_progress.json`；`launcher.log` |
| 阶段完成与复用SHA | `stages/{native_assets,observations,representation,geometry,planning,execution}.json` |
| 观测完成清单 | `observations/OBSERVATIONS.json`；逐对象`observation.json` |
| CGR共享前向进度 | `representation_stage/BATCH_REPRESENTATION.json`；`logs/representation.log` |
| 每slot几何状态 | `POOL_PREPARATION.json`、`pools/<object>/g<gesture>/d<index>/` |
| 臂准入与未执行null | `planning/queue_manifest.json`、逐slot`preparation.json` |
| 两对象实际派发与收尾 | `execution/PROGRESS.json` |
| 实际执行/步数/正负/null | `execution/RESULTS.json`；逐对象`execution/<object>/RESULTS.json` |
| 训练集唯一汇总入口 | `execution/train_records.jsonl`，仅train有效标签 |
| 独立验证标签 | `execution/validation_records.jsonl`，禁止训练/调参/选阈值 |
| 全量标签审计 | `execution/all_records.jsonl`，包含两split，仅供审计，不直接作为训练输入 |
| 配额、超额与欠额 | `collection_counts.json`，按split×gesture分别统计 |

每个实际记录还保留 `execution/<object>/trials/<output_identity>/{training_record.json,trial_binding.json,bound_evaluation.json,RESULTS.json}`；原生`evaluation.json`和`trace.npz`位置从执行回执的`execution_dir`读取。独立初态恢复后各次执行，不是连续同现场反馈重抓。验证网格组与全部姿态/手型/深入标签不得流入训练或超参数选择；需要调参只在40训练组内另划分。

## 负例、null与恢复边界

合格实际抬升成功为1，完整且可判定的合格实际失败为0。**普通有效0继续后续slot**。无几何候选/声明预算内无臂计划是未执行null，继续；零步、未完成、执行资格不成立或基础设施异常不造负标签。

明确基础设施异常、未处理错误或可能执行但无法确证的丢失进程会停止新派发，已启动对象收尾。保存partial状态、实际步数、未派发及未知执行列表；actual只计真实`physics_steps>0`。不能删除输出重启来掩盖失败，也不自动重放执行状态不明的slot。已完成阶段仅在收据与冻结SHA核验通过时复用。

## 已保留的零步失败与发布修复

第一次表示启动遗漏上述动态库路径，MinkowskiEngine所需运行库未找到，**模型加载0、forward0、物理0、未知执行0**。原log、进度、启动凭据及`RECOVERY.json`保留于 `attempts/representation_missing_runtime_libs_v1/`；恢复仅继承正确环境，原协议/manifest保持，50份观测复用，没有安装新包或修改共享环境。

原生50资产中，一份缓存`source_audit.json`的1MiB后尾部出现151,061个零字节；保留的原native probe字节匹配最初声明SHA。修复在**新缓存路径**按原probe字节发布并重读SHA/JSON校验，旧坏缓存、旧summary和原失败全部保留，未重跑原生预处理或改变20µm门。完整凭据在 `../assets_preflight_v2/publication_repair_20261106_v1/REPAIR_RECEIPT.json` 与 `ORIGINAL_MISMATCH.json`。修复后50对象/128资产文件核验无错误，原生资产为24 ready、26 repaired；这些是资产状态，不是抓取成功数。没有证据把尾部损坏归因于缓存键冲突或确定的存储机制。

`../assets_preflight_v2/SUMMARY.json`是当前发布入口，SHA `6fc3a3e5648d018c99ea6cb5a0fdcf917a35207a76e525d83921829388d7d86c`；正式run的`native_preprocessing/SUMMARY.json`引用这些源结果。规范化physics profile身份与配置文件字节SHA分别核验，不能混比。轻量来源/静态合同凭据在 `reproduction/boya/evidence/collection_20261106/`。当前只运行采集流水线，不自动进入训练。
