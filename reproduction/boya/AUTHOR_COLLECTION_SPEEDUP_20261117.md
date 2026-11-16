# 采集提速：先纠正计时，再保留完整几何检查加速

用户要求参考作者的速度实际改进采集。本阶段保留原候选、g1预闭合映射、控制命令、250Hz、64次求解迭代、原碰撞几何、质量摩擦、完整逐点接触和提起判定，只改计算实现与独立计时。不启动正式扩采或训练，所有本阶段物理记录排除训练。

## 对原方法的参考与计时更正

[原论文](https://graspnet.net/anydexgrasp/assets/files/AnyDexGrasp.pdf)报告约0.5秒生成200个抓姿，另约20秒做CPU/Open3D碰撞检测；试错采集由真实机器人执行。这些是分段时间，不是作者完整一轮采集的固定耗时。作者没有我们新增的PhysX积分、逐步完整CAD观测和独立数值审核。

前阶段把`physics_step_update_s`主要解释成物理推进，归因过早。源码证据是原native generator在yield之前开始计时；外层`author_lift.execution_steps`在取得下一条native step事件之后执行上一帧的`runtime.after_step`。所以这个inclusive桶还包含外层观测、源网格变换和接触日志等工作，**不能当成纯PhysX或fetch耗时**。原报告本来保留了inclusive限制，本报告明确更正此前用户说明中的过度归因；历史计时、协议和原报告不回写。

新增`author_observer_partition.py`分别直接计时`sim.step`、完整observer、全顶点极值计算和JSON写入；后三项存在包含关系，不简单相加。新一轮完整验证结束后，以它的直接数据说明真实物理和额外检查各占多少。

## 官方Fabric建议的实际短对照

按[NVIDIA性能指南](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/physics-performance.html)和[Isaac Lab Fabric文档](https://isaac-sim.github.io/IsaacLab/main/source/concepts/physics-backends/physx.html#fabric)，独立比较CPU原USD回写与CPU Fabric直接状态通路。原runtime源码只读，Fabric版本只覆盖两个已匹配的建场表达式，原场景物理参数不变。两组各64个环境时钟步，前16步预热、后48步计时，不抓取、不出标签。

| 后48步合计 | 原USD | CPU Fabric |
|---|---:|---:|
| 直接`sim.step` | .712482s | .764302s |
| 其中`fetch_results` | .707143s | .757778s |
| 每步平均`sim.step` | 14.843ms | 15.923ms |
| 完整接触点 | 105216 | 105216 |

初始化实测状态、全部64帧关节/机器人link/物体位姿速度和全部有效接触字段逐值exact，独立审核PASS。Fabric对照没有得到加速（观察比值USD/Fabric=.9322），**未晋级采集入口**。该静止前缀不能认证运动、闭合或提起，也不能用14.8ms直接替代全程平均。

本阶段发现原生reset内部还调用两次`.004s update_simulation`，不在SimulationContext计数里。新计数钩子实际记录了每组2次，两组共128时钟步+4原生初始化=132积分步、2Kit，父wall275.701秒；无未知在途成本。新完整验证也保留单独原生初始化计数。历史旧报告只记录其原有口径，本阶段不凭新观测重写旧成本。

public目录`cpu_fabric_profile_20261117_v1`保留两组协议、精确源码归档、trace及完整变长接触buffer。线程数检查仅发现当前PhysX为8线程；后续因源码揭示inclusive桶混入观测，**没有再派发线程实验**。

## 已实现的全源几何计算优化

`author_fast_geometry_checks.py`用Numba执行全顶点/全三角面CPU循环，`fastmath=False`、不并行、不采样、不改网格。网格最高/最低点只需世界Z，不再每一tick创建完整XYZ世界坐标数组；仍检查每一个源顶点的有限值。独立射线仍遍历所有原三角面，保留原Möller–Trumbore容差、正向距离、地面和tie规则，避免96次大型临时数组分配。它不替换碰撞几何，也不改变物体运动。

0物理实测核对原两轮7376帧×每帧三体、共22128个源体姿态：所有源网格极值与旧记录最大差2.776e-17m，低于原1e-9m审核阈值；6个物体各32姿态与原函数配对计算，组件倍率4.98–6.22。两轮全部192条射线相对旧记录最大差4.441e-16m，原1e-10m门保持；另以原函数核对每轮4条射线的深度、hit和tie。组件倍率不能当成完整采集倍率。

对同一旧第二轮capture，用同一进程配置、cProfile、完整原门依次做完整审核：

| 完整capture审核 | 原版 | 新射线实现 |
|---|---:|---:|
| 总wall | 69.177s | 30.590s |
| 96条完整源三角面射线 | 43.308s | 4.566s |
| 全部750帧/3046112接触/2472原生凸体/RGB-D与来源检查 | PASS | PASS |

本次观察完整审核约2.26倍，射线部分约9.48倍；新版本预编译时间另外记录，未藏入上述组件计时。这是相同已保存输入的顺序对照，包含cProfile开销，不是随机化正式采集倍率。旧运行的404秒审核受当时运行条件等影响，不能拿它直接与本次30秒相除宣称13倍。两个完整审核都新增0Kit/0物理/0标签。

## 当前入口与有限完整验证

`author_fast_fixed_scene_collection.py`是独立opt-in入口。它沿用固定世界组织，仅替换全顶点极值计算和独立capture审核的射线实现；完整runtime auditor保持原版。CPU Fabric没有启用，CPU PhysX、原64迭代、几何和判定保持。

固定控制只派发manual index3的一轮，保留参考运行整个6物体银行、同一相机和原g1规则，重新拍照/推理/首选一次，不复用历史结果生成标签。不换候选或重抓。最多1Kit/8750环境时钟步，另显式最多4个原生初始化积分步；全部exclude_training。协议和精确源码归档在public `fast_geometry_world_control_20261117_v1`。

25项CPU检查通过（7个本阶段新检查+18个既有，2.691秒）；原代码、CanonDex和旧映射31/新映射2fit/旧435及八权重不改、不混数据。

## 完整实跑结果

单轮已complete，父进程从启动到独立执行审核结束共**589.350秒=9分49.350秒**。1Kit/1world/1camera/6银行体，750初始化+3818执行=4568环境时钟步，另原生启动积分2步，实际4570。原identity CGR载入1次/前向1次，评分头0、训练0。首选仍raw34/ff，提起026成功，原全部物理门、初始化审核和完整原版runtime auditor全部PASS。末源底101.800mm/升100.300mm、末连续受载1.608秒与旧对应场景相同；该持续时间只是诊断，不新增门。

核对旧对应第一轮与新一轮：真实RGB-D、全部751帧初始化位姿/速度、24列逐步指令/初值/掌姿/时间/phase和全部3818帧native trace的14个数组**逐值exact**。三体全程位姿、速度和载荷exact；源网格极值因运算顺序仅有最高2.776e-17m差。全部3818条接触JSONL的2,466,275,238逻辑字节SHA相同。这个固定控制没有通过换场景、换候选、改路径、降低物理或省接触制造速度差；也不能据单场推广所有场景的物理轨迹等价。

| 对应第一轮阶段 | 旧记录（秒） | 新记录（秒） |
|---|---:|---:|
| 建场、摆放、稳定、拍照和保存worker | 143.047 | 156.949 |
| 完整初始化审核 | 165.351 | 14.425 |
| CGR帧处理 | 4.701 | 3.363 |
| 碰撞、映射和机械臂准备 | 99.703 | 86.517 |
| 抓取worker（含收尾保存） | 311.405 | 252.916 |
| 完整执行独立审核，源码不改 | 80.360 | 69.846 |
| 对应阶段合计 | 804.566 | 584.016 |

新parent wall另包含约5.334秒启动/绑定/调度间隔。阶段合计观察约1.378倍；这是历史匹配控制，非随机化新配对，两次主机/缓存/重叠任务等条件不能证明完全一致。尤其原版审核本次配对测量69.2秒，明显低于旧404秒，说明不能把历史与当前的所有墙钟差都归因于改动。旧parent 2171.808秒涵盖**两轮**，本次589.350秒只涵盖**一轮**，严禁直接相除报3.69倍。执行器内部同口径wall为310.566→250.709秒，约缩短19.3%；其中纯源码改动的净收益仍应以更多受控对照确认。

## 物理与观测的直接分段

| 新运行的直接计时 | 秒 | 范围 |
|---|---:|---|
| `sim.step(render=False)` | 82.236 | 3818次真实推进，平均21.538ms/4ms物理tick |
| 完整observer | 95.998 | 3818次，包含后两项及全接触读取/解码、载荷/几何/早停检查 |
| 全源顶点极值计算 | 4.898 | 11457次，含ready的3次；每个顶点仍检查 |
| 完整接触JSON append | 27.324 | 3818条，包含在observer内 |

原native的inclusive桶新值180.127秒，与真实`sim.step`的82.236秒明显不同，直接证实前述计时归因必须更正。物理本身仍比实时慢约5.38倍；observer和2.47GB逻辑接触JSON处理也很重。当前改造已实测改善，但**尚未达到原作者实机的整体节奏**。后续应优先处理完整接触数据的解码/整理/存储通路和剩余准备成本，保留完整接触与原门；不能继续把全部时间归咎PhysX或盲目重复开关GPU。

## 可复用入口与收尾

通用入口为`author_accelerated_fixed_scene_collection.py`，无需调试历史或本次benchmark根，沿用原有限场景冻结方式；它与上述完整验证使用相同worker、observer和审核实现。额外0Kit静态冻结验证通过：两轮/6银行体/1Kit上限，capture→execute顺序不变；mock GPU身份fixture不能通过实际run的GPU身份检查。该入口仍是exclude_training的有限开发采集，不等同于正式训练配额。

```bash
PYTHONPATH=reproduction/boya /home/wangrenpeng/tendon-spin/.venv/bin/python -B \
  reproduction/boya/author_accelerated_fixed_scene_collection.py freeze \
  --source-batch /mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/manual_start_20261116_v2/batch \
  --geometry-template /mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/contact_pilot_20261116_v1/scenes/00/g1_contact/pipeline/READY.json \
  --root /mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/accelerated_user_v1 \
  --queue-indices 3 4

PYTHONPATH=reproduction/boya /home/wangrenpeng/tendon-spin/.venv/bin/python -B \
  reproduction/boya/author_accelerated_fixed_scene_collection.py run \
  --root /mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/accelerated_user_v1
```

本阶段共3Kit、4696环境时钟步+6个原生初始化积分=4702实际积分步；0未知成本/0抓取重试，只有1次完整诊断正例，新增正式fit0/训练0。两个物理控制根的source/input/archive绑定不变、自有进程组全退、4当前预约released、driver/slot/global锁可取、全局registry released，无后台任务。`HOST_FINAL_AUDIT.json`通过。

轻证据为`evidence/author_collection_speedup_20261117_v1/EVIDENCE.json`及`SUMMARY.json`；包含原失败路线的无加速结果、0物理完整审核对照、全源几何核对、完整新控制结果与直接计时。所有source mesh/原始接触/轨迹/RGB-D/模型继续public或ignored outputs保存，不进Git。正式扩采/训练/实机驱动均没有启动。
