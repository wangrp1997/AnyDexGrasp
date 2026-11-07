# AnyDexGrasp 独立复现工作区

## 2026-11-08 最新：公开训练/映射/闭合对齐诊断完成，原MSE换姿E0成功但整体仍未验收

用户授权对齐。原训练train_one_epoch与code100四头各2真实batch，初始化/前向/全梯度/Adam/BN/LR严格相同；435条canonical480/标签索引通过。8+8临时CPU更新无保存checkpoint，旧模型不改。公开训练默认240 vs 发布20、best保存vs固定100及真实手采样差异明确保留，不能称作者完整训练史已复原。Allegro公开闭合实际是10位置增量/1s+.2s保持，未使用tactile/torque反馈；此前笼统归因缺少作者触觉不准确。三件映射/flip/深入/闭合端点未发现接线错。

新独立proxy控制v2：A0粗/细正对照均成功+102.543/+104.478mm，三旧失败A3/A6/E0细代理仍0/3；5实际10500步/55.285s，视觉字节/初态/所有指令不变，代理仍非精确CAD。初v1漏已有EULA环境五路零Kit/零步退出，原日志保留、v2只修启动环境。十级闭合时间控制A0成功+105.768mm、三旧失败仍0/3，4实际8400步/35.073s，仅240close行变化；仍以冻结开手起点，非完整Allegro硬件等价。

新增严格原MSE评分入口evaluate_boya_original_recipe.py，只接受旧code100原final-ReLU MSE四头，拒绝preactivation。四已知场景复用缓存CGR，先冻结排名后几何，firstgeometry各一次受控腕物理：A0rank1/g6短暂离桌后失载，A3rank42/g1接近1175步推物早停，A6rank19/g7完整失败，E0rank0/g6成功+110.990mm/底74.094mm/持续双侧1.216s。1/4、7475步/38.956s，E0该输出fit17条=13正4负，不是无监督输出。评分/选姿会改变成败，但未证明稳定改善或唯一根因。

共13实际/26375步，原外部1/12和旧三件0/3不改，435和全部权重不变。13段正常速度RTX完成、零回放积分，统一outputs/boya_alignment_20261108_v1/index.html；7合同tests、训练数值及独立评分/实际指令/源承载/SHA/媒体核验通过。报告reproduction/boya/BOYA_PROTOCOL_ALIGNMENT_RESULTS_20261108.md，证据evidence/boya_protocol_alignment_20261108。新入口均opt-in、未晋级默认，CanonDex只读；整体论文复现未验收。本轮已收尾，无后台物理/训练，不自行重抓或扩样。只push相关轻量源码/证据，所有大输出不提交。


## 2026-11-08 最新：三件受控腕部诊断完成，0/3，勿继续归因机械臂

用户授权按计划隔离臂限制，独立outputs/egad_floating_three_20261108_v1已complete，protocol SHA832865f29b7ceb1500c73d5da819e33f08210b8408c62a95252c86ce425e776c。A6原firstgeo rank17、A3/E0原实际rank58/4，各一次2100步，共6300步，CPU三路28.610s；全部七门/native20µm门通过，实际0/3，零有效提起步。A6参考点升9.28mm仅转动，源最低点仍贴桌、lift无双侧承载；A3/E0闭合有载荷抬升仍滑脱。最大掌误差.148/.155/.080mm，A3原15.27mm与E0原2.214mm大幅降低仍失败，不能仅归因臂跟踪。

新增手掌子树前3无碰撞平移驱动，原手惯性/20DOF驱动/mimic/原闭合保留。Kp1e5/Kd1000/1000N、虚拟link1g为明确诊断，非真实机械臂。A3/E0全2100步20手DOF指令与旧执行逐float32相同，approach/close/hold/lift目标掌姿同；前800步在pregrasp开手保持替代旧settle+home。执行中零状态重置，.1kg/.6摩擦/桌/原凸体/物体向下.98N保持。原模型/训练435/排序不改，零新CGR/训练、失败目标单列，不改外部1/12。

只读真实承载处核对：15个正载荷接触的凸体投影都在源物体外，A3离源面1.48–4.18mm，E0 2.12–2.54mm，非别处全局误差；未换代理，未证明因果。后续须隔离接触几何/闭合承载/映射与排序，不能再单靠移动物体或臂规划解释已实际失败。未加额外成功正对照，不能称完整新执行器验收/算法上限。

三段235帧25fps9.4s RTX视频完成，videos/index.html含A3/E0旧完整臂对照，回放零积分。6tests、独立指令/全源网格承载/有限数值/17源205输入SHA通过。报告reproduction/boya/EGAD_FLOATING_THREE_RESULTS_20261108.md与evidence/egad_floating_three_20261108。**本轮已结束，无后台任务；不自行重抓/调参/扩样，若继续需新协议。** CanonDex只读，仅轻量阶段push用户origin。

## 2026-11-08 最新：冻结版本EGAD12外部测试已完成，1/12

用户要求停止围绕内部8件追满分并测试未用外部物体。outputs/egad_external12_20261107_v1全阶段complete，最终protocol SHA49e3f6b417be62976187db7f813a434fab24d1f2c67edd9cb1bb07c650bf4e91。官方EGAD49编号A/C/E/G×0/3/6固定12，source/米制mesh/digest与训练S40/旧DGN均不重叠，不能声称未见于上游CGR预训练。统一1源单位1mm/.1kg/.6摩擦/同XY，非验证的AnyDex打印尺度。

原CoACD403.93s后4超时/4停止/1完成/3未派发保留；首次新RGBD/评分/抓取前统一全12 V-HACD4.556s。3个数值非凸三角面部件使G3/G6原v2失败，全12/81组件统一原顶点凸包规范面1.811s后v3全ready，不加点/膨胀/缩放/删部件/不重跑V-HACD。AABB虽约2mm内，采样表面差最高16.62mm并有凹部填补，是重要近似限制；native20µm仅证明导入凸体保留，不能冒称源形状精确。旧失败/成本及preoutcome修订均保留。

12稳定落位+原cold750检查通过，15400初始化对象步；新12RGBD+原CGR共享1次载入/12前向，四32-fit preactivation头未变，0训练。2530query/8618允许组合，11/12有手几何；37完整臂准备，3ready/9null。A0各一次2100步成功+101.249mm；A3/E0各一次实际失败，末态都失去载荷未离桌；总6300抓取步。**完整结果1成功/2实际失败/8臂无计划/1手几何无候选=1/12**。8臂null29拒绝=19IK局部不收敛+10直接路径拒绝，非全局不可达证明，没做cuRobo障碍重规划。G6手几何全拒。没有根据结果移物/加预算/重抓/换对象，benchmark-only不入435。

三段235帧/25fps/9.4s正常速度RTX（含2失败）全部完成，在execution/videos/index.html；9无计划没有动作录像，observation_contact_sheet.jpg为12件新输入总览。33tests、1857唯一输入/源码SHA、原控制/时钟/源网格末态承载与媒体绑定独立复核通过，parallel独立审计确认四模型和435/302+/133-不变。冻结流水线437.859s不含资产准备/实现/检查；CPU4物理，GPU0表示/RTX，未用GPU1。

新外围external schema独立，复用原手域/排序/几何、prepare_object/verify_selected、native-v3 executor和原renderer，不伪装S40，也不使用旧兼容view。全输入/源码EVALUATION_FREEZE不再修改。报告reproduction/boya/EGAD_EXTERNAL12_RESULTS_20261108.md；轻量证据evidence/egad_external12_20261108。内部5/8不代表外部成绩，当前完整适配不足，8件被臂限制且代理粗，不能把1/12当AnyDex评分器/算法上限或论文复现。**本轮已收尾，停止对内部8/外部12继续逐件调参或重抓；若后续研究须独立协议隔离规划和代理限制。** CanonDex只读，旧批次不恢复，只push相关轻量文件。

## 2026-11-07 最新：训练死区对照已完成，内部8件5成功/3无计划

原code100评分的254个fit正例中113个预测严格0，最终ReLU直接梯度死区解释主要训练误差；fit-only BN重估无收益（fit .310835/internal .532770，0优化/物理）。新增明确的Boya训练适配train_s40_preactivation_control.py，用可微BN7 clone将原选中输出MSE作用在ReLU前，部署仍原Inference/ReLU。不是作者原训练，不改官方源码、435记录、32-fit/8-internal、seed/每epoch shuffle/优化配方/固定100轮。400更新、6.586s，fit .000644279/internal .403550785，fit正例零预测0/254；内部仍差于常数 .208669，不称可靠评分。

outputs/s40_preactivation_{control,domain,top1,execution}_20261107_v1全部complete。同8缓存910query/2379组合，362过门→114几何/6物体；11次臂准备、5ready/3null。每ready仅一次2100步，实际5/5全成功，统一分母5/8：007、021、044、066、069成功；008两IK/两直接路径拒绝共4预算耗尽、009/048无完整手几何。10500步、7门/11守卫全通过，原3/8与全部失败保留，不回填历史/不写回435。五个正常速度235帧/25fps/9.4s RTX视频全部完成，videos/index.html；无三件no-plan动作视频。零新CGR/EGAD；8件已多轮调试，非独立外部泛化。

薄入口evaluate_s40_head_control.py/run_s40_head_control.py绑定冻结旧helper与明确训练控制。旧code100 schema/string只是兼容格式，权威控制训练看TRAINING_RESULTS.json、实抓看HEAD_CONTROL_EXECUTION_RESULTS.json。兼容view顶层聚合metric复制旧值的报告残留不参与排名，正确指标/说明在evidence/s40_preactivation_control_20261107/INTERFACE_VIEW_METRICS_NOTE.json，勿改冻结源码或据旧view报指标。13tests及独立训练/桥接/完整控制轨迹、源网格末态/视频SHA核验通过。

并行映射只读探针未发现flip/depth/pad/close_q接线错误，但开手参考与非对称闭合有中点漂移。改用raw宽度最接近的闭合参考65点静态探针6/6→4/6（044旧成功也触桌），未晋级、无新增物理；更改映射会改变旧标签语义，不直接部署或给对象补丁。报告reproduction/boya/S40_PREACTIVATION_CONTROL_RESULTS_20261107.md。本轮已收尾，无后台训练/试抓；后续处理009/048通用映射及008臂接近，再固定协议做外部验证。整体论文复现仍未验收；CanonDex只读，只push轻量相关文件。

**2026-11-06用户纠正后的最高优先级状态：整体复现未通过验收。此前“忠实复现”判断撤回；官方离线输出核验与自建DGN上的Boya适配不能合并称论文复现完成。新独立score-first入口已修正post-CGR顺序；Boya训练物体身份/划分、单帧输入和执行协议仍未完整对齐；4/10与6/10仅为有限池适配记录。先做论文物体和划分的身份对照及部署顺序对齐，明确未对齐项，再谈新的训练/评测；不以扩大自建集代替对齐。本次更正不新增采集、训练或物理试验。以下按日期保留的阶段通过只对其局部范围成立。**

更新：2026-11-07。用户要求作为强基线仔细复现，先按照作者原方法/原手/原数据验证，再谈Boya适配；不能把CanonDex修复继续混入此仓库。官方上游https://github.com/graspnet/AnyDexGrasp.git，基线commit `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`，原手复现分支 `research/native-baseline-reproduction` 已推送至用户Fork；当前Boya适配分支 `research/boya-simulation-adaptation`。用户已授权 `origin=git@github.com:wangrp1997/AnyDexGrasp.git` 推送轻量阶段成果；作者remote仍为 `upstream`，不向作者推送。

原算法源码保持不变。复现说明与独立工具在 `reproduction/`，大文件/原始输出放忽略目录 `outputs/`、`papers/`；官方数据/权重专用根目录 `/mnt/public/datasets/AnyDexGrasp/`。不要把源码附带手开度表当作训练数据已下载。先读 `reproduction/README.md`、`SOURCES.lock.json`、`PAPER_CODE_AUDIT.md` 和数据状态。

当前已下载原论文PDF/TeX与官方源码，官方32权重635,102,941B（三手obj140/480下31决策权重+通用表示权重）与8份有效真实JSON约7.403GB已校验发布。31/31原决策权重均480→20、num_two_finger_angle=1，完整strict load和作者真实test记录前向通过，31/31文件名阈值P/R/F1全部匹配至4位小数；覆盖22手/类型、1440个不重复手/记录ID，重复评估1800条，不把它当物理抓起成功率或独立泛化。原训练默认240头与发布模型20头是配置差异，不再写发布权重不可用。完整CGR所需ME与原PointNet2已局部构建，真实GPU算子及完整表示单帧已通过：2048候选、480维CGR、首帧forward1.364s。DH3完整原identity+10增强、4个发布head评分/标定/碰撞/最终top1已实际跑通：1544→原top400→9碰撞准入，最终type2/90mm；单identity另400→59，两路top1相同。4路CPU按原模型/函数生成28点云槽13.206s，零物理步；完整11分支runner内部39.549s，未含进程启动/首次编译资产准备。真实bits/合并/选姿已独立复核，未执行机械臂/物理抓取；本机Python3.12/Torch2.11/cu130/Blackwell保留，共享Isaac未改，未新建Python环境。

官方没有可复用的桌面动态抓取仿真基准，PyBullet工具用于手模型FK/网格/点云生成。禁止把Open3D展示或接口前向称为仿真抓取复现，不运行连接真实UR机械臂的robot/collect脚本。先做无硬件的发布模型离线验证；将来仿真适配须独立声明，不替代原论文的真实试抓指标。

用户最新明确：CanonDex已由另一服务器继续，本任务仅做AnyDexGrasp，不再修改、执行或push CanonDex。用户随后仅要求pull最新，已完成（远端`21f3d66`，拖鞋修复由另一服务器完成）；本轮三份交接另有本地`4f511ce`未push，不继续处理。原暂停/259未测记录只作历史，不限制另一服务器的新授权工作。只有相关轻量代码/来源锁/文档可提交；数据、权重、视频、原论文及大输出不提交。用户已授权多智能体并行，避免并发修改同一源码。

网络已解决：先 `source /opt/nros_tools/nros_rc` 再 `nros-proxy-sg`（或sg2），仅当前进程生效；本机nros工具是shell函数，不是PATH里的nros-proxy可执行文件。旧localhost:7897当前不可用，不再把它当唯一出口、不再重复查Clash。public `CURRENT_STATUS.json`为当前状态，旧intake blocked JSON只作历史。三CGR标签ZIP合55.8GB目前为Google Drive文件下载配额限制；另一DH3 obj40 JSON在官方源有坏字节已隔离。不能把这些当代理失败。用户无法进入/空间不足，无需要求复制或扩容；大包仅从头重训表示才需要。已设每6小时一次、最多4轮后台配额重试，见public metadata；未授权读取Google账号秘密。

下一阶段先读`reproduction/SIMULATION_AND_CONTROL.md`及新增原DH3仿真就绪审计。原DH3 URDF/标定可做几何，但不能未经核验就当完整动力学模型；官方2D→12D仿射公式已核实可复用，355槽逐值一致；闭合position=0虽不在JSON/原Excel实测记录内，但可由原公式计算几何目标。真实force/speed仍是未标定的硬件寄存器值。先确定可复用闭合运动学/动力学范围，再做明确声明的独立仿真移植，不能报告论文物理成功率已复现。

## 2026-11-03 最新：原DH3独立单方块仿真通过，勿当完整论文物理复现

见`reproduction/simulation/README.md`及`TIME_RESOLUTION_RESULTS.md`。原2D→12D公式直接复用；helperless几何保留13实体/STL/惯性，355+96姿态全顶点误差0.122µm；精确V1零质量helper动态锁定，保留失败。固定50mm/100g方块真实RGB-D→原11分支323候选→原top400碰撞111准入→原top1 type4/63mm/40mm/[95,100]→物理执行接通。分支15.980s、runner含加载评分碰撞35.647s，不是到抬升耗时。

早期0.8Nm传动和粗步长虽然实际抓起，但耦合/速度失败；仅增加inner迭代无效。后续明确将传动作为理想结构约束（cap100工程数值尺度，并非硬件或严格λ上界），两个主动电机仍.6/.3Nm、腕仍每行30。固定cap/inner20/原门/所有输入与连续指令，只细化实际积分步长240→480→960→1920→3840Hz，每步全记录。最后两档均过原门：末态升高88.518/88.592mm，差.073854mm；gear残差.001712/.000454rad，峰速1.377/1.262rad/s，腕误差.104/.052mm。末态持续正法向接触，结构反力远未触cap；独立source/模型/观察/候选/逐步q/控制/接触/帧核验通过。没有附着、关重力或积分后reset。

原作者没有桌面动态基准，本移植仍是理想结构/动态凸包/有限世界腕脚本，非识别的硬件或UR臂。采样校正功proxy随细化减少但不是能量上界/被动性证明。不据一个方块宣称泛化成功率。未接Boya、UR臂或多物体评测。粗网格及所有失败保留，不改旧标签。

总98,460真实积分步/119s混合时间步仿真，含全部失败/隔离/空手；几何和媒体0步。最后两档执行内部含渲染23.494/34.217s，非完整端到端。正常视频`outputs/dh3_dt1920_cube_physics_20261103_v1/grasp_normal_speed.mp4`，205帧/20fps/10.25s；3840同目录模式保留数值对照。

当前通过执行器`dh3_time_refinement_smoke.py`严格绑定冻结计划，不是通用新场景入口。下一步先把通过的物理数值设置接成可复用场景接口，再验其它几何；不要重跑下载/31模型/依赖构建或扩大既有时间网格。仅轻量相关代码/核验/文档提交，大数据/模型/视频/轨迹/生成URDF留outputs/public，只有作者upstream不向其推送。CanonDex由另一服务器维护，本任务不再改/执行/push它。


## 2026-11-03 最新：通用场景接通，三个新形状原top1均成功

用户授权先测圆柱/球/长盒。预冻结直径50高90mm圆柱、直径60mm球、100x45x35mm长盒，各100g/同一XY附近/一种摆放/仅原top1一次。真实RGB-D→原11分支→原4head→原碰撞/argsort→DH3，3/3通过原门，末态100.329/92.466/99.455mm；末半秒960步全部持续接触。三个均独立选中type4/63mm/[95,100]，深度40/40/20mm，没有动态覆盖其他类型，不当未知总体100%。

新`primitive_scene.py`/`run_official_dh3_batch.py`/`run_dh3_scene.py`为可复用接口，已验收冻结controller源码不改。新cube回归18数组和trace/contacts/frames全字节一致。默认1920Hz/inner20/电机.6/.3/gear100/wrist30及原门，不再加时间网格。三帧实际只加载1+4模型/5次strict，10次RNG重放0不匹配，原forward/sampling/collision每帧实际执行。推理内部总123.051s，单物理含渲染31.018/27.081/28.389s，worker与后续推理重叠，不相加冒充端到端；没有新同帧冷/热配对速度结论。

原TinyRenderer圆柱/球显示有1mm外观膨胀等偏差，首轮`scenes/cylinder/FAILED_RENDER.json`零物理保留。仅显示改解析网格，原生动态碰撞/尺寸/惯性/摆放/相机不变，生产输入`scenes_v2`解析误差<1mm。native ray点差作为辅助诊断，未改任何物理/抓起门。不得修改`primitive_scene.py`后继续复用旧signature观察，需新生成场景。

完整目录`outputs/dh3_primitive_generalization_20261103_v1/`，统一`videos/{cylinder,sphere,long_box}.mp4`及index/contact_sheet/SUMMARY。每视频205帧/20fps/10.25s。四条物理与三条推理均保存产物独立核验通过。新增78720步/41s物理（含cube回归），历史累计177180真实步/160s混合dt，媒体/审计/输入失败0步。[报告](reproduction/simulation/PRIMITIVE_GENERALIZATION_RESULTS.md)。

本轮已停止，三个成功不表示UR臂/Boya/硬件动力学或论文物理成功率复现。下一步建议增加真实物体网格和摆放/手型覆盖，须按用户后续授权范围，不自动大批扩测；不要再重复构建依赖/31模型离线验证/已验收cube时间网格。仅轻量相关代码/回执/文档本地提交，大文件不提交，只有作者upstream不能push。CanonDex仍由另一服务器维护。


## 2026-11-04 最新：真实YCB六物体十二场景完成，8/12

用户授权在无机械臂情况下继续原DH3验证，六件原始网格×yaw0/90，每场景原top1一次，已完成。8/12通过：苹果2、香蕉1、泡沫砖2、瓶2、杯1；香蕉0°实际未抓起且峰速1.7438>1.5；杯90°/碗0°90°分别48/83/88候选经原碰撞全拒，零物理。9次177120步/92.25s，含旧失败累计354300步/252.25s混合dt；不把8/9实际执行成功替代12分母。

完整输出`outputs/dh3_real_mesh_generalization_20261103_v1/`（11月3日冻结，4日完成）；统一9视频`videos/`、`index.html`、`contact_sheet.jpg`。报告`reproduction/simulation/REAL_MESH_GENERALIZATION_RESULTS.md`、案例`REAL_MESH_FAILURE_ANALYSIS.md`、轻量凭据`evidence/real_mesh_20261103/`。两组独立审计通过，12行和失败均保留；三个无候选只声明已有阶段审计，不谎称存在top1/物理。

为与资产准备重叠实际为10+2顺序两组，各常驻5模型、共10次真实加载；推理与CPU物理重叠，实际并发峰2/上限3，非GPU物理批处理。正式管线386.434s不含之前40.38分钟原CoACD冷准备、开发和零步故障；没有同帧加速倍数结论。

五件各128部件CoACD；杯碗稀疏凹深最差保留97.75/98.66%，不代表全表面精度。苹果原CoACD中间递归超时，原日志/成本保留、仅本任务workers已停止；在苹果推理前采用单独120s预算V-HACD资产配方，1部件、果柄凹陷采样误差9.248mm，苹果两次成功有粗近似限制。米制scale1、原尺寸、COM XY[0,.08]、杯碗正放，100g/摩擦1/凸部件混合惯性为移植假设。

新增mesh_scene/prepare_real_mesh_assets/prepare_mesh_fallback、run_dh3_scene_cohort、audit_real_dh3_cohort、merge_real_dh3_cohorts及publish_real_mesh_media。原算法和接受controller未改，runner只添加mesh_scene许可。首输入误把保守AABB当实际几何，已按真实凸顶点比较修复且容差不增；启动器venv路径与一处变量错误均零步保留，正式run_nonapple_v2/run_apple_v2；原run_nonapple启动错误不覆盖。保持解释器venv路径，不要resolve()到basePython。

已知元数据错误：mesh_scene旧visual_boundary称纯色/未用原纹理，实际上OBJ导出保留原纹理。APPEARANCE_ERRATUM是执行后补录，六个导出纹理RGBA像素与原完全一致；当前source_assets未单列生成MTL/PNG。未来生成前应更正描述并补依赖，改source后需新生成场景，不能复用旧signature或谎称补录执行前存在。此次实际RGBD/候选/物理核验有效，冻结source和原始图像均保留。

目前仍未接UR/Boya，不是原论文实机成功率复现。不要重跑依赖/下载/31模型/方块时间网格/已完成12场景；不自动扩物体或把topK重试混进本轮成绩。下一步先看原碰撞全拒和香蕉单侧接触问题，控制/资产近似与网络候选责任尚未隔离。CanonDex继续不修改/执行/push。仅轻量本地commit，大输出不提交，只有作者upstream不可push。


## 2026-11-04 后续：四非成功场景机制诊断完成，原8/12不变

见`reproduction/simulation/REAL_MESH_TARGETED_DIAGNOSTICS.md`及`evidence/failure_diagnostics_20261104/`。统一静态图/既有香蕉帧/实际轨迹/负载对照曲线在`outputs/dh3_real_mesh_failure_diagnostics_20261104_v2/index.html`。新工具为diagnose_original_dh3_{collision,source_mechanisms,depth}、render_failure_diagnostics、run_banana_load_ablation和audit_banana_load_ablation；原算法与accepted controller未改。

两碗73/83、87/88种子来自桌面，原robot_dh3.py逐seed objectness过滤被注释；该发布demo还限制向下25.84°，不能据此认定整个论文架构不能侧抓。6代表原voxel mask重放逐位一致。杯#12唯一标记点在连续源手凸包扫掠之外，但未执行/未证明整个姿态可抓。原PCL15层法向扩展最多11.25mm，检测full-depth与原执行减position/100×5mm语义不同，忠实基线保留。碗#79执行源STL距桌+0.237mm、native凸几何含margin为−0.759mm，是边界；碗#83有14.616mm物体重合，不能全放宽。

香蕉0°接触→电机0.6Nm饱和→共同轴0.394rad停滞得到存档trace支持。固定一次6.25s/12000步负载移除对照，保留手姿/桌/自碰/两电机/九gear和全部前12000行原指令（逐数组相同），仅第一步前移除香蕉，主轴到1.154rad，4.9505s后指间自接触，无手桌。独立审计通过，明确非抓取成绩/非真实机构辨识。论文说4DoF2motors欠驱动，当前gear仅2独立坐标；四DoF计数口径/被动拓扑未明，不凭空去gear或加各指电机。累计366300实际步/258.5s混合dt；原12结果仍8/12，没执行被拒候选。

后续先单列明确版本的objectness过滤和collision/execution深度对齐对照，再确认DH3被动机构/受载模型。当前没有这些修复已救回的结论，不扩对象、不重跑旧12、不把诊断分割真值送入网络；不再重复本轮零步几何诊断和负载对照。原metadata appearance erratum未来新输入前仍需处理。v1静态回执宽度列错/v2已正；负载v1JSON tuple/list预检零步失败/v2完成，全部留档。CanonDex继续不动，只轻量本地提交，只有作者upstream不push。


## 2026-11-04 最新：用户授权明确仿真近似，弹性共同驱动已实测

用户没有DH3实机，明确不要把精确硬件标定作为前置，要求快速公开资料近似。已单列实现dh3_compliant_proxy.py：共同虚拟轴+三弹性支路有成对反作用，主动预算仍0.6/0.3Nm，原13实体/局部tip/outlet仿射保持；不是已辨识商业DH3跨指拓扑或原4DoF机构。原算法/accepted刚性controller/12场景8成功不改。

两版各5固定工装验证通过；原一指受阻全停变成两侧继续到达固定垫，v2末态0.781256/.432068/.781257rad。原抓姿香蕉0°/90°/cube各3版本对照，v2和同参3840Hz v3均实际提起约9–10cm，但全3旧速度/约束稳定门仍未过，不晋级默认，不报论文新成功率。新增298560步/124.75物理秒，全部9抓/10工装/失败/成本保留，零网络重跑；累计696060步/399.5秒混合dt。没有继续硬件识别、扩对象或重跑原12。

报告reproduction/simulation/DH3_COMPLIANT_PROXY_RESULTS.md，交付outputs/dh3_compliant_delivery_20261104_v1/index.html和videos/（12段含原对照）。工装独立能量/源审计已通过；继承两个汇总字段映射问题有erratum和后续工具修正，历史原回执不改。公开PDF/CAD归档outputs/dh3_public_source_archive_20261104_v1，不提交大文件。下一步先看保存视频/接触和物体审计，不重复下载/模型/工装/九抓或把提起等同旧稳定门通过。CanonDex不动，仅本地轻量commit、没有用户发布remote不push作者。


## 2026-11-05 最新：用户确认近似可用即可，原12场景统一补测完成

用户明确大差不差复现能用即可，不要求精确机构辨识或12满分；随后问是否官方数据复核，必须明确区分。31个官方权重在作者真实test记录上的P/R/F1四位小数复核早已完成；本次是六个YCB网格×两方向自建仿真，不是AnyDex官方实机测试集。

outputs/dh3_compliant_full12_20261104_v1/（目录沿用启动标识）已完成：固定model50e02b/runnercdc861、k=.8/d=.02/J=.001、3840Hz；7新执行全部held+2同参既有香蕉held+3原collision无候选零步=12。准入9/12，条件执行9/9，含无候选的场景完成9/12；原数值门1/9过，其余诊断保留，不将本地1.5rad/s当官方成功判据。原8/12不改，不宣称论文物理成功率。

四路补测102.559s，新275520步/71.75物理秒，复用78720步/20.5秒；独立核验全部354240步/92.25秒通过。9段同目录视频/总览index.html，三无候选只有输入图。报告reproduction/simulation/DH3_COMPLIANT_FULL12_RESULTS_20261105.md和PAPER_EVALUATION_ALIGNMENT_20261105.md；轻量evidence/compliant_full12_20261105。不要重复跑已完成7或把复用算新增；别再为精确硬件/速度门过度校准。后续原官方数据/推理协议为主线，仿真功能验证单列，杯碗原候选覆盖问题尚在。仅本地轻量提交，CanonDex继续不动，不push作者upstream。


## 2026-11-05 最新：用户Fork已接通，Boya完整Isaac采数链路取得一正一负

用户明确按论文Algorithm1适配Boya，复用CanonDex已能抓取的完整UR16e+Boya Isaac Sim后端；不是另造PyBullet手，不调用CanonDex旧经验检索/抓姿采样/优化。原作者硬件采数在这里替换为声明的Isaac仿真采数，通用CGR权重冻结，Boya评分头尚未训练。已找到完整后端资产并只读复用；所有新代码在reproduction/boya，所有新产物在AnyDex outputs，分块缓存也重定到AnyDex，Python -B，不改/执行旧CanonDex实验批次。

新50mm/100g立方体RTX RGB-D -> 官方原表示一次真实GPU前向2048候选 -> 源distal pad新标定 -> 四个几何候选 -> 四次新臂准备（两零步拒绝） -> 两次实际2100步执行。首条原行561/未翻面未抓起label0保留；第二条1346/翻面末态+103.201mm、底面离桌102.594mm、持续承载1.636s，label1。两条原七门全过；总4200物理步/16.8s，一物一观测两次独立试验，不是泛化成功率，不是重抓协议或学习后性能。

接口已验证原16手指+2腕+6臂、4mimic及21个手link；9模板297状态/924参考槽，模板4缺拇指对侧标注明确不构造。正式标定用原five_finger_wrap.json的distal pads，旧中节v1诊断和两轮几何零步拒绝保留。1.7开口扩展遵循原碰撞代码，限幅显式记录；本轮四准入均未clamp。固定手型1/附加深入0，两侧翻转，未用DH3评分头、方向/45–99mm rig门或旧经验。新标签绑定原行/2261/官方CGR480/开度/深入/翻面/commands SHA/真实回执，零步和中止不记失败训练标签。

首轮新输入来自Isaac，旧通用forward回执example_boundary仍有官方demo固定文案；冻结原文件保留，另存input_scope_erratum.json。原DH3所有结果不改。报告reproduction/boya/INITIAL_ISAAC_RESULTS_20261105.md和README.md。下一步先冻结多物体/手型/深入采集协议再采更多标签/训练小head，不重跑这两次，不把两条样本充作有效Boya训练集或已复现原论文成功率。仅轻量阶段代码/证据推到用户origin，资产/模型/观测/视频/轨迹不进Git。


## 2026-11-05 当前：八物体小批已实际后台启动

用户授权8形状×手势1/6/8×每组最多2实际、总48抓上限；已冻结public源对象/姿态/scale0.1、源表/代码/输入SHA，AnyDex自己的 outputs/boya_pilot8_20261105_v1，后台PID2579933，两路CPU完整Isaac臂执行。源码父提交a85e6c9加本轮PREPARED.json逐文件冻结；本轮轻量阶段提交会覆盖同一源码字节。不要把旧CanonDex暂停限制误读成禁止本轮已授权AnyDex采数，也不要恢复旧259。

8个统一斜视原网格观测已完成、最终冷进程28.215s，早期SDK类型/Camera Fabric诊断版本保留；CGR常驻仅load一次、8forward共16384原候选，入口19.082s。10总项/每物体16项独立复核通过。24组合2CPU几何准备51.974s，22组合共148准入，碗g1/g8无候选原样保留。形状只选材，不入算法。

前向使用v1协议；任何新物理前另冻v2 SHA72da233f9e3657043819f9baa75c0fd3b852624745b92510a97f5f4b8e377677，仅启用旧后端cold reset后声明初态恢复（原20µm门前/零额外积分），8对象、相机、特征和候选不变，未重跑前向。试验各自独立，执行中不reset，和现场连续重抓协议不同。

首批已产生真实正负标签，但本轮尚在后台，任何启动快照都不是最终48或泛化成绩。以本机progress.json/cost/round回执为准，正常实际失败继续，不在成功后取消第二候选；无候选/臂拒绝/null不填负例，真基础设施异常paused_infrastructure停止新派发。死轮保留，不重跑；显式execute恢复先查活进程。不修改任何冻结Boya py或源输入、不重复启动；必要修复应先停止派发、保留旧协议后另版本执行。Boya评分器尚未训练。报告reproduction/boya/PILOT8_20261105.md，轻量证据evidence/pilot8_20261105；视频/轨迹/图像/权重不提交，只推用户origin。

首段已暂停：19实际/39900步，16正3负；5臂准入零步null。杯子g1第一轮原生导入分块043误差36.040µm超20µm，原生审计确认零物理步，原unknown账本保留并追加诊断；并非杯子抓取失败。原主进程已退出。23槽未派发，原48槽保留；5段RTX视频已完成。当前正用已有通用保形预处理处理剩余资产，不改CanonDex/源几何/候选/门。修复后另版本接续，扣除所有旧轮/索引/预算，不重跑19实际。轻量证据含19条独立核验与暂停原因。

2026-11-05 已修资产并实际接续：`outputs/boya_pilot8_20261105_v2`，PID2671029，两路CPU，协议v3 SHA4c3cf085d98904f38f73aceac6d3c31b896a181b434c24b96e5cad54a077dcb3。继承25终态/19旧抓取，父目录只读，源9886d7a及旧SHA全保留；仅剩余轮/索引可派发。四资产通用保形细分通过/零物理，原20µm门不放宽。新source已冻，不再热改/重复启动。杯子接续已实际产生两条成功，旧导入null不改。旧成本/new segment_cost分列，最终以v2/progress为准；CGR/观测/池全复用，未训练Boya head。页面在v2/index.html，原五视频引用旧文件，追加新物体录像。

**八物体小批最终已收尾，勿重复恢复：** v2/progress=complete_with_infrastructure_errors（仅保留历史资产null），24/24组合、36实际29正7负、75600步、12槽null；7对象有正例，碗无实际执行，香蕉2正4负。旧19条无重复、751冻结项及36×19实际检查全过。最终摘要/36行原训练JSONL/审计在v2目录，已有8段代表/成败视频（另补香蕉失败片）。本轮不自动扩批或训练，Boya head仍未训练。后续应优先诊断碗候选覆盖和香蕉稳定性，保留当前固定名单/失败/成本。


## 2026-11-05 最新：规模采集接口已接通，先三手型300训练标签

用户授权先接通常驻/并行采集并询问手型选择，尚未启动完整50物体采数或训练。作者先人工预定义手型（DH3关节分箱、Allegro/Inspire人类抓取分类），采数可按手型固定，部署论文方法再由各型评分头竞争；公开Inspire默认另有随机旁路，详见 `reproduction/boya/COLLECTION_PROTOCOL_20261105.md`。本阶段先用1五指握/6二指捏合/8爪式；三型在试抓前按形态及标定可执行性选择，未证明三型充分或最优。

正式50名单为 `outputs/collection40_20261105_v1/selection_v2/object_selection.json`，SHA487017083e79eb1323f243946ddf2b931dac6bd87d86f52632a254ba5226f60d：40train+10development-validation、50唯一mesh、两组及旧8开发物体零重合。源名称形状层仅选材/报告，不参与抓姿/手型决策。旧v1球头锤分类错误和修正差异原位保留；50静态资产预检通过不等于原生动态验收通过。

目标每型100实际成败训练标签，共300；验证每型20、共60。额外深入0/10/20/30mm在新 `build_collection_pool.py` 真实移动后重做完整几何；旧depth0 pilot不改。40×3×4=480训练槽及10×3×4=120验证槽为有限首轮上限，目标不足报告欠额、不造标签/按结果换物体。每手型独立480输入/20输出(5raw深入×4额外深入)评分头，尚未训练。

新 `prepare_collection_queue.py` 解耦有限CPU规划及原指令绑定，`execute_collection_queue.py` 支持同资产波次内CPU常驻serial或GPU隔离parallel，批大小1..16、总队列不受128限制；**跨波次仍新进程**。只读复用现有臂/资产物理后端，不改CanonDex，不使用其旧经验/采样器。`run_collection_pipeline.py` 串联零步原生资产预处理、单RTX进程观测、单CGR常驻模型、多CPU几何/规划及显式执行。默认prepare停在计划，dry-run零Kit/模型/物理；详见 `COLLECTION_PIPELINE.md`，不能把dry-plan说成50件完成。

实际新后端验收在 `outputs/boya_collection_backend_20261105_v1`：CPU常驻两个新候选完整4200步，两正，提起86.104/99.070mm；一次建场、第二抓前一次独立重置、两次原生审计，端到端101.403s/循环57.922s，指令/绑定全部通过。GPU0真实GPUDynamics+broadphase双环境在每路431记录步后因明显慢安全中断，862环境记录步+至多2在途未确认步，全部null；sim.step351.461s/隔离guard0.719s，无完整GPU抓取或提速结论。共享GPU有其它负载，不外推独占硬件本征倍率。两后端配对记录全benchmark_only，不计300训练配额、不重跑旧36。默认物理先CPU resident，CGR仍GPU。

准备22tests与执行12tests通过，后端18项独立审计通过；轻量证据在 `reproduction/boya/evidence/collection_20261105/`。完整300+60目标采集未启动，旧pilot36/29正7负及碗无实际记录保持。下一步按新版本运行准备、审查候选覆盖和真实吞吐后采数，不恢复旧pilot、不宣称Boya评分器已学会自动选型或达到论文成功率。

补充最终只读核验：CPU常驻两抓43项独立复算通过，旧pilot751冻结项、36记录和29/7标签保持。主入口600槽dry两次/复用通过（零子进程/Torch/Isaac导入/前向/物理），凭据 `evidence/collection_20261105/pipeline_dry_audit.json`。阶段源码/轻量证据可push用户origin，完整50采集仍未启动。


## 2026-11-05 最新：九手势对照完成，原三型选择需扩展验证

用户要求实际验证手势选择。`outputs/boya_gesture_comparison_20261105_v1/` 已完整收尾：原八个开发物体、同观测/原CGR/标定、depth0、ff、500/8几何及每格2实际/每轮4臂/总8臂预算。旧1/6/8的36实际只读继承；新增2/3/5/7/9/10共48格，47有几何/1无候选，166次臂准备后69准入计划、27未执行null。**69实际57正12负、144900步，无基础设施错误；合旧共105实际86正19负。** 旧摘要/36行JSONL SHA不变，无旧抓取重放；未新增网络前向或训练。完整九型矩阵及统计见 `reproduction/boya/GESTURE_COMPARISON_RESULTS_20261105.md`，事前物理评价口径与SHA在 `GESTURE_COMPARISON_20261105.md` 和启动凭据中保持。

新增各型正/实际：2=8/9、3=4/9、5=11/13、7=13/13、9=11/12、10=10/13；#7覆盖7/8，六个有两条有效候选的格均2/2，末态抬升70.861–100.723mm。香蕉新增2/3/5/7/9各两候选均成功，10为1/2；旧1/6各1/2、8为0/2。**原1/6/8不能继续称充分或最优；#7三指捏合是下一版扩型的优先项。** 当前8件开发集的物体观察覆盖仍7/8：碗九型均无完整臂准入/无实际抓取，不是九型实际失败。#2等的条件成功比例高但准入覆盖较小，不能只按成功比例选择。

新执行两对象CPU并行、同对象常驻serial，共7次建场/62次独立试前恢复/69次原生审计，恢复积分0；物理执行墙钟1671.011s，准备453.926s（与几何有重叠，不直接相加宣称端到端）。所有新轨迹2100步、命令/绑定及516源码754输入保持；独立从trace/源网格复算末态离桌、承载、抬升及限位/mimic/穿透三门，69/69标签一致。其余四门绑定原native保存证据，不宣称重新采集缺失的测量。轻量证据在 `reproduction/boya/evidence/gesture_comparison_20261105/`，大轨迹/JSONL/源资产不提交。

**下一步先按新版本把#7纳入候选采集型集（建议1/6/7/8）并在独立40train+10validation上验证；尚未启动正式50物体采数或训练。** 当前 `run_collection_pipeline.py` 的既有三型600槽协议和dry-run仍是历史冻结配置，不能直接称已改四型或把本轮开发105标签计入300正式配额。后续修改须生成新协议/目录，不改旧dry-run/本轮结果；八件单姿态对照不证明泛化、三指通用最优或13/13的总体成功概率。CanonDex不修改/不恢复原批次。


## 2026-11-06 最新：四型正式50物体采集已后台启动，勿重复派发

用户同意继续。旧九型开发105实际/86正19负及原三型600槽dry协议完整保留。新入口 `reproduction/boya/run_collection_pipeline_v2.py` 已冻结，源码SHA `82cb5deadbe82d6cbcded446a241d3244f1b3114ed11694a830d163ad3aab6d9`，协议文档 `COLLECTION_PROTOCOL_20261106.md`、选择静态凭据及所有输入/源码SHA绑定。四型固定 `[1,6,7,8]`×四额外深入0/10/20/30mm×原50件=800槽，640train/160validation；目标每型100/20合格正负标签，合计400/80，不保证有限槽全部达标，不把开发105抵扣配额。旧50名单SHA `487017083e79eb1323f243946ddf2b931dac6bd87d86f52632a254ba5226f60d`、40/10分组与源几何保持。

**正式目录 `outputs/collection40_20261106_v2/run_v2/`，后台all PID3322110，2026-11-06 01:02:53北京时间启动，协议SHA `20414374d11c0ed4df68b339e51a38318901bb3b34119cb7b127adcb3e820703`。** 已完成50原生资产准入与50真实RTX观测，单CGR共享模型/50前向正在运行；启动时尚无正式物理标签或训练结论。后续当前状态以 `pipeline_progress.json`、`representation_stage/BATCH_REPRESENTATION.json`、`POOL_PREPARATION.json`、`planning/queue_manifest.json`、`execution/PROGRESS.json` 为准；不要因本段启动快照重新建批次。源码运行中保持不变，禁止自动重放状态不明的执行。

原生资产外置预处理50件24原生ready/26精确细分后准入、0物理；source union/scale0.1/max_failed32/原20µm门保持，初轮/接续成本均保留。发现单件杯子cache/source_audit从1MiB起末151061字节全零，但原probe完整且SHA精确等于旧声明；保留坏cache/旧result/旧SUMMARY，在独立缓存恢复原字节，不修改声明SHA/几何，50件128artifact复核通过。外置 `assets_preflight_v2/SUMMARY.json` 当前SHA `6fc3a3e5648d018c99ea6cb5a0fdcf917a35207a76e525d83921829388d7d86c`，正式native阶段已采用冻结。勿再次改该发布记录。

RTX观测先与独立资产发布修正并行，50帧78.560s（stage80.460s），零显式物理，不再渲染。首次representation因启动器漏继承现有LD_LIBRARY_PATH而在0模型/0forward/0物理时退出，完整attempt移存 `run_v2/attempts/representation_missing_runtime_libs_v1/`，明确零执行后同协议/manifest继续。正式LAUNCH已继承pilot的os_libs/OpenBLAS/CUDA13/torch库路径，不安装依赖或更改共享环境。native profile语义SHA和config文件SHA分别绑定，train/validation分文件导出，actual只计steps>0；独立29项合同及runtime14项通过。

本轮物理固定CPU两个对象worker并发，每对象serial resident batch16；每槽最多4臂候选，只执行首个准入候选一次，普通有效负例继续，infra停止新派发并收尾。最终 `execution/train_records.jsonl` 与 `validation_records.jsonl` 分开，混合审计流为 `all_records.jsonl`；无候选/未执行/资格不明为null，不能伪作负例。所有额外记录保留，400/80欠额显式报告；validation不能调参或选手型。目前尚无Boya学习评分头。CanonDex保持只读，不恢复其批次。运行与恢复说明见 `reproduction/boya/COLLECTION_PIPELINE_V2.md`，大输出/图像/轨迹/数据不提交Git。

补充同轮启动快照：50/50 CGR前向已完整完成、单模型加载，当前 geometry_running；轻量 `evidence/collection_20261106/launch_snapshot.json` 记录45 ready/16 no_candidate（800槽中间状态）、正式物理标签仍0。无候选是准入null，非抓取失败；后续仍以活动进度为准。


## 2026-11-06 最新：原800槽准备完成，执行步数适配修复后已续跑

原run_v2于02:49停止：800槽610手几何/190无候选、431臂准入/369null；3对象27实际22正5负/56700步。Snowman原native g7d0为2126步（仅approach延长），旧采集入口固定2100导致纯prepare拒绝，0Kit/0物理；其他两个尚未执行计划亦有2124/2142步。原记录不覆盖，所有旧源码保持。

新独立execute_collection_queue_v2.py绑定各原plan完整长度/phase/原数组，新resume_collection_execution_v3.py核验431=27继承+404明确未执行、无unknown后，**新活动 outputs/collection40_20261106_v2/execution_resume_v3 已于北京时间10:44:39启动，PID3977466，协议ed51a65b24860b25c7b2116de209761522e62e04089039aec42605bef5a9a773**。两个对象CPU并发，对象内常驻；旧run_v2进度仍是历史停止，后续看新PROGRESS.json/RESULTS.json，勿重复启动或修改新冻结源码。原27条只读继承不重抓，观测/CGR/几何/规划不重做。当前运行启动快照非新完整抓取结论，尚未训练。

原准入上限train各型85/89/89/90=353，validation17/21/21/19=78，总431，原400/80目标必欠额，不能补null为0或按验证结果换对象。最终新活动目录train_records.jsonl/validation_records.jsonl分别合并旧27和新增；旧开发105不抵配额。10执行合同、Snowman16原计划prepare-only、5账本负例及27轨迹独立核验通过，详见reproduction/boya/COLLECTION_RESUME_20261106.md和evidence/collection_resume_20261106。所有轻量修改仅AnyDex，CanonDex仍只读。


## 2026-11-06 最新：官方四头训练工程验证通过，正式训练与实际对照已自动接续

原resume_v3采集仍运行，未改采集源码。新prepare_boya_training_data.py保留32/8/10 mesh分组，默认只接受final complete；原canonical480复用一次，单raw_depth*4+extra标签。新train_boya_decision.py AST复用原Inference/MSE/PolyLR，四独立480→20，正式双阶段各20epoch（内部32/8只报告末轮→全部40重训），不按external10调参。固定论文20epoch/lr1e-4+公开code WD5e-4/Poly.99组合，明确配置差异。旧文档“20=5角度×4深入”为口误，正确5 raw depth×4额外深入；实现一直正确，冻结旧文档不热改。

GPU0已用116真实记录跑通双阶段各2epoch工程smoke，16optimizer steps、trainer内部3.393s，8state_dict严格重载过，benchmark_only=true；不冒称正式模型/抓取提升，零新CGR/物理。

**自动交付PID4050257已于北京时间11:35:53实际启动，活动outputs/boya_decision_delivery_20261106_v1，plan SHA48fcfa71b97edbe02c25e5251c7433d6cde258dc7a4ae36b24109cc2881ec8d0**。当前waiting_for_collection，final complete后自动dataset→正式四头训练→外部10对象baseline/learned无标签top1推荐冻结→两策略各一次实际试抓（benchmark_only不入训练）→最多4个RTX视频。此处是同一431已准入池的对照，不能声称完整未过滤算法或广泛泛化结果。既有标签只有排名冻结后才能读作离线配对，不代替实际对照。

后续看DELIVERY_STATUS.json/DELIVERY_RESULTS.json和reproduction/boya/TRAINING_DELIVERY_20261106.md。watcher每15秒核producer/PID/源SHA，12小时有限等待；infra/dead/unknown明确blocked、不重放，媒体失败单列。正式训练和实际对照尚未开始，勿把工程smoke当最终模型。不要启动重复watcher或热改冻结prepare/train/eval/run/executor源码。轻量证据evidence/training_delivery_20261106，大文件不提交，CanonDex仍只读。


## 2026-11-06 最新：已知任务早停分类已修，97未执行计划已实际接续

原resume_v3在12:44左右停止，320合格241正79负/672068步；sem_Ball_730cf06c9c029ae45e1b0bd500a24ffc的14实际均187步，原生object_pushed_before_closure(闭合前XY>30mm)主动终止，共2618步，完整执行不成立而worker正常收尾。原队列把未成标签统一当infra，自动交付v1于12:44 blocked。旧431分区320eligible+14已知tasknull+97未执行，全部成本674686步保留。

新helper按明确原生reason、原source/input/commands/binding SHA和完整trace prefix认证known task stop（非对象特设），null不补0、不重播，未知/infra仍stop。新resume_v4只派12对象97原未执行（19train/78validation），新prepare_data_v2保留32/8/10和原480/20，正式ledger允许431=eligible+certifiednull。最多417有效(339train/78val)，不凑原400/80配额。

**当前活动采集outputs/collection40_20261106_v2/execution_resume_v4，PID103505，14:34:14启动，protocol4b3c93e6f82d88226e1a1b5d43e6a504014e5d3587a774dc99e2b4d4f3fc502e；活动交付outputs/boya_decision_delivery_20261106_v2，PID108089，14:37:18就绪屏障后恢复，plane17332fb2b2b811334a8e87d4359776c1a0e2719f15da1788b9eb8319c130fbc。** 新watch v2等final再自动四头训练/10物体真实top1对照/视频；其已认证tasknull在实际任务指标中未成功但不作训练负例。旧watch启动竞态的零阶段日志保留attempts，未重启采集。

当前训练/对照尚未完成。后续只看上述新PROGRESS/DELIVERY_STATUS，旧v3/v1是历史blocked，不要覆盖、重播或热改冻结source。14采集/9数据/11watch合同及真实prefix/ledger验证通过，详见reproduction/boya/KNOWN_TASK_STOP_CONTINUATION_20261106.md、evidence/known_task_stop_20261106。官方学习核心、物理门、旧320/14记录全部保持，CanonDex只读，大文件不提交。


## 2026-11-06 最新：首轮Boya学习评分器、实际对照和视频已完成

活动v4采集已完整收尾：800槽=369准备null+431实际，431=413合格（295正118负）+18认证task null；876649物理步，unknown/unexecuted均0。合格记录336train/77external，原400/80目标欠64/3，null未补0。原40/10分配不换物体，内部32/8/10 mesh分组不泄漏；40训练对象中实际38个有标签，各head实际36或37组。旧v3/v1阻塞、全部早停及成本保持。

`outputs/boya_decision_delivery_20261106_v2/`已于北京时间15:22:18 complete，正式四个1/6/7/8评分头及内部四头均20epoch，官方Inference/MSE/PolyLR核心复用，CGR冻结。配置明确为论文20epoch/lr1e-4/batch128+公开code WD5e-4/Poly.99，非逐行原脚本默认。外部10不参与训练/epoch选择；336条拟合，160optimizer更新，trainer内部3.031s。

10留出物体从同一原78个已准入候选中无标签top1排序冻结后，各策略新实际执行一次：原CGR分数4/10、Boya学习分数6/10；20条完整2100步/共42000步，无null或进程错误。相机和球本轮0→1，原本轮4个成功保持；相机同原计划原采集成功、本次baseline提起后滑落，离线旧配对5/10对6/10不可混作新实测4/10对6/10。样本少且已有重复差异，不宣称稳定20个百分点提升或达到论文实机成功率。

四段原模型RTX视频已完成：`outputs/boya_decision_delivery_20261106_v2/videos/index.html`，相机/球各一组原排序失败与学习排序成功；均235帧/25fps/9.4s、回放零物理。413数据/18证书/分组及四头、无标签排名冻结通过；20次实际152项核验和媒体SHA/结构通过。轻量结果见`reproduction/boya/FIRST_LEARNED_RESULTS_20261106.md`和`evidence/boya_first_learned_20261106/`，大文件不提交。

下一步先分诊剩4例：3例在原有限实际池里无成功记录，浅碗有1/3成功候选却未选中；前者检查候选映射/落位，后者研究排序。已有10件结果保留，不能调参后继续当独立验证，不自动追加大批/重播旧成功。当前采集和交付均已结束，不要再启动watcher或把旧启动段落当当前进度。CanonDex保持只读。


## 2026-11-06 最新：Boya适配验收已完成，数据范围与部署顺序须明确

本轮50物体来自自建DGN（40训练分配/10外部留出，实际38训练mesh有标签），不是原论文S40/L144/T30集合；官方真实JSON/原手发布模型复核与本轮Boya/Isaac试抓分开。不能仅因40数量相同称原物体集复现，也未证明与CGR预训练无资产交集。

只读端点FK重算431条映射最大误差2.306e-16m；四失败对象28条原canonical480及raw_depth*4+extra索引精确一致，开/闭命令绑定通过。四例实际均完整2100步但未提起：Bottle闭合推移45.761mm；Orbit食指被桌面阻挡；Utana推移50.241mm且中指被桌面阻挡；OXO两指99.6%以上闭合但推开18.033mm后失去接触。Utana池内#8原成功82.025mm却分数0，#7失败0.05348，是小数据排序缺口；另三件池内无成功，不证明物体不可抓。原6/10和所有失败/成本保持。

确认当前Boya先每槽CGR前500→几何keep8→最多4臂/首个准入，再学习重排，与原部署先多型多深入评分再最终碰撞不同。整体复现未通过验收；局部接线检查通过不能称完整官方部署已复现。下一实施应复用缓存和现有head，将评分移到候选压缩之前并冻结新的共同有限预算；接线不需先重训。真实Boya标定/限力控制仍为公开声明近似，闭合扫掠或新控制是另立改进。本轮没有新物理/前向/训练，未实施下一阶段，勿自动重播10件或启动大批。报告reproduction/boya/ADAPTATION_ACCEPTANCE_20261106.md，完整只读产物outputs/boya_adaptation_acceptance_20261106_v1，轻量凭据evidence/adaptation_acceptance_20261106。CanonDex保持只读。


## 2026-11-06 最新：原物体资产对照与score-first独立入口已交付

用户授权继续对齐。独立deploy_score_first.py默认锁定发布DH3的1000→全型/四额外深入评分→top3500→严格float32 score>.85→top400碰撞，原单flip；排名freeze先于标定/几何/臂，不读旧431/78队列或试抓标签。相同CGR/type一次head20输出再gather，原Inference与安全权重检查复用。8功能检查与独立16000索引/排序/1000canonical/SHA复算通过。可选paper-testing-budget-boya-port-v1不是完整原论文参数，不能静默回退/降低阈值。旧代码和6/10保持，尚未将此入口静默接成旧批次默认。

原DGN训练香蕉缓存仅作工程检查：4000headquery/16000组合→400几何检查全部table拒绝、零臂/零抓取，最终null如实保留；不是新成功率。rank/geometry_v1/geometry_v2三个产物outputs/score_first_alignment_20261106_v1全保留，共12000headquery、零新CGR/训练/物理，v2内部5.205s不含启动。v2加入原RGBD/T_WC/缓存输出交叉绑定且数值选择不变。新arm stage只接通prepare_one单行兼容/共享预算/绑定核对，因无几何准入尚无实际新臂验收。不要把接口通过称抓起成功。

S40从本机DexGraspNet2.0中GraspNet scene0000–0099身份表重建40ID，40原/简化模型和URDF均有SHA，属于派生来源，尚未与官方模型归档/AnyDex CAD逐字节核验。官方GraspNet API已核对train0–99和getObjIds语义。不要套DGN_5k scale.1或25/106/130mm过滤。EGAD官方49个A0–G6已下载ZIP/校验/解压到public/AnyDexGrasp/object_alignment_20261106_v1；原打印尺度/质量仍未知。L144/T30精确身份及约150全测试/选13DexNet表尚未取得，论文CAD说明upon request。四个Boya head仍来自旧DGN，未开始原S40新采集/训练/评测。

报告reproduction/OBJECT_SET_ALIGNMENT_20261106.md和reproduction/boya/SCORE_FIRST_ALIGNMENT_20261106.md；轻量凭据分别在reproduction/evidence/object_alignment_20261106和boya/evidence/score_first_alignment_20261106。并行代理两项中途服务429限流，已由主代理接回完成，不当成文件权限或资产失败。整体论文复现仍未通过。CanonDex只读、大资产不提交。


## 2026-11-06 最新：S40原尺度派生场景与两件真实采集链已通过

S40全部40件按官方尺度依据scale1，原nontextured.ply不减面转换OBJ，逐顶点/拓扑保形；现成CoACD是显式碰撞近似（最大AABB边界差2.13347mm），不等同作者CAD或20µm的CAD近似证书。独立资产3195项通过。资产public/AnyDexGrasp/boya_s40_assets_20261106_v1，原40全train，无旧DGN尺寸筛选/伪10测试。

新prepare_s40_scenes.py/render_s40_observations.py/run_s40_pilot.py独立入口；原DGN源/四头/6对4结果不改。固定000/002、四型1/6/7/8、extra0，8槽均有几何、22次臂准备后5准入3零步null，5条实际全2100步/10500步正例。原始末态升高95.971/93.622/99.545/96.673/97.329mm；不能写8/8、泛化率或完成原实机复现。采集不加载旧DGN head、不用部署.85门；几何优先采数本身符合论文结构，score-first修的是部署。仍为双flip/固定深入/Boya位置闭合/孤立场景移植。

outputs/s40_alignment_pilot_20261106_v1/PROGRESS=complete，338.802s；场景首备70.963s另计。GPU0单模型两次CGR，CPU几何/臂双worker及两对象常驻独立执行，非GPU物理。14项源/renderer测试过；实际两帧RGBD输入字节核验、563源码/62输入freeze过；物理2444项/756文件SHA与5条数组/提起/索引通过。七门中六门trace重算，臂支撑仅原evaluator最大值和checkpoint重判，完整接触流未保存的边界保留。

两段原模型RTX视频在本轮videos/index.html，235帧25fps9.4s、回放零物理，媒体核验通过。完整审计在本轮audit，轻量evidence/s40_pilot_20261106；报告S40_COLLECTION_PILOT_20261106.md。原评分头仍只用旧DGN训练，本轮5正0负不训练新评分器。

扩大前修下一版本通用无候选null分类（当前geometry非零会暂停，本轮8槽非空未触发）及renderer source_inventory路径freeze（当前40mesh/urdf/pose冻结、inventory已独立读回）。保留本轮冻结源码/输出，不热改重跑。后续按S40每型100次论文预算明确多姿态/深入与null成本，EGAD留到模型/尺度协议冻结再测试。整体复现仍未通过；L144/T30/完整实物测试身份等缺口不变。CanonDex只读，大资产不提交。


## 2026-11-07 最新：用户授权扩全40，S40共640槽已实际后台启动

新运行outputs/s40_collection_20261107_v1，PID1658959、启动2026-11-07T11:05:27.373395+08:00，入口run_s40_collection.py --phase all。40全train/四型1,6,7,8/四extra，共640槽；继承pilot的8个d0槽（5实际正例10500步+3零步prepare null）、仅新增632槽。两件原观测/CGR不改，其余38件新渲染/共享单CGR。每型100有效标签/合400是目标不保证，全部欠额、实际负例、null与额外有效记录保留，不结果替换。预冻32/8仅S40内部诊断分组，不是外部测试。

source_inventory路径/hash已补冻结；合法无几何非零经证据接成null，非正常异常停派发；v2 executor支持原完整变长approach，已知task早停经证书认证为已消费terminal null，unknown不重放。记录导出binding/精确label/steps/eval/command重新比对，并核row/serial cohort积分账。41测试通过，源码/输入/manifest 573/1162/3读回通过。不要修改任何运行冻结py/源资产，勿恢复旧DGN或重复派发。

CPU32–95，native4/geometry4/arm8/execution4；CGR GPU0、物理CPU对象内wave常驻，命令长度可拆多cohort，不保证每物体一次Kit。启动核验时40件原生资产全部完成，已进入观测阶段；没有新抓取率或新训练结论。后续看pipeline_progress、stages、planning/queue_manifest与execution/PROGRESS/RESULTS，最后collection_counts与train_records；文件未产生不补结果。协议S40_COLLECTION_PROTOCOL_20261107.md、启动S40_EXPANSION_LAUNCH_20261107.md及evidence/s40_expansion_20261107。

本轮只启动采集，无自动S40 head训练或EGAD评测；整体论文复现仍未完成。旧pilot5次/3null和DGN6/10全部保持，各段成本分列；CanonDex只读，大文件不提交。


## 2026-11-07 最新：S40采集收尾、四评分头训练完成，原始部署内部检查未通过

outputs/s40_collection_20261107_v1已于北京时间12:43:25 complete，97.971min；不再是后台运行状态，不恢复collector。640槽=240有效（165正/75负）+50认证早停null+350准备null；290实际、509766物理步（含旧pilot5次10500步，新285次499266步）。unknown/unexecuted为0，原400目标欠160。实际31件有标签/77种canonical480；28/40至少一次成功只属采集覆盖。pipeline_progress旧physics_steps=0是历史空字段，以physical_steps/execution最终账为准。

用户问后续，已实际完成outputs/s40_decision_20261107_v1/dataset、training与raw_internal_diagnostic。专用S40 loader保留原40全train、预冻32/8原source visual mesh SHA；196fit/44internal，EGAD未使用。复用原Inference/选中输出MSE/PolyLR和既有train_head，固定混合配方20epoch/lr1e-4/batch128+WD5e-4/Poly.99；四内部头再四all40头共160optimizer更新，trainer内部8.529s，8state_dict strict reload。内部加权MSE0.445549高于fit正例比例常数0.253756；all40包含内部44条，不作heldout。

仅用32-fit头，从8帧缓存原始CGR按发布DH3先评分→全局top3500→严格float32>.85→top400几何，32000query/128000组合/3200检查，几何准入0：3199table、1visible_scene_point_in_hand_hull。1423个入选候选的手型×目标输出在fit无直接监督；不是概率，也未隔离单一因果。不降阈值、不mask、不切回filter-first，不用旧ready池或标签排序。零新臂/物理/CGR，尚不能部署，不报新成功率。

50次早停均在接近前，完整手体保守距物体至少177.762mm；静置50步时物体已滚动/倾倒，旧观测失效，不是手推物体。5件初始化早停、4件纯准备null、3件完整执行全负分开。先统一稳定落位后再观测/CGR，并按新摆放/抓姿/raw-depth×extra覆盖补采、与旧640另记；不能改50null成负例或重复成功凑100。当前未启动补采、新物理或EGAD；不要误把“下一步”当已启动。原DGN4/10对6/10与所有旧失败保持。

14合同测试、真实8帧排序/几何集成通过，结果为0准入。详S40_COLLECTION_COMPLETION_20261107.md、S40_DECISION_RESULTS_20261107.md与evidence/s40_decision_20261107；大输出/模型/数据不提交。整体论文复现仍未通过，CanonDex只读。


## 2026-11-07 最新：稳定40件通过，覆盖补采与自动训练已在后台

用户明确“好的开始”。实际稳定v2 outputs/s40_stable_supplement_20261107_v2/settling 已40/40 ready；同source mesh/COM/惯量/.1kg/.6friction/.004dt/.98N与原Native20µm。统一连续窗口stationarity+native twist sanity、再750步3s cold-recreation，最大位移0.134280mm/转角0.004005rad。v2 113250 object steps，原过严瞬时速度v1探针200000保留并已归档源快照，初始化累计313250；零抓取。后续trial独立从实测稳定pose与明确声明zero velocity重置，不称连续live。新pose WXYZ/T/trace/native初始化链绑定，旧50null不改0。

新采集 outputs/s40_stable_supplement_20261107_v2/collection，PID2110437，北京时间16:01:16启动，协议c0ae291aec2132d93bf0c7721fd0e15ba01c2e798d8d4f9e60f68a5a9d513037。40全train/原32+8/640全新槽，排除旧290实际canonical480×g×extra含terminal null，不按0/1值选样；raw1000按5深度各200、每档≤64原geometry，总≤320、每档最多2/keep8、arm4。40新RGBD已完成、共享单CGR运行；几何4/臂8/执行4对象CPU并发，CPU32–95/GPU0。未有本段新最终试抓结果，不提前报成功率；不重复collector、不热改冻结py/资产。原640/240/50/350账和旧DGN成绩保持。

自动训练等待PID2120549，北京时间16:07:27，活动同父training_delivery。8h有限等待，final complete后export补采→严格两源merge旧240与新有效→固定20epoch四内部及四all40模型。source trace/path重复直接reject，protocol命名空间trial ID保留local/source。异常/unknown不重播，不把null补0。当前waiting，无新训练结论。EGAD未使用。

同时确认旧make_queries/raw_s40诊断遗漏作者pre-head approach/width/workspace联合mask；3200全拒不能归因训练不足，也不是完整AnyDex部署失败。原DH3离线链保留mask，其9准入不受影响。已新增boya_prehead_domain纯函数和14tests，与官方真实AST数值比对；Boya workspace/cone/raw-width必须显式，尚未冻结具体参数/接部署。自动交付本轮只到训练，部署评测仍pending，不重跑已知缺mask的旧诊断冒充完整验证。

新40合同测试（6+9+6+5+14）和真实旧帧coverage/原240源merger/稳定manifest串联通过。详S40_STABLE_SUPPLEMENT_20261107.md、S40_PREHEAD_MASK_REVIEW_20261107.md及evidence/s40_stable_supplement_20261107。仅轻量相关代码/证据push用户origin，CanonDex只读、大数据不提交。


## 2026-11-07 最新：S40补采157有效后被时长合同阻断，38条未执行已安全恢复

原outputs/s40_stable_supplement_20261107_v2/collection已640候选完成/195ready/445准备null；157实际=116正41负/329700步后停止，非批次完成。原0603unknown和35未派、旧训练watcher blocked保持，旧240+本轮157共397有效源记录，未重训。

060_g6_d1原计划2166步，home原生750→816，approach仍500；v2误假设仅approach可延长。原源码AST+原URDF+保存端点重建三home数组/timing精确相同，060在prepare遍历前未保存PREPARED/无worker/仅.queue.lock，证明3零抓取物理；35有原not_dispatched及无提交/输出证据。不能泛化“无文件=未执行”。

新v3薄executor复用固定SHA v2所有控制/worker/记录函数，只显式适配home/approach/lift原生时间缩放完整合同及原终止证书的同一依赖。原数组、速度/加速度/jerk、物理门不改；新exporter_v2先安装合同再原导出。195原计划全部过、9测试过、060实际prepare-only3trial/2cohort零物理过。

新活动outputs/s40_stable_supplement_20261107_v2/collection_resume_v3，PID2371155，2026-11-07T19:04:09.621434+08:00启动，protocol ba712cc75c3467457436b78b47eb438658958b9b8abae31981faa70a6a4ea076；四对象并行只派38首次执行，原157/329700步继承不重抓，不重render/CGR/几何/IK。原640、源40/alltrain/32+8保持，旧文件不改，新派生JSONL只改context追溯新旧protocol。

自动等待PID2371156，同父training_delivery_v3；新final完成后exporter_v2→两源merge旧240与本轮有效→固定20epoch训练。此刻新训练/部署成绩未产生，EGAD未用，Boyaprehead具体域配置待冻结。后续看新目录，不重复旧watcher或新恢复，不热改任何已冻结py；所有未完成/失败与成本保留。报告S40_TEMPORAL_RESUME_20261107.md及evidence/s40_temporal_resume_20261107；只轻量push用户origin，CanonDex只读。


## 2026-11-07 最新：补采与重训已完成，评分门诊断未通过

`outputs/s40_stable_supplement_20261107_v2/collection_resume_v3`已complete：640槽=195有效（137正58负）+445准备null，无terminal/unknown/unexecuted；409604物理步，继承157/329700，新38/79904。collector和training_delivery_v3均已收尾，不恢复watcher/重派。旧阻断及输入不覆盖。严格合并旧240后435=302正133负、35有标签网格，原40全train/32+8保持，364fit/71internal；四内部+四all40头固定20epoch已完成160更新、strict reload通过。

内部加权MSE .554678仍差于fit-prior常数.208669。独立原代码/数据检查未见480/标签/输出索引、Inference/Loss或漏权重接线错误。论文100次为每型在40物体上总计，约100/型已同数量级，不能只怪量少。每头仅20batch更新、7BN计数20，train/eval差异真实但不是已隔离唯一原因；未调参/重训。EGAD未用。

新增独立score_s40_domain_candidates.py接已有prehead helper，源推导单一policy在evidence/s40_domain_diagnostic_20261107/DOMAIN_POLICY.json先冻结；共同标定宽度交集为保守Boya端口域，不是最大开口，workspace为camera frustum world AABB/table裁剪，approach保留相机Z>.9语义。25tests通过。原8帧16384raw→89域内→356queries/1424组合；原float32>.85有38组合，但38全table拒绝、0几何/0臂/0物理。旧raw3200全拒记录仍留，不称完整官方部署失败。

独立diagnose_s40_domain_feasibility.py只读同一冻结输出，对全部1424组合做几何对照，17准入来自3件，最高score .494948，全部被原.85门挡；另5件该域内无准入。对照不是降低部署阈值，不覆盖0准入结果，17不是抓取成功。新诊断9.010s+对照29.380s额外成本；零新CGR/物理/训练。完整outputs/s40_domain_diagnostic_20261107_v1及s40_domain_feasibility_20261107_v1；报告S40_SUPPLEMENT_AND_DOMAIN_RESULTS_20261107.md。只读原内部27 ready进一步发现12条source row域内，但全部flip=True而official=False，完整query重叠0/27；双flip采集和单flip部署存在覆盖差异，不读标签。整体复现/新部署仍未通过，不自动用EGAD调参或追加大采集；下一阶段候选覆盖和训练协议需先对齐。CanonDex只读，只push轻量相关文件。


## 2026-11-07 最新：双翻面/公开代码配方已接通，内部8件中2件实际top1成功

用户问下一步后已实际推进：新增evaluate_s40_boya_port_v2.py显式双方翻面，与采集27 ready的canonical480/flip/depth逐字节一致，旧DH3单flip不改。新增train_s40_code_recipe.py，AST绑定公开默认100epoch/lr5e-4/batch128/WD5e-4/Poly.99，原435/364fit71internal/32+8，四内部头400更新、固定final100，未all40；outputs/s40_code_recipe_20261107_v1已complete。fit MSE .310881、内部.531875仍差于常数.208669，不把增轮数当评分已解决。

同一冻结prehead域/原float32>.85：旧20轮双flip712query/2848组合→81过门→0几何；code100双flip→396过门→9几何（0081、0215、0443）。新部署检查outputs/s40_dual_flip_{paper20,code100}_20261107_v1已完成，零新CGR。共同宽度/相机cone仍为保守Boya域，非完整硬件能力范围。

prepare_s40_port_top1_v2.py从新冻结排名按每对象最多4个几何候选找首个arm-ready，显式绑定原native-preflight/044保形repair。v1/v2各3次纯准备共6次（相同3候选重复），成本全保留；最终2ready/6null。run_s40_port_trials.py已实际各执行一次、两个CPU独立进程并行：021 rank33/型6/flipFalse成功+90.343mm，044 rank11/型8/flipTrue成功+102.281mm；完整2×2100=4200步，原七门/控制绑定全过，末态仍承载，无实际重试、null或unknown。两条benchmark-only不入435；两者翻面均与各自DH3 official相反，但不宣称已隔离翻面必要性。

**内部8件仍是2成功/6无最终计划（5几何null、008路径拒绝），不是新2/2或外部100%成功率。** 完整outputs/s40_port_top1_20261107_{v1,v2}及s40_port_execution_20261107_v1。两段原模型RTX录像已完成，videos/index.html，235帧25fps9.4s，回放零物理、trace/eval/MP4结构SHA绑定；执行加录像165.203s，准备/训练/推理另记。21项必要tests通过。全部阶段已收尾，不再当后台running或重复派发；EGAD未用。报告S40_BOYA_PORT_V2_RESULTS_20261107.md。下一步Boya各手型开口/接近方向候选覆盖及008路径问题需通用适配，评分误差仍高；不自动扩大采数或外部测试。CanonDex只读，只push轻量相关源码/证据。


## 2026-11-07 最新：Boya每型候选域完成，内部8件3成功/2实际失败/3无计划

用户问下一步后已完成通用手型候选域适配与真实单top1检查。boya_hand_domain.py以四型各自原×1.7/clip/nearest-slot和真实21link开手桌面支撑替代旧DH3相机方向锥/四型共同宽度交集，保留raw严格(0,.099)、.5mm桌面余量、双flip、四extra、原top1000/3500/严格>.85/400及完整可见点碰撞。评分前有明确支撑查询成本，不称零几何；全点云仍冻结评分后。源顶点凸包索引仅加速线性最低点，32真实姿态与原桌面谓词全一致、最大差1.39e-17m，无容差放宽/ID规则。

outputs/s40_hand_domain_20261107_v1已complete：同8缓存/同code100四32-fit头，910query/2379组合→240过门→93完整几何准入，6/8有候选（旧3/8/9准入），0桌面复核冲突；22.393s。prepare_s40_port_top1_v3.py共9次原完整臂准备、28.706s，5ready/3null。其queue完整落盘后的末尾print缺json导入保留，未来用绑定SHA的prepare_s40_hand_top1.py薄CLI，不改冻结源码/不重复准备。

outputs/s40_hand_execution_20261107_v1已收尾：run_s40_port_trials_v2.py四CPU进程，5件各一次2100步，总10500步。021/066/069成功+90.696/79.424/97.779mm；008和044实际失败，007前4臂计划拒绝、009/048几何null，故本轮统一协议**3/8，不是3/5，也不能补旧044作4/8**。全部7原门/11守卫通过，无terminal null/unknown/infra。5段正常速度RTX视频在videos/index.html，235帧25fps9.4s，执行+媒体329.938s，回放零物理。

044有明确回归：旧成功raw1401/g8/flipTrue/extra0在新池仍几何通过、rank10/score1.2142498493未变，却被新raw1875/g6/flipFalse/extra20mm rank7/score1.3258295059排前并执行失败。旧候选未丢，不用旧标签强选；单例未隔离唯一物理因果。044末态有双侧力但最低点仅1.36mm，是翘起；008闭合仅食指受力、无拇指承载。旧2/8保留，不回填本轮。

009/048本域全1/10组合绕过分门仍完整几何全拒；.099门外仍在workspace组合支撑6880/7008次也全拒。源网格都在相机/深度范围。证据限于冻结映射覆盖不足，非硬件不可抓证明；不降碰撞门或加对象规则。根代理16tests过、5条实际控制/末态全源顶点测量/视频绑定复核、557源输入SHA全匹配。原435/302+/133-、32+8、code100头不变；零新CGR/训练/EGAD、benchmark-only不回写。

报告reproduction/boya/S40_HAND_DOMAIN_RESULTS_20261107.md，轻量evidence/s40_hand_domain_20261107。下一步重点评分排序/训练部署候选域，而非再盲目扩采；009/048另需通用几何适配，007仅有限臂预算耗尽。此轮全部完成，无后台任务，不重跑成功/自动外部评测。8件已用于多轮开发，不称独立泛化；整体论文复现仍未通过，CanonDex只读，只push轻量相关文件。
