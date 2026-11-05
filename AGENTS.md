# AnyDexGrasp 独立复现工作区

更新：2026-11-05。用户要求作为强基线仔细复现，先按照作者原方法/原手/原数据验证，再谈Boya适配；不能把CanonDex修复继续混入此仓库。官方上游https://github.com/graspnet/AnyDexGrasp.git，基线commit `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`，原手复现分支 `research/native-baseline-reproduction` 已推送至用户Fork；当前Boya适配分支 `research/boya-simulation-adaptation`。用户已授权 `origin=git@github.com:wangrp1997/AnyDexGrasp.git` 推送轻量阶段成果；作者remote仍为 `upstream`，不向作者推送。

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
