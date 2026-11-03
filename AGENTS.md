# AnyDexGrasp 独立复现工作区

更新：2026-11-04。用户要求作为强基线仔细复现，先按照作者原方法/原手/原数据验证，再谈Boya适配；不能把CanonDex修复继续混入此仓库。官方上游https://github.com/graspnet/AnyDexGrasp.git，基线commit `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`，本地分支 `research/native-baseline-reproduction`。上游remote为 `upstream`，没有用户自己的发布remote，不向作者仓库推送。

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
