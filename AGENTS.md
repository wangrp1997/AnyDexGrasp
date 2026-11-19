# AnyDexGrasp 独立复现工作区

## 2026-11-19 固定手指闭合的受载对照已冻结

用户要求验证形成对向受载后保持手指目标。固定旧成功场00及新批失败场03/15，分别覆盖闭合时和提起时失去受载。按各原bank重建native MJCF，原摆放、1000稳定步、每场原完整24列命令与腕姿，入闭合前复制并逐值核对配对原生积分状态。两臂组各2/4环境。实验组只在同一物体拇指与其余合力均≥1N持续8帧后保持上一次实际给定的16手指位置，FAJ1/FAJ3和六臂位置/前馈及时间保持，掉接触后不再收紧。无新规划、CGR、录像、回位或训练，最多6次诊断抓取/207040原生环境积分。CPU预检同体连续性、臂腕保持及无二次闭合均通过，按同一规则读取旧轨迹会于00/03/15第1913/3027/2204步触发。完整执行及审计结果以`outputs/boya_mjx_closure_pair_20261119_v1/COMPARISON_RESULTS.json`为准。

## 2026-11-19 空池后备新批已收尾，补回4场几何候选，0场通过臂规划

seed2026111906的16布局中8场摆放空，8场实拍及CGR。原碰撞池4场为空，后备均补回候选，场06/07/08/11分别2/1/8/7条。独立核对两次筛选的原始行、特征及场景点逐值相同，保留进手方向和原深度，新增深度为0。7次后备臂规划均在声明的六轴IK种子内未收敛，保留为null。原4个非空场中3场规划及执行完成，03选rank1，14/15选rank0，另1场规划null。共13臂计划、3次抓取、0次重试。3次均未提起，回位通过。03/14对向受载在闭合阶段丢失，15进入提起后同一物体对向受载仅4帧。最大穿透1.711/2.314/2.666mm，三次原1mm诊断均未通过，按现有5mm硬停及其余六门归为有效评估负例。

采集含审核876.372s，三条负例仅用于评估，正式数据仍为117条。8460控制步、1082880原生积分、16921行调用日志、0未知积分。52源及归档、5进程组退出、4槽释放、6锁可取及全局registry释放核对通过。无录像、训练、标签晋级或后台任务。主结果`outputs/boya_mjx_cgr_empty_pool_rescue_20261119_v1/RESCUE_RESULTS.json`，几何及接触轨迹轻证据`reproduction/boya/evidence/author_mjx_cgr_empty_pool_rescue_20261119_v1`。

## 2026-11-19 空池后备已接入，三场旧图调度验收通过

原碰撞筛选后有候选时沿原CGR顺序最多查5项。仅原池为空时，用同一张RGB-D、原CGR与实测臂初值，按原翻转标志绕进手轴转180°，设追加深度为0，保留原预测深度与原特征，复用原碰撞和完整臂规划门，最多查2项。每场最多执行一次，观测与规划之间状态保持。已存07补回8候选且首选臂规划通过，08仍空，00原池2候选且原首选通过，未触发后备。三个实际CPU子进程退出、槽位均释放，两个ready的实测状态绑定和动作审计通过。11项现有CPU测试通过，预检0物理/CGR/训练。

新批根`outputs/boya_mjx_cgr_empty_pool_rescue_20261119_v1`，seed2026111906固定16布局，8摆放通过。16环境单轮，最多40臂计划、8次抓取、2380800原生积分，0抓取重试。全部为评估，排除训练，无录像。最终以`RESCUE_RESULTS.json`和`PIPELINE_FINISHED.json`为准，分别记原候选和空池后备的几何、计划、提起结果。预检轻证据`reproduction/boya/evidence/author_mjx_cgr_empty_pool_rescue_20261119_v1`。

## 2026-11-19 空池恢复离线通过2场完整臂规划

固定已结束CGR后备基线的4个空池场04/07/08/14及2个已提起对照场00/09。六场原映射和体素决策逐值复现。空池试原翻法、保持进手方向的绕X转动及不翻转，各试原有0/1/2/3cm追加深度。仅保持进手方向并令追加深度为0，保留原CGR预测深度，场04/07/08/14分别有1/8/0/3个过原手碰撞门。每个补回场最多2个原CGR高分且不同原始行做完整臂规划，6尝试3项ready，落在07和14两个场景。3项原手动作审计和已存实测状态绑定通过，均用保持进手方向及0追加深度。原阻挡场08扩大至同帧已存Stage3的13个种子和130个角度/深度组合，3行宽度合格，原体素门仍全部拒绝。Stage3初次CPU重建旋转容差未覆盖2.384e-7浮点差，第二次旧目录重名，均0物理，独立v3完成。

本次为已存观测诊断，0新增CGR/物理积分/训练/视频。后备生产入口尚未改；建议下一次有限仿真仅给原空池启用已验证的向下进手和0追加深度，保持碰撞及完整原规划门，区分计划就绪和实际提起。原记录和正式117条未改。结果`outputs/boya_cgr_approach_depth_recovery_20261119_v1/FINAL_REVIEW.json`，轻证据`reproduction/boya/evidence/author_cgr_approach_depth_recovery_20261119_v1`。

## 2026-11-19 小批CGR后备基线完成，补回1个场景并实际提起

v2固定16场，8摆放空、8实拍和CGR。只用原首选有1/8可规划，原排序内最多试5项达到2/8，实际只调用7次候选臂规划。场0首选提起022约96.020mm，场9首选被固定段几何门拒绝，第2个候选提起058约98.642mm。两次均完成回位和原只读审核，按原功能规则合格，最大穿透1.850/2.856mm，原1mm诊断均false。另4个空池和2个全无规划场景保持null，不计抓取负例。小批验证了实际后备收益，未跑总体成功率估算，数据仅供评估。

v2采集含审核819.840s，6260全局步/801280环境积分，含v1启动故障128000共929280。原1批审核通过，12521行native调用无漏计，50源及归档、5个进程组退出、6把锁和4预约释放核对通过，无录像/训练/后台任务。结果在`outputs/boya_mjx_cgr_fallback_20261119_v2/BASELINE_RESULTS.json`及`FINAL_REVIEW.json`，轻证据见`reproduction/boya/evidence/author_mjx_cgr_fallback_20261119_v1`。下文保留启动与恢复记录。

首个物理根v1在1000个全局稳定控制步、128000环境积分及8次CGR后，调度代码因漏导入subprocess在Popen前停止，0规划进程、0抓取。所有三条隔离回执均为spawn_attempted=false；主机进程及锁核对后仅释放该批保留，原回执归档在`resource_failure_archive`，原source/协议、RGB-D及日志保留。新`resume_author_mjx_cgr_fallback.py`补上实际调度依赖，独立真实CPU子进程检查通过，0物理。新根`outputs/boya_mjx_cgr_fallback_20261119_v2`只绑定原16个布局，重新稳定并继续相同实验，最多8次抓取，不重抽布局或重试抓取。新的运行状态以v2的`PIPELINE_STARTED`、`collection/RUN_STARTED`及终态为准；成功回执会同时计入v1的128000已消耗积分。

同一批已存RGB-D/CGR使用全部源网格，对照作者DH3 type1原查表及Boya虚拟预闭合参考。前5个原空池的141个匹配原始行中，DH3在目标处有136个最低点低于桌面超过1mm。549个匹配Boya分支中447个在深度位移前、547个在目标处低于桌面超过1mm。仅用已保存的开手参考重算目标仍有541个。原采集代码在俯向筛选后翻转0/1列，5个原空池151个原始行中的91个翻向上方。诊断保持同一翻转决定并按作者部署版本改用绕CGR +X转180度的1/2列，原体素门重算只救回1个候选，仍有4个空池。上述为静态端点及离线筛选，0物理、0新CGR，未接入本批采集。几何审计v1少PyBullet导入路径后退出，v2使用本地既有overlay完成。结果见`outputs/boya_cgr_table_clearance_20261119_v2/SUMMARY.json`。

已接通原CGR后备规划。沿原手碰撞筛选后的CGR顺序最多查5个，每场在原生状态不推进期间选择首个通过完整臂规划的候选，只执行一次。旧026图像核对前三项为null/null/ready，原行、特征、姿态逐值一致且live动作绑定通过。旧000空池停止，0臂规划。新根`outputs/boya_mjx_cgr_fallback_20261119_v1`于已推送的`6a75f20`派发，seed2026111905，12fit及4internal，8布局通过摆放，16环境一轮，最多40计划及8次抓取，2,380,800环境积分上限。无录像、训练或标签晋级。终态看`BASELINE_RESULTS.json`和`PIPELINE_FINISHED.json`，不能用派发记录代替结果。统计同批top1与后备的规划覆盖，物理仅执行实际选定候选。

## 2026-11-19 离线定位候选覆盖与桌面净空

上一轮新布局对照按排序分歧入组，8个首选仅2个通过规划，无法得到双侧有效物理结果。本轮在相同5个多候选场景核对全部15个原候选，复用8个已有计划回执，补7个原规划。原CGR首选与原评分头首选各在1/5场景可规划，继续按原CGR顺序检查可恢复至4/5。新增可规划项为014 rank4和026 rank2。每池至多1个可规划项，新增两项尚未执行，不据此报告成功率。所有原输入、碰撞门和计划配置保持，0新物理、录像、CGR、训练。

共享观测批的24个空池及累计100fit采集的108个无候选都在开手碰撞筛选清零。固定前5个空池全部549个手部映射分支中，547个目标张手点云最低点低于桌面1mm以上。回算取消全部深度位移后仍有442个，只有raw深度时有542个。原体素筛选输出逐项复现。靠近桌面点只是按高度诊断，480个分支单凭高度带外点也超原阈值。下一步优先核对CGR→Boya掌姿与开手桌面净空，同时用原CGR有上限的后备候选做小规模基线验收。先报告可规划率与实际提起率，再在可执行池中验证评分。详见`outputs/boya_mjx_shared_pool_coverage_20261119_v1/FINAL_REVIEW.json`。已结束，无后台任务。

## 2026-11-19 共享RGB-D配对已收尾，4组分歧、2次抓取

固定128个新布局中42个通过摆放，37组双方稳定并共用同一帧和复制后的积分状态。其中24个空池、8个单候选池、5个多候选池，5个多候选池中4组首选不同。用完队列后正常停止，达到4组，目标为8组。8个首选中CGR与评分头各1个通过完整臂规划。CGR在draw87提起021，末源底130.120mm、升高125.480mm，最大穿透1.992mm，按既有功能规则合格，原1mm诊断未过。评分头在draw8完成3590步抓取，未提起。两次均回位。其余6个首选保持null，4个固定端点或非HOME段碰撞，2个局部IK种子未收敛。没有一组双方都有有效抓取结果，当前不能估计纯抓取排序收益。

采集与审核耗时2161.923s，6批原独立审核均通过，24532全局控制步和3140096原生环境积分完整核对，49065条调用日志、0在途和未知积分。51项源及归档核对通过，25个所属进程组退出，7把锁可用，5项资源预约和全局registry释放。工具7463 exit0，无后台任务、无录像或训练。正式117条数据不纳入这两条评估标签。结果和资源回执在`outputs/boya_mjx_shared_pair_20261119_v1/COMPARISON_RESULTS.json`与`FINAL_REVIEW.json`，轻证据见`reproduction/boya/evidence/author_mjx_shared_pair_20261119_v1`。下文为已完成批次的启动记录。

用户同意继续8至12组首选不同的配对。执行根为`outputs/boya_mjx_shared_pair_20261119_v1`，常驻MJX，16环境，固定seed2026111904。从已有internal对象池抽128个新布局，42个通过摆放检查。按抽样顺序录取前8个排名不同的场景，每组CGR首选和评分头首选各一次规划，最多16次抓取。评分使用前一阶段固定末轮预激活MSE权重和发布ReLU推理，平分沿用原CGR次序。规划空与无效执行保持null。配对样本仅供评估，全程不录像。

配对两边使用同一模型、同一控制步的复制积分缓冲、一张RGB-D和一次CGR及碰撞筛选。复制后、规划后均核对GPU缓冲。GPU小缓冲测试通过，0物理积分。历史219场同一相机数据通过CPU双规划检查，2个候选分别选rank0和rank1，均规划ready，0新物理。新入口`run_author_mjx_shared_pair.py`于已推送的b737188启动。运行进度以`collection/PROGRESS.json`、`WAVE_PROGRESS.json`、`BATCH_COUNTERS.json`和`timing/process.log`为准。最终看`COMPARISON_RESULTS.json`和`PIPELINE_FINISHED.json`。该批次上限14284800次原生环境积分，有限布局用完可少于目标。当前无正式采集扩配额或新训练授权。

## 2026-11-19 最新排查：原评分训练出现32个持续ReLU死区，同数据对照解卡但泛化仍弱

用户要求排查并问“原网络结构／是否数据太少”。明确原正式模型就是发布480→20结构及原ReLU后MSE配方；117条累计数据含79正38负，“1个”仅是16内部验证中11正例有1个过0.5，不是只采到一次成功。新增`author_mjx_head_diagnosis.py`，根`outputs/boya_mjx_head_diagnosis_20261119_v1`，不改任何旧冻结source/input/archive、原模型、旧435或CanonDex；无新采集／CGR／录像。

117行原record/candidate/execution SHA、标签、raw2261→发布canonical480（原函数＋独立half-swap/roll）、raw_depth×4+extra逐行exact；原Loss全部20地址独立检查过。fit100=68正32负，internal16=11正5负，全三体对象隔离保持。fit仅12/20输出档位有标签，index12占52/100、internal14/16，另internal2行index8；internal没有完全未监督地址，不能用地址全缺解释这16行。原selected epoch56的32个fit正例严格0、其选中预激活的ReLU后MSE直接梯度0；不等于BN共享／其他参数全部无梯度。只用fit重估BN，内部TP仍1/11，MSE .632079→.634225，非主要修复。

预声明4个同seed2026111903/同first100/同batch128/原Adam-PolyLR/100epoch CPU对照，各100真实更新，共新增400诊断更新。原ReLU重放全部100epoch train/internal loss及选中epoch56全部tensor exact；第10..99轮相同32正例持续非正预激活，原fit损失约.32。比较统一固定末轮99而非挑最佳：原ReLU fitTP36/68、internalTP1/11/FP0；仅BN7 bias初值+1（结构／损失均原样）fit58/68、internal4/11/FP0；训练预激活MSE fit68/68、internal4/11/FP1；Softplus输出MSE fit68/68、internal4/11/FP0。最后两者改变训练配方，不能称原作者结果或默认替换。原骨干/state_dict形状不改，独立published inference重载用于评估兼容；Softplus的推理激活不同，不能拿原ReLU推理冒充同一输出。

训练预激活MSE末轮fit MSE .000965（batchstats4.21e-7），泛化仍差；Softplus internal MSE .315352仍劣于fit正例率.68常数的.2149。训练死区是实测问题，小样本／覆盖与泛化问题仍在，不能说仅修ReLU就可靠、增加数据一定解决或内部4/11就是实抓成功率。旧435分支11月7日曾独立发现同类现象，本轮新域单独重做，不搬旧5/8成绩／旧权重进来。

2个数值CPU测试通过（发布forward exact/BN probe不写原buffer、末激活隔离及死区梯度），诊断tool86054 exit0，训练run段37.754s，另只读audit入口`audit_author_mjx_head_diagnosis.py`核原源/archive/117行及4末轮权重。原正式100更新保留，诊断400单列；不自动部署诊断头、继续物理或追加训练。报告`AUTHOR_MJX_HEAD_DIAGNOSIS_20261119.md`与轻证据`evidence/author_mjx_head_diagnosis_20261119_v1`，轻源码／证据push用户origin保持工作区干净。大型权重／数组和原始日志继续ignored，不能改已派发新诊断源码。

## 2026-11-19 最新收尾：累计101fit＋16internal已采满，正式100样本头训练完成，召回仍弱

`outputs/boya_mjx_cgr_to100_20261119_v1`已target_complete，原tool79363 exit0，实际父wall19009.471s＝5小时16分49.471秒；不是派发至晚查的10小时。432场实际处理，368未启动槽标quota_reached_not_started，不当失败。110次抓取与回位，97有效（63正34负）、13无效保持null；另164placement_null/108planning_no_candidate/50planning_null_geometry。新增85fit＋12internal，与旧正式16fit＋4internal按同域引用合并为117条＝101fit（68正33负）＋16internal（11正5负），旧诊断/对照/PhysX/435均未混入。达到配额停止准入，在途第101条fit负例保留。

27个wave完整raw独立审核全PASS，97有效样本均审核通过；13invalid导致原aggregate all_complete_grasps_pass=false必须保留，不说110抓原严格七门全过。原1mm诊断、5mm硬停止及其他六门不改。202268global controls/25890304native环境积分，NativeCall404537行begin/end完整、unknown0；完整28920360接触/873952trial rows匹配。54源/输入/archive exact；111登记owner/PGID全退，6锁可取、4预约released、registryreleased，无采集后台。原CAMPAIGN_RESULTS/merged MANIFEST的training_updates0是采集阶段历史事实，不回写。

按累计100后重训的既定流程，新增独立`run_author_mjx_fit100.py`及`outputs/boya_mjx_fit100_20261119_v1`，原author_training/公开源保持。固定first100 fit＝68正32负，全部16internal仅作验证/原选模；第101fit不用于本cap100。seed2026111903，原100epoch/Adam/480→20/selected-output MSE/PolyLR，正式非development，100实际optimizer更新。epoch56选中唯一best_0.5；原发布inference strict重载及16行保存输出exact。训练主体6.720s，含导入/核验的run段41.565s；这是CPU训练耗时，不是采样倍率。tool38200 exit0，0新物理/CGR/视频。原sandbox RUN_STARTED PID3是本地命名空间标识，不能作为host PID3查询/操作。

0.5门在16内部验证TP1/FP0/TN5/FN10，正例召回1/11，当前效果弱；这16条参与过选模，不能称独立外测、模型增益或新头真实抓取成功率。新权重未自动替换原CGR采集策略；下一步先查分数及多候选同观测排序，再做独立新布局评测，不因本状态查询自动追加物理/训练/改门。新训练source/input/archive已绑定，不热编辑或重跑同root。两个阶段均未录像，必要RGBD/原完整轨迹保留ignore，不删除原始证据。

最终报告`reproduction/boya/AUTHOR_MJX_TO100_RESULTS_20261119.md`，轻证据`evidence/author_mjx_to100_20261119_v1`与`evidence/author_mjx_fit100_20261119_v1`；轻代码/文档push用户origin并保持Git干净。下文“运行中/尚未训练”均历史快照，以最终结果与独立FINAL_REVIEW为准；不要重启已结束根。

## 2026-11-19 新授权：启动累计100有效fit标签常驻扩采，不录像

**实际已启动，当前不是仅准备状态。** tool session79363，调度host PID/PGID2119361、GPU controller2119614，PIPELINE_STARTED绑定已push commit4a4b7d5且worktree_clean=true。54源/输入/archive exact，40物体银行已建立并进入首wave稳定；启动核验时已完成69248native环境步并继续推进（在途另128，不作最终成本）。此时累计已审核fit16/目标100、新有效0，剩84，仅启动快照。global registry现在属于该root并active，不要将本批当旧已结束任务释放或重启。以实时BATCH_COUNTERS及最终CAMPAIGN_RESULTS为准，无录像。

用户在“已有16fit，还差84，沿原CGR补到100”方案后明确“那你要启动吗”，现在授权启动该目标。root outputs/boya_mjx_cgr_to100_20261119_v1；new runner run_author_mjx_to100、新author_mjx_quota_collection/progress/data。原CGR固定g1、原250Hz/0.5ms×8/物理/映射/规划/完整接触与审核均不改；正常有效失败也计fit，null不计，无逐体修复、换候选/失败重试/录像/新训练。旧20正式数据的16fit4internal作为冻结基线，诊断/对照标签不加入。

固定seed2026111902，预抽800个新场景（640fit/160internal，4fit+1internal交错），静态480ready/320placement_null。800是最大派发容量而非必须采完；16world整个队列只创建一次机器人/模型/相机/CGR，边界换物，原物理回位保持。累计已有16＋新有效fit完成原回位>=100即停止新wave准入，等待所有完整审核后确认配额，保留在途超额标签，不伪造恰好100。PROGRESS/BATCH_COUNTERS另列processed与unstarted，未派发槽不能当采样失败计有效率。

本批独立上限119040000native环境积分（50wave×18600control×16×8），collect+audit父超时64800s，单计划/审核仍1800s；80GiB空闲空间guard。若800容量/步数/存储边界先到则合法收尾并明确target_not_met，不自动扩预算或重跑。达到目标只自动导出/同域合并，训练本阶段仍0，后续再做。merged/保留原trial/record/hash引用及全部overshoot，正式训练未来按原first100前缀取样。只合并明确formal_training_eligible且相同scientific source/物理/标签域，拒绝重复trial及全三体fit/internal泄漏。

为避免800布局反复读取全源文件，新增纯准备PreparationCache：老文件按dev/inode/size/mtime/ctime复用SHA、最近1秒文件始终重哈希防粗时间戳快速同长改写，源顶点只缓存不修改；准备前后完整未缓存SHA再核。原确定性采样/验证函数及源文件不改。6项CPU测试过（3新配额/缓存＋2既有Progress＋1partial），旧真实场景original/cached完整字典exact，旧20数据分片合并数组exact/重复trial与不同物理拒绝，均0积分/0新标签。早期静态缓存测试发现时间戳粗粒度已在冻结前修复；首merge夹具漏audit引用的失败v1保留，v2补只读symlink后通过，不改生产源或标签。

CAMPAIGN_READY/collection PROTOCOL已冻结，启动以PIPELINE_STARTED/collection RUN_STARTED与native journal为准；此条不把静态freeze当已采集。新源/输入/archive核对后、代码push且Git干净再派发；源/输入/归档/CanonDex及历史数据均保持。来源报告AUTHOR_MJX_TO100_20261119.md；进度优先看collection/BATCH_COUNTERS的cumulative_audited_fit_records、remaining_audited_fit_records，最终CAMPAIGN_RESULTS/merged/MANIFEST或PIPELINE_FAILURE。

## 2026-11-19 最新完成：评分头新布局实抓闭环通过；本次排序收益不可识别

root outputs/boya_mjx_policy_compare_20261119_v1已complete，tool43087 exit0。16固定internal新布局抽样8placement_null保留，8ready全部各跑cgr/head，共16world/8对；权重冻结best_0.5.pt、原碰撞/规划/动作/物理/标签保持，0训练/重试/视频。总wall950.902717s（15分50.903），collector+audit941.678297s。CGR组assigned8/executed0/eligible0/success0/null8；head组assigned8/executed1/eligible1/success1/null7。13槽碰撞筛选后无候选，另外2槽规划几何拒绝，head15唯一候选实抓成功并原回位通过，完整原raw审核PASS。新1条仅exclude_training测试，不加入旧20/模型。旧16fit小模型100更新不改，不自动替换原采集排序或追加数据。

核心结论：16个live候选池13空、3个只有1候选，0多候选、head改变top1数0。8对的网络输入及raw候选特征exact均0；自动汇总semantic_original_pool_exact=6仅六对共同空集合，**非6对可比较候选**；nonempty exact pool pairs0、both qualified pairs0。原自动head_task_wins=1只是含空结果的任务计数，不能称head胜过CGR或提升12.5%；原统计字段不回写，FINAL_REVIEW另明ranking_benefit_identifiable=false。独立物理稳定后的观测不同，原CGR组未执行与head组成功不能作纯排序因果比较。两边都只有0/1可选，任何给定可用池内CGR和head排序决策相同。当前只证明真实新观测→模型打分/选姿→规划→实抓/回位/审核功能闭环，尚无评分收益证据。

另独立SHARED_POOL_DECISION_COMPARISON已对16个新观测逐个使用同一个原始pool分别算两种策略，16/16决策一致（13共同空、3共同选唯一候选），head_changes_top1=0，0额外物理。这个同输入决策对照有效，但不伪造未执行物理结果、不推论未见多候选的模型优劣。成功graspnet_021末源底107.459mm、参考升103.471mm；原1mm诊断false，按既有functional通过，不称严格七门均过。

8844global control/1132032native环境积分，17689行NativeCall完整begin/end未知0<2380800本批上限；35094phase rows/2868357全部contact匹配，独立native-palm FK max5.237115e-7；all_complete_grasps_pass=true是唯一已执行抓的原functional资格，不自动解释成所有16槽原七门过。49source/input/archive/冻结头核验通过，6登记owner/PGID全退、6锁可取/4预约released/registryreleased，无本轮后台任务。实际video文件0，逻辑文件8.158GB主要完整JSONL，保持本地ignore不删原历史证据。

最终报告AUTHOR_MJX_POLICY_COMPARISON_RESULTS_20261119.md，轻证据evidence/author_mjx_policy_compare_20261119_v1；源代码b474966已push，结果轻文档另push并保持Git干净。后续先沿原CGR采集积累有效正负，再在有多个候选、共享观测且独立新布局的评测上验证头；新总配额未获答复，不自动扩到100/300/1000。下文“运行中/冻结未派发”均历史快照。

## 2026-11-19 当前新授权：固定小样本头与原CGR排序做新布局实抓对照，不录像

用户“好吧对照下再说”授权有限比较，不是追加大规模采集/训练。新root outputs/boya_mjx_policy_compare_20261119_v1，固定seed2026111901先抽16个internal新布局，8placement_null保留，8ready全部用于8对/16槽（每对cgr/head，固定RNG随机分配相邻lane位置），只按建场几何准入，无CGR/IK/抓取结果择场。来自原internal对象池，场景未参与训练/选模，但不称新物体域外测。

新author_mjx_policy_compare_worker只在原碰撞筛选后CGR top10内按已冻结pilot inference_checkpoint（best_0.5.pt）selected-depth分数重排，ties保留原CGR次序；baseline仍原top1，额外head打分只作shadow记录，不影响选择。候选payload/姿态/闭合/碰撞/机械臂规划/原物理门均保持，无换候选fallback/失败重抓/训练/录像。opt-in新controller复用原hour.collect_wave及v9/native_batch v1/F.settle/F.motion与完整独立闭wave审核，仅注入新规划worker，不编辑旧源码。Ranker在各worker加载一次固定头、CPU推理；原CGR仍常驻且原每帧恢复同post-init RNG不变。

两边各自在独立lane按相同源布局稳定/实拍/CGR，初始化坐标配对，不声称稳定后状态/观测或物理轨迹逐值等价。最终逐对核对实际网络输入points/coords、raw候选与特征、语义候选池是否exact；有差异须报告、不得当纯排序因果收益。汇总比较assigned8的任务成功、可规划/实际执行/合格样本、paired wins以及双方合格子集；空/无效仍null，诊断exclude_training不并入旧20或任何模型。

静态freeze完成，16env一wave最多2380800原生环境积分，独立本次预算、最多16次抓取/回位、0重试/训练/video。2个ranking单测及一个旧真实候选池接续检查通过（0新物理）：baseline字典逐值不变，head从原4候选中选rank1，所有candidate SHA与trace保留；这只是工程检查，不是比较效果。新48左右source/input/archive需在实跑前核对，root还未派发时不得宣称已实抓。

入口run_author_mjx_policy_compare freeze/run，COMPARISON_READY/INTENT已落盘；调度事实看PIPELINE_STARTED、collection/RUN_STARTED及BATCH_COUNTERS，最终POLICY_COMPARISON_RESULTS/PIPELINE_FINISHED或PIPELINE_FAILURE。保持工作区干净后启动，轻成果push用户origin，旧source/input/archive/CanonDex/权重/20标签均不改。报告AUTHOR_MJX_POLICY_COMPARISON_20261119.md。当前冻结源码已绑定，后续不热编辑。

## 2026-11-18 最新：用户优先跑通闭环、明确不录视频；小样本真实训练与推理已完成

用户改为“失败物体先不管，先把流程走通后续提高成功率”，随后问进展/采集ETA并明确不需要视频。此后所有新流程video_recording=false，不启动录像/逐步渲染；必要单次RGBD观测仍保留。此前100场实际video0，磁盘文件逻辑体积28.777GB，其中JSONL25.506GB，不把轨迹接触日志误称视频；不删除或覆盖旧冻结原始证据。

当前100场早已完成，16fit/4internal有效20条；上一轮只给了下一步计划、尚未训练，本轮按先通闭环启动独立real-label小样本pilot。root outputs/boya_mjx_head_pilot_20261118_v1；新增author_mjx_head_pilot复用原components/CanonicalDataset/train_one_epoch/eval_one_epoch/save_model，原100epoch/Adam/selected-output MSE/PolyLR，16fit=14正2负、4internal=2正2负，0平衡/旧域合并/换seed。正式100/300/1000 sample-cap函数及历史frozen源不改，本次明确limited_data_pilot，不以16条充当正式100条实验，也不套仅synthetic的development_epochs捷径。

实跑CPU单线程100epoch/100次optimizer更新，训练主体wall2.869812s，参数SHA变化；原验证规则选择0.5/0.7权重，另存显式last_pilot（非自动晋级部署）。三份checkpoint weights_only重载、原inference与source eval对4行输出[4,1,20]逐值exact，20行真实候选输出有限。0.5所选头在4条内部验证TP1/FP0/TN2/FN1，仅小样本流程证据，不称75%总体成功率/外测或新场景提起效果。0新增physics/CGR/video；训练进程已退出，无本轮后台任务。旧采集manifest上的training_updates0作为历史保持，新pilot报告累计100更新。

当前已通真实采集/标签/导出→小样本训练→保存重载→已保存真实候选评分，live_candidate_ranking_integrated=false、new_scene_execution_complete=false，未把模型接入新场景规划/实抓，不能称完整训后实抓闭环已验收。优先接推理与少量新场景验证，暂不逐体修到成功；正常有效失败仍作负例，未执行/无效仍为空。

用户“多久采集完”未说明新的有效样本总目标，异步问题已给三选项：先完成训练推理再定扩采／补到100有效fit／补到300有效fit；未答不自动追加队列或大物理预算。按上批16fit/约71分44的单批比例，补到100fit尚缺84，粗略同条件约6.3小时，仅情景估算，布局和宿主负载会变，不是实测采满ETA。论文正式cap不作自动采集目标。

报告AUTHOR_MJX_HEAD_PILOT_20261118.md与轻证据evidence/author_mjx_head_pilot_20261118_v1；checkpoint与scores留ignored outputs，新轻源码/报告push并保持工作区干净。此前“训练0/未开始训练”是pilot启动前历史，不回写原成果。

## 2026-11-18 最新完成：100新场景扩采收尾，20正式标签16正4负，训练0

root outputs/boya_mjx_fresh_expansion_20261118_v1已完成，无后台采集/审核/后处理/训练。tool18366原调度exit1是已预知的逐体统计object_outcomes KeyError，原collect+audit与staging export均exit0；独立watch40508 exit0按精确错误完成缺失元数据，原PIPELINE_FAILURE/冻结source/input/archive/标签均保留，0物理重试。正式完成回执EXPANSION_METADATA_COMPLETION.json、formal/MANIFEST及FINAL_REVIEW全部通过；不存在原EXPANSION_RESULTS，不能只据旧PROGRESS.json motion误报仍在运行。

100场/80fit20internal，63ready/37placement_null，63新RGBD/CGR/规划，21实际抓取及21原控制回位通过；20有效正式样本=16正4负（fit16=14正2负、internal4=2正2负），不合并旧cohort。另27planning_no_candidate（全部CGR有候选但碰撞筛选后0，未进入臂IK）、15planning_null_geometry、1invalid_execution，合80null，不标负/不补抓。15规划空中11固定端点或非HOME段碰撞、2首次RRT连接后的逐帧/导数门拒绝、2局部IK种子未收敛（75/80，非不可达证明）；前轮已回答的前7规划空确实全部IK已解，不能把后半新增2例倒灌为此前原因。71执行1902/2351步遇原5mm停止，深度5.063880mm，labelnull且返回通过；因此aggregate all_complete_grasps_pass=false须保留，不说所有21抓均过数值门。20导出行均独立审核与原canonical480→20 head消费/无梯度检查通过，权重不变/训练0。

7wave原完整raw审计PASS，46342global control/5931776native env积分，NativeCall全92685行连续begin/end、未知0<16665600本批预算；完整接触和逐源底/载荷/FK/命令/回位均复核。collector+audit4260.605295s，原调度至统计错误4285.983127s，formal元数据文件mtime距派发约4303.899s（约71分44，仅近似全流程，非独立monotonic计时）。用户晚查时的279分钟是派发至查询，不能称实际采集用时。32登记owner/PGID全退、7锁可取、5预约released/registryreleased，无本批后台任务。

40种抽样/38种ready；29种只代表有效场景中出现，实际机器人接触及手载荷涉及9种，成功提起7种：000=1/002=4/014=2/021=1/022=4/057=3/066=1。不等于40种均被尝试或均有正负；未触碰旁观体仍不是负例。现正式fit16/internal4，旧PhysX等域不合并，旧4800/追加新批次/正式训练均未派发。代码截至f52eb9e已push，收尾报告与轻证据另小提交push并保持Git干净；原大RGBD/轨迹/模型继续ignored。

报告AUTHOR_MJX_FRESH_EXPANSION_RESULTS_20261118.md，轻证据evidence/author_mjx_fresh_expansion_20261118_v1。下一批先依据布局/碰撞候选淘汰和逐体接触缺口决定方案，本条不授权追加物理或训练。下文“运行中”均历史快照。

## 2026-11-18 新授权：先push并清理，再启动100新场景扩采

实际已启动：root同下，tool session18366，调度host PID/PGID1164791、GPU采集1164862；PIPELINE_STARTED绑定已push的c99d2110c8a7fa45eab8bec717adcd4533fbc328且worktree_clean=true，48绑定源码均tracked/归档exact。38种银行已建好，首wave9个ready槽进入实际GPU稳定推进；0新标签时已确认原生完成94848步且仍推进，非完成计数。最终须查原回执，不把该快照当最终结果。

启动后只读复查发现旧run_author_mjx_expansion.coverage对不完整执行缺object_outcomes会在最终统计KeyError；原物理/标签是正确null，不能因此重跑采集或改已绑定源。新增独立finish_author_expansion_metadata与1个partial CPU测试，已有25检查合计。watch仅等待原调度退出：原成功则不操作；仅遇该精确KeyError且原collect/audit/staging-export完整通过时，独立复核已生成formal NPZ所有科学数组/逐record/hash/原audit，再exclusive补缺失OBJECT_COVERAGE/MANIFEST并写EXPANSION_METADATA_COMPLETION，原PIPELINE_FAILURE和数据保留、0再积分/重试。缺结果计unavailable，不标负；其他失败原样保留不自动重派。watch登记METADATA_WATCH_STARTED/FINISHED/FAILURE，同目录；当前未结束须当后台任务，不能报告无后台。新增后处理源码另commit/push，采集的原48源/输入/archive不改。

用户明确“代码push了吗，大文件和无关文件ignore然后保持工作区干净后开始扩采”。授权范围为前述首批100全新场景（不是保证100有效标签），80fit/20internal、目录32/8对象隔离、g1、16常驻world；不恢复旧4800或训练。新入口run_author_mjx_expansion/author_mjx_expansion_collection，复用已验收hour collect_wave及v9/native_batch v1物理、照片/CGR/实测起点原规划/逐250Hz全字段/功能标签/原回位与闭wave完整审核；新Progress仅去掉小时准入截止。保留1mm诊断/原5mm停止及其他六门，不称严格七门均过或PhysX等价。新增24项CPU检查通过（3新＋21既有），0新增物理测试；完整运行仍须查实际回执。

root outputs/boya_mjx_fresh_expansion_20261118_v1；seed2026111802在任何物理前冻结，100独立抽样覆盖目录40，但63ready场景仅含38种，040/046只在37个placement_null中，不调seed或补布局追覆盖/成功。每场至多一原候选/执行，null消耗配额。16环境7wave最多16665600原生环境积分，独立新预算；0重试。40全覆盖的是抽样意向，不能宣传全部已实际尝试或都有正负。逐体报告分出现、机器人实际接触、手载荷、成功提起，接触未提起只诊断、未触碰旁观体不标负；全局label仍任一原三体被手提起。训练/新权重/训后外测0。

工作区先提交轻量代码/报告/证据至用户origin，outputs/大模型/轨迹/视频/缓存ignored；不删除原失败/诊断，不改历史source/input/archive或CanonDex。启动前入口强制git工作区干净并记录实际commit。是否实际派发与当前进度以PIPELINE_STARTED、collection/RUN_STARTED、collection/BATCH_COUNTERS、PROGRESS及最终EXPANSION_RESULTS/PIPELINE_FAILURE为准，不能把本授权记录当已开始/完成。完整原audit＋staging export通过后另建formal/records/NPZ/MANIFEST提升为本独立MJX功能域正式数据，原staging exclude_training和所有历史记录不回写，不合并旧cohort。正式训练仍须后续授权。

报告AUTHOR_MJX_FRESH_EXPANSION_20261118.md。下文“无后台任务/正式未启动”是此前收尾快照，新批次须检查上述root的实际状态；不要重跑已派发root。

## 2026-11-18 最新收尾：真实一小时39落盘/36时限内已审核，用户接受速度；40物体尚未逐体覆盖

用户明确实际跑一小时，已完成root outputs/boya_mjx_actual_one_hour_20261118_v1（session25548 exit0）。北京时间16:22:27.986244–17:22:27.986244/精确3600s半开窗口含初始化：runtime完整有效标签39（25正14负）；其中36在时限内也完成回位和完整独立审核（22正14负）。事件monotonic_ns独立recount与现场exact，最终39全通过审核。截止后的9条不计入39；在途完整收尾后总48抓/48回位、30正18负，独立export34fit14internal，原head eval过/权重不变0训练。总wall含在途+剩余审核/导出4408.394964s，不用它重定义3600s窗口。先前18.4是短批次折算，不是实测连续一小时；本轮审核与后续采集后台重叠，也不能全归因物理提速。

4关闭wave审计PASS：38872控制/4975616原生env、6507824contact、391692分阶段行匹配、独立palm FK max5.700351e-7。物理沿已验收native_batch v1/entry_v9原settle/motion/evaluation、原250Hz/0.5ms×8/24位置+6FF/完整17字段/原functional政策（1mm诊断、5mm停止及其他六门）；非严格所有抓取七门True，48返回原七门全过。1model/1GPUupload/1renderer/18bank/64新照片，CGR1load64forward，各机器人初始化一次0reset。19登记owner/PGID全退、6锁可取、4资源槽released/registryreleased，当前无本次后台任务。实际4975616<本次独立19046400预算，NativeCall journal全块完成未知0；此前阶段不混计。

用户认为一分钟多一条可接受、询问正式扩采：按整机39/3600平均92.3s/有效标签，严格时限内审核36/3600为100s/条，这是整机出样成本非单环境单条等待。建议复杂并行重构后置，可进入按新布局分批扩采；曾建议100新布局/批，但尚未获明确启动和正式总配额，不自动启动新的正式批次/旧4800/训练。

用户又问40物体正负分布，已核源CATALOG确40（fit32/internal8），并明确回答**目前不是40种都已有正负**。本小时7重复布局预载18种，39有效样本实际仅涉及15种、另25目录对象未入本cohort有效样本；实际成功提起5种：000=9/014=4/021=4/026=5/057=3次。object_outcomes保留真实物体结果；全局label是任一三体提起的抓取成功，不能将legacy target_object_id或场景里没被抓的旁观者当逐对象正负。scene cooccurrence正负分布不是逐对象抓取标签。后续正式采集应覆盖全部40、分清出现/实际尝试/实际提起和缺类，保留自然失败、不强行凑双类/不依失败改控制。不混旧435或其他域/诊断cohort来填覆盖。

报告AUTHOR_MJX_ACTUAL_ONE_HOUR_20261118.md，ACTUAL_HOUR_RESULTS/FINAL_REVIEW/SAMPLE_TIMESTAMPS/OBJECT_COVERAGE_WITHIN_HOUR在本轮root；轻证据evidence/author_mjx_actual_one_hour_20261118_v1。大轨迹/RGBD ignored、本地保存未远程发布；source/input/archive/CanonDex保持。下文“正在运行”均历史快照。

## 2026-11-18 当前实跑：用户明确要求真实一小时计数，16:22:27–17:22:27北京时间，禁止用短批次外推

**最新：3600秒窗口已结束，只剩在途收尾/末批审核，禁止重启计时。截止内runtime有效标签39（25正14负），截止内回位且独立审核通过36（22正14负），源HOUR_COUNTERS在elapsed60.56min确认；截止后初始已有3条tail不计入39。GPU918494/session25548仍在完成第四轮已启动的12抓及回位，前三wave审核已PASS，第四wave结束后需审计/导出/独立recount和资源收尾。review_author_mjx_one_hour.py已写，需在wrapper退出后用host权限执行核PGID及事件计数；未做新物理回放。最终结果尚未生成，不能把尾部加到一小时。**

用户指出“实际一小时绝对不止18条”，明确授权一次真实60分钟测试。新root outputs/boya_mjx_actual_one_hour_20261118_v1、工具session25548，GPU controller host PID/PGID918494；window monotonic_ns1809090849813602..1812690849813602，unix1794990147.9862444..1794993747.9862444。从进程派发前开始，包含冷初始化，半开窗口[start,deadline)。截止后不准启动新抓取，已在途抓取/回位完整收尾，尾部完成独立计，不加进一小时数字。不得提前结束后用倍率报一小时。

16常驻world/1机器人初始化每环境/原18银行，128槽循环全部七个placement-ready布局（0旧标签复用、每次新照片/CGR/原实测起点规划/动作/完整接触与原functional门），仅速度诊断、非独立128新布局，正式训练0。物理沿已实跑native_batch v1与entry_v9原settle/motion/evaluation函数，无新binary/分组异步控制改动。变化为计时准入和已关闭wave的完整独立审核后台并行（prepare≤3/audit≤2），各wave独立不可变JSON段+原绝对float32时钟offset，审核不截字段/不采样。主数为截止前完成有效标签、原回位PASS并独立审核结束的唯一样本；次数字为runtime有效标签、完整抓取、回位和截止后尾部。

author_hour_metrics四CPU测试过（截止排除、晚审核不追溯、null/重复不计、失败回位不算已审核）；3个旧真实tick offset9346的独立审核通过，0新增积分。新hour源/输入/archive均冻结，128槽最多19046400原生环境步，为本次明确一小时测试独立上限，不挪原5M阶段；0重试/0训练/不晋级正式配额。source/input/archive/旧根/CanonDex保持。新source run_author_mjx_one_hour/author_mjx_hour_collection/author_hour_metrics/author_hour_audit/audit_author_mjx_wave_snapshot。截止后需等在途收尾和剩余审核、导出、核所属PGID/锁/registry释放，再报ACTUAL_HOUR_RESULTS里的实测数。

## 2026-11-18 最新检索/收尾：并行主因在整批屏障与主机同步；32槽完整48分33/20标签，新增IO修复仅离线通过

用户再次明确“单环境也不止，检索并行是否没发挥”。已查官方MJWarp/mjlab和作者DexGraspNet原代码：本项目物理核确实16world并行，但entry_v9逐环境照片/CGR后等整批CPU计划，motion等最慢返回才补一wave；native_graph每4ms host拷贝+synchronize_stream，外层逐点解码/逐环境CPU观测+重复JSON。不能称16路持续采集/不能把整体慢归物理，也不能把20.316%计时范围占比当GPU利用率。Dex原validator先simulate一段后读contact，其use_gpu_pipeline=False，非完整臂/相机/逐250Hz全字段合同，不套其速度。

完整root boya_mjx_continuous_queue_control_20261118_v1已collect/audit/export均exit0，session49647结束：32新照片/CGR/规划、20完整抓取13正7负、8nocandidate/4nullgeometry、20回位原七门过；各环境仅初始化1次/换物2次、1model/18bank/1renderer。collect parent1917.017888/外部1925.387044，audit981.679487/export5.675468，完整2912.789502s=48:32.790，20有效诊断/24.718573每小时；独立13fit+7internal，0训练/不合并旧cohort。原JSONL13061330009字节，18702控制/2393856原生、158874分阶段行/3511341点contact全部独立PASS，palm独立FKmax5.176980e-7。原head实际eval过/权重不变。functional保留1mm诊断/5mm停止及其余六门，不说严格原七门全部抓取过。

source/in/out/archive/CanonDex不改。原全scratch同动作196608＋换物32＋本组2393856＝本修复阶段2590496原生步、未知0<5000000；旧阶段成本/未知另列。FINAL_RESOURCE_REVIEW确认12登记owner/PGID全退、3预约槽released、5锁可取/registryreleased，无后台GPU采集/训练。旧“正在运行”段均历史快照。

零积分完整allocation统计与原审核SHA/点数一致：nacon峰614对4352，单world nefc峰316对8192；候选两倍峰向上取2幂2048/1024，仅诊断建议，未运行/不保证未来需求界，保留原overflow不得截字段。root boya_mjx_allocation_review_20261118_v1。

新增无损Journal（4CPUtests过）176实录记录CPU写盘三次轮换中位JSON .455370s、syncbinary .058659/async .063399s，7.763/7.183倍仅该组件，含worker drain/fsync；全部字段/浮点/顺序exact。bulk_contacts 12真packet/1162点同原17字段及派生值canonical exact、函数2.266倍，仅转换函数，0积分。roots boya_lossless_journal_control_20261118_v1、boya_mjx_bulk_contact_control_20261118_v1。

新native_batch_v3/entry_v10接bulk/asyncbinary，显式补回JSON的NaN/Inf拒绝；ExitStack在构造/关闭失败时也收尾所有phase writer，原native cost仍逐行JSON。Native step真实包fixture和3类非有限拒绝通过0积分；v2旧fixture源保持。新auditor_v2兼容json/binary。CPU迁移root boya_mjx_binary_storage_review_20261118_v1对旧remaining_v2完整35282行logical SHA exact，1153480147→714398632字节；binary原独立审核18016分阶段行/232280contact/原门/标签及summary与原JSON exact，0新增积分/标签；其276256是旧物理来源计数，不能再加到本阶段。

outputs/boya_mjx_binary_entry_static_20261118_v1仅freeze前16输入/16world，0Kit/积分/派发；新源/输入/archive核过，最大2380800，与已花2590496合上限4971296<5000000。不能称entry_v10物理吞吐已验收。主调度整批等待和250Hz主机同步仍未改；下一架构重点为独立环境组持续任务队列、GPU成块推进/原停止判据在设备保留、全字段缓冲异步读回/观察/审核，不能让已拍照等待计划的world继续积分破坏绑定，不靠增加环境数或放物理门。正式单型配额仍未冻/训练0，不自动启动该静态根或正式扩采。

报告AUTHOR_MJX_PARALLEL_BOTTLENECKS_20261118.md、AUTHOR_MJX_EXECUTION_FIX_20261118.md；轻证据evidence/author_mjx_execution_fix_20261118_v1。源码/说明/轻证据本地保存，原RGBD/逐点轨迹仍ignored，未远程发布。

## 2026-11-18 当前提速修复：原全scratch staging开销已实测，native单图微对照2.578倍，连续采集已结束/独立审核中

用户指出慢是代码问题。本轮查明旧author_mjx_graph_staging在每个0.5ms子步对全部in-out scratch输入输出复制；16环境每4ms控制8step+forward合553649468字节staging、9次FFI，不能归咎纯GPU物理。新author_mjx_native_graph直接调用已装同一MJX-Warp内核，单CUDA图含8step、一次完整独立observer拷贝+forward，保留全部17接触字段/EFC/原物理及functional门，不热改旧冻结源。三段各128原保存控制、16环境，四模式共196608原生步：old热段19.373629s/native7.516407s，2.577512倍仅热执行段。受载机器人q最大差.001475573rad/v .512053rad/s，旧自身重复也有差，不能称逐值等价；original limit/mimic样本过不代替完整审核。observer_copy候选因同进程JAX lowering缓存未证实际替换，计时剔除。微对照root boya_mjx_execution_path_control_20261118_v1已结束。

新native_batch使用稳定buffer、仅观测lane更新CPU运动学、逐行native费用/全状态日志。2world换物probe root boya_mjx_native_bank_check_20261118_v1已PASS/32原生步：机器人/其他world/时钟/ctrl/warmstart exact、graph地址稳定、0reset/0CPU积分。旧八槽比较16/64只派一wave，不代表每环境持续一小时只一条，不能用总标签/16解释单环境实际产能。

新有限root outputs/boya_mjx_continuous_queue_control_20261118_v1已freeze32槽/16env/2wave并启动，工具session49647、GPU host PID/PGID136066；循环使用全部七个旧placement-ready布局，独立于抓取结果，非32独立新场景，所有照片/CGR/原实测起点规划/抓取/返回重新执行，不复用旧标签。entry_v9及native模块已归档绑定，原冻结源/CanonDex不改。队列最多4761600原生步；加微对照196608与probe32，共上限4958240<本修复阶段5000000；0重试/0正式训练/正式扩采配额仍未冻。通过run_author_mjx_queue_control完成collect→原独立audit→诊断export，不先报完整采集倍率。旧阶段已收尾费用及未知上界单列保持，不混入本阶段。

最新collect已complete/GPU136066退出，当前仅原只读audit host PID/PGID209884及wrapper135997等待（session49647）；未再积分。32新拍照/CGR/规划、20完整抓取13正7负、另8nocandidate/4nullgeometry、20回位原七门全过。parent1917.017888/外部1925.387044s，18702控制/2393856原生；加两个检查本阶段2590496<5000000。推进＋GPU观察/读回389.467255s占parent20.316%，接触解码154.563342s（旧字段contact_decode_and_kinematics_s实际仅decode）、setup80.769334、照片200.234470/CGR45.371332、两波规划worker最长合519.820050、未分区外层526.792105s，不能继续主归因物理/不能把余量全说JSON。JSONL总13061330009字节。原CPU模型XML SHA与旧16 exact；v9 comparison_underfilled沿旧八槽描述值True，实际本轮每wave16输入、各环境2次分配；只读review另解释，不回写冻结字段。完整audit/export/关闭回执尚未完成，不报最终标签率或总体加速；报告AUTHOR_MJX_EXECUTION_FIX_20261118.md和轻证据已记录组件结果。

## 2026-11-18 最新收尾：完整MJX八槽＋1/16/64全流程均PASS，16批次标签吞吐1.662倍，训练0

所有本轮任务结束，无后台采集/训练。新comparison_v4三模式完整collect/audit/export均exit0，env001完整1624.649298s（27:04.649）5标签2正3负/3null、11.079314标签/h；env016 977.738825s（16:17.739）5标签2正3负/3null、18.409824/h；env064 1240.897455s（20:40.897）5标签4正1负/3null、14.505631/h。相对单环境MJX观察倍率16=1.661639、64=1.309253；同8输入、一次固定顺序、实际新照片/候选/路径不同，高模式仅7有布局、64含57空闲环境全部计成本，不当纯物理/同动作倍率或满载产能，不推采满ETA。各次五个实际返回原七门全过，抓取沿既有functional政策（原1mm诊断、其他六门与5mm停止保持），不称全部严格七门过。

每模式1共享模型/18银行体/1renderer/7新照片与CGR/7原规划/5抓，各环境robot初态一次、0reset；所有source scene与静态MODEL.xml SHA exact。全局控制33790/8402/9320，原生环境270320/1075456/4771840。原指令24位置＋6源前馈/实际float32、完整GPU原生17字段及EFC/float32每原生时钟、source极值/载荷/XML FK/lift/返回/真实RGBD和CGR输入均独立复算PASS；匹配33790/37586/37102行，接触357399/648675/662562。实际原评分头输入与loss eval/no_grad过，权重不变0训练。三个各5行NPZ独立，原八槽先行v3＋remaining两个NPZ共5行也独立；不合并旧CPU/PhysX/旧31/435/权重，不算正式新增。

原HOME交接从当次实测六臂通过原native --initial-arm-state传入原RRT/几何/retiming；N1slot2实际偏名义25.911331µrad>旧20µrad假设而计划起点对实测exact，抓取/返回已跑通，不放门、不写robot q。运行时opt-in entry_v8/live_batch_v4/full_comparison_v4使用WARP_STAGED_EX及author_mjx_graph_staging补全部既有in-out数组拷贝，依赖/内核/物理参数不改。微型检查原全qvel阈值仍因物体角Y .012425而失败（默认重复也.00615449），原exit1和RAW保持；只读合同复核不是原阈值PASS或完全等价。当前三个有限完整模式正常跑完，不保证任意长运行稳定。CPU镜像0积分/接触解算，GPU独立post-forward观察；原生字段不当PhysX filtered合同。

先行八槽原成本607680；旧默认图速度组SIGSEGV只有8848 durable控制/70784已知原生，原先0..8上界遗漏1MB日志缓冲已更正：default buffer1048576/textchunk8192/minimal row4927，最多214缓存行＋1在途块，另源首波最多9104控制，故未知0..1720、成本70784..72504。audit_author_mjx_buffered_interruption独立零积分复算，输出boya_mjx_log_buffer_review_v2；旧错误review/protocol字段保持不回写。含1280微检查和正常N1合342384..344104<原1190400，0实际超支。全部三个比较连此前开销6189680..6191400<13094400；连先行主验收本轮6797360..6799080<14284800、0Kit，历史JAX未知0..8等更早阶段单列，不说全项目未知0。

最终host核11个实际GPU根/15 owner与PGID全退、13预约released、21所属锁及GPU0共享锁可取、registry released。原DH3paused、正式单型配额未冻、训练/新模型/外测0，不自动扩采或复跑已派发根。9不同CPU tests（5新＋4既有）过0积分；所有旧source/input/archive/失败/CanonDex保持。完整报告AUTHOR_MJX_FULL_COLLECTION_20261118.md，summary及FINAL_RESOURCE_REVIEW在outputs/boya_mjx_full_comparison_20261118_v4；轻证据evidence/author_mjx_full_collection_20261118_v1。源码/说明/轻证据保存在本工作区，大RGBD/轨迹/逐帧接触仍ignored，未作远程发布。

## 2026-11-18 最新运行：1/16完整对照均PASS，16观察提速1.662倍；最后64环境组执行中

当前仅outputs/boya_mjx_full_comparison_20261118_v4/env_064运行，GPU host PID/PGID3479318、工具session88211，64各机器人初态一次、1共享model/1renderer/18银行体，MODEL.xml与1/16 SHA exact；7有布局槽1000控制稳定后新RGBD/CGR7/规划7，5槽ready（0/1/2/3/5）并行执行，4和6无候选、7placement_null。MAX9523200原生步，所有空闲环境积分/完整接触/状态均计。未完成前不报64性能、标签或满载吞吐。

N16工具session6637已exit0：同八槽5抓2正3负/3null、5返回全七门过；8402全局控制/1075456原生步。完整wall977.738825s=16:17.739，collect723.247813（parent716.983466）/audit252.617269/export1.867882，合格标签18.409824/h。独立audit37586行/648675接触/palm-FK max4.934244e-7，源/命令/判定/导出/head eval通过、权重不变训练0。对N1完整1624.649298s/11.079314标签每小时，观察wall及标签率比1.661639；同输入但各模式稳定窗口/新照片/候选与命令不同，单次固定顺序，非纯物理或同轨迹倍率。主验收旧5条和三个性能cohort独立，正式配额/训练0、DH3paused。

先前默认图崩溃费用最新上界仍是日志缓冲修正后的70784..72504（未知0..1720），不是旧0..8。audit_author_mjx_buffered_interruption零积分独立复算，与手工review一致；新结果outputs/boya_mjx_log_buffer_review_20261118_v2/RESULTS.json，含1280检查和完整N1合342384..344104<1190400。旧错误review/protocol/source/失败保持，不能回写；N16/64独立预算不受此修正影响。

## 2026-11-18 最新：单环境全流程27:04.649通过，16环境运行；崩溃成本上界因日志缓冲已纠正

新comparison_v4/env_001已完整成功退出/审核/导出，工具session69999 exit0。1model/1world/18bank/1camera/robot初态一次/0reset、7新照片/CGR与7规划、5抓取2正3负、3null（geometry/no candidate/placement），5次物理返回全七门过；33790控制/270320原生步。collect外部1438.932056s（parent1434.500734）、audit183.794614/export1.917731，完整wall1624.649298s=27:04.649，5条合格诊断=11.079314/h，slots17.726903/h。原head输入/loss eval过、权重不变、训练0。原1mm仍诊断，functional不是原严格七门全过；旧八槽主验收4正1负另两个根保持独立，不合并这些性能重复数据。

旧comparison_v2/env_001 crash成本原估70784..70792低估，因Path.open默认二进制buffer=文件系统block1MB，旧journal并未逐行flush。新outputs/boya_mjx_log_buffer_review_20261118_v1/CORRECTED_INTERRUPTION_REVIEW（0积分）实测buffer1048576/textchunk8192，保守最小完整global行4927字节，最多214缓存行+1在途块，另源首轮phase最多9104控制/下一capture不存在，故已知70784＋未知0..1720，总70784..72504。旧错误review及protocol保留，新review明确取代旧上界。原N1freeze按72072（旧上界+1280检查）扣，真实先前上界73784；实际完成N1含所有旧失败/检查合342384..344104<1190400，0实际超支。不能再写未知0..8或总0未知。

当前仅comparison_v4/env_016运行，GPU控制器host PID/PGID3437526，工具session6637。静态MODEL.xml与N1 SHA exact；同八槽（7有布局），16各robot初态一次，稳定共1000控制后全部7照片/CGR完成，7CPU规划在3进程内进行，GPU当前暂停积分保观测绑定。全部私有字段staging/原控制/functional门保持。N16独立max2380800原生步；N64同root/env_064仅freeze，max9523200，须等N16完整审核导出后顺序派发。高模式空闲环境全部计成本，不能报满载64吞吐。主验收607680已知步、原DH3paused/正式配额未冻/训练0保持。

## 2026-11-18 最新运行：完整八槽验收PASS；固定buffer/完整原生字段版单环境基线运行中

主验收607680原生环境步、原八槽4正1负3null、v3的3条与remaining_v2的2条独立NPZ及两原始审核PASS保持，正式配额/训练0、DH3paused。旧默认单环境速度组exit -11，8848控制/70784已知原生＋末块0..8不确定，INTERRUPTION_REVIEW已核host owner/PGID3308071全退及锁/预约释放，不能当有效基线。

图模式检查v1字段写法错误在JAX tracing退出0步；v2默认STAGED_EX没拷私有字段，512步后nacon0失败；v3新增author_mjx_graph_staging，只扩充既有in-out数组的stage_in/out拷贝，依赖/内核/物理参数/数组schema不改。三模式WARP、原样重复、修正STAGED_EX各256，共768步。修正所有观察nacon/nefc与默认一致，末各10点/17原生字段/约束载荷恢复；机器人q差2.384186e-7rad/v差1.622527e-5rad/s。混合qposmax8.658506e-6、全qvel .012425在graspnet011角Y（默认自身重复也.00615449）；原0.001全qvel阈值仍失败、原脚本exit1保持。STAGING_READ_ONLY_REVIEW只确认读回合同与实测残差，不把原阈值改成PASS、不声称完全数值等价或长运行崩溃已修。已知检查共1280步、0抓取/标签/训练；最大native预算按70792+1280=72072扣除。

新author_mjx_live_batch_v4/entry_v8/full_comparison_v4均已freeze。当前仅outputs/boya_mjx_full_comparison_20261118_v4/env_001实际运行，GPU控制器host PID/PGID3372871，工具session69999；每环境初态一次，首槽750settle过/原CGR与实测起点原RRT ready，正在完整抓取。N1原1190400总预算扣最坏72072，剩1118328；N16=2380800/N64=9523200，两者在同comparison_v4仅freeze，必须等N1完整计时与审核结束再顺序启动。新三模式均同八槽/同固定缓冲+完整私有字段/原functional门/无候选补抓/0训练，高模式空闲积分全计，不称满载64吞吐。原comparison_v1/v2/v3根和所有失败/source/archive保持，不热续、不以旧组替新速度。

## 2026-11-18 最新纠正：单环境速度基线CUDA图崩溃，暂停高模式；运行固定缓冲兼容检查

之前“基线运行中”的owner3308071已exit -11/PGID全退，root comparison_v2/env_001没有RESULTS/OWNER_SCOPE_FINISHED。首槽完整4127指令正例已RECORD落盘，返回在8848全局控制tick时崩溃，原始日志/源码保持；原GPU日志8848完整行/无截断=70784已知原生步，末未返回块0..8不确定，故比较阶段成本70784..70792，不能报未知0或完整速度。INTERRUPTION_REVIEW已由host确认所属组退/预约released/全局锁可取，0新积分，原registry随后由新有限检查合法接管。该首正例暂不计入原主验收5条合格导出。16/64尚未启动，不能按旧entry派发。

崩溃日志为wp_free_device_async failed to get capture graph及CUDA901，唯一根因未证明，不归因接触/动作。官方MJX与已装3.11源码提供WARP_STAGED_EX稳定buffer模式，避免JAX地址变化反复捕获。新author_mjx_live_batch_v4/entry_v8采用该官方选项，只改执行缓冲/图模式及回执，物理/控制/门保持；尚未完整实跑。新author_mjx_full_comparison_v4计划所有模式全新同版，并从原单环境1190400预算中扣旧崩溃最坏70792及兼容检查512，不按结果换候选。旧comparison_v1/v2/v3和旧来源保持，只能作freeze/失败证据，不能热续。

check_author_mjx_graph_staging首次在JAX Python tracing读错DataWarp.contact字段而退出，未编译/派发积分，0原生步；TRACE_FAILURE_REVIEW保留。新check_author_mjx_graph_staging_v2 root boya_mjx_graph_staging_check_20261118_v2当前运行，SOURCE旧v3同一loaded tick5375/lane0，原WARP与新WARP_STAGED_EX各32控制×8native=合512上限、0抓取/标签/训练，每块前后持久化NATIVE_COST；这是运行时数值检查，不证明长运行崩溃已修。通过后才freeze/派发新的完整1/16/64，所有先前成本另存且扣预算。已完主八槽607680原生/4正1负3null/两NPZ独立/audit均PASS保持，原DH3paused、正式配额未冻/训练0。

## 2026-11-18 最新运行：八槽补组全部审核结束，完整单环境基线运行中

上一条进行中为历史快照。remaining_v2已complete/parent997.744406s，2world/1model/1renderer/各robot初态一次/0机器人reset、9bank/3照片/CGR3/规划3/2抓均正、2物理返回全部七门通过，两波共17266控制/276256原生环境步。只读audit PASS：18016行/232280接触点、GPU掌姿对独立FKmax4.938043e-7，0再积分；独立导出2fit正例，internal0故最低划分检查拒绝（正常保留，无补例/训练）。与旧v3合原八槽4正1负3null，两个根NPZ保持独立；先前224失败＋331200＋276256=607680原生步，低于原1190400。原GPU host PID/PGID3262349已正常返回退出，当前不再属于活动组。

用户授权的完整1/16/64对照已启动单环境：outputs/boya_mjx_full_comparison_20261118_v2/env_001，wrapper host3308007，GPU控制器PID/PGID3308071，完整工具session21121。新v6相机为全源面Numba96射线（七场672条既有回归过），首槽capture6.184s，尚未完成完整八槽。此计时含初始化、完整抓取回位、日志、审计与导出，不能拿组件或旧2.210倍替代。

高模式使用新comparison_20261118_v3，N1只读复用v2正在执行的同一root，不重跑；N16/N64新root/env_016与env_064当前均仅freeze。v7相对v6仅修正freeze中stage_native_budget对高环境数仍写旧pilot1190400的元数据错误：新N16=2380800/N64=9523200，与原max_native_environment_steps一致，保留旧pilot reference1190400；6个实际运行函数AST exact（RUNTIME_CODE_EQUALITY.json），动作/物理/完整验收不改。v2的N16/64仅freeze未派发且有该旧元数据，不使用。高模式新入口author_mjx_full_comparison_v3.py逐模式run，各固定八槽/0候选重试/0训练，高模式有空闲环境且全计成本，不称满载吞吐。N16/64必须等N1完整计时结束再顺序派发；尚无其性能结果。原DH3仍paused，正式配额未冻。

## 2026-11-18 进行中：用户“开始”授权八槽完整MJX验收及1/16/64完整流程对照

固定六个旧ready布局＋两个固定新抽样，最后一槽placement_null，不补抽、不按抓取结果换候选。新增author_mjx_live_batch_v2/批量入口v3完整六臂/手/一次初始化/对象银行/原生GPU逐点接触/相机/CGR/原规划/原24位置与6前馈/物理返回。CPU MjData仅运动学/渲染，0 CPU积分或接触求解；GPU noslip0/CCD1µm独立域，原1mm穿透门保留诊断、原5mm停止及其他六门保持，按既有functional policy判0/1/null，不混CPU/PhysX/旧权重。

full_collection_20261118_v1序列化模型失败0积分；v2刚体模型flex/elem/vert未分配数组解码失败224积分，末7th tick包未完整落盘，保留失败。v3原生331200步/10350控制，首轮3抓（4125/3069/2326控制）2正1负＋1nocandidate，3次完整物理返回全7门过；第二轮3新拍照/CGR/规划完成，因实际回位wrist1偏31.395µrad超过旧固定HOME接口20µrad拒绝，尚无第二轮执行。父1174.126874s，不当正常完整吞吐。READ_ONLY_AUDIT已PASS：26590分阶段行与全局日志/487551接触点逐值核对、全source极值/载荷/独立FK/原门/lift/返回重算一致；GPU掌姿对独立FK最大4.837784e-7。3抓原1mm门均false（1.293/2.030/2.113mm），不可误称原严格七门全过；其他六门和既有functional门过。独立导出3x480、2fit+1internal、标签[1,0,1]，原480→20/loss eval/no_grad通过，权重不变、0训练。

未放HOME门或写机器人回HOME。原native支持--initial-arm-state，新增author_mjx_measured_planning将当次照片实测六轴输入交原RRT/几何/retiming，原bind_live_plan离线通过0积分。remaining_v1仅freeze未启动，因先前成本漏计改独立v5/remaining_v2；后者当前GPU0控制器host PID/PGID3262349，2常驻环境只完成旧4..7四个未执行槽，不重跑旧首3抓，原三体源布局绑定不变。先前331424原生步＋本组最多595200<原1190400预算。第一波1nocandidate/1正，物理返回6056控制7门全过，第二波待完成。原v3与本组不同物理世界，不称八槽一进程连续完成。

完整对照root boya_mjx_full_comparison_20261118_v2仅freeze，未启动，1/16/64各相同八槽；高模式有空闲环境并全部计成本，不能报满载64吞吐。v1对照仅freeze旧慢射线版也未启动。新v6/live_batch_v3仅把相机射线接到既有Numba全三角面扫描，原96条/像素中心/源几何/门均保留；七个已保存RGBD共672条对原Trimesh报告命中exact、深度max3.33e-16m差，0积分/新标签。新版本冻结独立，不热改运行中v5。9项MJX CPU tests过0积分。原DH3仍paused，正式配额未冻/训练0；当前有本轮GPU任务，勿并发启动旧根或绕过GPU0 lease。

## 2026-11-18 最新：MJX主要接触漏检由CCD精度引起，GPU适配已修复，完整采集尚未验收

用户问CPU/GPU差异能否修复，已固定四条件128原生步成对诊断，无接触128步qpos差8.615e-7；同受载初态q/qvel exact、驱动力差9.93e-5但原GPU缺ff/mf/rf distal—014三组接触，约束力最大差3.08645N。GPU source CCD1e-10与float32不适配，改MuJoCo默认1e-6m后接触恢复、约束力差.005184N；CPU该初态CCD两设置的qacc/约束力exact。四指CPU10.901022/4.083208/4.229009/3.786703N，原GPU仅thumb4.242700，其余0；修后10.901093/4.083098/4.228950/3.786740。仅改solver tolerance无效。128步混合qpos差.314631→.000106718；纯机器人关节差.314631→.000029617rad，不能把混合qpos统一写rad。

新增author_mjx_compatibility仅作用GPU模型copy，ccd_tolerance至少原生默认1µm；CPU保持1e-10，其他数值/物理/动作/几何/验收门不改。这是碰撞算法收敛设置，非放穿透门。原冻结source不热改，独立worker/continuation_v4实际1/16/64两轮全部通过，仍每环境初态写一次、换物边界28关节q/qvel/time exact、预载16银行体/完整六臂。64两轮5203.446环境步/s、CPU2354.366，2.210倍；旧5479/2.327不迁作新结果。parent64.000s复用已有Warp缓存，不作采集倍率或ETA。仍为闭合短段，无新拍照/CGR/规划/完整抓取回位及逐帧全部接触。

残差明确保留：64 env0两轮最大关节差.000148324/.002207457rad，但全部可比偶数环境max .000159292/.012840041rad（末约.736°，旧同口径.327435968rad），不能只挑env0报完全一致。端点限位<.000380/mimic<.000393rad仅端点，不当全程七门通过。noslip仍关闭独立域，原CPU采集不改，训练数据不合并。零积分六姿态回归又恢复ffdistal—021的.783N接触；014 close125尚有.103718N拇指弱接触缺失（CPUdist -10.8355µm），未证明唯一根因或全部差异无法避免；不为每条负例调控制或追正，也不以CPU轨迹exact作为新域成功前提。

新根boya_mjx_contact_diagnosis_20261118_v1、boya_mjx_warp_persistent_benchmark_20261118_v4、boya_mjx_ccd_contact_regression_20261118_v1均complete；诊断CPU512+GPU512及矩阵GPU42120=本阶段43144原生env步<44000，回归/4CPUtests0积分，0Kit/完整试抓/CGR/标签/训练。此前阶段43680..43688单列，历史JAX首块未知0..8不抹除。6所属host PID/PGID退出、4锁可取/registryreleased，无后台。报告AUTHOR_MJX_CCD_FIX_20261118.md，轻证据evidence/author_mjx_ccd_fix_20261118_v1。CanonDex/历史冻结/失败/正式数据/权重保持；下一项是剩余碰撞差异与完整有效性/标签审核，不自动正式扩采、训练或恢复DH3。

## 2026-11-18 当前：MJX-Warp 1/16/64常驻环境换物短段通过，完整采集尚未GPU化

用户澄清要MJX GPU并行、每环境机械臂仅初始化一次后换物，并明确开始测试。最新语义也纠正此前逐条失败即调控制的优先级：正常完整失败应保留负例，不要求先力闭合或每条抓起；保持固定协议/不按失败重抓，按整批正例数与覆盖评估。正式配额仍未冻结，不自动训练/扩采。

完整原v3归档模型UR16e六轴+Boya28机器人hinge/30执行器+16银行体，nq140/nv124/254geom。先MJX-JAX真CUDA/完整put_model通过；预分配94552接触槽/379896约束，240秒首次编译及8步首块超时，无有效吞吐。未继续纯JAX16/64。随后官方MJX-Warp独立v3实跑1/16/64各两轮，从固定旧2628闭合快照起、重复原30驱动控制、每轮32x4ms/每控制8x.5ms、另8步预热。每环境机器人只初态写1次，轮界限换两组三物体；q/qvel/time边界原始数组独立exact、模型保持，无机器人重置。没有新拍照/CGR/规划/完整抓取/回位/逐帧完整接触，不将局部测试当整个采集或训练标签。

两轮合计CPU2354.366环境步/s；MJX-Warp1/16/64为122.408/1411.371/5479.466，相对.052/.599/2.327倍。64第二轮9458.116仅该轮，不能替换两轮结果。预热64.793/2.230/2.281秒，后两复用Warp缓存；最终parent120.702秒。一次顺序实验、共享宿主负载，不推总体均值、线性倍率或完整采足ETA。

原noslip5在MJX-JAX无承接字段且MJX-Warp不支持，本测试显式noslip0且CPU匹配，不改原采集域；GPUfloat32/CPUfloat64及碰撞实现有差异。64末态MFJ3与CPU差.327230rad，GPU保存端点限位0/mimic最大约6.26e-6rad只属端点，不称物理等价或完整准入，也未证明单一根因。无需以CPU轨迹exact作为新域每条抓取成功前提；后续仍须完整有效性/接触和标签审核。

初版Warp make_data设备Model参数及下一版共享buffer错误broadcast均0积分退出，旧根/source/archive保留；独立v3按官方vmap(make_data(MjModel))及先shape-check修正。四根outputs/boya_mjx_persistent_benchmark_20261118_v1与boya_mjx_warp_persistent_benchmark_20261118_v1/v2/v3；三CPU各520+成功GPU42120=已知43680，纯JAX首块未知按0..8，合43680..43688<冻44000，不能报未知0。0Kit/完整试抓/CGR/标签/训练；1换物CPUtest过且0积分。新安装在ignored outputs独立MJX/MuJoCo3.11/JAX.12/Warp1.14，原tendon-spin依赖不改，初次解析4.0目录0物理。10所属host PID/PGID已退、5锁可取/registryreleased，无后台。报告AUTHOR_MJX_PERSISTENT_BENCHMARK_20261118.md，轻证据evidence/author_mjx_persistent_benchmark_20261118_v1。CanonDex、旧物理源/失败/正式数据保持，不自动继续原DH3分支。

## 2026-11-17 最新目视反馈：用户指出挤压后脱手，核实HOLD仍追最终角度，未改控制

用户看视频后“挤压之后脱手了”。本轮只读核v3场景1/raw1722/mf/014：close2418..2667、hold2668..2717；2552首次拇指+其他共同受载，2628拇指峰10.876N，hold开始2668食FFJ3实际.4774/目标.8757rad，差约.3983rad，MFJ3实际.5594/目标.8256；保持指令从首到末完全不变，但实际手指仍收拢追该固定目标。2694最后共同受载、物体Y速度-.432m/s、源底24.73mm；2695载荷全0、源底24.95mm；2718 lift首帧全0载荷。与用户所见挤出相符，不是提起阶段才掉落或改标签高度门能解决；不能单靠这些确定闭合幅度、接触方向/几何、摩擦哪项为唯一根因。

接下来优先研究接触后限制继续闭合、保留预紧力与柔顺维持；不能简单首次触碰就零力停住，也不能宣称已有接触停止/力控完成。最新只为目视反馈/只读分析，0物理/CGR/标签/训练，未自行改关节轴、腕轨迹、闭合控制或启动正式采集。视频及v3原数据保持。

## 2026-11-17 最新：三段实际状态视频已交付，等用户自行目视分析，不改控制

用户明确“给我看视频我就能分析为何闭合/保持丢接触”。中断前一句含“轴偏/横向角速度恢复纳入联合控制”，最新重述是先看视频，本轮不据此实现或宣称完成控制修改。新增render_author_mujoco_failure_video只读v3三次TRACE，实际28关节q与三物体pose恢复到独立MjData，mj_kinematics/mj_camlight更新几何，0 mj_step/forward dynamics/CGR/标签修改/训练；没有新抓取。

outputs/boya_mujoco_failure_videos_20261117_v2 complete：trial01/raw1722物体014、trial00/raw1412物体011、trial05/raw1450物体021各651帧/25fps/26.04秒，合集1953帧78.12秒约4.61MB，所有个别与合集完整解码。全approach/close/hold/lift，原250Hz日志隔2步，无插值，五倍慢放；index.html原速=5倍播放文件并支持逐帧。左近景/右俯斜，前臂仅画面隐藏、右palm透明.22，th/ff/mf/rf/lf红蓝绿紫青；橙色按实际最大接触选作观察对象，不作为网络输入或重选抓取。载荷/物体世界XYZ角速度/源底从原日志直接显示，彩色轴是URDF palm link frame，不是虚拟pad/CGR参考中心。三组预览已目视看清；物体pose误差0、掌姿ASCII矩阵误差回执保存，不称同步相机实拍。

v1预览被前臂遮挡，主动核PID/PGID/startticks后SIGINT只中断自有rendergroup，原失败/RENDERER_SOURCE留存。v2 RESULTS complete/ownerfinally结束，工具外层最后exit143原因未证明，不冒称exit0；已独立解码与源SHA核验、host两个所属PGID全退、3锁可取、registryreleased，无后台任务。两个静态渲染模型/0积分，不计新采集建场或吞吐。视频大文件ignore，轻证据evidence/author_mujoco_failure_videos_20261117_v1；说明AUTHOR_MUJOCO_FAILURE_VIDEOS_20261117.md。当前等用户自行分析，不自主调映射/关节轴/角速度恢复或派新试抓/正式采集。

## 2026-11-17 最新决策：用户质疑正例产出，正式扩采未启动，先查提起前接触丢失

用户在讨论论文单型100条后指出当前成功率恐不够。已明确纠正把“有限管线功能通过”误当“适合正式采集”的推断；100/1000＋额外internal仅讨论，正式配额/采集/训练均未启动。v3现有3完整试抓全负不足估总体成功率，也没有同物理版本正例证明；不拿v2的1正例混入平衡、不放门或用更多负例凑100。

只读核v3三条close/hold/lift全状态：raw1412/rf对011、raw1722/mf对014、raw1450/ff对021都曾闭合接触并有短暂拇指+其他手指同时受载；但所有lift期间三体thumb_ticks与bilateral_ticks均0。014在hold源底最高25.324mm、021在close最高16.590mm后回桌，未形成持续提起；不是仅原50mm末升门太严。只能定位接触在提起前丢失，尚不能唯一归因映射、继续闭合挤出、控制或摩擦/接触模型，不能说已证明必须训练头才能抓起。后续先核闭合/保持的接触维持，再以固定未按结果挑选的批次核正例产出，未派新物理/追加试抓。

outputs/boya_mujoco_failure_diagnosis_20261117_v1及轻证据evidence/author_mujoco_failure_diagnosis_20261117_v1保存全分阶段统计/末手q对target。0积分/CGR/标签修改/训练，前阶段成本/原CLOSURE/数据均保持。当前无后台采集。论文100每型是作者条件下少样本结果，不是Boya同量保证；不能把原文训练后测试成功率当其采集起始成功率。

## 2026-11-17 当前：采集管线有限功能验收PASS，v3新增2fit+1internal全负，训练0

用户“好的收尾吧”授权完成限位修正版完整抓取/回位/连续换物及实际数据导出，不是正式扩采或训练。author_mujoco_acceptance新增冻结原manual所有ready索引3/4/6/7/10/12共6场（4fit2internal），最多80000控制/640000源.5ms积分/6抓/6CGR/6规划，每场一次0候选重试。v2仅stiff_limit，完整6场1合格正例（015真实提起）/2nocandidate/3init_unstable，17032控制136256积分/3CGR1抓，parent243.140412s；其正例独立不混入新物理域。源归档/结果保持。

核物体发现旧MuJoCo按重叠凸块推断COM/I，而原PhysX明确用源mesh；009 COMy偏3.65mm/048偏2.68mm。新增author_mujoco_object_dynamics迁回源mesh COM/I/质量.1kg，几何不改；原friction[.6,.02,.0001]在condim4不启用rolling，新增仅table-object维度6pair启用既有.1mm rolling，原混合solref/solimp/摩擦数值保持，hand-object仍4。**这是显式新接触物理域，不能说完全保持原物理或PhysX等价。** 无新增sleep/吸附/焊接/人工零速/观测后物体写入，原门/动作/候选不调。独立3个旧不稳定初始化，一世界4000控制32000积分：两个internal各750过，fit7仍2500不稳，0CGR/抓/标签；结果failed保留。

v3同6场source-inertia/table-rolling已complete：1world/1camera/1reset/1robot初态写入/16bank，5新照片/CGR5forward1load/5规划/3抓取。fit3 2925exec+3025return负；fit4 3217+3317负；fit6nocandidate；fit7init2500fail；internal2nocandidate；internal4 3177+3277负。三抓全6项非penetration原门及回位全7门过，最大joint violation0/.000604592/.000506556rad；pen .756276/2.696441/2.963233mm，后两原1mm false保留诊断，原5mm stop未改；按用户独立functional policy才label0，不冒称原全部严格门过。0候选补抓/追正，全部空标签保留不填0。v3新增独立2fit+1internal=3负（无正），不混v2正例/旧PhysX2fit/31/435或旧权重。

finish_author_mujoco_acceptance实际CLOSURE functional_acceptance_pass：逐帧6250init+9319抓取+9619返回控制的RGBD/source rays/CGR输入hash/24位置6前馈/完整接触载荷/源极值/独立XML FK/七门/lift/返回重算一致；0审核积分/CGR。export/gesture_1.npz实际3x480 float32/label[0,0,0]/int64原depth*4+extra，三体fit/internal源身份隔离、不可pickle、保存重载exact。原训练数据最低2fit+1internal验证器过；原480→20模型/原loss对实际fit/internal eval/no_grad过，权重exact不变，0 backward/optimizer/训练。另probe实际逐条数据过；它只测格式/前向，不放原formal training数量条件。本批3负远不足训练，正式单型配额/生产队列仍未冻，新模型/外测0，新入口仍有限验收，不自动转正式生产。

v3 parent735.265653s=12:15.266，setup102.813218；六outer128.922417/277.745198/34.876352/18.930855/25.942412/123.681532，抓执行26.706082/35.760682/26.645207s、返回9.677420/10.237791/10.833550s。父计时不含Python前置导入/源验证及另行只读审核，不能除3报含全部审核吞吐、套旧PhysX倍率或推采满ETA。v3 25188控制201504原生积分，本轮v2+初始化+v3合46220控制369760积分/4抓/8CGR；另37与39suite微型限位各1600共3200积分，总本轮372960，活动三体原生789840，0Kit/未知/同版本重试。39 tests过；新clock按整数计数/每tick增量/浮点累加界校验，缺.5ms仍拒绝。对象单测初无compiler.angle字段0积分已修。最终0积分16体归档ASCII质量/惯量、207 table-only pair及stiff限位检查过，序列化tol明确非内存exact；另1静态model编译0积分。

两个GPU控制器host PID1946191/PGID1946190及1990506/1990505均退出；初始化CPU诊断raw PID3/PGID1属sandbox，不能拿host systemd当所属进程；已依执行返回/owner结束/锁释放核验，混命名空间初审误判已说明且0信号。4相关锁可取、registryreleased，无后台任务。报告AUTHOR_MUJOCO_COLLECTION_CLOSURE_20261117.md，轻证据evidence/author_mujoco_collection_closure_20261117_v1约867KB，大图像/接触/轨迹ignore。所有旧source/input/archive/raw/失败保持；原版DH3物理仍paused未派，不能误称本轮完整复现原实机。

## 2026-11-17 当前：限位响应独立修正通过，未重跑抓取，训练0

用户追问URDF机械限位并要求不再卡住后，核实编译THJ2 limited1/range[0,1.5708]/LIMIT与CONSTRAINT均启用，但joint.solreflimit仍默认[.02,1]；此前geom.solref.003是接触，不作用于关节挡块。新独立author_mujoco_joint_limits对全部有限hinge统一timeconstant=2native_dt/critical1、solimp不改；源.5ms时即.001s。按官方refsafe边界，不按结局挑参，不改URDF/质量惯量/驱动限矩/mimic/摩擦或动作，不裁剪q/qdot/target，显式新求解器域。World可选rigid_joint_limits、acceptance freeze可选--stiff-joint-limits，默认false，协议及源码绑定，旧根/历史归档不热改。

outputs/boya_mujoco_joint_limits_20261117_v1固定原验收模型，2模型×3空手/外载fixture，各200控制×8积分=1200控制/9600原生。21组非限位响应数组exact、6例无接触/原生warning0。原开手全28关节最大.010007629→.001347146rad（THJ2 .007467271→.000820370）；THJ2下限-1Nm .064423809→.001031698、上限+1Nm .064423809→.001029340；每原生子步统计，3新例均原.002rad门过。外载是自由度诊断扭矩，不能当实际持物接触/全抓取或回位认证。

35项mujoco tests过，新增3限位检查；两次微型测试各1600真实原生积分共3200，与机器人9600分列，总12800，0Kit/CGR/新抓取/标签/训练，不混入此前采集148332积分。独立stiff_limits_static freeze源绑定/opt-in及STATIC_FIXTURE拒绝run通过0积分。所有任务已返回无后台。报告AUTHOR_MUJOCO_JOINT_LIMIT_FIX_20261117.md，轻证据evidence/author_mujoco_joint_limits_20261117_v1。原015提起10.26cm但越限的验收仍failed/labelnull，完整抓取尚未重验，不自动续旧失败、扩采、训练或报总体收益。

## 2026-11-17 当前：采集验收未通过，真实015提起10.26cm，THJ2越限，0训练标签

用户“好的验收”授权有限管线验收，不是全量/训练。新增独立author_mujoco_acceptance/sample，固定原manual队列首两个ready fit索引3/4与首两个ready internal10/12，不按结局更换，最多4抓/4CGR/4规划、48000控制/384000源0.5ms积分，0重试。新functional_sample_v1按用户允许轻微穿透，把原1mm接触门保留诊断，原5mm深碰撞停止/其他六门/有限完整执行/原生警告0保持，正常完成未提起才记0，初始化/无候选/异常/无效执行空标签。原v1仅object-object豁免与所有旧冻结/失败不回写，不称原严格物理等价。

outputs/boya_mujoco_acceptance_20261117_v1实际1world/1camera/1显式reset/1robot写入/10体bank，2capture各750/2CGR forward1load/2规划/1抓取。fit3几何候选0，labelnull；fit4固定raw260原3591步完整，将015末底103.726218mm/升102.566836mm，末th/ff/mf/rf/lf11.619709/4.497066/4.203483/3.867724/0N，但THJ2下限0，settle第5步-.003224rad首越限、hold3059步-.036704rad最大；原penetration2.223393mm诊断，其他五门过。functional也因jointlimit拒绝，记录invalid_execution/labelnull，不把真实提起说成完全未抓起，也不自动写正标签。原命令开手目标与实测初态有跳变，但更大越限在受载hold，不能仅归因交接或只调整初态宣称修好。

3691步完整物理回位只有joint_limits失败，THJ2最大.014695rad；控制并未提前中断，回位末motor误差.010618rad。已完成执行记录在return前落盘，原总控失败/trace保存，后两internal未派发，不能称四场已完整验收。1500初始化+3591抓取+3691返回=8782控制/70256原生积分，活动三体原生步122184，0Kit/未知/重试。parent151.280584s含setup21.206118，次capture14.751846/CGR1.562702/prepare53.012076/execute26.163312/return8.074630；无合格标签，不能除4报吞吐或套旧PhysX倍率。此前20298控制/78076积分与本组合29080/148332，再早3946旧MuJoCo/旧PhysX单列。

read_only audit复算RGBD rays/CGR输入/24位置6前馈/源极值/完整接触载荷/XML FK/原7门与lift/回位exact，PASS仅记录一致。新export已实际重载0行，consumer因实际fit/internal不足blocked；原版480→20模型/loss在二值CPU夹具eval/no_grad前向过、权重exact不变，0训练，夹具不计标签。32项mujoco CPUtests过，新增5样本/1稳定窗口；窗口测试初临时目录错误0积分已修。capture可按原阈值750..2500有限延长、只看末250支撑，不reset；实际两场均750过，延长未动态实跑。新标签政策不合并旧fit/31/435或权重，正式quota未冻。

报告AUTHOR_MUJOCO_COLLECTION_ACCEPTANCE_20261117.md，轻证据evidence/author_mujoco_collection_acceptance_20261117_v1。所属PID1833595/PGID1833594退出、owner结束、registryreleased/两持有锁可取，无后台任务。收尾修正CLIfailed退出非0，0积分，源归档保持，POST_RUN_SOURCE_REVIEW绑定前后版本，旧根不能热续跑。下一步需独立修正并验证THJ2开手/受载/回位行为，不能放门追正或自行扩采。原版DH3原场景对照已做推理/0积分准备，用户“等下”后未派任何原手物理；勿误称已完成动态原手复现。

## 2026-11-17 最新用户规则：物体间穿透只记录，以真实提起受载判成功，0新物理

用户明确“物体穿透应该没事吧只要能抓取起来就行”，覆盖此前对物体—物体接触强制原1mm否决的要求。新增独立author_mujoco_lift_policy及rejudge，schema anydex_boya_object_overlap_diagnostic_lift_v1；仅两端为不同原物体的穿透改为诊断，手—物/物—桌/未知接触及其余原门保持、原5mm深碰撞停止保持，物理参数/控制/完整接触不变。原7门和新acceptance门并存、旧冻结/失败/数据不回写；新collector freeze必须明确绑定当前policy，旧协议不可热跑当前变更。以原源底离桌5mm/末升50mm/拇指+其他手指真实受载判提起，不要求五指齐触、不新增持握秒数。

既有fresh_v3/raw87/ff 0积分重判：object-object最大1.538419mm不再单独否决，其余接触最大1.224830mm仍超原门；所有物体均未提起，observed_lift_task_success和新diagnostic_success仍false，不能仅改判定把失败写成成功。正式fit/训练新增0，未重跑物理/CGR/抓取，稳定门也未改。新增6CPU tests过、累计44不同CPU用例；新版两轮静态freeze过、STATIC_FIXTURE禁实跑。报告AUTHOR_MUJOCO_LIFT_POLICY_20261117.md，轻证据evidence/author_mujoco_object_overlap_policy_20261117_v1。原阶段78076积分/20298控制成本不变，无后台任务。

## 2026-11-17 当前：新观测首轮常驻全流程62.92秒，物理准入未过，已收尾，训练0

用户“好的开始/继续”授权两轮连续诊断与新RGBD/CGR/常驻规划接入。新增author_mujoco_camera/representation/collection/source_options/return及owner、独立只读audit。原冻结源、失败和CanonDex只读。连续保存动作replay_v2已complete：1world/1显式reset/1robot初态写入，3818exec→3918物理返回→换物→3558exec，parent45.466632s，11294控制/原生积分。首抓原7门及提起过，返回7门过；次抓jointlimit .016136rad/penetration2.337462mm未过，不能称合格负标签。0新CGR/正式标签。

相机启动一次，1280x720/K原标定，源三角面视觉mass0/mask0与原碰撞分开；修复URDF discardvisual默认丢纯视觉。前检v4 1world/1camera/2布局/0tick，96射线深度median .214/.230mm、P90 .262/.291mm，视觉加入前后13组质量/惯量/约束/驱动全精度数组exact。前检v1模块路径0world、v2错误用六位XML重载作exact参考、v3纯视觉被丢射线失败均保留0tick；v2-v4各1静态world，不隐去编译成本。

fresh_v1在750×4ms初始化后011原稳定门失败，CGR1load0forward/0抓取；fresh_v2改独立源XML求解器/接触配置，0.5ms native×8保持每4ms原指令，750控制/6000积分，稳定/RGBD/原CGR1load1forward过，但新回执遗漏args字段被原collector拒绝，0抓取。接口修复和0物理只读校验保留。源XML版本复制timestep/tol/CCDtol/solver/cone/impratio/noslip及手/地面contact solver字段，不复制重力/桌面位置，原质量/摩擦/驱动保持，仍不是物理等价认证；与旧DGB接触候选明确分版。协议17500控制tick/140000原生积分，不混步长成本。

fresh_v3 1world/1camera/1显式reset/1robot写入：首轮新拍照→原CGR1forward→原g1/raw87/ff→2952步执行→3052控制返回，首轮outer62.917879s，capture7.960752/CGR1.933032/prepare31.651556/execute14.723239/return6.047653s；一次setup18.017472/CGRload1.203678另列。执行loop12.426064s，native5.116413/observer6.548350s。首轮未提起任何体，7门中仅penetration失败：036—011最大1.538419mm，首次2425hold。次轮750初始化后014原window平移.227026mm/转角.007608rad失败，未拍照/CGR/抓取。parent94.795423s含未完次轮，不能除2报平均，也不能套旧589s/250s报有效采样倍率；“12s”只属旧保存动作余量。

独立只读终审0积分：RGBD/source rays/CGR输入hash/原24目标及6前馈/接触载荷/源极值/XML FK/原7门/控制返回均重算一致；一致性PASS不等于物理准入PASS。当前full_collection_qualified=false，不放门、不补抓/重复已失败抓取追正。接触和稳定性须独立标定，不能从036—011一条接触断言单一根因。新正式fit0/训练0，旧映射/新映射2fit/旧435与八权重不混，不恢复正式配额。

本阶段总20298个4ms控制/78076原生积分/149226活动三体原生步，0Kit，CGR2forward/3load、3抓取尝试（2保存+1新观测）、未知积分0；fresh两初始化失败不抹除，0抓取重试。原上一阶段3946 MuJoCo积分、旧PhysX成本单列。38不同CPUtests过、0积分；最终owner/回执修正仅静态freeze，STATIC_FIXTURE拒绝run，未重派物理。

fresh_v3错误把本进程套入必须Popen的ResourcePool.stage，收尾隔离spawn_attemptedfalse/pgidnull；原隔离/registry/slots归档。主机核验本轮PID1579931/start171625020及PGID均不存在后审阅释放3槽，registryreleased。新author_mujoco_owner以本进程lease管理生命周期，异常/互斥/fixture有测试，不伪造子进程状态。无后台任务。报告AUTHOR_MUJOCO_FRESH_COLLECTION_20261117.md，轻证据evidence/author_mujoco_fresh_collection_20261117_v1；大trace/RGBD/model ignored，原冻结及失败只读。

## 2026-11-17 当前：严格一次重置MuJoCo候选与常驻规划，分别通过，未集成两轮新观测采集

用户强调真实工作台只初始化一次、可换物体，不能把旧单Kit/世界每轮robot.reset+750初始化当完成。新增独立author_mujoco_persistent_world：1活动Model/Data、1显式mj_resetData、1机器人初态写入；启动原适配器中间2编译+最终银行1编译均公开。6预载物体，边界只写物体pose/vel/外力、停放未用体，机器人q/qd与时钟exact保持。仅预载银行换物，未实现任意新CAD热加载；当前版另拒绝身份/几何变化、非法位姿和活动轮换物。

one_world_prefix_20261117_v1两场各64home hold、共128MuJoCo tick，换物2.823/1.046ms，含原生点读取/JSON循环36.202/34.045ms，setup另14.029s；句柄/机器人位置和非零速度/时钟exact。不能把短hold当完整抓取倍率。persistent_replay_20261117_v1原有限两份计划/17500tick/0重试，实际第一条3818步完整后因误调用不存在LIFT.evaluate失败，第二条未开始、0返回步。原错误/轨迹/源码保留；当前改evaluate_lift及补生命周期guard，未重跑修正版物理。

只读full audit0额外积分复算第一条：原24目标+六臂前馈exact、全部7门过，mimic .000764700rad、penetration .975336mm、掌4.814383mm/.007781544rad、臂支撑/基座漂移0；026末底102.370365mm/升100.651363mm，末五指5.219590/5.519123/.650040/0/0N，受载1.616s仅诊断。3818帧67574原生点，源极值/载荷复算差0，掌对XML FK1.4433e-15。诊断正例但正式fit0/训练0。parent24.862611s含world setup12.310721s，其余12.551890含完整观测/JSON/后处理，不能叫纯物理或新RGBD/CGR完整采集，更不能除旧589s报倍率。

新后端显式变更几何/接触/约束/求解器：补源手XML4mimic equality solref/solimp，采用公开DexGraspBench手接触/elliptic/noslip2/impratio10，控制/积分仍4ms而非源XML .5ms。未获PhysX20µm凸体/filtered全字段认证，不称物理或轨迹等价，不放门/重抓追正。新增author_planning_reuse+compiled_kinematics单CPU拥有模型/网格/精确局部FK与状态审核缓存，绑定全几何/物体pose/排除对/tol/精确state；旧源/CanonDex只读。

同2份保存输入原规划161.014982s，basic127.906375/static95.108418/compiled84.143496s，最终观察少47.74%、1.9136倍，仅规划。各6科学数组/候选/26完整几何报告exact，路径相关2SHA各自独立验真。原计时复用、非随机完整采集倍率。初缺libusb0规划和初比较器SHA误拒都保留，独立审核过；当前比较器修复。author_reused_fixed_scene_collection已接常驻规划/资源槽/有限单worker与退出清理，2轮6bank静态freeze v3过，mockGPUfixture禁止run；该入口仍保留旧PhysX每轮robot复位/750初始化，不等于严格一次reset新世界。

29不同CPUtests过、0积分。阶段实际0Kit、3946MuJoCo环境/11838活动三体tick，0额外只读审计tick/重试/新CGR/RGBD/正式标签/训练；另一次全资产静态world0tick，微型单测构模不作采集world。新增正式fit0，新映射2fit及旧26fit+5internal、旧435/八权重不混。尚缺新世界连续两轮物理返回+换物+freshRGBD/CGR/规划/抓取完整验收与outer计时，不自动扩大到正式配额。所有已启动CPU进程返回、无后台任务。报告AUTHOR_PERSISTENT_WORLD_REUSE_20261117.md，轻证据evidence/author_persistent_world_reuse_20261117_v1，大trace/model/输出忽略，旧冻结源和失败不改。

## 2026-11-17 当前：完整同动作提速控制通过，纠正“物理等待”归因，新增训练0

用户要求参考作者速度实际提速。先查明原native physics_step_update_s跨generator yield，包含外层after_step完整观测/网格/JSON，不能当纯PhysX。此前主要归因物理过早；历史报告/证据不改，本条纠正说明。CPU Fabric独立2Kit×64tick对照实跑，后48步USD .712482s/Fabric .764302s、状态及所有有效接触exact，未提速且未晋级。各另实读2个native reset update_simulation积分，合132步/parent275.701s；线程实验未派发。原生初始化不在SimContext计数，本阶段单独计，不能混旧历史口径。

新增author_fast_geometry_checks以Numba/fastmathFalse全顶点Z扫描、全三角面Moller射线循环，0采样/近似/物理变更；原250Hz/64迭代/原几何/质量摩擦/完整接触/全部门保持。0物理核22128个旧三体姿态，极值最大差2.776e-17m；192射线最大4.441e-16m。极值组件配对4.98–6.22倍。完整同一旧scene01 capture审核新顺序对照69.177→30.590s约2.26倍，96射线43.308→4.566s，全部750帧/3046112接触/2472hulls/RGBD和SHA均PASS。不能用旧404s/current30s虚报13倍，不把组件倍率套完整采集。

public fast_geometry_world_control_20261117_v1单轮已complete，1Kit/1world/1camera/原6银行体，freshRGBD/CGR1load1forward、raw34/ff/3818步提起026、原capture及原版完整runtime audit PASS。与旧v4首轮实拍/751初始化帧/24指令全数组/3818 native trace14数组/三体pose与velocity及loads exact；源底/顶最多2.776e-17m差；3818接触JSONL逻辑2466275238字节SHA exact。末源底101.800mm/升100.300mm、末受载1.608s仅诊断。新parent589.350s=9分49.350；对应阶段合计804.566→584.016s观察1.378倍，仅历史匹配、非随机化完整采集倍率；旧parent2171.808含两轮，严禁除以新一轮报3.69倍。执行器wall310.566→250.709s。capture worker156.949/audit14.425/CGR3.363/prepare86.517/execute worker252.916/runtime audit69.846s；未改的阶段变快也不能都归因代码。

独立author_observer_partition实测3818次sim.step仅82.236s，完整observer95.998s（含全源极值4.898/接触JSON append27.324s，不相加），原inclusive桶180.127s。物理平均21.538ms/4ms tick，仍慢约5.38倍，完整接触解码/整理/2.47GB逻辑JSON也是大开销，不能继续把全部时间归咎PhysX。仍未达到作者实机总体节奏，不报采足ETA。

通用入口author_accelerated_fixed_scene_collection.py使用相同fast worker/observer/audit，无调试历史依赖，仍有限诊断exclude_training；2轮6bank的静态freeze通过0Kit/mockGPUfixture不能实跑。25CPUtests（7新+18既有）过。阶段3Kit/4696环境时钟+6native初始化=4702积分，0未知成本/抓取重试，1诊断正例、正式fit新增0/训练0；此前新映射2fit、旧26fit+5internal、旧435与八权重不改不混。正式单型配额未冻，不恢复旧4800或启动训练/实机。

两新控制根source/input/archive保持，自有PGID全退、4预约released、所有锁可取/registryreleased，无后台任务。报告AUTHOR_COLLECTION_SPEEDUP_20261117.md，轻证据evidence/author_collection_speedup_20261117_v1，完整接触/轨迹等大数据public忽略；阶段轻成果commit/push origin，CanonDex及原冻结源码只读。

## 2026-11-16 当前：同一物理世界连续拍照/抓取两轮已验收，整体加速未证明

用户明确改为固定场景模拟现实，为后续实机采集组织做准备。新增独立author_live_world/contract/collection/service及只读audit，通用入口author_fixed_scene_collection.py路由v4修复层；一次创建原native机器人、地面、相机和6物体银行，每轮边界才摆放/复位。capture→execute沿同一sim/robot/camera/actors，0清场重建/0观测后物体位姿或速度写入；实测非零速度保留，原750初始化/稳定窗口/250Hz/全部物理门/完整接触/源底面承载保持。初始手/腕重绑定为拍照实测值，原六臂初值及24列逐步目标、掌姿、时间、phase exact；原50settle接预张开，原250close/50hold/lift不变。CPU PhysX连续求解器/银行结构改变，不声称轨迹等价。CanonDex和原冻结源码不改。

public live_world_control_20261116_v4已complete：固定manual index3/4两轮，1Kit/1world/1camera/0stage重建/2边界摆放，CGR1load/2forward、0评分头。首轮raw34/ff/3818步正，026末源底101.800mm/升100.300mm，末th/ff/mf载荷5.62156/5.88714/.31162N，rf/lf0；第二轮raw296/ff/3558步负，014末源底.555mm/末五指载荷0、三体无提起。两次完整capture和runtime audit均PASS、所有原物理门通过，负例保留、不换候选重抓。照片时钟750/5318、执行末4568/8876，原native入口均bound_measured_live_state_without_reset，句柄和相机标定exact。两条仅exclude_training诊断，新增正式fit0/训练0；此前新映射2fit与旧映射26fit+5internal及旧435/八权重不自动合并。

v4父wall2171.808s=36分11.808秒，静态freeze另15.538s；capture worker143.047/286.207s，capture audit165.351/404.183s，prepare99.703/118.546s，execute worker311.405/485.429s，runtime audit80.360/123.752s（首轮audit与次轮capture重叠，不相加冒充outer）。实际native sim.step+asset更新614.993s/接触读取13.709s，只是inclusive时间桶，不归因具体kernel/驱动。旧首轮setup44.298→26.728s但loop190.140→249.236s；新旧次轮候选及步数不同，不能只套setup倍率或旧cold26:54含中断outer报加速。整体尚慢/无净加速证明，不外推采足ETA。

v1缺cam全局、v2原all审计跳过/World/Object主物体、v3新增瞬时角速度.02门误拒原稳定窗口，均各750初始化/0抓取，原错误/源归档保留。v2显式all+object补足原20um审计，v4仅移除新加截断、重算原稳定window并保留实测速度，未放原物理门。全阶段4Kit/2250旧失败初始化+1500新初始化+7376执行=11126环境步/33378活动三体步；6银行体时钟机会数66756非awake工作量。四parent wall合2971.734s不含版本间开发/冻结，未知成本0/抓取重试0。18CPUtests（13新+5既有）通过，通用入口静态freeze通过0Kit，不能拿mockGPU fixture实跑。四版source/input/archive绑定不变，全已登记PGID退出、预约released、所有锁可取/registryreleased，无本轮后台任务。

报告AUTHOR_FIXED_SCENE_COLLECTION_20261116.md，轻证据evidence/author_fixed_scene_collection_20261116_v1；两轮真实RGB与首轮实测姿态图在ignored outputs，0回放积分。固定流程组织已完成，真实Boya驱动/标定/时延状态检查尚未接；仿真CGR/规划时暂停积分保观测绑定，不能声称实机可以暂停。正式单型扩采协议/配额仍未冻结，不自动恢复旧四型4800、训练或继续追加物理。阶段轻量成果push用户origin，大数据public忽略。

## 2026-11-16 当前：新预闭合g1映射已接完整采集，两场均提起并审核通过，新集2fit，训练0

用户“好的继续”授权先接新映射并做1–2场小批次，本轮固定复用未启动manual_start_v2中index3/4两新布局，均fit；不是授权正式全量。独立author_contact_pipeline贯穿原source碰撞/动作合同/桥接/机械臂准备/原PhysX执行，原冻结源码与CanonDex不改。沿原g1闭合轨迹按raw宽度求虚拟预闭合参考；1.7只选原预张开，按ff/mf/rf/lf首个碰撞清楚分支选姿，无GT调姿/历史标签调参。执行仍原250close至公司包握目标+50hold，不在虚拟参考停手，不要求五指齐触。COLLISION_BRANCHES与独立XML FK/命令审核另存，13合同/映射测试+3接续测试通过。

public `contact_pilot_20261116_v1`已complete：fit00003 raw34/ff/3817步提起026，末源底101.796mm、升100.203mm，末th/ff/mf/rf/lf载荷5.69679/5.93571/.02753/0/0N；fit00004 raw1929/rf/2681步提起014，末底95.825mm、升94.099mm，末载荷11.77012/5.15991/4.41343/4.78929/0N。两次原全部物理门、完整逐点接触及独立数值审核PASS，新增2条fit正标签、internal0；末受载1.608/1.472s只诊断不加门。选rf不要求该指独占对向载荷，仍按拇指+其他合计。新集export/gesture_1.npz含2x480特征及全三体身份，重载exact；旧映射26fit+5internal=31与旧435均不自动合并。正式训练/新模型/训后外测仍0，新单型正式配额与扩采队列尚未冻结，不自动扩大或恢复旧四型4800。

总控首次在第二场CGR前退出143，原因未证明。第一场标签与第二场capture/audit已完成；第二场仅RESOURCE_WAIT/预约，无STAGE/process.log，原Popen前持久化顺序+主机全PID/PGID核对证明未派发。INTERRUPTION_REVIEW保留原资源状态，审阅释放唯一未启动预约后，独立continue_contact_pilot只完成剩余第二场；0初始化/执行重试、最多剩余1Kit/8000步，原协议/源/记录不回写，最终明确controller_was_interrupted。继承total_scene_budget16/64容量字段不作派发范围，实际只rows[3,4]/g1及更严格4Kit/17500步预算。两阶段所有已登记PGID退出、资源released、3锁可取/registryreleased，无后台任务。

实际4Kit/1500初始化tick+6498执行=7998环境/23994三体步，0未知成本；2原identity CGR、0评分头/训练。完整outer1614.294s含中断恢复，不作正常吞吐；原执行stage273.880/464.457s，未证明整体加速或采满ETA。1ms仍仅映射。旧fit00013纯准备v3另得5210行ready、独立参考宽度/掌姿核对通过，0物理不计新标签；v1 libusb路径和v2新适配源编辑导致SHA拒绝均保留0物理。首成功图outputs/boya_contact_pilot_visual_20261116_v2/actual_execution.png用真实native link/三体pose+源网格，无回放积分，fixed palm按源URDF从pmbase恢复；初版缺合并palm停止保留。不能以两场称总体成功率/净映射收益。报告AUTHOR_CONTACT_PILOT_RESULTS_20261116.md，轻证据evidence/author_contact_pilot_20261116_v1；大数据public忽略，阶段轻成果push用户origin。

## 2026-11-16 最新语义更正：预闭合构型用于定位，执行仍继续闭合，不要求参考帧先夹住

用户质疑姿态/宽度后明确“也不用太闭合，因为后续继续闭合也能接触到物体”。按此继续：raw宽度对齐的是预闭合参考与目标掌姿，不是接触完成/五指齐触或最终受载状态。统一g1五指握，拇指必须真实接触受载、其余部分可参与；不要强行让虚拟参考点贴CAD后就报抓稳，也不能把未实现的接触停止控制说成完成。开手/接近需清楚；闭合阶段允许手指接触物体，物体会改变实际关节运动，自由FK接触后轨迹不是实际轨迹。前阶段自由FK的负z是风险/未准入，不足以单独判真实必撞桌或失败；后续须判首次物体接触与桌面接触顺序及实际控制，不盲目放门、抬高目标或全体加大闭合。

已做只读图/距离核验，无追加物理：保存目标掌姿确实旋转17.337315°、平移21.624655mm（ΔXYZ +.741923/-10.889495/-18.667998mm），相同手指q的全部源网格刚体变换差3.33e-16m，有坐标轴叠加图outputs/boya_contact_palm_delta_20261116_v1；尚无机械臂/内腕IK/执行。参考点距83.867550mm是两选定指腹点距，不是全手包络/真实受载开口。源CAD诊断所选拇指点在015内距表面.942128mm，食指点在外距表面18.016246mm；015 watertight且winding-consistent，winding分别约1/0。其余分支世界两参考点相同（相同CGR中点/轴/宽度），不构成新接触证据。不能推论整根食指没碰物体，也不从两点距离直接决定闭合幅度。

初始朴素nearest_point诊断遇源物体4个零面积三角形NaN，原失败/部分JSON保留0物理；新增audit_contact_reference_width以三角面或退化边点有限距离测量，源资产不修不改。render_contact_wrist_delta只画已保存掌姿差，audit不修映射，不用CAD反向调该候选。轻证据evidence/author_contact_reference_diagnosis_20261116_v1；本阶段0Kit/物理/IK/标签/训练，旧31与新映射基准/源文件保持。几何准入实现尚未继续/未运行任何新采集，当前不得将预闭合数值匹配称完整抓取验收。

## 2026-11-16 当前：用户指定接触构型对齐/单一五指握，几何映射已实现约1ms，尚未执行准入

用户看图后指出开手参考对齐不能保证弧线闭合后接触对齐，要求按预计接触构型的目标宽度求手腕；统一五指包握、拇指必须接触，其余允许不足四根参与，不锁死食指/不强求五指齐触；强调不能每候选优化几分钟。此为最新新分支设计，旧四型冻结数据和成绩保留，不能拿旧31自动充当新映射数据。当前尚未冻结新正式采集/训练规模，不因单型原型自动改变旧配额或派采集。

新opt-in author_fast_contact_alignment已做g1原五指16电机/4mimic单闭合进度、513状态表/四个可选对向分支/固定2次批量精确FK修正/闭式刚体掌姿；接触目标用raw宽度不乘1.7，无物体mesh/GT目标/ learned优化；有限步后数值不合格直接无解，不作无限重试。拇指必需和对向选择是规划目标，未声称实际受载。1µm宽度tol仅数值精度，不是物理门。8CPUtests过（含源独立FK、非单调首根、原闭合/全五指/mimic、无解不夹紧、输入不改）。全部4分支CPU首call4.2335ms，129单call中位1.0557/P95 1.1044/最大1.2878ms；源合同＋表一次.238825s；2048保存raw五次batch中位60.0584ms，摊销.029325ms/CGR。时钟不含进程导入/IO/碰撞/IK路径/物理/渲染，不能说整体采集毫秒级。1276有宽度解/4282分支不是物理准入或成功率。64分支独立XML FK复算pad8.33e-17m、参考矩阵5.55e-16。

outputs/boya_fast_contact_alignment_20261116_v1/visual有原始参考/旧手腕闭合/新手腕同闭合构型与先张开图。固定raw1322/ff闭合40.724%达83.86755mm；旧掌姿闭合后参考中点偏20.632mm、方向17.337°，新姿态数值对齐。**示例仍不可送执行**：4分支各33个完整源网格自由FK采样最小地面z分别ff -2.9515mm、mf -.6103mm、rf -1.4186mm、lf -1.1593mm（部分在开手即触桌）；不把纯宽度解当可用抓姿。该独立floor诊断每分支约43–57ms且不含资产加载、不含在映射计时，不证明连续路径/自碰/邻物/完整臂/真实接触。不得为本例手动抬高手腕目标、放门或强行派试抓，后续须按通用规则完成几何准入。

本轮0Kit/物理/CGR/IK/新标签/训练，新模块production_ready=false、未接collector，不改原冻结source/CanonDex/权重，原有效26fit+5internal保持。新bench与render均独立诊断，完整raw/表/图在ignored outputs，轻证据evidence/author_fast_contact_alignment_20261116_v1。前文“新接触对齐未实现”已由本条几何实现更新，但不表示动态闭合、抓取或完整流程已验收。

## 2026-11-16 候选图已交付，当前等用户自行目视诊断，不启动采集或改映射

用户明确要“原始可用候选与Boya转换后对比图，我来诊断”。已用同一旧fit00013场景生成outputs/boya_candidate_comparison_20261116_v2/index.html：g6/raw1210历史成功、g1/raw1322历史失败、g7/raw1322未执行；3组2040x1740俯/侧视、3组2560x850原RGB-D静态投影及3组全源网格Plotly可旋转左右联动3D。左CGR标尺不是DH3实体手，没有同场作者实机成功证明，不能标成“作者原始已可用”；右为保存pool实际掌姿及原16手关节FK/全部258086三角面，无显示降采样。计算参考点不宣称硬件接触标定；投影图明确静态合成非手实拍。原图/大HTML继续ignore，只push轻代码/README/证据。

原绑定和6PNG全解码/尺寸/历史标签读回通过，同raw1322原始俯视面板pixel exact，源帧到映射参考最大矩阵差5.73e-8。v1因理想候选掌矩阵与pool保存roundtrip约1.74e-8差而逐值断言失败，源码/失败保留0步；v2只接受<=1e-7保存数值roundtrip并用原pool实际pose绘图，未改物理门或原source。HTML生成完成，未做浏览器运行验收。本轮0Kit/物理/CGR/IK/新标签/训练，有效31与模型保持。

纠正前面“13项HOME路径未通过”易误导：所示g7原上层bounded_HOME实为fixed_segments_geometry的approach1270..1299/30样本lfdistal—graspnet005源凸代理相交，最大0.046500060mm，RRT未开始。只能称该例原几何门拒绝，不说全局不可达/真实三角面穿透或原CGR必然无用。图中红小指仅涉事link，不伪造接触点。当前用户要自行诊断，交图后不继续派试抓、调参、改映射或启动此前只prepare的manual队列。生成器render_author_boya_candidate_comparison.py，轻证据evidence/author_candidate_comparison_20261116_v1。

## 2026-11-16 采集交接：新手动队列只准备，未运行，不自动继续实验

用户要求说明如何自己启动及是否沿用原论文。已核原Algorithm1/公开DH3采集代码，明确CGR/top500/固定型/随机extra/source flip/原碰撞逻辑对齐公开代码，但Boya/UR16e/PhysX/S40派生场景/相机域/完整臂规划与自动提起标签属适配，不能称作者原实机环境或完全复现。新增prepare_author_collection_start只生成布局与冻结协议，author_verified_collection_execute只给原串行E.execute安装已通过的verified journal，原控制/物理/判断不改，12项既有执行/日志测试通过。README顶部已有可复制运行/查看/导出命令。

public根manual_start_20261116_v1静态2fit+2internal全4placement_null，结果/协议保留，0Kit/物理。另预备固定manual_start_20261116_v2为8fit+8internal/seed2026111601，16场/最多64槽，6布局ready、10placement_null；相同seed保留原4布局结果，没有物理重试或自动补例。源/输入绑定通过，协议71c312818b94484e95dfe38d0c3d65c5f95f9563e992fdb26628e7eaedd5f105，当前零stage/ledger，0Kit/物理/新标签/训练。新队列未经实际采集，不能宣称已出标签或新入口实跑通过。给用户启动命令是原batch run --max-scenes4（前3null保留，第四场首次ready，最多四型各一次）；不自动执行或扩大预算。starter默认回执的max-scenes1只处理第一条null，用户交接命令以README的4为准。

配置不是正式4800全量；当前有效数据仍26fit+5internal=31，正式训练/新模型/训后外测0。MuJoCo失败支线保留但本轮未继续调参/物理。只有用户执行run或明确要求启动才开始此新队列，当前无本任务后台采集/训练。轻证据evidence/author_collection_start_20261116_v1，代码/说明/证据阶段push，public大输出不入Git，原冻结源/CanonDex不改。

## 2026-11-16 当前：MuJoCo一条完整原计划40秒，但闭合物理失败，不能正式采集

用户要求继续确认候选后端。新独立author_mujoco_control/full_replay及只读audit完成一次固定历史正例fit_00009/g1；原3389行24位置指令及原六臂float32速度前馈exact，原THJ4/THJ3的0.6Nm覆盖恢复。24位置+6前馈执行器合力后按原关节限矩，28机器人关节/4mimic，原三体初态/质量.1kg/每体-.98N/250Hz及原手28exclude保留，0调门/换候选/重试。此为显式后端/接触/几何变化，不是物理等价认证。

3389步全返回/31313接触点，函数内部wall40.125878s（不含进程导入与后续只读审计），其中mj_step1.832797s/观测25.064796s。观测用copyData+forward独立末态Data，不改live状态；点全存明确诊断字段，非PhysX完整filtered合同。新原七门仅5过：mimic误差.026422327rad>.002，穿透4.831803mm>1mm，未提起任何物体；原同计划七门全过、主物体升99.114mm且受载。新首失败2594close为mimic，2652close为穿透，最深2747为thdistal—graspnet014；hold/lift均无机器人—物体接触，掌姿/限位/固定基座/无臂支撑/完成性过。不得把40s或1.83s比旧472s报有效采集倍率或ETA，不晋级新后端。

原后处理因MuJoCo累计时钟与step*.004差1.049827e-12s、超过原1e-12容差而报错，原RESULTS/runtime_error/raw不改。只读审计仅在派生评价行用原整数步网格，另存RECOMPUTED_EVALUATION，0再积分；首次未适配时钟的审核失败保留说明。raw重算PASS只表示3389指令/源底/载荷/七门完整一致，非物理门PASS；源底/载荷误差0，实测掌姿对独立XML FK1.332268e-15。11CPUtests过。本轮0Kit/3389MuJoCo环境/10167三体步，含旧两前缀候选累计3429环境/10287对象步；0新训练标签/训练/CGR，累计有效数据26fit+5internal=31，正式4800及新模型/外测0。

原默认软mimic、接触求解与机器人凸包表示尚未和原PhysX标定，不能断言单一根因。下一步集中核原约束及闭合接触/几何，不放宽门或追逐本条已知成功。报告reproduction/boya/AUTHOR_MUJOCO_FULL_REPLAY_RESULTS_20261116.md，轻证据evidence/author_mujoco_full_replay_20261116_v1；原轨迹/模型在ignored outputs，原冻结源/失败/CanonDex不改。完整物理已结束、无后台任务，不自动扩大预算或生产。

## 2026-11-16 当前：参考现成后端落成紧凑MuJoCo原型，仍未完整标签准入

用户要求参考开源采集实现并解决异常慢。本轮逐代码核DexGraspBench MjHO/GetaGrip/DexGraspNet及本地AnyDex fork，沿已装MuJoCo3.11直接源网格建场实现独立author_mujoco_preflight/scene_preflight/prefix_smoke/source_filters，无新依赖/Kit/正式采集。定位现有PhysX max_extent.02/vertex64等适配将机器人/手分到12681凸体（上臂3316等）；新候选同源URDF机器人30网格、三体场景总52/53。不能单凭数量归因或声称几何/物理等价，旧20um门仍未在新后端认证。

两场全部7138行24电机指令映射exact，28机器人关节/4mimic，静态36状态/1476 link FK最大1.332e-15，三个对象源初态/各.1kg核对。0积分静态检查通过，7CPUtests过；4次静态开发失败0步及源码保留。新显式摩擦/求解器/约束差异未校准，原ML候选/评分/旧数据不改。

首20步数值通过但独立检查发现漏迁原手XML排除，结构自碰12.393mm/速度16.11，不具原碰撞语义。仅按原XML补齐28组exclude后，另独立20步正常：mj_step .003040s/读回.001417s/函数内部6.363170s/345点/穿透.329270mm；内部计时不含进程导入/最初校验/协议落盘，不是完整采集wall。两版合0Kit/40MuJoCo env/120对象步，0完整抓取/标签/训练；不能报完整倍率/采满ETA。点数全存但诊断字段不是原PhysX完整合同，旧trace fields_dropped:false解释限制见CONTACT_SCHEMA_NOTE，不能拿它认证原全字段。

候选production_ready=false，尚缺完整动态提起、原源底/受载/七门与独立审核对应验收，未自动派完整试抓或扩采。累计有效新数据仍26fit+5internal=31，正式4800/训练/新模型/外测0，原PhysX成本单列且冻结源/失败/CanonDex不改。当前无后台任务。报告reproduction/boya/AUTHOR_REFERENCE_BACKEND_PROGRESS_20261116.md及轻证据evidence/author_mujoco_reference_adapter_20261116_v1；模型/轨迹outputs忽略，阶段轻成果push用户origin。

## 2026-11-16 当前：修复版完整常驻对照通过，但未加速，本组慢1.62%，已收尾

用户明确追加1Kit后，public stream_collection_bound_control_20261116_v1已完整完成。复用cold1098.021376s，与新resident1115.850096s（18:18→18:36）比较，多17.828720s/1.623713%，冷/常驻比.984022。固定两场原g1，2capture×750及2exec3389/3749均完整，4原独立审核全PASS；两场8组件/50数组RGBD/原CGR/候选/科学指令exact，标签均[1,0]，数值准入通过但不证明物理轨迹等价。单组旧cold复用有顺序/缓存/宿主负载影响，不推总体均值或4800ETA。两组都有相同流水/CGR常驻，不是全流水相对旧8路采集的净收益。

新prebind服务确实单Kit走完capture0→capture1→execute0→execute1/closed，旧XML失败保持。explicit_context_clears2+execution_scene_rebuilds1，执行asset_setup仍2，跨几何资产未复用；单Kit没有带来可见完整吞吐收益，保持opt-in，不晋级默认/替换旧8路。相同R.execute jobwall冷472.406735/307.336563、新473.377583/337.195573；含建场/执行/存储，不是纯积分或kernel。保存scene0 native审计机器人/手12681凸体、含模板主物体13916，cooked max53/over64为0；仅几何事实，不单凭数量归因慢或推旧GPU回退。后续先量物理建场/推进/接触/存储成本再优化，不恢复生产。

本修复组1Kit/8638env/25914三体步，0重试/未知；10所属组全退/3槽released/5锁可取/registryreleased。连前轮冷与失败合6Kit/18776env/56328三体步/30所属组全退，无后台任务，失败成本不抹除。23不同CPU用例通过（原21+2新；本轮6中4复用不重复计）。全部诊断exclude_training，累计26fit+5internal=31，正式4800/训练/新模型/训后外测0，未派下一批。报告reproduction/boya/AUTHOR_COLLECTION_STREAM_BOUND_RESULTS_20261116.md，轻证据evidence/author_collection_stream_bound_20261116_v1；原待审批/运行中段为历史，冻结源/失败/CanonDex不改，轻成果push用户origin。

## 2026-11-16 当前运行：用户明确追加1Kit，修复版常驻两场对照进行中

用户在预算问题中选择“追加1次Kit完成比较”，仅授权修复版两场g1常驻组、最多17500env/1Kit/0重试，复用已经完整的cold1098.021376s，不重跑cold或恢复旧失败。新增author_collection_stream_continuation独立freeze在public stream_collection_bound_control_20261116_v1；旧control/failed结果只读绑定，新observations/plan/execution在全新目录。SOURCE冷组通过路径路由只读用于比较，无旧结果搬运/改写。新手模型预绑定已实际落盘且在capture前完成，目前尚未完成首条抓取，不报加速。

本组2新增协议检查+4复用绑定检查共6过；前轮21中的4绑定检查重复，因此累计23项不同CPU检查，不能加成27。已有5Kit/10138env均保留；本组额外最多17500env，组合上限27638仍小于原35000env，但Kit总预算经本次明确批准增至6。所有新结果exclude_training，累计31条训练兼容标签不变，正式训练/模型/外测0。不得因用户批准1Kit再扩组或重试；完成后独立收尾与轻成果push。

## 2026-11-16 当前：冷组18分18秒完成，常驻组绑定失败，未获得有效加速比

用户“比较下”授权原固定两场/g1对照，公共stream_collection_control_20261116_v1已实际结束。cold完整1098.021376s/4Kit/2×750init/3389+3749执行步，2初始化及2原抓取数值审核全PASS，1正1负仅诊断exclude_training；resident351.079361s因失败停止，1Kit完成两capture及各独立audit，2CGR/2ready，但首条execute0步/infra/labelnull，第二条未启动。两组8组件RGBD/原CGR/候选/科学命令exact，不代表resident完整执行已验收。不得用351与1098相除冒称3倍加速。

根因为观测先导入native hand模块捕获CanonDex/third_party旧XML默认值，后续bind_hand不能改已捕获默认参数。新增独立author_collection_service_bound仅在任何capture/Kit之前用原backend真路径调用原bind_hand，拒绝stale imports并逐执行核计划引用；旧源/失败不改。4新+17既有=21CPU检查通过；两个真实无Kit前检分别重现旧默认拒绝与正确XML读取（22关节=18主动含2腕16手+4被动），首次错误len16断言及更正另留档，0新物理。修复版尚未实跑，不宣称完整链路已修复。

本轮合5Kit/10138env/30414三体步（3000init+7138execution），初始化10020812接触，unknown0/重试0/新增训练标签0。20所属组全退，4槽released/6锁可取/registryreleased，无后台任务；累计26fit+5internal=31，正式4800/训练/新模型/外测0。原最多5Kit预算耗尽，已询问是否追加1Kit/最多17500env，仅修复版两场g1常驻组并复用旧cold基线，未获答复前不启动。不得重启本轮失败任务或自动补例。

报告reproduction/boya/AUTHOR_COLLECTION_STREAM_COMPARISON_20261116.md，轻证据evidence/author_collection_stream_comparison_20261116_v1。上一节“待权限/0物理”是启动前快照；CanonDex和全部历史冻结源保持，阶段轻量成果push用户origin。

## 2026-11-16 当前：完整采集常驻入口已实现，17项离线检查通过，真实对照待权限

用户要求落实成熟加速方案，新增独立author_collection_service/client、author_stream_representation、author_collection_stream_control。单拥有线程的Kit衔接capture与execute，CGR原逐帧循环/模型常驻，第二场观测与第一场规划可重叠，执行与前一条原数值审核可重叠。观测之间仍重建SimulationContext/场景，尚未实现跨观测USD/原生资产缓存；不能宣称全部资产复用或物理等价。原750初始化/250Hz/三体完整接触/原命令/判定保留，物理仍CPU。

7新增+2verified+8resident=17CPU检查通过，含真实无Kit子进程生命周期/持槽/并行审核/退出释放；mock非物理性能。工作区outputs/boya_stream_collection_control_20261116_v1已freeze两个既有fit场景00009/00013、仅g1、cold4Kit→resident1Kit、最多35000env步、0重试/补例，全部exclude_training。两模式均含相同流水与常驻CGR，只能测Kit复用的成对效果，不能归为全部流水相对旧入口净收益。原数组/候选/命令与独立原审核通过前不得报倍率；单组已知场景不推采满ETA。

公共目录freeze两次在启动前因自动权限审核超时未派发；普通沙箱nvidia-smi无法连接驱动，已发明确固定预算权限请求，等待回复。当前本阶段0Kit/物理/标签/训练，累计仍26fit+5internal=31，正式4800/训练/新模型/外测0；无真实常驻对照在后台。报告reproduction/boya/AUTHOR_COLLECTION_SERVICE_STATUS_20261116.md，轻证据evidence/author_collection_service_20261116_v1。历史冻结源和CanonDex不改。

## 2026-11-15 当前：原文／现成加速方案核查完成，尚未新增性能实现或物理批次

用户质疑速度并要求查论文／网络与创新空间。本轮查原论文、作者采集源码及NVIDIA／DexGraspNet一手资料，0Kit／物理／标签／训练／依赖安装。已确认原采集网络与机器人在循环外创建；IsaacLab官方资产复制／隔离、DexGraspNet多环境GPU验证、cuRobo GPU规划均有现成实现。最新官方cuRobo v0.9.0（2026-11-04）特定RTX5090基准4.4ms规划不迁移为本项目实测；本机Python3.12，curobo未装，不能拿旧0.7.x兼容约束代替当前版本核验。

不能把此前27–33天现状情景当算法下限，也不能把“没有找到直接可用适配”当新颖性证明。下一优先是完整生命周期常驻／资产复用和流水安排，再独立评估GPU规划；同组三体四型常驻已实测1.52倍，八路CPU/两audit已集成，不能重新宣布它们未实现。尚缺跨阶段／跨几何复用和全流水集成。原GPU单/双20tick约91% fetch等待仍未归因，官方GPU cooking/凸体限制CPU回退只是待查假设，不可声称已发现根因或只改读回就能解决。

本轮按92阶段时间区间核：建场/观测/初始化审核16.506min，规划8.759min，执行+抓取审核扣重叠20.667min，CGR .432min，其余1.554min。论文/官方方案不等于本机性能承诺；改变规划器、碰撞几何或物理参数须独立版本，不回写原命令／证据。候选分布、完整字段与原数值判定保持。论文创新需在现成方法基线上证明新方法和收益，常驻/GPU批量/批量IK本身已有先例。

报告reproduction/boya/AUTHOR_ACCELERATION_LITERATURE_REVIEW_20261115.md，来源与阶段成本证据evidence/author_acceleration_literature_20261115_v1。只读研究不改变上轮已完成尾批：累计26fit+5internal=31，训练/新模型/外测0，无后台任务。轻量文档push用户origin，CanonDex和所有运行冻结源保持。

## 2026-11-15 当前：原尾批15场景全部完成，新增6fit+4internal，累计31条，训练0

用户要求先跑完再估时，原index17..31/seed2026111501、15场景/60槽全部完成并导出。外层2875.152003s=47.919200min；新增10标签=6fit+4internal/8正2负，fit g1/6/7/8=2/2/1/1（正1/2/1/0），internal各1且全正。累计26fit+5internal=31：fit8/6/6/6、internal1/1/2/1，四型internal均不再为零；旧435/八权重保持，不自动合并旧集，训练/新模型/训后评测0，未派下一批。

60终态10labelled+15nocandidate+15geometry+20placement_null；无新增infra/完整性错误/重试/补抽。10capture各750tick，独立全点审核全PASS，7500init/22500对象步/20748079初始化点；2CGRload/10forward/0手型头。10真实执行32723步/98169对象步/128797269点，10原首次完整数值audit全PASS，tmpfs raw/隔离压缩/全读回seal与最终raw SHA通过，原CPU250Hz/.1kg/.6/三体/24→28/全部判定不变。合20Kit/40223env/120669三体步，未知0；1395host采样实见8Kit/2audit，92所属stage组及父/observer全退，14槽released/16锁可取、registryreleased，无后台任务。

独立收尾核1280去重绑定/NPZ480特征与标签/全三体split/证书/13旧数据权重/原15标签及前批6导出，0重复完整点流审核/新物理。新增5CPU/mock检查含真实ExpandedPool审核槽，mock非成绩。原32预算终结：初失败4槽+旧64+新60=128，合16生产试采标签12fit+4internal，加原15成31；52placement+25geometry+29nocandidate+16labelled+6infra槽（初单场依赖4槽和旧坏gzip2槽）。三段37Kit/已报告71685env/215055三体步，原初父unknown_capture1和ERROR0tick仍分别保留，不称全历史未知0；原坏流及失败不回写。

每型1000fit+200internal仍缺3974fit+795internal。按本轮15场景和原32预算的混合组成/实测产率保持、24h连续运行，两个条件情景约33.08/27.16天补满；这是敏感性情景而非置信区间或工期承诺，不能把4800除总产率、把原三段wall和当连续实测或直接与旧fit-only15.02天比较。当前混合配比会超采internal，若以后另冻结split预算需重新估时；本轮不为此扩采、凑比例或训练。

报告reproduction/boya/AUTHOR_PRODUCTION_TAIL_RESULTS_20261115.md；轻证据evidence/author_production_tail_20261115_v1。启动代码ac9d053已push，最终轻成果继续push用户origin；大输出public/tmpfs忽略，CanonDex只读。下段“当前运行”是启动快照，现已结束，不重启尾批。

## 2026-11-15 当前运行：用户要求先跑完原剩余15场景，再估计采满耗时

用户“先跑完再顾及吧”授权执行原32预算内的尾批。新production_tail_20261115_v1只取原index17..31，7fit+8internal/15场景/60槽，原seed2026111501/原三体spec顺序不变，10ready+5placement_null，两窗8/7，0重试/不补抽。新author_production_tail私有AST仅改剩余预算/窗口/新入口并接author_verified_execution；执行raw写/dev/shm/anydex_production_tail_20261115_v1，原完整接触隔离压缩+全读回+seal后才可准入，capture仍原完整初始化及独立审核。原250Hz/质量摩擦/24→28/三体/全部物理判定不变，最多8CPU执行/2原审核/3prepare/1observation，固定GPU0不动GPU1。

5项新增CPU/mock检查通过，覆盖8/7尾窗导出、证书exit0但failed停后窗、真实seal篡改拒绝、真实ExpandedPool audit槽普通Python进程退出释放；未另加诊断物理批次。真实依赖预检通过，561源码/归档冻结；协议04552e9a7ef2db6cf8c7e1b21a022014a816d695cbd4a62f1557b1e29597d2ea。run/父PROGRESS仅窗末更新，已进第1个实际capture，实时状态读窗口ledger/STAGE和HOST_OBSERVATIONS。新增标签/成本/总wall等完成后报告，不能把本快照当最终计数；启动时累计20fit+1internal，训练/新模型/外测0。

保持旧失败/坏gzip及所有冻结源码，CanonDex只读，大日志public/tmpfs忽略。首infra/完整性错误停新非audit派发并收尾，不重抓/凑比例或扩原预算。先完成这15场景，再按实际随机产出估时，不直接外推旧15天或诊断倍率。报告reproduction/boya/AUTHOR_PRODUCTION_TAIL_PROGRESS_20261115.md；阶段轻量代码/证据push用户origin。下段“不自动恢复15”为此前验证阶段，已被本轮用户明确运行要求替代。

## 2026-11-15 当前：tmpfs完整执行日志及四份首次原数值审核通过，生产尚未恢复

用户要求先验证。独立author_storage_path_validation冻结/tmp与tmpfs各4路，每份原完整3024帧/1895914265字节，8份全部写入/隔离压缩/两次独立全读回通过，0Kit/0物理，外层147.836573s；两路径都未复现损坏，不能归因原ZFS故障或声称tmpfs普遍更快。新author_verified_full_control_v2仅补真实协议缺的venv解释器绑定并改raw spool至/dev/shm，四路同一已知g6模板各一次原3024步，4Kit/12096env/36288三体步，四份封口和最终raw SHA/无空洞通过，原控制命令/250Hz/质量摩擦/全部逐点接触与判定保持。

原控制器执行后误以仅支持Kit的ScalingPool申请audit，任何审核子进程启动前失败，原stopped及未启动quarantine保留。另独立author_verified_saved_audit使用原ResourcePool两audit槽，对四份保存结果各做首次原完整数值审核，4 full_integrity_pass/0额外Kit/物理/重试。物理组233.001461s、独立审核225.544539s不含修复交接，不外推生产吞吐。v2冻结源码不回写，其原整体控制器仍stopped；不能重启本轮或拿四次同模板诊断当四型新数据。新7项CPU检查含真实audit池普通Python子进程释放验证通过。

本轮新增训练标签0，累计20fit+1internal仍21，训练/新模型/训后评测0，剩15场景/60槽未派发。13旧数据权重及原15标签核验入口/上一批6标签导出NPZ绑定复核保持。收尾确认原audit全未spawn、所有所属组退出，独立留复核证据后在全局锁内释放滚动registry；原隔离事件/停止回执不删改。6槽released/9锁可取，无后台任务。tmpfs raw易失，持久gzip/seal已fsync及全读回，不证明断电恢复或随机长期可靠性，原故障根因仍未知。

报告reproduction/boya/AUTHOR_STORAGE_PATH_VALIDATION_RESULTS_20261115.md，证据evidence/author_storage_path_validation_20261115_v1；大数据public/临时目录忽略、CanonDex只读。当前完成执行日志验证，下一项是将该方案和正确审核资源池集成到原有限生产入口；capture日志仍原路径，本轮未验证其新集成。不自动恢复15或全量4000+800，不扩大诊断预算。下段为前轮失败快照，不表示本次仍未通过执行日志验证。

## 2026-11-15 当前：日志完整性防线已实现，完整验证发现原始空洞，生产仍停止

用户授权推进日志修复。独立 author_verified_contact_journal／contact_compressor 保留完整逐行原字节，写入前逻辑SHA、raw spool、隔离标准库压缩、fsync及全解压读回；只在seal通过后交原pointer，失败无重试。18 CPU/mock通过；独立与空Kit各原／新四路、每份1024帧399120740字节，共16份全通过，1Kit/0物理。但完整四路各3024步诊断未通过，不能称已修复或恢复剩15场景。

完整诊断4Kit/12096env/36288三体步，四份原runtime/trace/contact计数均3024，0完整数值audit/训练标签/重试。worker02 raw JSON先失败，整组停止；00/01压缩器中断，03已封口及原finish但退出时取消，原stage失败全部保留。停机只读比较：00偏移1310720000..1311768576为1MiB全零；02偏移25165824..26214400与51380224..52428800共2MiB全零，SEEK_HOLE一致；01/03全raw SHA与写入前exact。四份1.896GB原始文件及失败原样留档，不修补、补压缩或重抓。/tmp/public报告ZFS和1MiB块，但根因仍未证明；独立四路同完整原始文件复制/fsync读回全过，0Kit，未复现间歇故障。

本轮合5Kit/12096env/36288三体步，新增标签0，累计仍20fit+1internal=21，训练0。两组外层153.007164s/244.883271s不含开发交接取证，不外推产率。5槽released/8锁可取、所属组全退、registryreleased，无后台任务。完整控制器未执行audit_one还存在实际继承协议缺inputs.python的问题，CPU fixture未覆盖；冻结版本保留，未来须独立修复并核真实协议，禁止重启本轮。未实施tmpfs方案或新物理批次，剩15场景/60槽未派发，原坏gzip和所有冻结源/CanonDex保持。

报告reproduction/boya/AUTHOR_CONTACT_INTEGRITY_RESULTS_20261115.md，轻证据evidence/author_contact_integrity_20261115_v1。当前只证明能拒绝坏日志，生产准入仍失败；先验证可靠写入路径，再考虑有限完整验证和剩余原预算，4000+800/训练/外测均未启动。用户追问为何耗时后，本轮停止追加实验并收尾，不继续扩大诊断。

## 2026-11-15 当前：生产试采新增6fit有效标签，接触gzip损坏后停止，15场景未派发

用户批准32场景/128槽有限新随机试采。首capture因freeze未继承原clarabel路径0tick导入失败，原protocol acc3f763.../1Kit/53.460996s停止保留，原父unknown_capture=1保留。恢复原已验依赖并实际子进程预检通过后，仅原未启动index1..31在独立d1c9dbb3...协议接续，首失败不重跑、不重抽、不扩原预算。接续前两窗64槽=6labelled+14nocandidate+10geometry+32placement_null+2auditinfra；剩15场景/60槽未启动，原首次失败4槽另列合原128。接续1792.661695s=29.877695min停止，两段run wall合30.768712min，首启动至接续停机span39.281108min含8.512397min依赖修复交接，不代表完整随机产率或加速倍数。

新增6条fit全正，g1/6/7/8=2/1/2/1，internal新增0；已在接续run/export独立导出，parent结果仍stopped，另PARTIAL_EXPORT_RECEIPT不改旧OUTER。累计实际新标签20fit+1internal=21，原14fit+1internal/旧435不自动混入该6条数组。正式4000fit+800internal/训练/新模型/训后评测均0，旧13数据权重SHA保持。

新fit scene6的g6/g8原首次审核分别报JSON/deflate异常与invalid distance code，原审计CLI退出0但证书failed，父准入正确拒绝，标签null而非负例。g6第742行JSON异常，进一步纯gzip读289406976解压字节后报invalid literal/length；g8至第719行报invalid distance。原压缩字节SHA/size与producer pointer相符，/tmp字节相同副本同处失败，确认现存流不可用，损坏根因未证明。原数据/证书/失败不修补或重抓，未重新认证两标签；源/命令/完整点门不放宽。当前保持停止，不再自动派发剩15或新物理批次。

合17Kit=初失败1+capture8+execution8；capture6000tick/18000对象步、8独立初始化审计全PASS/15080831点，2CGRload/8forward/0评分头。8执行25462步/76386对象步含坏g6/g8的3509/3679步，完整trace tick/最终成本对齐；6份原数值PASS共33994219执行点，2份失败不认证gates。接续报告31462env/94386三体步，接续成本未知0；初父保守unknown_capture1和ERROR报0tick分别保留不合并为“全未知0”。前缀独立核1556文件与6条NPZ/标签/全三体/split/features/证书通过，0新积分/完整数值重审。

host实见峰4Kit/2audit，初2+接续74所属stage组、两父/两观察进程全退；接续10槽released/12锁可取，初2槽/4锁释放，registryreleased，无本轮后台任务。CanonDex只读，所有运行冻结源/归档保持。59项不同CPU/mock覆盖通过（原39、新生产13、接续4、后续修复3；重复不累计）。独立future v2补证书status检查以在audit退出0但failed时立即停新派发；bootstrap_v2恢复原环境并Kit前真实依赖预检，显式新seed。v2仅CPU/metadata验证、未新物理运行，未解决gzip损坏根因，也未扩大本批预算。

报告reproduction/boya/AUTHOR_PRODUCTION_PILOT_RESULTS_20261115.md，轻证据evidence/author_production_pilot_20261115_v1；初59e9bd1已push，收尾轻成果继续push origin，原始轨迹/接触/NPZ留public/ignored。下一准入项是日志写入前逻辑SHA与关闭后持久readback等完整性机制和独立验证，再考虑剩15；internal三型仍零产出，整体4800ETA未知。不得把本批说成跑完32或用诊断倍率预测采满；不自动恢复采集/训练。下方启动快照均为历史。

## 2026-11-15 启动快照：固定32场景有限生产试采，首个0tick依赖失败保留，31未启动场景独立接续

用户批准下一步并明确“开始”：16fit+16internal、128手型槽、固定seed2026111501、原首选每型一次、0自动重试，不启动全量4000+800或训练。新author_production_collection独立生产schema，原八路诊断/冻结源保持。只在原数值证书和独立初始化全点审计通过后准入0/1，null不作负例，原15/旧435不自动混入导出。4窗口、最多8CPU冷session/2audit/3prepare/1observation；parent整批持GPU0全局lease，GPU1不选；原250Hz/.1kg/.6/原三体/24→28/native20um/完整接触和所有门保留。52项不同CPU/mock覆盖（13新+39复用），另4接续测试通过，非实测成绩。

初32随机场景19ready（12fit+7internal）/13placement_null（4fit+9internal），不补抽。首protocol acc3f763... 在首capture因freeze未继承原clarabel覆盖PYTHONPATH与LD_LIBRARY_PATH报ModuleNotFoundError，outer53.460996s；1Kit，ERROR/PROGRESS报0初始化tick/0对象步，0CGR/规划/抓取/标签/训练。原父表保守unknown_capture=1保留不回写，失败不是抓取负例。初轮2所属stage组及父/观察进程全退、2槽released/4锁可取，源与失败全保留，无物理重试。

独立author_production_continuation以固定SHA私有AST仅改剩余31场景/124槽及尾窗7；新协议d1c9dbb3...严格取原queue index1..31，15fit+16internal，4窗8/8/8/7，不重抽/不改布局/不重跑原index0。首失败四槽与新124槽合原128预算。精确子进程依赖预检通过，恢复原已验inherited_env且绑定clarabel包。接续运行public根production_pilot_20261115_continuation_v1/run；初根production_pilot_20261115_v1/run保持stopped。已完成首个新capture750tick，首次独立初始化审核进行中；新标签数/完整wall/internal覆盖须读实际ledger，不能从此启动快照推断完成。

首infra/完整性错误停新非audit派发和后窗，已启动安全收尾并记成本；1200s/stage，控制器4h后停止新派发，持久STOP_REQUEST触发STOP_INTENT；RUN_STARTED后拒绝重启。该批为原32有限预算内的新训练标签试采，不自动扩预算、换候选、凑比例、训练或外测。报告reproduction/boya/AUTHOR_PRODUCTION_PILOT_PROGRESS_20261115.md、轻证据evidence/author_production_pilot_20261115_v1；大输出public/ignored，CanonDex只读，轻量阶段push origin。父PROGRESS仅窗末刷新，窗内读stage/ledger/host。下段是此前已知两场景加速对照，不能当本随机试采结果。

## 2026-11-15 当前：八路执行与两路首次审计已接独立采集入口，固定完整流程15.10分钟

用户授权继续集成后，新增opt-in author_pipeline_collection/run/observe，私有AST编译原固定SHA collector，仅改调度/分组/输出路径，不修改冻结源或CanonDex。固定旧scene1+4重新build/capture/CGR/prepare，最多8CPU执行+2原首次audit，0重试/8槽全部诊断排除训练。外层905.730148s=15.095502min，旧同组resident1765.610592s=29.426843min，历史wall比1.949378/约减少48.7%；非新配对或随机完整采集倍率，不能据此除旧15.02天或推4800总ETA。原默认collector仍不变，新独立入口已实际接通完整链。

八条执行25656步、62194441接触点、8原数值audit全部PASS，6正2负与旧标签相同；父labelnull/exclude_training=true，新增训练标签0。独立主机447采样实见8Kit/2audit、八journal同区间增长、27次执行与审计同存，stage重叠56.941966s。新分组为8冷session各一计划、每个1Kit/1资产/0reset；旧为2热session各四型，不能说八路热复用或物理轨迹等价。原250Hz/.1kg/.6/CPU物理/24→28/完整三体接触/native20um/全判定门均保持，仅GPU0场景/RTX/CGR，不动GPU1。

两capture独立审核各一次PASS，1500初始化tick/4500对象步、3487508初始化接触、1111/1135hulls、每场96全源射线/RGBD/三体承载通过。与原开发两场初始化/RGBD/34表示数组exact，8候选各36科学字段与池/命令数值数组exact；新路径来源SHA元数据保留变化，不冒称所有NPZ字节或轨迹相同。父ledger独立核745文件/8原证书/三体split/features/成本/排除训练通过，没有第二遍执行点流审计。全轮10Kit/27156env/81468三体步，0未知成本，1CGRload/2forward/0评分头/训练。额外只读capture审计在运行期间做，其竞争开销未从实测wall扣除。

10新CPU/mock与原resident/wave合39tests通过；首次组合误写不存在模块的1导入错误留档，更正后39通过才冻结/派物理。29所属stage组/父/观察进程全退，14槽released、16锁可取、registryreleased，无本轮后台任务。555项源绑定/24组输入和源归档通过，13旧数据权重与15原标签不变，累计仍14fit+1internal。报告reproduction/boya/AUTHOR_PIPELINE_COLLECTION_RESULTS_20261115.md、轻证据evidence/author_pipeline_collection_20261115_v1；大输出public/ignored，轻量commit/push用户origin。

该新入口硬限最多2场景并排除训练，正式4000fit+800internal/训练/新模型/训后评测仍0，internal三型仍零产出。约31.80条/小时只属本组全ready已知条件，不能当随机产率。下一生产协议须独立冻结对象/场景与尝试预算/训练准入，再由用户决定启动；不自动补例、重抓或恢复生产。下段“collector尚未切八路”是集成前阶段历史。

## 2026-11-15 历史：CPU1/4/8前缀与CPU1/8完整对照通过，执行容量显著提高，collector彼时尚未切八路

用户授权CPU常驻并行路线。独立cpu_prefix_control_20261115_v1固定同旧scene4/g6各100tick、1→4→8进程，共13Kit/1300环境步/0重试。前缀包络5.021568/6.408663/8.860896s，19.914/62.416/90.284envstep/s，实测峰值重叠1/4/8，共同重叠5.022/4.118/2.093s；组wall60.191/60.350/65.336s。独立前缀审计24.187898s通过后，按事前min(前缀增益,含启动增益)、至少1.10、5%内优先小规模规则选8，不看标签选并发。

另冻结cpu_full_control_20261115_v1：CPU1一次→CPU8各一次，同原3024命令，9Kit/27216env/0重试。完整执行包络143.979669/168.788235s、组wall196.751887/225.859960s，对应吞吐6.824157/含启动6.968987倍；已知同模板控制，非随机完整采集倍率，不能与旧resident1.52相乘。9完整原门/独立全数值审计全PASS、9诊断正例排除训练，不报模型9/9成功率。8并行保存q与primary pose相对本次单路exact，不外推全物理等价。CPU实际MBP/ordinal-1/CPU完整接触；原250Hz/.1kg/.6/24→28/原速度/三体向下力/native20um及全点日志均保留，CanonDex/冻结源不改，仅GPU0服务建场。

首次独立audit每份一次。单路audit109.115760s，与CPU8实际执行重叠41.785691s；八份审核最多2子进程，wall435.359416s、各worker合861.146544s，原全点数值审核不删减。组执行＋audit阶段成本合305.867647/661.219376s，折算诊断吞吐11.770/43.556条/小时、比3.700650；它是成本合计，不是连续完整采集wall。含开发/交接间隙至证书实际488.684938/789.109227s单列，不抹掉间隙或套采满ETA。全部审计私有投影有额外成本，非旧resident原audit成本。

本轮合22Kit/28516独立时钟暨env/85548三体步，62326116接触点；无在途未知，audit新增Kit/积分0。6前缀+6完整新检查及2复用full检查过。host最终22Kit+9审核进程组退出、16槽released、20锁可取、registryreleased，无后台物理/审核。旧13数据权重/15新标签绑定不变，正式新标签/训练/模型/外测0，累计14fit+1internal。

另只读64旧槽：fit14labelled/7geometry/16placement/6nocandidate/1infra，internal1/7/8/4；14geometry均frozen_endpoint_or_non_home_collision且HOME搜索跳过，保存路径10条仅approach-floor、4条其它或跨段碰撞；10nocandidate原碰撞前21–61候选，碰撞后0。未新规划/放门/换候选/计负例，internal三型仍0。

报告 reproduction/boya/AUTHOR_CPU_SCALING_RESULTS_20261115.md、轻证据 evidence/author_cpu_scaling_20261115_v1。现有resident collector仍硬性最多3路/串行audit，本次独立8路控制＋2路审核不冒称已接生产全流水。下一项是opt-in采集器并发/审核生命周期集成及新有限全链路队列；未自动追加该队列或正式4000+800，旧15.02天仅旧条件情景、整体ETA仍未知。轻量阶段提交push用户origin，大输出public/ignored。

## 2026-11-15 当前：Fabric 双关闭小对照完成，默认已满足该建议，未测出新加速

用户授权按网络官方建议开始小规模验证。新增独立 author_fabric_execution/control/audit，复用原 scaling/vector/profile、原命令及完整逐点日志，不改冻结源或 CanonDex。协议 fabric_control_20261115_v1 固定原 scene4/g6、baseline→both_disabled 各100tick、2Kit/0重试/每worker600s，仅 GPU0 UUID。实读关键发现：baseline 在 App 后就已 fabricUpdateTransformations=false、fabricUpdateVelocities=false、fabricUseGPUInterop=true，两组积分时亦均如此；这是相同实际配置的重复，不是有效开关干预，不能套上游7倍或把 wall 比称关闭 Fabric 的净收益。

两组前缀58.417396/58.662117s，fetch54.553399/54.575174s约93%，接触读取.233291/.275146s；组wall124.023485/120.059069s，控制器244.123901s，不含事后独立审计。精确2Kit/200shared/200env/600三体步，native尝试=返回=全观测100/组，无在途未知。独立审计6.879916s核1286绑定、200trace/200contact/431400点，原24→28和速度目标及所有14个native数组、全三体entry/轨迹exact。原接触列表100行均有顺序差异，原audit保留false；另2.643654s只读核每tick/每类完整精确多重集合100/100相同、每组215700点，不删零力/字段/重复项、不取整；不能说原始逐行相同或完整轨迹等价。

后置官方单prim兼容查询每组13846凸网格prim（机器人12711、primary328、另两体807）全返回GPU compatible，查询前后stage layer SHA和100tick不变；未调用完整CheckStage/reset/detach/cache release。prim不是唯一cooked hull，不证明所有接触/约束kernel都在GPU。原三体1135hulls/native20um保持。无任务相机/visualizer，原一次forward_fabric保持，RGBD capture未改。8新CPUtests+30复用检查过，mock不冒充实测。

本轮0完整抓取/新标签/CGR/训练；只认证前缀完整性，旧14fit+1internal及13旧数据权重文件SHA不变。2自有Kit进程组全退、当前slot released、3锁可取、registry released，无后台物理。报告 reproduction/boya/AUTHOR_FABRIC_RESULTS_20261115.md，轻证据 evidence/author_fabric_control_20261115_v1，大输出public/ignored。fetch内部归因仍未知；CPU/GPU1/4/8代码已具备离线检查但尚无实跑，未自动追加该预算。正式扩采/训练仍0，旧fit15.02天条件情景及4800总ETA未知不变。

## 2026-11-15 当前：GPU单/双环境物理接通，主要耗时在fetch，尚不能加速大采集

用户追问为何不接GPU，已实际实现独立author_gpu_vector_execution与官方clone/原共享barrier；原commands/三体几何/250Hz/质量摩擦/完整逐点接触与判定逻辑保留，CPU→GPU为显式后端变化，不宣称轨迹等价。真实GPU0 UUID固定，NVML0对应minor1，GPU1其它任务不干预。两次实读均GPU dynamics/broadphase、tensor ordinal0、full filtered6 CUDA；1135hulls原native20um通过。原v1绑定导入顺序失败1Kit/0步留档，v2只提前原bind_hand，11CPUtests过，CanonDex只读。

public gpu_vector_control_20261115_v2原单3024+双3374/组1800s/0重试，实际单路过慢，另存取消意图后仅子PGID SIGINT安全收尾：937已返回sim.step及937全观测，第938在fetch_results中断是否完成未知，成本必须[937,938]tick/[2811,2814]三体步，标签/raw为空，不认证完整gates/抓取失败。close前718.326s/父719.738s；后组未派发。唯一partial审计13.864s核937trace/937contact/2021298点、696SHA/前937原28目标exact，通过的仅完整性，未调用完整执行audit。

另独立gpu_partition_profile_20261115_v1协议ff421bb1dd8f7269a37315d141b3f10379b4e3bc0a8468c50ed27c5a97926f80，复用原step_limit，各20tick单g6与双g1/g6一次0重试，完整接触和原条件不变，均partial_profile/labelsnull。single前缀12.0385s、fetch11.0209s、接触读取.04509s；双前缀13.6242s、fetch12.4350s、读取.10859s，跨环境接触0，主要物理等待约91%。没有额外CUDA同步；inclusive不可相加，不能进一步未经实测归因GPUkernel/driver/CPU接触生成/几何/参数。原barrier可预写第21命令但不积分。两组outer169.313s，2Kit/40shared/60env/180三体步，不能外推完整lift/采集ETA。现成packed readback可参考但不是本次主要瓶颈，停止本轮追加批次/调物理/降精度。

本阶段共4Kit、[977,978]shared/[997,998]env/[2991,2994]三体步，0完整抓取/标签/训练/CGR；32组件CPUtests及3+15mock控制场景通过，mock非实测。4Kit子进程组全退、3当前槽released、4锁可取/registryreleased，无后台任务。557运行source/原input/旧13数据权重/原15新标签绑定均不变。总报告reproduction/boya/AUTHOR_GPU_CONTROL_RESULTS_20261115.md；轻总证据evidence/author_gpu_control_20261115_v1，另有partial/短前缀独立证据。当前GPU接通但未证明加速/完整提起准入；常驻CPU旧fit约15.02天条件情景不改，internal三型零产出、整体4800总ETA未知。正式扩采/训练仍0，累计仅14fit+1internal，不自动恢复生产；阶段轻量push用户origin，大输出忽略，CanonDex不改。

## 2026-11-15 历史：常驻双路完整fit对照完成，正式全量采集等待用户决定

用户最新要求先解决常驻/并行并实测耗时，再由其决定是否启动采集。本轮只冻结3场景/12槽/最多12执行的小诊断，0重试，全部exclude_training；正式4000fit＋800internal没有启动，不自动扩大预算或训练。新author_resident_execution/collection/benchmark已实现，31项CPU测试通过，源码提交fadc12a已push用户origin。fit scene1＋4重新build/capture/CGR/prepare后，两实际Kit各常驻执行四型，共2建场/6边界reset/25656步/6正2负，8原numeric audit全部PASS，与旧对应步数和标签一致。

新fit完整outer1765.611s=29.427min；旧同两scene各自stage跨度合2688.671s=44.811min，观察约1.523倍，仅历史范围比较，非新配对或随机完整采集倍率。随机开发11fit产率4/3/3/4与实测成本迁移情景：各1000fit约3667scene/15.02天连续运行，同工作量执行无重叠约20.20天；internal5scene三型零产出，4800整体ETA未知。不承诺小时级、不套worker数或执行段倍率。GPU0为场景/RTX/CGR服务，实际物理仍CPU；官方GPU多环境批量物理未接通。固定internal小对照已complete，完整outer578.404s=9.640min，g7一次冷执行正例/3242步，g1/g6为null_geometry、g8为no_candidate，不是内部热复用证据。合计9次实际/28898步/7正2负、3无计划，另2250初始化tick/6750对象步；全部9次原audit通过，未知成本0。根归档首次误加总数28998，保留原归档并有ERRATUM_TOTAL_STEPS更正，原逐条/逐run回执不变。最终host34相关PID退出/13槽released/14锁可取，无后台任务，不重复派发。

热复用并非全部物理轨迹一致：实际机器人/全三体初态exact，机器人settle/HOME/approach exact、关节首差在close；物体preclose微差最大43.738um，接触后某原失败轨迹差149.905mm/179.916°。原门与8标签保持，原因未证明，不能说无偏/物理等价。改变几何的同Kit重建仅CPU测试，不扩本轮预算验证。当前新collector硬性诊断排除训练，正式扩采须另冻结生产协议/准入与有限预算。

总报告reproduction/boya/AUTHOR_RESIDENT_THROUGHPUT_RESULTS_20261115.md，动作/初始化/ETA分别独立审计。累计有效新采集标签仍14fit＋1internal=15条，旧435/八权重不改，正式训练/新模型/训后评测0；S40派生40物体32/8未扩大。本轮不改冻结源码或CanonDex，只push轻量代码/文档/证据，大输出继续忽略。下列“常驻尚未实现”等是阶段历史，现已完成同组三体四型的实际常驻对照，不能与GPU向量化混同。

## 2026-11-15 历史：两个固定开发预算已结束，累计15条新标签，正式训练0

主线为公开代码＋发布模型的Boya无箱稳定提起适配，先采集真实标签、再训练、最后冻结外部评测。旧development/batch_v1已结束：40槽产生14条fit标签=10正4负，g1/6/7/8为4/3/3/4，另10null_geometry、3no_candidate、12placement_null、1准备infra。新development_waves/waves_v1固定fresh6也已结束：24槽=1条internal g7正标签＋4null_geometry＋7no_candidate＋12placement_null；3次完整capture共2250初始化tick/6750对象步，实际执行3242步，0infra/0未知执行成本。累计14fit＋1internal=15条有效新标签，不自动并入旧435；正式训练、新正式模型与训后外部评测均0。这些是采集标签，不能报成新模型成功率。

fresh6实际最大Kit重叠仅1，不能以fresh6本身证明双Kit并发。另立的吞吐benchmark已完成：旧成功g1/g6各一次3139/3008步，两原numeric audit通过，两标签均正但明确排除训练，不能混入上述15条。相同两条旧执行stage合471.457s，新并行执行span237.844s，约1.982倍；这只是历史串行对照的执行段倍率，不是新配对完整采集倍率，含两审计总302.148s。独立host观察确认2Kit同时存活、同区间两journal增长、GPU0 FD与资源slot。benchmark协议SHA080e470ce022322b7b4a7ec73afadb7edba6bc568d3912682b4e3bb148d11d5d，最终证据见新总报告。旧NUL配置失败原样保留，原因未定且不重试。正式目标仍每型1000fit＋另200internal，当前不足；正式总场景/尝试预算尚未冻结，不自动扩采或按正负比例补例。

后续架构参考已检出的kywch/pixi常驻环境/模型循环与本机官方IsaacLab场景克隆、批量状态和相机接口；常驻/官方向量环境尚未实现或运行，不再追加局部数学优化。具体复用接点见reproduction/boya/AUTHOR_RESIDENT_REUSE_PLAN_20261115.md；阶段总报告为reproduction/boya/AUTHOR_PARALLEL_COLLECTION_RESULTS_20261115.md。完整采集、标签及成本以public原ledger/PROGRESS和轻量审计为准。下面均按日期保留历史快照，其中“运行中”“仅freeze”“新增0标签”“搬运投放”等不表示当前状态，也不恢复已被用户撤销的投放要求。只更新滚动文档，不回写冻结源码、历史证据或失败；阶段轻量成果commit/push用户origin，大数据忽略，CanonDex只读。

## 2026-11-14 历史快照：public固定开发采集当时已有10条有效标签，7正3负

接续快照：四型新fit标签g1/6/7/8为3/2/2/3，共10条=7正3负，internal仍0；原10场景固定预算继续运行，不能把7/10当训后成功率。新增三场景wave调度/独立导出，23项组合检查通过，但尚无实际并发或加速证据。独立fresh6仅freeze（3fit+3internal/seed2026111402，3ready+3placement_null），0启动；因旧批次index5/g6准备时BACKEND.json在写后SHA绑定到读取间变为2150字节全NUL，保留infra且0执行/不记负例/不重试，原因未定，暂不启动新wave。旧预算的其余有效任务继续按原规则执行，未修改冻结源码。正式目标仍每型1000fit+200internal，不自动混旧435；正式训练/新模型外部评测均0。当前数量继续以ledger为准，以下首标签段为较早快照。

用户授权/mnt/public/datasets存大文件。新专用根/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1，轻代码/证据push，原始数据/轨迹/接触/模型不进Git。新场景构建/批量/无损capture与execute/独立标签审计已并行交付；开发queue预冻结8fit+2internal/seed2026111401/三体同split/每型固定首选一次，7ready+3placement_null不补例。原32+8按全三体隔离，新增NPZ三体身份列和训练数量门；目标仍每型1000fit+另200internal，旧435不自动混入。

development/batch_v1已freeze并完成前缀1：017/046/011真实750init tick/2250对象步、2274278接触、1648原生凸体/RGBD/源射线/GPU0独立核验通过，官方CGR1load/1forward、0评分头/训练。首场四型同raw816均规划null，原因是固定4cmapproach末段指尖碰地板（最深5.861/1.499/4.801/6.750mm），HOME端点clear、RRT未搜索；不能归为HOME不可达或报0/4抓取成功率。

队列index1三体002/005/060实际capture/CGR通过、四型均ready，g1已实际3139步，独立8163718接触/24→28指令/全三体底面与承载核验通过，产生首条fit正标签；002源底97.945mm/升98.787mm、末尾受载1.572s为诊断非额外门。这是采集正例，不是新模型测试。其它固定任务继续运行，PROGRESS仅场景末刷新，型内实时计数读ledger。首条链路通过不代表全四型/内部/批量完整验收。正式4000fit队列/正式训练/新模型外部评测仍未开始，旧435/八权重不变。

运行源码已freeze，队列971e661c...，batch协议完整SHA见阶段证据。后续继续读实际PROGRESS/ledger，不从本快照推断剩余场景结果。报告reproduction/boya/AUTHOR_COLLECTION_PROGRESS_20261114.md；仅相关轻量源码/报告/证据阶段push用户origin，CanonDex不改。下文此前“无后台任务/批量未实现”仅指旧阶段。

## 2026-11-14 最新实际状态：无箱采集组件完成，新场景四型均停在IK，尚未采集或训练

用户再次强调先采集、再训练、最后评测，不能把采集入口开发当作训练后的抓取评测。新独立author_lift_plan/author_lift/author_lift_pipeline已实现和测试；原transport默认保持历史任务。新无箱三体场景实际750初始化tick/2250对象步，三体稳定、有地面承载，原生1633凸体导入门与RGB-D核验通过，渲染0积分；新identity CGR仅1次载入/1前向，0手型头。

outputs/boya_full_data_lift_20261114_v1冻结四型g1/6/7/8、各1原采集首选、seed20261114。四型均为新raw1538/extra0，全部在原八seed的预抓IK局部不收敛，未进入HOME/闭合/提起，0实际抓取、0新标签、0正式训练；不能报0/4抓取成功率，也不能据此证明全局不可达。它与旧有箱raw879不是同一候选，不能把旧计划改名复用。未换候选、加seed、放宽门或重抓。

当前第3步采集闭环仍未验收；4000fit只是已定目标，正式多场景队列/对象划分/预算尚未冻结或启动。下一步补通用批量采集协议和入口，先确保可产生真实正负标签并逐项验收，然后采足四型各1000fit及另留internal，按公开配方训练，再冻结外部评测。采集试抓与训后评测必须在日志和用户说明中明确区分。当前收尾无后台Kit/物理/训练，旧435/八权重不变。详见reproduction/boya/FULL_DATA_LIFT_PROGRESS_20261114.md及独立准备审计，仅轻量阶段push用户origin。

## 2026-11-14 最新扩采目标：四型各1000条实际训练标签，内部验证另留

用户指出现数据远少于完整设置不可作为最终交付。主线数据规模目标现明确为Boya g1/6/7/8各1000条有效fit标签、共4000条实际训练记录，物体隔离的internal标签另外保留，不把4000条总标签再切分后冒称有4000训练样本。100/300每型只作相同冻结顺序的嵌套少样本对照，不再把435旧标签或容量参数当最终扩采完成。论文主打的DH3少样本设置仅400条，旧435与其同量级；“远少于”只针对完整约4000条训练规模，不能笼统说少于论文所有设置，更不能推断数据越多就必然修复成功率。

新增author_dataset_scale.py独立检查四型实际fit数量、全局trial唯一和跨手型物体隔离；author_training正式模式不足请求sample_cap立即拒绝，在创建模型/输出和优化前停止，不再将sample_cap1000悄悄训练成不到1000条。旧集数量审计为364fit/71internal，对4000fit目标缺额3636；这只是旧集计数，不认证旧记录符合新任务。仍须先完成无箱提起闭环及标签兼容核验，再冻结正式对象/尝试预算/序列后扩采、训练及内部检查，最后冻结外部验收。新无箱lift模块三agent并行推进，未完成的实跑不得写成成功。本轮不得自动把null计负例、凑正负比例或根据试验成功数改采集预算。

## 2026-11-14 用户最新纠正：主线只要求稳定提起，不需要投放箱

用户明确“我们不需要投放箱啊，只要能提起就好”。当前主线为Boya稳定提起仿真适配；这条指令替代下文历史记录中的搬运、固定释放位及投放箱验收要求。停止围绕箱体、释放位或搬运末段继续开发和调参。公开代码＋发布模型仍是算法/训练对齐依据；提起任务与作者实机搬运指标的差异明确保留，不因此声称论文物理成绩已复现。

实际进度：第1步原算法数值组件已核验，第2步Boya计算动作合同和原采集接口已实现并接通到臂准备；第3步新的无箱稳定提起采集闭环尚未验收，第4步正式扩采/重训和第5步冻结外部验收尚未开始。旧入口具备提起执行能力，但当前author_pipeline/author_scene仍绑定transport任务及箱体，不能仅改文档后冒称已经切换或直接运行它。后续先接独立无箱场景、提起路径与标签任务绑定，并核源网格离桌/持续受载、原控制门和输入SHA，再冻结采集队列、扩采、训练及内部检查；不再先拿旧模型做外部成绩测试。

当前训练数据仍为旧435条=302正133负，g1/6/7/8分别98/120/117/100，364fit/71internal。此次完整对齐阶段新增正式标签0、新正式重训0；新RGB-D/CGR候选、点云、初始化步和规划ready均不计训练样本。论文DH3少样本设置为4型×100=400条，完整设置约4型×1000=4000条/144物体。435仅比少样本总数多35，不能称多于论文完整数据，也不代表每型或采集分布已等价。旧435标签是否复用按映射/闭合/物理/提起定义和来源审核，不因撤销箱体就丢弃，也不未经审核混入新集。

历史搬运源码、运行目录、碰撞失败、成本和冻结证据保留，不回写为提起任务或新的成功率。详见reproduction/boya/LIFT_SCOPE_AND_DATA_STATUS_20261114.md及五步交付清单。本轮范围/数据核验不新增物理、CGR、标签或训练。阶段相关轻量成果commit并push用户origin，大输出继续忽略，CanonDex只读。

## 2026-11-14 后续并行集成：三型绕障接近通过，仍在第3步搬运验收

用户要求多agent加速，动作/路径、采集集成、搬运后端三路并行，root独立核保存数组。原4cm四型首选仍raw879/rank73/nonflip/extra10mm，无重跑387原碰撞查询或端点IK。六笛卡尔路线共40次IK均null保留；新统一RRT-Connect(seed20261114/.25rad/.05rad/goalbias.2/1200iter/32connect)四型均28iter连通，首连接无第二解。初次全HOME拼接jerk31.90–32.28>30，四条均拒绝；g8另15tick手指碰000，最深.415224mm，保持null不派搬运。只对原几何清楚g1/6/7另冻结确定性统一伸时，原connected_path NPZ字节SHA不变，无新搜索/IK，HOME2180/1887/2025步，全tick全场景清楚、jerk28.941/28.799/28.991<30。原初态/预抓24列及掌姿终点/18非arm/50settle/500approach exact，root独立核导数通过。ready只代表纯规划，不能报新成功率3/4或0/4。

固定release已去除compatprimary物体GT偏移，显式scene/bin/SHA sidecar固定世界掌XYZ[-.4,-.25,.35]、参考HOME方向，SHA43223161bf53e4fa40335960e1b51185320646c8ee38f03793f2bea4fed7522c；这是Boya仿真适配非作者硬件TCP。首次缺16电机的真实FK失败0物理保留，修复后完整FK及独立XML误差0。post刚体门现对全部3源体+5箱体部件对称，不再漏primary；仍非动态物体预测。新pipeline补交原已验arm_collision_refinements.json到实际native父目录，同时锁来源/副本/combinedURDF/physics及其37唯一mesh，执行前后SHA复核后才生成可训练标签，防止基础设施/完整性失败遗留有效记录。

新continuation复用原候选和已准备HOME，三路各一次退4mm/回预抓/固定投放/松手准备已完成，原transport IK96×3全返回，但post全部null_geometry：g1/6/7 object2126/2413/2263+self4185/4857/4505；open及close/hold/retreat4mm/wait/回预抓全清楚，碰撞仅transport_bin/release/settle。首碰上臂-箱x正墙，三路release终点同时自碰及箱壁碰，最大约69mm；仅换中间路径也不能修此末端分支。0ready/0infra/0物理/0标签，不称新成功率。协议47f65a07...、并行wall27.639s；root独立源/命令/释放目标审计与74tests通过。后续先核同fixedrelease从统一参考初值的IK分支，不改release/候选/门。

后续同fixedrelease单HOME参考初值另协议一次原IK/13nfev收敛，末端自碰消失但上臂仍撞箱x+/y+，源STL也真相交，仍无clear目标；不是全局不可达证明。87tests通过(74集成+13新调度/固定目标)，新统一RRT+same-path retime driver已独立实现但尚未接当前pipeline；不得说正式采集全自动验收。下一阶段先按机器人/箱体共同几何另协议标定固定释放位，不按物体成功结果逐件调目标。

结果以reproduction/boya/AUTHOR_HOME_RELEASE_INTEGRATION_20261114.md及新轻证据为准，原74a78c9阶段证据不回写。正式新数据集、四头重训、新正式模型、冻结外部验收尚未开始；旧435/八权重13文件SHA复核不变，原作者源码不改，CanonDex只读。前三型绕障时长发生变化，不可称执行时长不变。已授权对齐和扩采，不重复问许可；每阶段轻量成果push用户origin，所有大输出继续ignore。

## 2026-11-14 并行集成：新入口已到完整臂准备，搬运未验收，正式重训0

用户明确要求多个agent加速推进。动作语义、采集/桥接、搬运后端并行实现，独立源码/几何/表示审核后集成。当前五步交付：第1数值组件完成，第2动作计算合同与新采集入口已实现并实际核验，第3接近/固定投放/搬运尚未通过，第4正式新数据＋重训、第5冻结外部验收未开始。不能再说只有数值函数，但也不能说已经完整对齐或重新训练。报告reproduction/boya/AUTHOR_COLLECTION_INTEGRATION_20261114.md、五步清单已更新。

ActionContract核验四型363槽、57历史执行239.4万个20手DOF目标exact；源pad表面/法向对应但仍未人工/硬件验收，6自由闭合触桌含旧成功只诊断不加门。新author_collect用未过滤单帧原top500/固定型/采集单flip/随机extra/全原碰撞，source采集flip第0/1列与部署第1/2列严格区分。真实native24列=6臂+2腕+16手，4mimic只派生审核，不得替换指令格式。原250close/50hold补回approach-only后，1600步六数组与旧真实计划逐值一致；已移除无用世界Z lift准备门，新增全体物体3cm预闭合推物早停及非法臂/箱体支撑门。Kit关闭前先落盘全部标签/成本/错误。

outputs/boya_author_scene_20261114_v4/capture完整三件开发S40 000/002/005同场动态落位/RTX，53inputs+518sources锁定、1633hulls原native20µm、质量.1kg/摩擦.6、每体-.98N、末250/250有地面载荷、全RGBD/96射线独立通过。源底仍比桌高.685–1.881mm，载荷是凸代理接地，别说源CAD精确接触。v2同快照接触流全空，只接受几何/稳定性证据；v1非法USD名字、v2缺clarabel、v3构造前contact设置被覆盖均零物理失败保留。两次成功初始化共1500tick/4500对象步，2真实RGBD/2官方identity CGR；两次图像/轨迹/候选数组相同不能冒称2独立场景。观察/执行明确冷重建实测pose+零twist，执行中不reset。新场景仍单一固定布局，非随机杂乱覆盖。

新原10cm四型队列outputs/boya_author_pipeline_fourtype_20261114_v1：2048→世界域97，完整原碰撞96/97/97/97→4/1/3/3，四型原首选raw879/nonflip/extra10mm不换。四条原native IK得到1300采样点，但全部手指穿邻近000而no_plan，无arm/self/floor违规；最终grasp1299清楚，预抓799及home/approach中段有源三角真实相交。不能叫全局不可达或新成功率0/4；0实际抓取/搬运标签。已发现backend旧.10错误继承：作者collect_dh3.py469实际.04，detect与grasp_and_throw均传同变量，pipeline以SHA+AST核验并强制.04，新受控队列v2独立保留。默认release还按compatprimaryGT平移，已确认适配bug；作者throw固定，需新显式固定投放协议修复，禁止把有此bug的ready计划执行/晋级。

最终40根集成tests通过(5.951s)，4cm分支21tests/26子测试另过。新4cm v2四型预抓及grasp端点全清楚、500approach全0违规，仍因home中间段手指-000实碰而4no_plan；无transport派发/试抓/标签，不能叫0/4成功率。首次合并测试缺repo PYTHONPATH的2导入错误、旧A0后段IK不收敛及其原receipt全NUL（独立返回副本有效、原文件保留）均明确留档，未伪造正常或重跑物理。旧435/八权重SHA复核不变，旧lift不并新transport；正式数据集/训练更新/新正式模型/外部验收均0。第一阶段8b3d559证据不回写，后续源码按精确归档解析；CanonDex只读。每阶段相关轻量代码/报告/证据commit并push用户origin，大输出忽略。继续修通用接近和固定投放，不为凑ready放门、移物或换候选；对齐和后续扩采已授权无需重复问。

## 2026-11-14 较早进度补充：仅数值组件阶段完成，完整新流程尚未接通

当前按reproduction/boya/ALIGNMENT_DELIVERY_CHECKLIST_20261114.md五步交付：第1步独立数值组件完成；第2步Boya动作语义＋新采集入口未完成，第3步多物体搬运未完成，第4步新任务数据＋正式重训未开始，第5步冻结外部验收未开始。不要把author_protocol/author_collision/author_training接口存在或8tests通过写成完整流程完成。当前阶段新增标签/正式数据更新/新正式模型均0；3960万点是几何资产，100/300/1000每型是容量规格而非已启动队列。训练目标已选定，不能再写论文/代码配置待选择。下一项实现是动作语义及新采集入口，随后真实动态多物体与搬运投放小闭环，再采兼容任务数据、训练和内部检查，最后冻结外部验收；用户已授权这些工作，不重复索取许可。

本次补充仅更新状态与交付标准，未运行新采集/训练/物理。第一阶段evidence/author_alignment_20261114/EVIDENCE.json的源码及文档SHA绑定提交8b3d559eac06230cb191de3f4fd053bddf0287df；滚动AGENTS/README/验收说明变化后用git快照核对旧SHA，不回写冻结证据。阶段轻量成果须commit并push用户origin，保持工作区干净，大输出忽略，CanonDex只读。

## 2026-11-14 当前：用户选定公开代码＋发布模型，完整对齐与扩采已授权，第一阶段组件完成

用户明确选定“公开代码＋发布模型”为主线，论文冲突另列，并授权可增加训练数据；后文历史“停止扩样”不撤销本次新授权，不要再次问许可。仍须先完成手型/任务/数据流程对齐，再正式采集、重训与新外部测试，不能又拿旧模型先测。

新增author_protocol/author_hand_clouds/author_collision/author_training及只读审计均opt-in，原源码不改。原top500固定型随机深入、全原碰撞类与原train/eval/save函数已独立接通；主线最终ReLU选中输出MSE/100轮/lr与WD5e-4/batch128/PolyLR.99，angle1/20输出为匹配发布权重的显式配置。Boya仍为计算参考表/位置驱动仿真替换，不能叫硬件等价。DH3采集4cm/允许2点、部署5cm/0点；源采集翻第0/1列而部署翻第1/2列，差异保留，勿混用Allegro参数。

四型g1/6/7/8×33共132状态，全21手link原三角面各采300000表面点，再1–5mm下采样；3960万是几何采样不是训练标签。outputs/boya_full_alignment_20261114_v2/hand_clouds约628MiB忽略。独立XML/NumPy FK核验2772link变换/660点云有限值与边界/363槽绑定，最大矩阵误差2.220446049250313e-16；未审计力学可抓性。8tests/25子测试通过，300条确实进入原训练而非旧first100截断；原验证P/R选型及安全发布推理strict重载通过。0正式数据更新/新模型/采集/CGR/物理，开发合成8更新含初失败、6临时合成checkpoint已删除。原435/八权重/成绩保持，CanonDex只读。

来源审计11个流程差异，pipeline_ready=false。下一步接新采集路径并验收Boya接触参考/接近方向/闭合语义，去掉不忠实的评分前手几何门；实现随机多物体动态场景及搬运投放标签；固定对象划分/预算/观测初态绑定后采集100基线及显式300/1000每型扩展，再正式训练及内部验收。旧lift435不得自动当transport标签混入，已有EGAD只作开发；还没有新成功率。报告reproduction/boya/PUBLIC_CODE_ALIGNMENT_STAGE_20261114.md，轻量证据evidence/author_alignment_20261114；阶段轻量成果应提交push用户origin并保持工作区干净，大输出不提交。

## 2026-11-14 当前优先：新六件仅诊断，完整对齐与重训验收未完成

新六件2/6、1/6是旧模型接原11分支增强后的诊断，不是完整算法对齐并重新训练后的验收。11月14日没有新训练，使用11月7日原MSE四头/preactivation四头与旧435条。以局部增强接通及旧物体可规划作为新物体开测依据，推进过早。当前停止新增外部试抓、扩样和基于本轮结果的调参，先回到算法/采集/手型映射/训练/执行链路核对。来源与待验收清单见reproduction/boya/REPRODUCTION_READINESS_20261114.md。不要把后文局部核验通过误读为整体算法完成。

正确接续：明确目标协议及所有差异；验收Boya接触参考与动作语义；判断旧435可复用范围，映射/闭合/标签若变更应重采受影响记录；按最终数据/划分/架构/损失/模型选择规则训练及内部核验，再冻结版本做新外部测试。仅补增强不必然需要重训CGR；不得用盲目重训替代数据和动作核对。本轮六件已用于诊断，不再当未使用的最终验收集。旧协议/输出/证据/失败均保留，状态更正0采集/训练/物理。

## 2026-11-14 最新：原增强接完整臂新六件完成，原配方2/6、修改版1/6

用户授权继续主线。outputs/boya_augmented_arm_external6_20261114_v1全部complete，研究协议SHA a956f2182f88e1b01016af6807c86dbe5270e166380c171e4b3befebd8144ca7，运行协议SHA 23a682bdc19a28e1fae3a74a7133c7b886305e3b41858aaf837a845b2e3a4605。先旧A3/A6/E0×两增强评分纯规划6/6 ready、0物理，再冻结EGAD A/C/E×2/5新六件；源文件和米制摘要与Boya训练/旧DGN/旧外部12/上一组六件无重叠，不能称CGR预训练未见或语义形状全新。

新6 RGBD、原11分支，两套旧评分分别选姿。原MSE（Boya按公开原配方训练，非其它手官方权重）2成功/0实际失败/4无计划=2/6；preactivation 1/0/5=1/6。A2两路、E5原MSE实际成功；所有12条件在首次物理前冻结，每条件最多一次，9无计划留分母。三次各2100步/七门/native20µm通过，共6300步/25.2物理秒，另7500初始化对象步；独立重建所有手命令、28实际关节目标/速度/掌姿、源顶点底面/双侧承载通过。没有新物体identity物理对照，不能与旧1/12或3/6相减声称增强净收益，也不能报完整总体3/3。

新72 CGR前向/7载入、135168增强候选；6 identity数组完全相同。56810查询/1136200评分输出独立CPU复算误差0、6668完整点云几何检查。最多4候选/条件，实际42次准备、3ready/9null；115嵌套拒绝分支中29只有手指-物体违规，不能全称机械臂够不到或全局不可达。原8+16 Halton/最多4新IK分支固定预算；没有放宽碰撞门或追加重规划。原手域/映射/闭合/阈值/原完整臂/.1kg/.6/粗凸代理均不变，原435/八权重/CGR不变，0训练/回写。六件代理源表面采样差最高27.357mm，native20µm仅验导入保留。

15tests含7策略拒绝子测试及独立分支/评分/物理前绑定/物理审计通过。3段正常速度235帧/25fps/9.4s录像、705帧全解码/源绑定通过，0回放积分；入口outputs/boya_augmented_arm_external6_20261114_v1/index.html。旧纯规划v1重复哈希输入时人工中止，0规划派发/0物理，v2缓存重复路径并保留前后核验；初独立媒体cv2解码不可用，保留源码回执后用已有imageio-ffmpeg检查原视频，0重渲染/物理重跑。新六件正式阶段无基础设施失败或重试。

报告reproduction/boya/AUGMENTED_ARM_EXTERNAL6_RESULTS_20261114.md，轻量证据evidence/augmentation_arm_external6_20261114。本轮完成并停止，无后台物理/训练/回放；仅opt-in，不自动重训、重抓、调阈值、改映射、扩样或晋级默认。旧成绩/负结果全部保留，整体论文复现仍未验收；CanonDex源码不改，仅轻量相关源码/证据推用户origin，大输出忽略。

## 2026-11-14 最新：原增强首选实抓对照完成，现用0/3→2/3，原MSE1/3→1/3

用户授权继续主线，outputs/boya_augmentation_physics_20261114_v1已完成。研究协议SHA696a288f3bcf99faea71cbd5461cca2bf82133098d01735df6edf7ea6abc18aa，正式physical_runtime_v3/protocol.json SHA54f914f347f2a741c24ba48e447733b7d14b88c5b3c255eb363567bcdbeaee9f。三件已用A3/A6/E0各原identity/11分支×原MSE/现用preactivation，A0上一阶段离线复用且单列。预设主比较原MSE仍1/3（identity E0成功换成增强A3）；次比较现用0/3→2/3（增强A3/E0）。A6两增强同姿态均失败；A0四条件全失败，旧coarse成功是另一姿态，不能声称本轮正对照成功或新的执行器完整验收。

新三件各1CGR strict载入/11前向，共33前向67584候选，三件identity全部数组与旧缓存相同；恢复/特征/碰撞云独立核验通过。15,408 canonical查询/308,160输出CPU复算误差0，24正式+24审计旧头加载；2268完整几何检查。原映射/闭合/严格>.85/1000/3500/400/八权重/435不变，无新RGBD/训练/标签回写。所有16几何首选在程序化历史匹配和物理前冻结，先验旧结果已知，不称独立盲测。

16条件物理严格去重14，7历史+7新一次。新增7均2100步/七门/native20µm通过，14700步/58.8物理秒；历史13775步，合计28475，原A3 MSE1175步approach推物早停仍失败。新成功A3 MSE/preact源底102.816/109.881mm、连续末态承载1.608/1.504s，E0 preact源底26.870mm/0.692s。其余新末态失载；E0 MSE参考点虽升90.170mm但源底1.981mm，不算成功，两个A0曾离桌后失载。原受控腕/粗凸代理/.1kg/.6/初态/20手DOF及五阶段均不变，无完整臂。

事后只读核验：A6已知成功仍在增强准入，MSE名次328、现用389；E0 MSE两个旧成功仍准入名次20/33而新增首选失败。A3旧成功现用增强名次1347被400预算排除，本次另一个新增姿态成功。补增强确实影响成败，但不是评分已可靠、唯一根因已解决或总体成功率。旧外部1/12、新六件3/6、固定池5/57及全部负结果保持，整体论文复现仍未验收。

初准备漏A0原MSE同数组历史，派发前physical_history_v2修正8新6旧→7新7旧，16首选/14姿态不变，原准备0派发。v2首波4因原生cooking审计缺clarabel在积分前exit1，0命令/评价/轨迹/试抓步；全部日志与0步初始化保留，停止后续派发。零步诊断v1场景通过、v2显式禁止积分捕获缺依赖，v3补依赖后通过20µm与全部积分前流程、保护停在首个sim.step，0积分。按现有声明clarabel==0.11.1，仅项目outputs/native_deps/clarabel_0111_overlay，官方PyPI SHA核对通过；不改共享环境或CanonDex。正式v3只加PYTHONPATH及输出目录，7队列/控制/物理不改，无已实际抓取姿态重试。未来调用原执行器须先检查此隔离依赖；不要重做已完成队列。

12tests（2新+10回归）、独立分支/全评分/排序/旧几何/全2100指令重建/全源底与承载/原输入SHA通过。7新RTX+7历史共14段录像全25fps、正常速度/源绑定/全帧解码通过，回放0积分；入口outputs/boya_augmentation_physics_20261114_v1/index.html。报告reproduction/boya/AUGMENTED_PHYSICS_RESULTS_20261114.md，轻量证据evidence/augmentation_physics_20261114。正式物理调度100.916s；表示/离线/故障及媒体成本分开。全部任务已结束，不自行重抓、调阈值、改映射、训练、扩样或晋级默认；仅相关轻量源码/证据推用户origin，大输出忽略，CanonDex源码保持不变。


## 2026-11-14 后续：A0两输入×两旧头离线部署对照完成，无新物理

用户问下一步，已先完成现成A0的2×2离线接入验证：outputs/boya_augmentation_deployment_20261114_v1，协议SHA1ad0711f30698deeaf3a194721a58f8b7311ac2ae370aabc5a1726ad14558315。原identity/11分支缓存×code100原MSE/现用preactivation各四旧头；冻结原映射/闭合/严格float32>.85/top1000/3500/400。手域raw38→484，查询226→2729，可用深入600→7724。短名单MSE138→400、现用185→400；完整几何准入MSE59→159、现用103→134，共1123完整点云检查。四组排名在任何完整点云检查前全冻结，原桌面前缀单列10880/134016次。

两个identity首准入均原行1369/g6/flipTrue/rawdepth3/extra0；两个增强首准入均分支1/行223（合并2271）/g6/flipTrue/rawdepth2，原MSE extra1、现用extra0。首准入评分名次1基2/11/1/52。不是臂ready或物理成功；不能把59→159/103→134当抓起提升。原准入在增强下仅保留23/59、19/103，两个旧首准入仍保留，说明预算挤出旧姿态。A0是已用正对照，不是新外部评测。

独立canonical5910查询/头118200输出/排序与门复算误差0；旧现用头226×20和103准入全部同序，3新tests过。92原输入/源码/模型/资产不变，原八权重/435/CGR及默认不变。0新CGR/训练/臂计划/物理；CanonDex仅原Boya只读FK/源几何，无项目任务或修改。正式8头加载，中止/恢复审计各8，共24次CPU旧头加载。独立审计初版逐行重复解压NPZ，精准SIGINT退出130，源码/日志/回执保留audit_interrupted_v1；仅审计改成一次读数组后恢复，四正式评分/几何未重跑。runner89.597s不含开发/测试/审计，不作速度比较。

报告reproduction/boya/AUGMENTED_DEPLOYMENT_RESULTS_20261114.md，轻量证据evidence/augmented_deployment_20261114，本地入口outputs/boya_augmentation_deployment_20261114_v1/index.html。全部任务结束。下一阶段建议对固定A3/A6/E0复用观测作同样离线对照，再独立协议冻结首选后受控腕每条件一次，A0单列；本轮尚未启动，不自动重训/改映射/调阈值/物理重抓。g8几何语义另立，不同轮混改。整体论文复现仍未验收，仅轻量相关内容推用户origin。

## 2026-11-14 最新：参考 fork 已检出，几何与 11 分支离线对齐完成

用户授权参考跨手适配。三个固定版本真实 Git 检出在 outputs/reference_repos_20261114：Jonas own_hand、kywch pixi、kitt server；轻量来源锁见 reproduction/evidence/fork_adaptation_survey_20261114。参考工程方法，不借用其它手模型、不运行第三方机器人脚本。报告 reproduction/FORK_ADAPTATION_SURVEY_20261114.md。

新 outputs/boya_reference_alignment_20261114_v2 complete，协议SHA7d55300aab23d9258eafa49c487ea94446556da195a243a1f209c9eb58439c41。独立 NumPy/XML URDF FK 297状态/924槽，矩阵误差最大1.8263e-14，未发现表格/FK接线错误。源pad五点仍candidate_pending_human_review。四型开度均单调闭合，但自由闭合中点开→闭移动42.403/46.318/50.742/33.880mm，接近轴19.864/42.684/19.406/110.066度，g8相邻槽最大34.068度。是静态源标注语义，不是实际物体/手掌漂移或失败因果；未更改映射、动作或标签，g6两点不能唯一确定三角面法向。

新增run_augmented_representation.py opt-in，复用冻结原CGR/helper，AST原增强/反射恢复，全部10矩阵在identity前生成，RNG顺序推进，排除DH3 rig门。单个已用A0旧RGBD，1strict模型/11前向=22528候选，identity17数组与旧缓存全相同；合并来源/全部特征/独立坐标恢复/canonical480通过。通用预几何top1000有106identity/894增强，不是Boya手域后准入或成功率提升。原identity场景云保留给碰撞。21tests，59输入/源码/资产/旧模型哈希不变；原八头/435/CanonDex不改，0手评分头/训练/物理。11前向内部1.1968s、runner27.9884s含序列化，无速度倍数结论。v1静态解析误拒fixed关节零axis保留，0模型/前向/物理；v2仅修独立解析器与测试，无推理重试。

报告reproduction/boya/REFERENCE_ALIGNMENT_RESULTS_20261114.md，轻量证据evidence/reference_alignment_20261114，图表outputs/boya_reference_alignment_20261114_v2/review/index.html。入口未晋级默认，无后台任务，不自动新物理/重训/调阈值；旧57评分漏选及1/12、3/6、所有负结果仍成立，整体论文复现未验收。新手几何更改会改变旧标签动作语义，不能直接套fork闭合锚点/偏移覆盖。只提交轻量相关内容，大输出忽略。

## 2026-11-12 最新：固定57候选关键验证完成，三件均有成功，现用评分首选0/3

用户授权验证“候选没有好抓法还是评分没选中”。outputs/boya_fixed_pool_20261112_v1物理complete，协议SHA c519b2be601e58fa88101aabb745848f8164301c6748f8df252e64772d77cc32。A3/A6/E0两套旧评分全部已保存几何准入61来源，严格物理数组去重57=31/11/15；不只用漏掉E0成功的旧preact35池。15最早同条件历史复用+42新各一次，成功分别1/31、2/11、2/15，总5/57；新增成功A3 s1642/g8、A6 s1573/g8、E0 s1842/g6。没有改映射、闭合或执行条件，三件均存在能抓起的候选。

先冻结目录/排名/队列，三路同池比较。preact top1/top3/top5含成功物体0/0/1（分母3）；原MSE 1/1/2；CGR 0/1/1。最好成功名次（1基、对象内）A3 6/29/8，A6 4/4/2，E0 9/1/10。E0两成功preact分数.813900/.804619均低于原>.85门；A6新成功preact0，A3唯一成功原MSE0。旧8头strict CPU重算2280输出与保存值误差0，0训练/新CGR/435回写。原MSE是Boya按原配方训练的头，不冒称官方其它手发布模型。

结论：此固定池明确存在评分/阈值/选姿漏选，不能继续笼统归因为Boya抓不了或必须先重做视觉/闭合。只有5/57成功，不能说评分是唯一根因；池此前受两评分/阈值/预算限制，事后任一成功3/3不是可执行策略或整个CGR空间覆盖，更非外部泛化/完整臂成绩。旧1/12、新六件3/6、共享评分失败及闭合平移负结果不改；整体论文复现仍未验收。

原受控腕/0.1kg/0.6/原粗凸代理/初态/20手DOF及五阶段不变。42新82688步+15旧30575=113263步，新增330.752物理秒，调度373.342s。50完整执行七门全过，7原approach推物保护早停（6新1旧）计失败，仅complete门不过。52失败末态源底<5mm且双侧失载，但57/57曾指接触、48/57曾双侧受载，不可描述成没有接触。A0粗正对照单列复核，排除57分母。

24tests/5子测试、23源/7097输入/42job、所有指令独立重建、全源底/承载及独立评分指标审计通过。准备空列表类型断言、审计相对路径两次零物理调用失败留档；正式物理无基础设施失败、无重试。报告reproduction/boya/FIXED_POOL_RESULTS_20261112.md，轻量证据reproduction/boya/evidence/boya_fixed_pool_20261112。57段正常速度录像全部完成（15历史+42单段RTX），50段235帧、7早停按实际帧数，全25fps/源绑定/全帧解码通过，0回放积分/0媒体失败；入口outputs/boya_fixed_pool_20261112_v1/index.html。媒体调度410.909s，无后台物理/训练/回放。原8权重/435/CGR及CanonDex不变，新增工具opt-in，不自动训练、调阈值、重抓、扩样或晋级默认。仅轻量源码/证据推用户origin，大输出忽略。

## 2026-11-12 最新：闭合几何平移对照完成，1/4→1/4，停止此规则

用户授权尽快验证一套统一局部几何修正。outputs/boya_closure_refinement_20261112_v1 complete，协议SHA17b480f304beeb22f33bd4803842418a47aff417c7257f3bd63b7f94d2b26ddb。已知A0 coarse正对照及原受控腕A3/A6/E0，原候选rank17/58/17/4各baseline/refined一次，共8实际16800步/67.2物理秒；两路均1/4，只有A0成功，三旧失败净恢复0，预设stop_this_rule，不晋级默认。

新增closure_translation_refinement仅用旧单帧可见稀疏点云+手pad闭合FK；125世界XYZ各±10/5/0mm，11共同闭合时刻，min_t max(双侧前向加权最近距离)+.1位移范数，原开手桌面/21扫掠门。四件静态目标均改善，实际无收益。选中Δ A0[-10,-5,0]、A3[-10,-10,10]、A6[5,0,-10]、E0[-5,-5,-5]mm。原姿态方向/评分/手型/开闭端点/20手DOF指令/粗凸代理/.1kg/.6物理不变；仅掌平移，可改变有效插入深度，勿称几何深入不变。优化无源物体mesh/真值pose/历史接触或标签；源mesh仅原执行及事后测量使用。可见面距离不含受载闭合/力平衡模型。

A0参考升高102.543→105.678mm，源底108.577→108.210mm、合格连续末尾1.648→1.632s；三失败两路全0合格离桌步、末态五指失载，源底约.794/.662/.699mm。8条完整2100步、七门/native20µm全过；12tests、独立静态目标/全部20DOF指令/全源顶点底面/承载核验通过，26源码/6934原输入/28派生/8job SHA不变。原435/八权重/CGR/旧外部1/12及新六件3/6保留，0训练/新CGR/回写。受控腕与粗代理的已知物体诊断，非外部泛化或完整臂新成绩。

八段正常速度235帧/25fps/9.4s视频已完成，统一outputs/boya_closure_refinement_20261112_v1/index.html，回放0积分。初RTX在第三段SyntheticData崩溃exit139；原2段/半成品/日志保留。媒体恢复v1无ffprobe、v2 cv2无codec均零新Kit，v3用现成imageio-ffmpeg、6单段进程一次完成，不重跑物理；delivery_videos为2原+6新回放。准备101.935s、执行调度145.387s、仅v3媒体恢复128.683s，边界见报告。报告reproduction/boya/CLOSURE_TRANSLATION_RESULTS_20261112.md，轻量证据reproduction/boya/evidence/boya_closure_refinement_20261112。停止追加平移/评分器补丁、调参、重抓或扩样；整体论文复现仍未验收。本轮已收尾无后台物理/训练；仅相关轻量代码/证据推用户origin，大输出忽略，CanonDex不改。

## 2026-11-11 后续：共享评分器离线对照完成，未过停止线，勿自动试抓

用户授权先固定映射用原435条比较共享评分器，未新增采数/物理。outputs/boya_shared_score_20261111_v2 complete，协议SHA e99b57abcd9e33743134c211ff133c6da2b26c3df0848c6f1058397862dfc54a，结果SHA e859377fe8e016a6cb2be1815793de0721b6b1d4ced5ab8fc994be3fee4c66d7。新496输入=canonical480+手型4/raw深度5/extra4/flip1/宽度2，仅预执行binding白名单。原32/8声明有标签实际29/6，364fit/71internal不变；混合正负物体22/5，同观测混合组31/4(21/4物体)，空3/2不补负例。

完整32声明物体按seed20261111分4fold8，训练/检查246/118、273/91、277/87、296/68，364恰一次折外。fit-only标准化连续482列、物体等权BCE/AdamWD.01；条件线性17参数/共享线性497参数各1000步lr.01；MLP496→64→1有31873参数、三个固定seed各300步lr.001、概率均值。CV选型落盘后才final32-fit/internal诊断，旧8头只在71内部前向，未把旧fit输出当OOF。

选中shared_mlp64：CV物体AUC .665181、条件线性.576345，gain .088835/22物体11胜2平9负，95%配对对象开发区间[-.038558,.221386]；同观测.622657 vs .545584；macroBrier .239261差于常数.214301。内部物体AUC .752564优于旧原MSE .555604/preact .349744，但条件线性.716557已较高；同观测新.727679略低条件.729167，macroBrier .272021差条件.204459/常数.213963。shared_linear内部.775714/有限混合池首选1.0未被结果驱动改选，不能称5次新成功。10预设门5未过，stop_no_physical_promotion；只有开发有限已执行池排序，不是物理成功率或独立外部泛化。

v1首个condition模型已1000更新，fit_meshes的np.str_导致weights_only安全reload失败；原progress因计数在reload后显示0，明确erratum而不改原回执。独立v2只保存时转Python元数据标量并即时记录fit历史，tensor/超参/fold/seed/门不改；失败和恢复首模型tensor存储字节相同，未放宽安全加载。v2正式25模型14500更新，加失败1000共15500真实数据更新；单测另48合成更新。runner106.441s含保存/评分/末尾哈希，不含前置校验/开发/独立审计。

26tests及原binding输入/25checkpoint独立前向/全预测指标/分组/来源/选型时序/门/成本复算通过，6650inputs/12sources不变。原8权重/435/CGR、旧1/12与3/6不改，0新CGR/物理/外部对象，CanonDex未改。报告reproduction/boya/SHARED_SCORE_RESULTS_20261111.md，证据evidence/shared_score_20261111，图表outputs/boya_shared_score_20261111_v2/review/index.html。本轮已收尾无后台任务；停止继续调参/挑seed/重训/试抓，不晋级默认；整体论文复现未验收。仅轻量源码/证据推用户origin，权重和NPZ等大输出不提交。

## 2026-11-11 最新：有限诊断与新六件验收完成，3/6，未晋级默认

用户授权尽快产出小闭环：纯规划、多候选诊断、单一统一修改、新6一试。新outputs/boya_fast6_20261111_v1已complete，协议SHA74632d0f8ff8981417c05fe22e5fc1cfc7db33ff02e6dd6301923d42f81ca52c。EGAD A/C/E×1/4固定6，与训练/旧DGN/旧EGAD12源和米制digest无重叠，不证明未见于CGR预训练。原scale.001/.1kg/.6摩擦/同XY；统一原V-HACD+规范凸面，源表面采样差最大33.479mm，非精确CAD；native20µm仅导入门。

新6原初始化7500对象步，6RGBD/1CGR载入6前向，1789query/6003组合→1095shortlist/197手几何，6/6有几何；15候选/360预抓IK/47路径→5ready/1null。五个ready全部baseline直接通过，额外分支本轮恢复0。A1/C1/E4各一次成功+99.587/+83.416/+89.669mm；A4/E1实际失败，C4四候选无完整计划，统一3/6。五条10500步/七门/native20µm过，无早停/重抓。E1参考点虽升88.061mm但源底.633mm、末态失载，失败；原视频字幕“物体升高”为参考点位移，勿误报离桌。原1/12不改，不能把不同对象的3/6当策略提升。

旧29候选纯规划单列outputs/boya_fast_planning_20261111_v1：15预抓IK+4连续IK+10碰撞；十碰撞均approach违规、五条仅手指-物体，不能全叫臂可达性问题。新planning_recovery原8seed+固定16Halton、最多4新IK分支，恢复E3rank59/G0rank4，零物理；928预抓IK/83路径/490.242s。0中转触发，未验证cuRobo绕障。20手DOF指令/四动作目标不变，home原限速延长88/16步；legacy seeds仍8，实际预算看EXPANDED_IK_SEARCH/PLAN_METADATA_NOTE。新入口opt-in、未晋级默认。

9新候选诊断outputs/boya_candidate_diagnostic_20261111_v2：按原preactivation几何排名、排除14历史实际物理数组相同姿态，A3rank36/80/95、A6rank22/26/33、E0rank1/2/3，每条2100步共18900；仅A6rank33/g1成功+66.037mm/源底59.809mm/连续1.22s。其它8失败，所有七门过。它是已知物体多候选受控腕诊断，非外部top1分母；成功后仍完成冻结队列，不回写435。

两次基础设施失败均保留：9候选v1因sandbox设备/缓存受限，Kit/USD已启动但0试抓步；v2仅外层设备权限/输出定位变化。新6首次表征漏LD_LIBRARY_PATH，0模型/0前向；resume_boya_fast6.py独立恢复原external12库环境和新batch回执，复用6RGBD，无重编译/下载/重新初始化。原协议/source不改，progress追加恢复且保留旧失败。

本轮14实际29400步，另7500初始化对象步；全部14段正常速度视频，回放0积分，统一outputs/boya_fast_delivery_20261111_v1/index.html。27tests及独立选择/排名/规划/全指令/源网格底面/末态承载/全帧媒体核验通过。报告reproduction/boya/FAST_BOYA_RESULTS_20261111.md，轻量证据evidence/fast_boya_20261111。原8权重/435=302正133负/364fit71internal/CGR与CanonDex源码不变，0训练；整体论文复现仍未验收。本轮已收尾，无后台物理/训练，不自动重抓、加预算或扩样。仅轻量源码/证据推用户origin，大输出不提交。

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
