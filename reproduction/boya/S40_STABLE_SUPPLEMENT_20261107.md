# S40稳定落位、覆盖补采与自动训练已启动

2026-11-07。新活动 `outputs/s40_stable_supplement_20261107_v2/collection/`，北京时间16:01:16启动，PID2110437；协议SHA `c0ae291aec2132d93bf0c7721fd0e15ba01c2e798d8d4f9e60f68a5a9d513037`。自动训练等待PID2120549，活动同父目录 `training_delivery/`。本报告为启动，不是最终采集、抓取成功率或整体论文复现结论。[启动与实测凭据](evidence/s40_stable_supplement_20261107/README.md)。

## 稳定落位实际完成

所有40件均使用同一真实桌面/原compound几何、mass0.1kg、friction0.6、原COM/惯量、dt0.004和物体专用0.98N重力；原Native20µm门保留。为并行初始化，只对原spawn两个函数做对象prim路径命名空间替换，碰撞/动力学数值不改，CanonDex源文件只读。每组10个空间隔离对象，4组CPU进程并行，不是GPU抓取并行。

第一版严瞬时twist阈值探针40件跑满各5000步，200000 object steps，没有发布稳定manifest。末1s位姿跨度小，但原生瞬时接触twist仍超严阈值；保留全部失败、源码快照和成本。新版本以连续位姿平稳为主并保留twist sanity：至少500、最多5000步；250帧窗口内最大位置跨度0.2mm、最大转角0.005rad，原生线/角速度峰值上限0.02m/s、0.5rad/s。初始化标准是新工程协议，不能描述成原作者或原抓取门未改的精确复用。

随后明确cold recreate：只在段间把实测稳定pose及声明零速度作为新初态，再保持750步=3s，全段最大位移≤1mm、转角≤0.005rad且末窗口仍平稳。**40/40通过**，包括原018/020/056/060/070早停对象；最大实际位移0.134280mm、最大实际转角0.004005rad。新版本113250 object steps/11325 shared ticks；加旧探针共313250初始化object steps，零抓取。这是稳定初始化验证，不是40/40抓起。

manifest、每对象新WXYZ pose文件、原始实测pose/twist、完整trace、3s保持证据同步绑定。正式每次抓取仍从稳定实测pose及**明确声明零速度**独立重置，不声称保留非零速度连续现场。重置前实测速度不覆盖成0；正式原native初态审计、完整臂控制与物理准入保持。稳定pose之后重新生成40帧RGB-D和40次共享原CGR，不挪用旧观测/候选。

## 有限补采预算

原40全outer train、源scale1、原32-fit/8-internal源视觉mesh SHA逐项沿用；无EGAD/旧DGN尺寸剔除。四型1/6/7/8×四extra0/10/20/30mm，共640新槽；每槽最多一次首个臂准入试抓，不能重复旧成功凑数。

历史只使用已执行的特征与档位覆盖，不按旧0/1结果选样。排除旧290实际的 canonical480×gesture×extra，含50认证早停；五个raw-depth分层各≤200来源行、总raw1000。每raw档双flip总≤64原几何检查，合≤320/槽；每raw最多2个几何准入、总keep8，原arm最多4。优先欠监督档位与特征差异，所有普通准备null仍不补0。100有效/型是目标，无法保证补齐20输出或每件成功。

CPU32–95，几何4/臂8/物理4对象worker；GPU0一个共享CGR进程。各对象内按原命令长度的homogeneous wave复用场景，跨wave仍可能建场，不能称GPU全物理并行或每对象只建场一次。新完整路径/执行门全部重算，只复用source/profile/artifact一致的零物理Native预处理。

## 自动接续与当前边界

`watch_s40_supplement_training.py`已实际等待本轮collector；最多8h，异常/未知停止，不重播。完成后自动导出新有效标签，与旧240两完整dataset合并：同40/alltrain/32+8，trial_id加源protocol命名空间，保留原local ID/source manifest；重复实际trace SHA或执行源路径直接拒绝。全部旧正负/早停/成本保留。然后固定20epoch内部32-fit/8check及all40四头训练，CGR冻结，EGAD未访问。

另发现上轮 `score_s40_raw_candidates`/`deploy_score_first.make_queries`遗漏原作者head前的接近方向/宽度/工作区联合mask，不能把3200全拒绝归因训练不足或称完整原部署失败。[更正与源码对照](S40_PREHEAD_MASK_REVIEW_20261107.md)。原DH3离线链执行了原mask，其9准入结果不受影响。

已新增纯函数 `boya_prehead_domain.py`，正确转换世界approach轴及XYZ，保持原source indices，要求显式Boya workspace/cone/raw-width配置；14项测试含与作者真实AST边界比对通过。尚未选定Boya域参数、未接入冻结部署或实际抓取；不静默复制DH3相机矩形/45–99mm宽度。**自动交付本轮只到合并与训练，部署评测待这一显式域配置冻结后另行验收**，不重复已知缺mask的raw部署作为完整结论。

40合同测试通过：稳定6、覆盖9、runner6、merge5、mask14。真实旧帧coverage8候选/169几何、旧240 merge源读取、全40稳定pose/trace/WXYZ串联已过。启动时40新RGB-D已完成、CGR在运行；最终阶段以本机pipeline_progress/stages为准，不提前填写新标签或成功率。

## 接续入口

- 活动采集：`collection/pipeline_progress.json`、`planning/queue_manifest.json`、`execution/PROGRESS.json`及最终`execution/RESULTS.json`。
- 自动交付：`training_delivery/STATUS.json`，最终`merged_dataset/manifest.json`及`training/TRAINING_RESULTS.json`。
- 初始化：`settling/objects.json`、`RESULTS.json`、各group完整trace；旧v1目录单列保留。

不要重复启动collector或watcher，不热改任一已冻结Python/资产。追加独立模块不改变当前source freeze；大数据/图像/轨迹/权重/视频仅留outputs，不提交。CanonDex仍只读。
