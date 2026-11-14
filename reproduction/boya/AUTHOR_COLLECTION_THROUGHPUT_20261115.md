# 采集吞吐与下一步最小切口（2026-11-15）

这里只读统计现有回执，没有重试旧失败、启动新模拟或修改冻结源码。新工具 `author_collection_profile.py` 读取 stage/ledger/audit 小回执，输出耗时、完成区间重叠和已有接触计数；不重新解码大接触文件，也不认证训练标签。

## 实测瓶颈

旧开发队列从首 stage 到末 stage 为 5295.205s（88.25 分钟），40 个预算槽产生 14 个有效 fit 标签，internal 为 0。状态为 14 labelled、10 null_geometry、12 placement_null、3 no_candidate、1 infrastructure_failure。按完整墙钟约 9.52 标签/小时；不能用这一个小队列保证正式集或 internal 有相同产出率。

| 已有阶段 | 数量 | 累计 stage 时间 | 均值 |
|---|---:|---:|---:|
| 执行 | 14 | 3119.46s（52.0min） | 222.82s |
| 独立审计 | 14 | 822.06s（13.7min） | 58.72s |
| 初始化＋capture | 7 | 492.67s（8.2min） | 70.38s |
| prepare | 28 | 1195.68s 工作量，实际各场景关键路径合计598.50s | 42.70s |
| CGR 表征 | 7 | 29.00s | 4.14s |
| build | 7 | 54.59s | 7.80s |

prepare 已有最多 3 路重叠，其累计工作量不能直接当墙钟。10 条几何 null 累计 prepare 工作量277.41s；3 条无候选仅11.57s。三场完全无实际抓取的可构建场景0/7/8，其总场景时间为438.59s。大头是执行和完整接触审计，不是 CGR。14 条审计共独立处理109,128,993个接触点。

执行 stage 的223s包含启动、场景/机器人导入、数值门、物理、接触读写、SHA和关闭，现有回执没有内部计时，不能把它全部归因于启动。首执行日志在 `30,965ms`（30.965秒）仍出现导入/材质警告，只能证明启动/建场存在可见成本，不能据此准确拆解其余时间。

## 调度收益边界

三 worker 的一般理想上限为3倍，真实收益尚未实测。按旧10场景的固定三场景波次重放，在每个场景耗时不变、没有资源竞争/并发减速和新开销的最乐观条件下，各波仍至少1418.35/1270.32/1225.87/0s，合计3914.54s（65.24min）。旧重场景分布不均，理论也只有约1.35倍。这是计算下界，不是跑出的并发成绩。

简单沿旧产出率外推4800个标签约21天，最终 export 再按旧审计均值逐条重算另约78小时；internal目前0标签，这个外推不能当完成承诺。即便采集达到理想3倍，单纯调度也不会把它降到几小时。

每个 child 当前独立依赖559个、约350.9MB，其中checkpoint307.6MB。formal freeze仍逐child重复原B.freeze的前后校验，千场景会有TB量级重复SHA读取。fresh6文件的首末冻结时间戳相差2.53s，只是已有hot-cache文件时序，不是正式冻结性能基准。更明显的附加成本是 export 再串行完整解码/审计每条trial；当前waves没有将全量export并行。

## 最小真实并发验证

使用已经冻结的 fresh6，父协议`c657176d0478391e2907289d0c454c57c05deaf3262ba1e3c30a69b991191c25`，不重试旧失败。其原序为：fit null/null/ready，internal null/ready/ready。第一波只有一个实际可构建场景，不能验证并发；全6预算第二波才有两个实际场景可重叠。最多3次capture、3次identity CGR、12次prepare、12次执行，实际null继续保留。即使全通过，也只验证两个实际场景重叠，不能声称验证了三个或3倍加速。

2026-11-15预检：父及六child所有SHA通过，旧串行已结束、40ledger完整、原全局lease可获取，无匹配活进程；fresh6无started/stage/quarantine。sandbox内nvidia-smi不能连接驱动，GPU硬件/显存需root以实际运行权限检查。随后是否派发与实际结果由root记录，本报告不把预检当运行。

## 三项优先优化

1. 先测fresh6的实际重叠、单条执行和审计是否因竞争变慢。下一版可用有界流水队列替代整wave barrier，仍按原index接纳首配额，保所有在途成本。这样先消除可明确计算的负载失衡；不改变候选或动作。
2. 对执行最大块做内部profile后，再实现常驻Kit：保App、每job销毁旧stage/物理上下文并用原构造器新建，完整750初始化和所有命令/门保持。先只省App启动，不立即跨job复用动态物理状态；验证通过后才研究机器人/凸体cooking复用。完整接触解码、序列化/压缩可能也是大头，应分别计时，而不是猜启动占比。
3. audit与执行流水并行，导出验证既有独立audit证书及其全部输入SHA，避免对完全相同内容再次解码数百万接触。保持独立全数值审计至少一次，不能以mtime或抽样代替；冻结共享依赖在同一校验轮去重。此方案需新协议/实现，当前冻结export不改。

## 常驻方案的最小可实施顺序

先新增独立profile适配器，在未来新协议中只计时以下原调用：App启动、ready/导入/原生几何门、命令生成、before_step、sim.step、after_step、ContactJournal序列化与gzip、finish/SHA、close。保原调用顺序、数值与完整流；当前fresh6已冻结，不能临时注入这个新源码。现有只读工具只能给整个stage时间，不能补造内部profile。

最干净的常驻服务是CGR：strict加载模型一次，逐job恢复原seed，仍一次identity前向、相同2048候选和全部17数组逐值对照。但当前仅4.14s/scene，即使完全消除此项也只省约0.55%，优先级低。

有实际收益的切口是独立resident execution wrapper，保留原一次执行标记、finalize-before-close和独立审计契约，job结束只销毁job场景，worker最终退出才close真实App。原wrapper会调用SimulationApp.close且可能os._exit，不能直接在原execute外套for循环。新worker必须有单独source pin、job输入/输出协议和完整初态读回；每job采集时禁止沿用上个job动态对象状态。worker持有Kit和全局lease FD，实际服务进程而非仅IPC客户端承担槽；崩溃的active job不重试。

capture服务之后可复用同类App生命周期，但仍逐job做原750tick初始化与一次RGBD，observation锁仅覆盖capture/CGR，execute保render=False。无需先将capture、CGR和execute合成一个巨大服务：先证明一个resident执行worker跨两个新job的数值/初态/资源守卫一致，再考虑多worker。未经这个开发核验，不降低物理质量/步数、接触完整性或碰撞门换吞吐。
