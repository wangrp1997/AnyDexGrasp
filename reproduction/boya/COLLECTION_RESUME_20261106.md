# 四型采集续跑：2026-11-06

原正式50物体800槽已完成：610有手几何、190无候选；431完整臂准入、369规划null。原run_v2在3对象27次实际（22正5负、56700步）后于北京时间02:49停止新派发，46对象未派发，Snowman在纯prepare文件校验阶段被拒，零Kit/零物理。旧统计与目录原样保留。

原生规划器为关节速度限幅延长了3条计划的approach：总步数2124/2126/2142，其他428条为2100。原采集适配器固定2100，误把合法原计划判作程序异常。新 `execute_collection_queue_v2.py` 只改完整步数绑定，仍要求原phase边界、时间网格、命令SHA及完整target/phase/desired-palm数组，闭合/保持/抬升长度和原物理门不变；未改原生规划器或原命令，不按对象打补丁。431计划及10合同通过，Snowman16条prepare-only通过。

新 `resume_collection_execution_v3.py` 在原run锁和新分段锁下，只执行已证明未执行的404条（47对象）；原27条从源记录只读继承。显式unknown、未核实的partial执行、已完成ID混入剩余账本等均拒绝，不自动重放。两对象CPU并发、每对象serial resident batch16；普通合格负例继续、程序异常停止新派发并收尾。

**活动目录**：`outputs/collection40_20261106_v2/execution_resume_v3/`。已于北京时间2026-11-06 10:44:39实际启动，PID3977466；协议SHA `ed51a65b24860b25c7b2116de209761522e62e04089039aec42605bef5a9a773`。原 `run_v2/pipeline_progress.json` 是历史停止记录，不覆盖；新进度为活动目录 `PROGRESS.json`，对象执行在 `execution/<object>/RESULTS.json`。当前启动快照为running且两个对象PREPARED已通过，不冒称已完成新标签。

最终活动目录 `RESULTS.json`、`collection_counts.json` 和 `train_records.jsonl`/`validation_records.jsonl` 合并继承27条与本段新标签，按冻结slot ID去重；原开发105不并入。`all_records.jsonl`只供审计，不直接训练。每型训练/验证准入上限分别85/89/89/90和17/21/21/19，共353+78=431，原400+80目标必有欠额；不按结果换对象或扩预算凑数，最终按型报告。当前仍不训练评分头。

完整660文件/1768项只读审计核对原27目标/phase/palm数组、记录/轨迹/命令SHA与原冻结源码，标签一致；独立重算末态离桌/双侧承载/抬升，未冒称重测所有接触门。账本5个负例合同通过；root又真实核验源输入和原431=27+404。审批服务曾timeout/用户并发上限，失败请求均未执行；结束并行审批后单独请求成功启动，无绕过审批或共享环境改动。

来源与启动快照见 `evidence/collection_resume_20261106/`。旧v1/v2执行器与旧run字节保持。后台未结束前不改冻结执行源码、不重复启动。
