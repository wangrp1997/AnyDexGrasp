# S40补采：原生移动段时长兼容修复与安全恢复

2026-11-07。原 `collection` 已完成640候选准备、195臂准入/445准备null；实际执行157条得到116正/41负、329700步后，于060准备阶段停止。旧3unknown/35未派发与blocked watcher原样保留，不把旧批次改成完成。加旧初轮240，当前有397条有效源记录；未完成合并/重训，不当435或新泛化率。

## 卡点及具体证据

060_g6_d1原计划2166步，home750→816、approach仍500。规划器原 `timed_segment`将3s视为下限，按原速度/加速度/jerk限制延长至3.264s。旧v2入口将所有延长归给approach，拒绝合法home时间缩放。从原源码AST、原URDF限位及保存端点重新生成三条home，数组与全部timing字段精确一致；不是改控制救成功。

060队列在全部check_trial完成前抛异常，尚未保存PREPARED或派worker；原执行目录只有.queue.lock，源/日志/提交manifest完整绑定，证明这3条零Kit/零抓取物理。其余35条同时有原final not_dispatched、无SUBMITTING/输出的证据。仅据缺文件或SUBMITTING不自动判断未执行。[轻量实证](evidence/s40_temporal_resume_20261107/zero_execution_diagnosis.json)。

## 新接口与验证

新execute_collection_queue_v3只读加载固定SHA的v2，复用全部prepare主体/execute/worker/finish_trial；唯一时长守卫按原segment_timing、连续phase、完整time grid、边界及n三方一致，分别支持home≥750、approach/lift≥500，settle50/close250/hold50保持。原速度比≤1、a≤3、jerk≤30与显式元数据限值保持，任何joint_target/desired_palm/phase/time数组都不改。

v3在独立进程安装时长函数至队列及原终止证书模块的对应依赖；原文件不修改，证书其余source/prefix/首停止谓词全部复用。导出v2薄入口先装该合同再调用旧导出器，避免扩展home计划若真实早停时被旧证书守卫误拒。wrapper、原v2和所有原依赖均冻结。

实际195份原计划全部通过新合同；9项测试涵盖真实816步home逐数组不变、相位/闭合/时间篡改及限值拒绝、全部原计划、终止合同一致、157继承/38分区与拒绝重复启动。原060执行准备实测3trial/2cohort、0物理通过；不冒称已抓起。

## 已实际恢复

独立活动 `outputs/s40_stable_supplement_20261107_v2/collection_resume_v3/`，北京时间2026-11-07T19:04:09.621434+08:00，PID2371155；四对象CPU并行，只派38条第一次执行。原157及其records/commands/traces只读消费，不重新观察/CGR/几何/IK。原640及32/8全train分组保持，证据化的3条零执行重新分类写入新protocol，旧unknown记录不覆盖。

新协议SHA `ba712cc75c3467457436b78b47eb438658958b9b8abae31981faa70a6a4ea076`。新JSONL的继承只改collection_context追溯新/原协议，原source record保持，导出器继续逐条重放原collect_record。不会把null补0。

新自动等待PID2371156，`training_delivery_v3/STATUS.json`；完成后新exporter_v2→合并旧240与本轮全部有效→固定20epoch四内部及all40评分头。旧training_delivery的blocked保持。本段尚无新的训练结果或部署评测；评分前Boya域配置/评测仍待单独冻结，EGAD未使用。

后续看新collection_resume_v3的pipeline_progress、execution/PROGRESS和最终RESULTS，不再将旧collection日志当活动进度。不重启已经派发的新恢复目录、不热改冻结源码。大图像/轨迹/权重不提交，CanonDex仅只读后端。
