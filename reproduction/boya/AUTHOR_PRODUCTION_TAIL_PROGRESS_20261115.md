# 原固定队列尾批：15 场景运行中

用户要求先跑完剩余队列再估计采满耗时。本批只取原 seed2026111501、32 场景预算内尚未启动的原 index17..31，**15 场景／60 手型槽，7 fit＋8 internal**；原三体身份、scene spec 和顺序保持，10 ready／5 placement_null，不重抽、不补例、不重抓。

运行根为 `/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/production_tail_20261115_v1/run`。协议 SHA 为 `04552e9a7ef2db6cf8c7e1b21a022014a816d695cbd4a62f1557b1e29597d2ea`，两窗 8／7；最多 8 CPU 执行、2 原首次数值审核、3 prepare、1 observation。GPU0 UUID 固定，GPU1 不选。窗口开始后父 PROGRESS 仍可能显示冻结快照；实时读窗口 ledger、实际 STAGE 和只读 host observer。

独立 `author_production_tail` 私有编译固定 SHA 的原 production adapter，只改剩余预算、窗口和入口；561 项源码绑定及归档已冻结。执行入口使用已验证的 `author_verified_execution`，原始接触写入 `/dev/shm/anydex_production_tail_20261115_v1`，隔离压缩并核完整逻辑字节后交原 pointer。原 250 Hz、质量摩擦、三体、24→28 命令、全部接触点／原数值判定不变，capture 仍走原完整初始化及独立全点审核。

标签准入同时要求原 runtime audit PASS、完整 capture audit PASS、执行 seal 通过及原 runtime／trace／接触行数一致。原审核退出 0 但证书 failed 时也立即停新非审核派发；使用原 ExpandedPool 两路审核，已用真实普通 Python 子进程验证该资源槽释放。5 项新增 CPU/mock 检查通过，包含 8／7 尾窗、原数组导出、失败证书停派发和实际日志篡改拒绝；mock 不算物理采集成绩。

本批已启动第一个实际 capture。新增合格标签及完整成本须以本轮原 ledger／回执和最终报告为准；启动时累计仍为 20 fit＋1 internal，旧 435／八权重保持。正式训练、新模型、训后外测为 0。首基础设施或完整性失败停新派发并收尾，不自动扩预算；完整完成后再按各手型／split 实际产出估计耗时。
