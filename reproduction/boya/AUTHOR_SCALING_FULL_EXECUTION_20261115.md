# CPU 完整轨迹吞吐诊断入口

独立 opt-in `author_scaling_full_execution.py` SHA256 `4095477e27f08a52ddd175669c55dcb75e122f282d9b97be5c4a8a68f1cbd5dd`，严格绑定已冻结 scaling adapter `4037bebc...`。此前 100tick 六组协议和源码不变；完整轨迹需要外层另冻结 CPU1/CPU8 对照协议后才能启动，本次实现与单测未启动 Kit。

```bash
python -B reproduction/boya/author_scaling_full_execution.py \
 --prepared-dir /ABS/FROZEN/SCENE4/g6 --output-dir /ABS/FRESH \
 --protocol /ABS/NEW_FULL_PROTOCOL.json
```

每进程仅 CPU 单 lane，执行原模板完整 commands；实际冻结 g6 为 3024 行。外层现成资源调度器负责单进程与八进程对照，本入口不创建额外调度器、不增加候选或重抓。复用原完整 shared clock、原控制和所有接触/源网格/28DOF/材料/向下重量/原门，仅通过已有 scaling 适配设置 CPU 设备准入与 CPU tensors 证据。未编辑 GPU vector/profile/probe/native、既有采集源或 CanonDex。

`GPU_VECTOR_EXECUTION.json` 保留原完整执行与逐 lane 原始 gates/诊断标签。新 `GPU_VECTOR_PROFILE.json` 和 `SCALING_EXECUTION.json` 在真正 App.close 前落盘；后者成功状态为 `diagnostic_full / complete=true / full_trial=true / prefix_ticks=0`，绑定 READY、commands、scene、两份底层回执及全部 source。`raw_label` 保留原诊断，正式 `label=null / formal_training_eligible=false / exclude_training=true`，不进入训练。完整时间保存 UNIX/monotonic 区间，同时保留显式同值的 `execution_prefix_*` 兼容别名；它们此处覆盖完整轨迹，包含 generator 收尾序列化，simulate/fetch 等分项另记。

成功必须原完整执行成立、returned/native-counter/每 lane observed 与原 commands 长度一致；异常保留已返回、已观测、在途未知与上下界，不把可能还在 fetch 的步数冒称完成。CPU 探针要求 GPU dynamics=false、MBP、ordinal=-1 及实际 CPU tensors，失败关闭。

2 项 CPU tests 0.116 秒通过：源码变更在启动前拒绝；原 shared-clock 完整推进 5 行合成 stream，逐行完成、原 raw label/gates 不变、正式标签为空，并在模拟 close 立即 os._exit 前保存 profile/sidecar 及 SHA 绑定。合成 5 行只核接线，不代表实际 3024 步物理验收；真实模板 3024 行来自先前只读预校验。本 agent 本轮 0 Kit/物理/IK/CGR/训练，未启动 CPU1/8 完整对照或正式 bulk。
