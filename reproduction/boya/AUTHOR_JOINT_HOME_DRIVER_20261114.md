# HOME 单次搜索与同路径时间调整入口

新增 opt-in `author_joint_home_driver.py`，供下一阶段自动接入。当前四型 RRT、三型时间调整及搬运准备回执保持原样；没有通过此入口重做实际规划或物理。

```python
from author_joint_home_driver import prepare_home_route
receipt = prepare_home_route(
    seed_plan_dir, scene_path, output,
    config_path=backend_json, archive_sources=historical_scene_sources)
```

`config_path` 为原完整 BACKEND JSON。`archive_sources` 沿用原场景源码路径到相同 SHA 历史归档的显式映射。`output` 必须是不存在的新目录。返回 `ready`、`null_geometry` 或 `infrastructure_failure`；只有 ready 的 `plan_dir` 有值，指向 `output/rrt/native` 或 `output/retime/native`。子计划仍采用原模块的 metadata、source/input locks 和命令格式；顶层协议及回执独立绑定完整调度，没有改写子回执。

启动任何子准备前，入口核对输入和固定核心源码，写出 `HOME_ROUTE_PROTOCOL.json`，冻结统一预算：RRT 准备一次、随机种子 20261114、1200 次迭代、每次 connect 最多 32 次 extend、0 IK；原策略的 `.25rad` L2 步长、每关节 `.05rad` edge 采样、`.2` goal bias 和全部原几何门均保留。只有第一次返回明确已知的 `first_rrt_connection_failed_final_tick_or_derivative_gate`、树已连接、最终全场景几何报告 SHA 正确且无违规、有限的关节位置门通过而速度/加速度/jerk 超限时，才调用原 bounded retime 一次。

该次 retime 使用原保存路径，同一路径文件 SHA 必须完全相同；内部沿用原最多 8 次时间检查（包含最初失败），统一时间倍率 `max(vratio,sqrt(aratio),cbrt(jratio))*1.01`。碰撞、预算耗尽、端点违规、未知 null、基础设施失败和原单段时间生成失败均直接结束，没有重新搜索、改路、换候选或 IK。时间调整自身返回 null/infra 也直接结束。实际最后全 ticks 的全场景审计仍由冻结子模块执行。

顶层 `operations` 保存每次尝试的阶段、子回执 SHA、完整准备时间和已报告搜索成本；`cost` 累计 RRT/retime 调用次数、已知搜索次数、子阶段时间，`wall_s` 包含顶层校验开销。异常发生时仍记录已进入的调用；无法确认是否执行过搜索时 `actual_search_runs=null`，不误记为 0。最终原生 plan、commands、path geometry 及依赖 SHA 再核对后才能返回 ready。

固定核心保持不变：

| 模块 | SHA256 |
| --- | --- |
| `author_home_route.py` | `1bf86feaccd7880128b54ace69b1289027d313e7f8a641bc44ef6aa00d0ff0d8` |
| `author_joint_home_route.py` | `cc390e0363696c5c65c368672a385c885f5ccb4ba450bdb1323d7766b2a0fcc6` |
| `author_home_retime.py` | `0beb0b8aa0640d6ced6c9ed3a7f3ad500fd6f335ccac17eaa8699986507bf884` |

10 个 dispatcher tests 通过，子规划器均为 mock。覆盖启动前冻结、正常单次 ready、只允许时间失败的一次 retime、真实碰撞/未知 null/infra 不追加、关节位置超限不调整、几何 SHA 被修改、retime 换路径、输入中途变化和异常成本记录。没有调用真正 RRT/IK/几何规划。

另对旧四型保存产物实际执行只读核验：每型 622 个输入锁全部通过；g1/g6/g7 的 eligibility 为 true，g8 因原最终细采样碰撞为 false。该检查 1.0231s，0 新 prepare/search/retime/IK/physics。证据见 `evidence/author_joint_home_driver_20261114_v1`；完整只读输出为 `outputs/boya_author_joint_home_driver_20261114_v1/READ_ONLY_ACCEPTANCE.json`。

该入口尚未接入 pipeline，也没有据此增加实际 ready 数、训练标签或成功率。全场景门仍是源凸代理和离散采样，物理动态/抓持/搬运成功仍须后续冻结执行验证。
