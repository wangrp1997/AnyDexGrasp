# 评分顺序修正：先对原始CGR组合评分，再做几何和臂准入

2026-11-06。新增独立入口`deploy_score_first.py`，旧采集入口、旧评分训练、旧6/10对照和所有冻结输出保持。此处修复已确认的部署顺序差异；**整体论文复现仍未通过验收**，原物体对照见[物体资产报告](../OBJECT_SET_ALIGNMENT_20261106.md)。

## 新路径与锁定版本

默认`released-dh3-v1`严格按固定官方`robot_dh3.py`的post-CGR顺序：原CGR分数前1000→原单一flip/canonical480→各Boyahead的raw_depth*4+extra四输出→所有手型/深入全局前3500→严格score>.85→前400进入几何→按全局分数顺序进行有限完整臂准入、首个准入者成为top1。

评分不依赖已选Boya开度/FK/palm，现有原Inference类、strict权重载入和head元数据校验直接复用。相同CGR/flip/type只前向一次，20输出随后选择四额外深入；不是对旧78条准入计划重排。阈值按原Torch float32比较，精确float32(.85)不会通过，无自动降阈值或采样回退。确定性分数并列采用源顺序；原torch topk的并列行为未指定，这一差异明确保留。

`RANKING_FREEZE.json`及全部全局排名在读取手部标定表、开度选槽、几何和臂调用之前写出。原RGB-D/相机/缓存CGR通过receipt和输入/输出SHA交叉绑定；试抓标签及原准入队列不进入排名。每个已准入几何候选生成独立单行pool，再调用原prepare_one做一次新臂准入；所有手型/深入共享显式臂总预算，缺省4次，只准备不执行。

可选`paper-testing-budget-boya-port-v1`使用论文测试段top100/top200及公开code四额外深入、原flip；其名字和记录明确是Boya移植变体，**不能把论文和发布code不同参数拼成已消歧的同一协议**。默认仍为完整发布DH3参数，无自动切到该变体。`--flip-policy both`亦为显式额外Boya搜索，默认关闭。

## 已实际验证的范围

8项功能检查通过：旧前500/keep8之外的高学习分候选可被选中、全部20深度索引、float32严格阈值、跨型/flip全局排序、标签置换不改排序、空池/非有限输入、原单flip/canonical及场景绑定、freeze先于几何。独立审阅发现并已修正float64阈值误放float32(.85)的问题，未加epsilon。

对一个**原DGN训练对象的已有缓存**做工程检查，不是原S40评测或新留出成绩：2048原CGR取前1000，4头共4000query，展开16000手型/深入组合，按发布参数得到400个几何输入。400个全部被桌面检查拒绝，零臂准备、零实际抓取，结果为`no_admitted_hand_geometry`。不把零抓取改成抓取失败率，也不因结果为空改阈值、放宽桌面门或换候选池。

最终`released_geometry_v2`内部5.205s（含head及几何，不含进程启动），此前rank1.450s、geometry_v1为5.363s全部保留；三次共12000head query，零新CGR前向、训练、物理步。v2增加缓存RGB-D/相机绑定检查，未改变数值选择；三次shortlist相同。head分数ReLU不限定到1，不能解释为已校准成功概率。

独立从NPZ的head输出用NumPy重建全部16000项target/index/排序以及严格阈值和前400；重新从作者函数计算1000行canonical480/实际flip，并验证所有冻结输入/输出SHA，均通过。完整产物`outputs/score_first_alignment_20261106_v1/`，Git只提交源码、测试和轻量证据。

## 使用与当前边界

```bash
python -B reproduction/boya/deploy_score_first.py \
  --observation OBSERVATION_JSON --frame-dir CACHED_CGR_DIR \
  --calibration CALIBRATION_JSON --training-results TRAINING_RESULTS_JSON \
  --training-split FROZEN_HEAD_TRAINING_OBJECTS_JSON \
  --output-dir NEW_OUTPUT_DIR --stage rank
```

`--stage geometry`在冻结全局排名后跑手/桌/接近检查；`--stage arm --backend-config CONFIG`进一步准备一个绑定top1或明确null，支持显式`--collision-refinement`复用源绑定资产修正，仍不执行。**本轮没有候选通过几何，所以新入口的真实臂准备路径尚未实际验收。**

Boya当前四头仍由旧DGN训练，未因新入口而变成原S40训练；本轮也没有新物体抓取。单帧CGR而非作者11增强、Boya标定/手型/控制、完整臂和手/桌几何后端、数据及成功标签均仍为适配差异。代码只修复post-CGR评分筛选顺序，不能据此宣布整体忠实复现。原6/10不能迁移为新路径成绩。后续需在明确的S40身份/尺度及独立采集协议下再训练、评测。
