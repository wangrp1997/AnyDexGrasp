# Boya评分训练与公开训练实现对齐（2026-11-08）

本轮**未发现当前code100训练的前向、监督索引、梯度或更新错接**。同一真实记录顺序和已发布20输出配置下，直接执行原`train_one_epoch()`与现有适配循环，四头各两轮的初始化、前向、全部参数梯度、Adam更新后状态及7层BN缓冲误差均严格0。无需据此改loss或重训；此前preactivation控制仍是明确改过loss的独立实验，不能冒称作者配方。

## 实际核验

- 原435条绑定记录的raw2261经作者`get_graspgroup_features`重新得到canonical480，与训练NPZ逐字节一致；label及`raw_depth*4+extra`均绑定真实记录。162条使用现代奇数角度bin，直接存canonical480有必要，不能再用旧loader的`12+2*angle`重复转换。
- 实际执行原Dataset类：先按手型过滤，按JSON插入顺序保留first100，再shuffle；测试集不截100；无类别平衡或正负重采样。24个旧angle×flip组合与部署转换误差0。
- 当前32-fit四头83/98/98/85，全部小于100且小于batch128，因此first100和原单样本尾批跳过**对当前code100没有影响**。历史all40头有120/117等记录，不能与原first100截取完全等同；本轮未使用或重训历史all40头。
- 两轮数值对照直接执行原函数，使用相同批次顺序。原始随机种子和DataLoader shuffle历史没有公开；本对照不声称复原作者随机轨迹。CPU临时原循环8更新+适配循环8更新，不保存checkpoint、不部署模型，零物理步/零CGR前向。

## 仍未完全相同的协议

| 项目 | 公开源/发布文件 | 当前Boya |
|---|---|---|
| 架构配置 | 训练脚本默认240输出；31发布头实为20输出 | 采用发布模型的480→20，非默认240实验 |
| 网络及目标 | 原Conv1d/7层BN/残差/final ReLU，选中执行输出MSE | code100完全相同；preactivation实验另列 |
| 优化默认 | 100epoch、Adam lr5e-4、WD5e-4、PolyLR .99每epoch一次 | code100相同；旧paper20为公开说明过的混合配方 |
| 模型选择 | 每epoch用test阈值P/R保存best | 固定final100，避免利用反复调试内部8件挑checkpoint |
| 样本/标签 | 作者真实手标定、真实试抓与搬运结果 | Boya几何标定、仿真抬升；接触代理/闭合仍待对齐 |

不能为模仿原脚本而在已反复使用的内部8件上搜索best epoch/阈值，也不能用内部/EGAD结果改变loss再称忠实复现。

## 435条的支持范围

原32-fit合计364条、内部71条，四头×20=80个输出槽中：**17槽没有fit标签、20槽仅1–2条，31槽同时出现过正负两类**。这只是监督覆盖诊断；不能据此证明每槽都必须补齐或它是全部失败的唯一原因。论文“每型100次”说的是一整型总量，不能错误变成“每输出100次”。当前每型数据量同数量级，但采样分布和对应的物理标签语义仍不同。

## 只读发布模型训练计数

另读31官方权重的7层`num_batches_tracked`及保存配置，详独立`OFFICIAL_COUNTERS.json`。计数只直接代表training-mode BN前向次数；原eval不会增加计数。作者保存的是模型而非optimizer state，无法严格从这些文件还原优化步数、重启/额外前向和完整采样历史。

本轮不以训练误差或内部成功率调参，不增加物体，不改变435条/旧模型/既有统计。对齐下一步应先由物理映射和闭合受控对照确认可执行标签语义，再决定是否重采；训练实现本身目前没有证据支持再打补丁。

复现命令（新输出目录）：

```bash
/home/wangrenpeng/boya-inhand-wm/.venv/bin/python -B reproduction/boya/audit_s40_training_alignment.py --output-dir outputs/s40_training_alignment_NEW
/home/wangrenpeng/boya-inhand-wm/.venv/bin/python -B reproduction/boya/audit_official_training_counters.py --output-dir outputs/official_training_counters_NEW
```

轻量证据：`reproduction/boya/evidence/training_alignment_20261108/`。审计没有任何新抓取成功率结论。

31文件的计数范围：Allegro19文件为19–120（仅1个超过100）；DH3四文件为54/130/96/801；Inspire八文件为35/36/37/41/77/126/390/560；全体仅6/31超过Boya的100。各文件7层计数完全一致，均为文件名epoch+1的整数倍。部分DH3/Inspire的8–14倍说明其保存历史不等同于当前first100/batch128的一批每epoch；不能据默认脚本恢复所有发布权重的完整训练历史，也不能断言Boya只是更新次数太少。

新增`evaluate_boya_original_recipe.py`独立可运行入口，固定使用原code100 report及其四头，拒绝preactivation/混合配置，并复用原score-first/domain/全可见点碰撞。仅允许已知A0/A3/A6/E0的缓存帧诊断，显式known-object、无训练写回，不冒称独立新评测。此交付只实现并完成3项合同测试，未运行该入口的几何或物理；后续是否运行及结果由主代理独立协议负责。原pipeline文件不改，进程内替换校验与标记在finally恢复，几何子进程仍复用原worker。

```bash
cd reproduction/boya
/home/wangrenpeng/boya-inhand-wm/.venv/bin/python -B -m unittest -v test_boya_original_recipe.py
# 新入口其他参数与旧external入口相同；其新manifest/protocol schema及已知身份标记见源码。
```
