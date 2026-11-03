# AnyDexGrasp 原方法复现

此目录属于独立仓库 `/home/wangrenpeng/AnyDexGrasp`，与CanonDex的抓取改进分开。官方基线固定在 `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`，来源与原论文散列见 [SOURCES.lock.json](SOURCES.lock.json)。作者算法/硬件入口未修改，仓库忽略规则仅补充本机大文件目录。

## 已实际完成

- 原始arXiv论文 `2502.16420v1` 的PDF与TeX源分别保存在 `papers/AnyDexGrasp_2502.16420v1.pdf`、`papers/AnyDexGrasp_2502.16420v1_source.tar`；本机副本已校验SHA256。
- [论文—代码核对](PAPER_CODE_AUDIT.md)：CGR表示、手势/开度标定映射、监督损失、推理筛选与真实试验指标逐项对照。
- [仿真/控制核对](SIMULATION_AND_CONTROL.md)：未发现官方桌面动态抓取仿真，论文实际为UR5+D415+三种真实手；现有PyBullet仅用于手资产准备。
- [独立运行检查](runtime/README.md)：官方原决策类CPU前向与原MSE反传、24种角度/翻转规范化一致性已实跑。输入是确定性合成数据、网络随机初始化、零优化步；只证明这些软件接口可执行。
- `/mnt/public/datasets/AnyDexGrasp/`已经建立。实际官方数据/权重状态以 `metadata/intake_20261103.json` 为准；源码附带三种手的标定表已单列 `source_tables/`，不计作试抓训练数据。

## 当前未完成

官方Google Drive连接超时/DNS异常，尚未取得任何预训练checkpoint或作者真实试抓训练记录。仅用原GraspNet数据不能替代作者新CGR标注。数据获取脚本与失败凭据保留，不能把HTML页面或空目录当有效模型/数据。

完整点云→CGR入口同时缺MinkowskiEngine、仓库PointNet2与KNN编译扩展；本机Blackwell与作者旧Torch/CUDA组合需要单独验证。现有共享Isaac环境未安装/降级任何依赖。因此**尚无发布模型推理准确率、仿真抓取成功率或原论文指标复现结果**。

公开训练默认输出240维、部署默认20维已实际验证不兼容；要先看发布checkpoint再选择对应配置，不能擅自截断输出头。Inspire脚本默认随机手势/深度，原论文学习版必须明示使用learned分支；不要直接启动真实机器人入口来测试网络。

## 接续顺序

1. 恢复官方数据源正常访问，优先下载representation checkpoint、原手decision checkpoint与真实记录，核对来源/大小/文件格式/SHA；再按需要取完整表示训练数据。
2. 检查checkpoint的真实网络尺寸和数据canonicalization，运行原手离线推理和作者固定划分的评分验证。
3. 建立匹配的独立运行依赖或透明记录Blackwell移植；不改共享Isaac/Torch、不用其它模型冒充CGR。
4. 原方法离线结果稳定后，再决定原手仿真执行或Boya显式适配。后者需要新手标定与该手成败数据，不是直接套用Inspire权重。

比较时至少分列首抓、有限重抓后的最终物体成功率、全部拒绝/试抓与耗时。原论文是逐次真实试抓成功率，不能与CanonDex最多8次后的物体成功率直接比较。当前CanonDex暂停交接已push `072adaa`，不因此恢复其剩余259件。
