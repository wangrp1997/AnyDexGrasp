# AnyDexGrasp 独立复现工作区

更新：2026-11-03。用户要求作为强基线仔细复现，先按照作者原方法/原手/原数据验证，再谈Boya适配；不能把CanonDex修复继续混入此仓库。官方上游https://github.com/graspnet/AnyDexGrasp.git，基线commit `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`，本地分支 `research/native-baseline-reproduction`。上游remote为 `upstream`，没有用户自己的发布remote，不向作者仓库推送。

原算法源码保持不变。复现说明与独立工具在 `reproduction/`，大文件/原始输出放忽略目录 `outputs/`、`papers/`；官方数据/权重专用根目录 `/mnt/public/datasets/AnyDexGrasp/`。不要把源码附带手开度表当作训练数据已下载。先读 `reproduction/README.md`、`SOURCES.lock.json`、`PAPER_CODE_AUDIT.md` 和数据状态。

当前已下载原论文PDF/TeX与官方源码，完成纯PyTorch接口/梯度/规范化检查；合成输入和随机初始化不是预训练推理、更不是抓取成功。Drive权重/训练数据获取未完成；完整CGR缺ME、PointNet2/KNN扩展，当前已有栈是Python3.12/Torch2.11/CUDA13/Blackwell，不能直接假设作者Python3.8/Torch1.13/CUDA11.7栈可运行。不降级共享Isaac环境。优先审计官方checkpoint尺寸再决定独立适配，不静默切片20/240输出头。

官方没有可复用的桌面动态抓取仿真基准，PyBullet工具用于手模型FK/网格/点云生成。禁止把Open3D展示或接口前向称为仿真抓取复现，不运行连接真实UR机械臂的robot/collect脚本。先做无硬件的发布模型离线验证；将来仿真适配须独立声明，不替代原论文的真实试抓指标。

CanonDex当前抓取失败/259件未测保持暂停，交接已push `072adaa`，原工程修复已push `f29f760`。不在此任务中恢复旧批次。只有相关轻量代码/来源锁/文档可提交；数据、权重、视频、原论文及大输出不提交。用户已授权多智能体并行，避免并发修改同一源码。
