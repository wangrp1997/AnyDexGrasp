# AnyDexGrasp
Dexterous grasp pose detection network built upon [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).

## Boya提速进展（2026-11-17）

新入口[`author_accelerated_fixed_scene_collection.py`](reproduction/boya/author_accelerated_fixed_scene_collection.py)保留固定场景、原物理和全部检查，编译全源几何计算以减少临时数组。完整同一份保存数据的审核从69.2降到30.6秒；新一轮同图像、同3818步命令的完整提起与审核通过，parent耗时9分49秒，执行器从旧对应310.6降到250.7秒。对应阶段总和约缩短27%，属历史匹配观察，不能当全量采集倍率；旧36分12秒是两轮，不能与新一轮直接相除。

同时纠正此前计时解释：旧“物理推进＋更新”桶包含外层观测。新直接测得物理82.2秒、完整观测96.0秒；仍需改善接触处理与存储，尚未达到作者实机节奏。原几何、250Hz、64迭代、质量摩擦和完整接触保持，原生轨迹与旧对应场景exact。25项检查通过。仅1条排除训练的诊断正例，正式扩采/训练未启动，新映射fit仍2条。运行命令与完整限制见[提速报告](reproduction/boya/AUTHOR_COLLECTION_SPEEDUP_20261117.md)，来源与结果见[轻量证据](reproduction/boya/evidence/author_collection_speedup_20261117_v1/SUMMARY.json)。

## Boya当前入口：固定场景连续采集（2026-11-16）

已实际跑通同一个机器人、桌面、相机和物理世界中的两轮“换物体→稳定→拍照→CGR→抓取→记标签”，拍照到抓取不清场、不重建、不重置物体。1Kit/1建场/1相机，CGR载入1次、前向2次；两轮分别提起成功和提起失败，初始化及完整执行审核均通过。两轮完整耗时36分11.808秒，**尚未证明整体加速**，主要耗时仍在物理执行和审核；这两条为排除训练的流程诊断，正式扩采和训练未启动。此前新映射2fit保持不变。

新入口为[`reproduction/boya/author_fixed_scene_collection.py`](reproduction/boya/author_fixed_scene_collection.py)，支持冻结有限场景列表后在一个物体银行中循环；不依赖调试失败记录。可复制的`freeze`/`run`命令、成本及限制见[固定场景报告](reproduction/boya/AUTHOR_FIXED_SCENE_COLLECTION_20261116.md)。原作者实机入口见下文；Boya硬件驱动和标定仍需另接。

[查看两轮实际RGB对比](outputs/boya_fixed_scene_rounds_20261116_v1/same_world_two_rounds.png) · [首轮实际提起图](outputs/boya_fixed_scene_first_execution_20261116_v1/actual_execution.png) · [轻量结果与来源证据](reproduction/boya/evidence/author_fixed_scene_collection_20261116_v1/SUMMARY.json)。原始大数据保存在public，不进Git。

## Requirements
- [Anaconda](https://www.anaconda.com/) with Python 3.8
- PyTorch 1.13 with CUDA 11.7
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) v0.5

## Installation
1. Follow MinkowskiEngine [instructions](https://github.com/NVIDIA/MinkowskiEngine#anaconda) to install [Anaconda](https://www.anaconda.com/), cudatoolkit, Pytorch and MinkowskiEngine.**Note that you need ``export MAX_JOBS=2;`` before ``pip install ``due to [this issue](https://github.com/NVIDIA/MinkowskiEngine/issues/228)**

2. Install other requirements from Pip.
```bash
    pip install -r requirements.txt
```

3. Install ``knn`` module.
```bash
    cd knn
    python setup.py install
```

4. Install ``pointnet2`` module.
```bash
    cd pointnet2
    python setup.py install
```

5. Install ur toolbox.
```bash
    cd ur_toolbox
    pip install .
    cd python-urx
    pip install .
    pip install -r requirements.txt
```
6. Install Allegro Hand.

   Install Allegro Hand upon [Allegro-Hand-Controller-DIME](https://github.com/NYU-robot-learning/Allegro-Hand-Controller-DIME).

7. Download model weights and data at [GoogleDrive](https://drive.google.com/drive/folders/1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo) and put it under ``logs/`` and extract zip files in ``logs/data/representation_model/graspnet_v1_newformat/`` to ``logs/data/representation_model/graspnet_v1_newformat/``.   
   Download the data from the [Graspnet](https://graspnet.net/datasets.html) web page and extract it to ``logs/data/representation_model/graspnet_v1_newformat/``

## Generating the STL file for dexterous hand.
```bash
    python command_generate_mesh_file.sh
```

## Training
```bash
    sh command_train_representation.sh # Representation model
    sh command_train_decision.sh # Decision model
```


## Collecting data
```bash
    python realsense.py
    sh command_collect_multifinger_grasp_data.sh
```


## Robot grasp
```bash
    python realsense.py
    sh command_robot_multifinger_grasp.sh
```


## License
The code is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) for non-commercial purposes.


## Bibtex
If you find the code useful, please consider citing our paper
```
        @article{fang2025anydexgrasp,
          title={AnyDexGrasp: General Dexterous Grasping for Different Hands with Human-level Learning Efficiency},
          author={Fang, Hao-Shu and Yan, Hengxu and Tang, Zhenyu and Fang, Hongjie and Wang, Chenxi and Lu, Cewu},
          journal={arXiv preprint arXiv:2502.16420},
          year={2025}
        }
```

