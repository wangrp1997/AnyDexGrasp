# AnyDexGrasp
Dexterous grasp pose detection network built upon [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).

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

