# 已验收的 CUDA13 / Blackwell 原生构建配方

这是2026-11-03实际成功步骤的轻量归档。作者网络/权重/pointnet2算法保持；只有ME依赖的3文件包装/API兼容补丁。原执行产物/完整编译日志留 `outputs/native_deps/`，此目录不收源码树、二进制、SDK或大日志。

## 已验收范围

- 原机器：Python3.12.14，Torch2.11.0+cu130，NumPy2.3.1，GCC13.3，RTX PRO6000 Blackwell sm120。
- 原Python入口：`/home/wangrenpeng/boya-inhand-wm/.venv/bin/python`，解析到已有tendon-spin运行环境；没有创建环境或安装/升级共享包。
- ME0.5.4编译21对象；原AnyDex pointnet2编译11对象。`MAX_JOBS=4`、`TORCH_CUDA_ARCH_LIST=12.0`。
- GPU验证：ME3×3卷积按坐标对齐CPU输出，最大绝对误差0；原FPS、CylinderQuery、group_points通过。结果原样复制在 `evidence/BUILD_ACCEPTANCE.json` 和 `evidence/native_validation.json`。
- 本目录迁移时未重新下载、编译或跑GPU；新路径包装只做语法/补丁dry-run检查。完整点云CGR验收在独立runtime报告，不能由这些小算子测试单独推出论文抓取效果。

## 固定依赖与下载范围

精确URL、字节数和SHA见 `components.json`；NVIDIA清单裁剪为 `nvidia_components_13.0.0.json`。

| NVIDIA官方组件 | 版本 | 下载字节 |
|---|---|---:|
| cuda_nvcc |13.0.48|26,732,376|
| cuda_cudart |13.0.48|1,487,372|
| cuda_cccl |13.0.50|987,548|
| cuda_crt |13.0.48|79,960|
| libnvvm（含cicc/libdevice）|13.0.48|43,941,232|

合计73,228,488字节。大cuBLAS/cuSPARSE/NVRTC包没有下载；继续只读复用当前Torch安装的 `site-packages/nvidia/cu13/{include,lib}`。若未来机器没有兼容CUDA13的现有Torch运行库，脚本会停止；不自动安装或修改共享运行栈。

当前容器不能加载host挂载的OpenBLAS0.3.32（要求GLIBC2.43）。实际改为Ubuntu官方OpenBLAS0.3.26两个`.deb`局部解包，合计12,075,504字节；不执行apt或dpkg安装。复用已有NumPy/SciPy wheel的Fortran/Quadmath库，通过局部链接提供所需soname。

ME作者源码固定：`chenxi-wang/MinkowskiEngine` commit `39edad65022ec3e56c994a743c0a87877c70933a`；源码归档5,664,881字节，SHA在清单。上游API修复参考 `kevintsq/MinkowskiEngine` PR637 commit `4f968c622b95cc1408c05b3a0cd2a26852d90577`。

## 复建

需要已有Python/Torch CUDA13、GCC/G++、`patch`、`dpkg-deb`、`readelf`以及Python requests/ninja/setuptools。以下步骤只向独立仓库 `outputs/native_deps/` 写入，不会安装到site-packages。

本机网络按需启用代理：

```bash
source /opt/nros_tools/nros_rc
nros-proxy-sg
```

在独立AnyDexGrasp仓库执行：

```bash
export ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
bash reproduction/native/build_native.sh
```

未来机器修改 `ANYDEX_PY` 为已有兼容Python入口即可；仓库、现有Torch/lib、CUDA wheel路径由脚本解析。GPU目标仍明确是sm120，不自动为不同GPU换算法/精度。该脚本顺序执行：

1. `prepare_sources.py`：校验并下载固定作者源码/Ubuntu小包，安全解包，创建局部OpenBLAS链接；应用本目录补丁；逐文件检查原pointnet2身份并复制工作目录。
2. `acquire_cuda.py`：按NVIDIA固定大小/SHA获取5个开发包，组装局部toolkit，复用当前cu13头和库。
3. `write_env.py`：生成仅当前复现进程使用的 `outputs/native_deps/env.sh`。
4. 下列原生构建命令与实际成功运行一致（源码准备已去除setup副作用）：

```bash
source outputs/native_deps/env.sh
cd outputs/native_deps/build_src/MinkowskiEngine
"$ANYDEX_PY" -B setup.py build_ext --inplace --force_cuda \
  --blas=openblas --blas_include_dirs="$ANYDEX_NATIVE/openblas/include" \
  --blas_library_dirs="$ANYDEX_NATIVE/openblas/lib"
cd ../pointnet2
"$ANYDEX_PY" -B setup.py build_ext \
  --build-lib "$ANYDEX_NATIVE/pointnet2/lib" \
  --build-temp "$ANYDEX_NATIVE/pointnet2/temp"
```

5. `validate_native_ops.py`：实际GPU/CPU sparse-conv比较及原CUDA采样/分组数值检查；它写新的 `outputs/native_deps/native_validation.json`。固定历史结果在本目录evidence中不覆盖。构建日志写带时间的新目录，不覆盖原首轮失败。

## 三文件ME补丁与原始错误

`me_local_compat.patch`直接针对固定作者源应用；每个文件前后SHA见 `evidence/me_local_patch_receipt.json`。

- `setup.py`移除启动时自动 `pip uninstall MinkowskiEngine -y` 和清空build缓存。不能未打此包装补丁就在共享环境运行setup，连build_ext也会触发卸载。
- `src/coordinate_map_gpu.cuh`局部将unique_ptr转换为shared_ptr，保留原deleter，避免CUDA13/GCC13 ADL歧义；没有改系统C++头。
- `src/3rdparty/cudf/detail/nvtx/ranges.hpp`精确采用PR637的profiling-only兼容处理，模型数值逻辑不变。

第一次真实编译还报 `cudaEmulationSpecialValuesSupport/MantissaControl` 未定义：初始13.0.48 redist头比复用cuBLAS声明旧。实际只把局部 `include/library_types.h` 链接指向同一现有cu13运行库的匹配原头；没有修改任何头内容。该纠正在CUDA准备脚本结尾保留。nvcc启动器复制到局部toolkit实际路径，以便相对定位已合并的nvvm/cicc/libdevice。

首次失败、host库不兼容、一次审批服务断连未执行及对应修正见 `evidence/initial_failures.json`。完整大日志保留原outputs，不提交。任何后续构建失败都应保留新日志再按实际错误修必要兼容，不能用CPU或随机采样替换原CUDA推理路径后称原方法复现。

## 原DH3生成器附加依赖

`PYBULLET_OVERLAY.md` 与 `prepare_pybullet_overlay.py` 归档官方PyPI3.2.7 sdist→当前cp312本地wheel，再与transforms3d0.4.2安全解包到独立`python_overlay`的实际路线。唯一包装补丁把PyBullet自带`2*cpu_count`线程池改为硬上限4，不改Bullet算法；不调用pip安装、不改共享包或env.sh。

本次附加下载81,885,138字节（PyBullet源码80,508,379 + transforms3d wheel1,376,759）；源码下载/解包实际7.843s，1轮构建/267对象/4 worker，编译日志文件时间区间约85.897s（未独立记录精确进程总墙钟）。本地wheel102,050,491字节，留outputs不提交。没有额外重编ME/PointNet2。成本和局限见`evidence/pybullet_costs.json`。

实际仅import、PyBullet DIRECT connect/空场景查询/disconnect，**零step**；该依赖供原DH3手模型FK/几何生成使用，不是已完成真实动力学抓取验证。4019个解包文件只保留在outputs，Git仅小清单、补丁、SHA与验收摘要。
