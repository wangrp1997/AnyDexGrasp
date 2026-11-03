# 原DH3几何生成器的局部Python overlay

2026-11-03实际完成：官方PyPI transforms3d0.4.2纯Python wheel；官方PyBullet3.2.7 sdist局部构建cp312/Linux wheel，包含pybullet_data。没有新Python环境、pip安装、共享包修改或env.sh改写。

官方PyPI3.2.7无cp312 wheel，实际发布的CPython wheel最高3.11；因此没有混用不匹配ABI，而是在root明确授权下局部源码构建。精确来源URL/大小/官方SHA见`pybullet_components.json`。源码80,508,379字节，SHA `042879db8d101ac7590dee475fc6aded508b85fe1273fdbbfde1d88bd200e14f`。

PyBullet setup自带线程池使用`2*cpu_count`，不读取MAX_JOBS。实际唯一源码差异为`setup.py`这一行包装改成读取MAX_JOBS并硬上限4；见`pybullet_build_workers.patch`和evidence中的前后SHA。不改变Bullet算法/C++代码。

实际构建命令（仅独立工作目录）：

```bash
export MAX_JOBS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TMPDIR=/home/wangrenpeng/AnyDexGrasp/outputs/native_deps/pybullet_tmp
export CC=/usr/bin/gcc CXX=/usr/bin/g++
# cwd: outputs/native_deps/build_src/pybullet-3.2.7
"$ANYDEX_PY" -B setup.py build \
  --build-base /home/wangrenpeng/AnyDexGrasp/outputs/native_deps/pybullet_build \
  bdist_wheel --dist-dir /home/wangrenpeng/AnyDexGrasp/outputs/native_deps/wheels \
  --bdist-dir /home/wangrenpeng/AnyDexGrasp/outputs/native_deps/pybullet_bdist
```

实际本地wheel为`pybullet-3.2.7-cp312-cp312-linux_x86_64.whl`，102,050,491字节，SHA `3a2d8f53e3659a0957304427ac7f921017199189e0e821292bc971d88a048839`；该SHA记录本次构建身份，未来重新编译的时间戳不同可能产生不同wheelSHA，不能把它当官方发布SHA。

后续机器可用已有兼容Python执行本目录`prepare_pybullet_overlay.py`，它复用相同固定包/唯一包装补丁并安全解包、拒绝路径穿越/符号链接/文件冲突。脚本不调用pip，也不会触碰已有ME/PointNet2。已有匹配wheel则复用，不重复编译。迁移时只做Python语法和补丁dry-run，未再次构建。

```bash
export ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$ANYDEX_PY" -B reproduction/native/prepare_pybullet_overlay.py
# 只在原DH3生成器进程额外前置此路径；不修改env.sh：
export PYTHONPATH="$PWD/outputs/native_deps/python_overlay${PYTHONPATH:+:$PYTHONPATH}"
```

实际验收：pybullet、pybullet_data、transforms3d均import成功；DIRECT connect返回0、空场景body数0、disconnect成功；**stepSimulation调用0、物理步0**。仅证明原生成器所需依赖可用，不是动态抓取仿真结果。完整轻量凭据在`evidence/pybullet_overlay_receipt.json`、`evidence/python_overlay_smoke.json`；完整编译日志/源码/wheel留outputs，不提交Git。

下载/构建成本逐项见`evidence/pybullet_costs.json`：总下载81,885,138B；1次构建、267对象、实际包装池上限4；日志文件区间约85.897s，非独立精确端到端计时。
