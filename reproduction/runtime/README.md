# Official AnyDexGrasp runtime checks

This directory belongs to the independent AnyDexGrasp checkout at original
revision `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`. Scripts locate this repository
from their own file paths. They require no CanonDex files or modules.
Original author code is unchanged.

Run from this repository with an existing interpreter of your choice:

```bash
python -B reproduction/runtime/probe_dependencies.py
python -B reproduction/runtime/audit_runtime.py
```

The current run uses
`/home/wangrenpeng/boya-inhand-wm/.venv/bin/python -B`; this is an existing shared
interpreter, not an installed reproduction environment. Outputs go to the
ignored `outputs/runtime_audit/` directory. No environment is installed or
changed, no robot entrypoint is imported, and no simulation/robot executes.

## Checks actually performed

`audit_runtime.py` imports the unmodified pure-PyTorch decision network and
AST-selects original loss and feature functions only to avoid unrelated
MinkowskiEngine/KNN imports or hardware entrypoint side effects. Source SHA256,
Python/Torch versions and model constructor dimensions are written with results.

* Original training network: forward/backward using eight deterministic
  **synthetic** 480-value feature vectors; original selected-depth MSE loss,
  all gradients finite, zero optimizer steps.
* Original inference decision class: forward returns `[8, 1, 20]`.
* Released training default has 240 final channels, while inference default
  has 20. Direct original `load_state_dict` fails on conv7/bn7 as recorded.
  Check real released checkpoint shapes before choosing training settings;
  no weight truncation or source fix is performed here.
* Training dataset and robot canonicalization match exactly on all 12 angle
  types x both 180-degree flips, under explicit mapping
  `robot_angle_bin = 12 + 2 * training_angle_type`.

These are software interface checks with random initialization and synthetic
features. They are **not** pretrained-model accuracy, paper reproduction,
training convergence, simulation success or real grasp success. At this point
no official checkpoint or official real decision record has been obtained.

## Original full representation path is currently blocked

`probe_dependencies.py` directly imports original representation and custom-op
modules and preserves real exceptions. Current shared Python3.12 /
Torch2.11+cu130 lacks MinkowskiEngine, GraspNet API, PointNet2 CUDA ops and KNN.
Bounded known-environment, binary and cache searches are recorded explicitly.
No nvcc, standard CUDA development toolkit or Python3.8/Conda runtime was found
at the searched locations. These are bounded probes, not a filesystem-wide
claim. Original `pointnet2/_ext_src/src/cylinder_query.cpp` explicitly rejects
CPU execution, so CPU-only ME does not make full CGR inference work.

The host GPU probe shows two RTX PRO6000 Blackwell GPUs (compute capability
12.0). Author Python3.8/Torch1.13/CUDA11.7/ME0.5 is not their native support
stack. [PyTorch2.7 introduced Blackwell/CUDA12.8 support](https://pytorch.org/blog/pytorch-2-7/).
[NVIDIA permits older CUDA binaries with usable PTX to run through forward JIT](https://docs.nvidia.com/cuda/pdf/Blackwell_Compatibility_Guide.pdf),
so legacy execution is not categorically impossible; each dependency must
actually be validated. No such full legacy runtime is available here yet.

A strict author-stack run therefore needs an isolated compatible environment
and supported GPU (or demonstrated complete PTX compatibility). A modern
Blackwell port would need ME/custom-op builds and equivalence validation and
must be identified as a port. Neither was installed in this audit.

The smallest runnable component after official downloads is the pretrained
pure-Torch decision model on supplied real 480-value decision features.
Preserve checkpoint SHA, match exact architecture and preserve official splits.
This component alone does not reproduce point-cloud-to-CGR or physical grasps.

Do not import the robot scripts for offline tests: they import hardware tools;
`robot_inspire.py` also releases with `RANDOM_GRASP=True`, bypassing the learned
selector unless explicitly changed in an isolated runner.
