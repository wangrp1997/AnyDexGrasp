# Official AnyDexGrasp runtime checks

This directory belongs to the independent AnyDexGrasp checkout at original
revision `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`. Scripts locate this repository
from their own file paths. They require no CanonDex files or modules.
Original author code is unchanged.

Run from this repository with an existing interpreter of your choice:

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$ANYDEX_PY" -B reproduction/runtime/probe_dependencies.py
"$ANYDEX_PY" -B reproduction/runtime/audit_runtime.py
```

The current run uses
`/home/wangrenpeng/boya-inhand-wm/.venv/bin/python -B`; this is an existing shared
interpreter, not an installed reproduction environment. Outputs go to the
ignored `outputs/runtime_audit/` directory. No shared environment is installed or
changed; later local dependency builds are isolated under ignored outputs.
No robot entrypoint is imported, and no simulation/robot grasp executes.

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
training convergence, simulation success or real grasp success. These initial checks did not load official weights; genuine checkpoint results
are recorded below.

## Initial full-representation dependency audit

`probe_dependencies.py` directly imports original representation and custom-op
modules and preserves real exceptions. At intake the shared Python3.12 /
Torch2.11+cu130 lacked MinkowskiEngine, GraspNet API, PointNet2 CUDA ops and KNN.
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

## Official checkpoint / recorded-feature runner

`validate_official_decision.py` is prepared for the official downloads:

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$ANYDEX_PY" -B reproduction/runtime/validate_official_decision.py \
  --checkpoint /path/to/official_model.pth \
  --records /path/to/official/test_single_point.json \
  --hand Allegro --finger-type 9 --split test \
  --output-dir outputs/official_decision_validation/allegro_type9
```

Use the hand and finger type belonging to the selected checkpoint. Omitting
`--records` inspects and strictly loads the checkpoint without fabricating
inputs. The runner first statically inspects pickle globals and only uses
`weights_only=True`, explicitly allowing the original pure-Torch decision
class and required Torch layer classes. Unknown globals stop execution;
there is no unrestricted pickle fallback. The exact checkpoint SHA, all tensor
shapes, head dimensions, author source SHA and runtime versions are saved.
Weights are never truncated or reshaped to bridge the 20/240 discrepancy.

For real records the original dataset `__getitem__` performs feature
normalization. Selected output follows the original loss's depth indexing.
Scores are ReLU regression outputs, not calibrated sigmoid probabilities.
Source labels and IDs are retained with per-record predictions; reported MSE
and threshold statistics concern the supplied feature/record component only.
`--split train` preserves the author's first-100-matching-record cap. A train
file's statistics are replay/fit statistics, never held-out accuracy. The
runner conservatively makes no independent-test claim even for a file named
test until source split and non-overlap with training IDs are verified.

Both full-module checkpoint serialization shapes (20 and 240) were software
checked with **locally synthesized random initialized fixtures**, clearly
labeled under ignored `outputs/runtime_audit/synthetic_serialization_check/`.
Those fixtures are not author weights and provide no method-quality result.


## Genuine released decision models: 31 / 31 validated

After the nros proxy was enabled, all 31 files published under the three
original hands' `obj140/480` folders were downloaded, retaining every model
without choosing a best checkpoint from test outcomes.
`validate_all_official_decisions.py` loaded every official state dictionary
strictly through the weights-only allowlist and evaluated the author's real
test features. Each hand JSON was parsed/canonicalized once.

**All 31 checkpoints reproduced their filename-declared threshold precision,
recall and F1 to four decimal places.** The released weights consistently have
480 input values, 20 outputs and saved `num_two_finger_angle=1`. Therefore the
240-output source training default is a configuration mismatch, not a need to
truncate released weights. No weight or author network source was changed.

Coverage is three hands / 22 hand-type groups / 1,440 unique original hand and
record IDs. Multiple weights on the same hand/type produce 1,800 record-model
evaluations; these must not be counted as 1,800 independent examples. CPU
execution of this component audit took 6.94 seconds, excluding downloading,
prior data remapping and full CGR computation.

The public raw JSON uses older grasp-type numbers. The original author's
`update_data.py:update_grasp_type` was executed on separate derived copies;
only imports of an unused hardware `grasp_types` name were isolated. Public
raw bytes, retained feature vectors and binary results remain unchanged.
For example, published Allegro type9 corresponds to old raw type10, not raw9.
Official preprocessing retains 400/520 Allegro, 240/240 DH3 and 800/1,000
Inspire records; all removals follow the author's supplied mapping.

These released test records may also have been used to select the released
checkpoints. The matching metrics establish component/configuration
reproduction, **not independent generalization or physical grasp success**.
There was no new training, no robot action and no simulated grasp.

Lightweight auditable receipts are in
`evidence/official_decision_models_20261103.json` and
`evidence/first_five_official_checkpoints_20261103.json`. Full per-record
predictions remain ignored under `outputs/official_decision_validation_all_obj140/`.

The complete original representation checkpoint also loaded safely as a
tensor dictionary (202 tensors, saved epoch19); this alone does not execute
its network. `run_official_rgbd_frame.py` prepares calibrated genuine demo
RGB-D, retains the original 5mm integer voxelization, original model forward
and original candidate parser while isolating only unused training imports.
A real input-only preflight and then a full genuine RGB-D forward have both
passed using the separately built native ME/PointNet2 dependencies. Original
FPS/cylinder/group operations execute; no Python numerical replacement is used.


## Full original representation single frame: executed successfully

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
source outputs/native_deps/env.sh
"$ANYDEX_PY" -B reproduction/runtime/run_official_rgbd_frame.py \
  --image-dir /path/to/the/original/graspnet_baseline/demo \
  --device cuda:1 --output-dir outputs/official_rgbd_frame_20261103_v2
"$ANYDEX_PY" -B reproduction/runtime/verify_rgbd_frame.py \
  outputs/official_rgbd_frame_20261103_v2
```

The official GraspNet-team demo's own `meta.mat` calibration and depth scale
produce 822,348 depth-valid pixels and 43,869 5mm voxels. The complete official
representation checkpoint loads all 202 tensors strictly. The original
network executes 2,048 seed points, 300 views, 48 angles and 5 depths; native
FPS executes once, cylinder query once, grouping twice. All saved outputs are
finite: raw CGR is `[1, 2048, 480]`, original parsed candidates `[2048, 15]`.

The first full GPU forward took 1.364 seconds; the timed runner body took
5.184 seconds including input, dependency imports/model load, voxelization
and saved arrays, **excluding Python process startup, downloading and native
compilation**. Peak PyTorch allocated GPU tensor memory was 521,414,656 bytes.
These figures concern this one software inference input, not general latency
or observation-to-lift execution.

Independent artifact verification confirms exact CGR/parser feature-slice
identity, exact seed/candidate XYZ identity, exact independent bilateral-score
reduction, and valid rotations (max orthogonality/determinant error 3.58e-7).
The candidates precede rig-specific camera/workspace/width and collision
filtering; they are not executable or successful grasps by this check alone.
No robot or simulator has run.

The first v1 attempt stopped at the removed Python3.8
`collections.Sequence` alias. v2 restores precisely the standard
`collections.abc.Sequence` alias in the runner process; original AnyDex source
remains untouched. The 3.12/Torch2.11/CUDA13 dependency compatibility build and
its limited ME patches are recorded separately in native dependency receipts;
this is not a claim of the exact author Python3.8/Torch1.13 runtime. Initial
failure and final successful receipts are both retained in `evidence/`.

## Original train / test record-ID audit

`check_official_record_splits.py` streams top-level JSON keys and binds whole
original files by SHA256, without loading several gigabytes of feature values.
Allegro has 3,865 train / 520 test IDs, DH3 4,262 / 240, Inspire 10,449 / 1,000.
All three train/test ID intersections are empty. This establishes record-ID
separation only; it does not establish object-level or feature-level
separation, and released checkpoints may have been selected on this test set.
No independent generalization claim is added.


## Original DH3 single-frame learned hand prediction

The original DH3 branch also ran with its own 0.15-0.72m depth domain and
0.099m parser width scale. Its original camera direction, width and workspace
gates retain 156 of 2,048 candidates; all four published DH3 heads evaluate
2,496 type/depth combinations. The original learned-score `>.85` gate retains
419; original calibrated width/pose mapping plus top400 retains 400 predicted
DH3 poses (types1/2/3/4: 60/103/110/127). No gate was relaxed and `RANDOM=False`.
The exact original `get_DH3_depth_type`, `flip_ggarray`, feature normalization,
`GraspGroup` data API and `DH3GraspGroup.from_graspgroup` are used.

The released `get_DH3_model` resets its model dictionary inside each class
loop; the independent runner deterministically loads all four unique
published heads instead of invoking that defective I/O loader. It changes no
head weights or scoring function. The pinned official API source is fetched
without package installation:

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$ANYDEX_PY" -B reproduction/runtime/prepare_graspnet_api_data.py
source outputs/native_deps/env.sh
CUDA_VISIBLE_DEVICES=1 "$ANYDEX_PY" -B reproduction/runtime/run_official_dh3_frame.py \
  --frame-dir outputs/official_rgbd_dh3_frame_20261103_v1 \
  --output-dir outputs/official_dh3_hand_frame_20261103_v1
"$ANYDEX_PY" -B reproduction/runtime/verify_dh3_frame_mapping.py \
  outputs/official_dh3_hand_frame_20261103_v1
```

`prepare_graspnet_api_data.py` pins commit
`eb57dd2092d8dbe05312a29c3d0c22f3226efbfc`, verifies fixed SHA256 for both
`grasp.py` and its license, and neither installs an API package nor imports
robot/NMS code. `evidence/graspnetAPI_data_source_20261103.json` records URLs
and SHA. On this server the network fetch can use `source
/opt/nros_tools/nros_rc` then `nros-proxy-sg`; existing checked files also
validate with `--offline`.

Independent mapping verification passes all 400 poses:
`T_hand * T_calibration = T_twofinger` maximum error3.33e-16, actuator lookup
error0, original .85 gate/top400 type-and-score order exact. The verifier uses
the exact original per-type width clamp; its first overly strict float32 vs
float64 width equality assertion was corrected (representation difference
only3.58e-9m), without altering a model or predicted pose.

This continuation took0.417s for learned scoring and2.849s inside its separate
runner, excluding Python process startup and previous CGR execution. The
original precomputed 3mm per-width collision PCL was absent, so this stage
explicitly stops at predicted hand poses. No empty collision mask was used;
no robot, simulated grasp or successful lift is claimed. Local generation of
those original collision assets is tracked as a following stage.

## Original collision assets and collision gate: executed

The previously missing hand pointclouds were generated locally from all 13
original DH3 STLs and its original URDF. Their bytes were checked against the
pinned author commit before generation. The published lookup table supplies
its original 12-joint configurations; original `get_meshes(if_source=False)`,
mesh assembly and `save_stl_and_pointcloud` execute without numerical changes.
Every slot retains the original 300,000 surface samples, 15 inward layers,
STL and all five 1-5mm voxel pointcloud scales. Per-slot Open3D random seeds
are saved. Four separate CPU processes each use a private PyBullet `DIRECT`
connection, with `stepSimulation` guarded against use: **zero physics steps**.
This is offline forward kinematics/asset generation, not a simulated grasp.

The original collision function expands two-finger widths by1.7 and remaps
the hand before checking. Consequently this frame needs **28** generated
slots, rather than the107 unique initial hand widths. Generation completed
in13.206s. The initial asset-v1 failed before generation because Open3D lacked
`libusb-1.0.so.0`; its failure is retained. Asset-v2 uses only the existing,
compatible libusb in a local loader directory, without changing system or
shared Python packages. Do not add the entire host library directory.

The local PyBullet overlay build/receipt is described in `../native/`; it
also contains `pybullet_data` and `transforms3d`. On this server the additional
Open3D library can be prepared with:

```bash
mkdir -p outputs/native_deps/os_libs
install -m 644 /opt/nros-host/usr/lib/x86_64-linux-gnu/libusb-1.0.so.0.5.0 \
  outputs/native_deps/os_libs/libusb-1.0.so.0
export LD_LIBRARY_PATH="$PWD/outputs/native_deps/os_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

This specific source library requires at most GLIBC2.38, below this server's
GLIBC2.39. Its SHA and origin are saved in
`evidence/open3d_libusb_source_20261103.json`. Another machine should supply
its own ABI-compatible libusb, rather than assume this host-specific path.

To generate the required original slots and run the unchanged original gate:

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
export LD_LIBRARY_PATH="$PWD/outputs/native_deps/os_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$ANYDEX_PY" -B reproduction/runtime/generate_dh3_collision_assets.py \
  --hand-run-dir outputs/official_dh3_hand_frame_20261103_v1 \
  --workers 4 --output-dir outputs/official_dh3_collision_assets_20261103_v2
"$ANYDEX_PY" -B reproduction/runtime/run_official_dh3_collision.py \
  --hand-run-dir outputs/official_dh3_hand_frame_20261103_v1 \
  --asset-root outputs/official_dh3_collision_assets_20261103_v2/assets \
  --output-dir outputs/official_dh3_collision_frame_20261103_v1
"$ANYDEX_PY" -B reproduction/runtime/verify_dh3_collision_artifacts.py \
  outputs/official_dh3_collision_frame_20261103_v1
```

Use fresh output directories to preserve earlier receipts. For additional
candidates, `--asset-root` on the generator reuses only existing SHA-checked
slots and generates missing ones. The collision runner also accepts explicit
`--scene-points path.npy`; its default reads the previous frame's original
identity-branch quantized input, not merged augmented points.

The real original detector checked all400 poses against43,869 scene points:
**341 rejected /59 admitted**, with no threshold relaxation or empty-cloud
substitution. Original parameters remain 3mm hand/approach voxels, 5cm
approach, width>.045m after expansion, zero scene-point overlap threshold,
no center adjustment. Original cumulative approach-copy behavior (offsets
0/2/5/7cm) is preserved, not silently corrected. The detector source omits a
`time` import; its AST scope supplies the real timing module. Open3D queries
are instrumented by a pass-through wrapper that returns their unchanged
results while recording inclusion bits for all400 candidates.

Original collision computation took15.339s, this continuation's timed body
16.157s. Independent verification reduces every saved real inclusion bit,
checks every decision/admitted pose and verifies every generated STL and
five-scale pointcloud SHA. It reproduces the59/341 split exactly. Lightweight
receipts are `evidence/dh3_collision_*_20261103.json`; full assets, candidate
records and packed masks remain ignored under `outputs/`.

Collision admission here is the author's visibility-based scene/hand
pointcloud test. It establishes neither mechanical-arm reachability nor
lifting success. No arm, simulator grasp, real robot, new training or
independent-scene evaluation has run.

## Complete identity +10 original augmentations

`run_official_dh3_augmented_frame.py` provides the original11-branch option.
It executes original `get_ggarray_features`, `augment_data`, candidate
recovery/flip/filter/merge and parses the real demo using its own calibration.
Only the original five rig-specific camera constants are replaced by that
frame's intrinsics/depth scale. The original random augmentation matrices,
seed and branch costs are recorded. The model is reused in one process.

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
source outputs/native_deps/env.sh
export PYTHONPATH="$PWD/outputs/native_deps/python_overlay${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/outputs/native_deps/os_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES=1 "$ANYDEX_PY" -B \
  reproduction/runtime/run_official_dh3_augmented_frame.py \
  --image-dir /mnt/public/datasets/AnyDexGrasp/examples/graspnet_baseline/280c215129f759ed8649cb4e89fc5dfee55f4f80 \
  --seed 20261103 \
  --asset-root outputs/official_dh3_collision_assets_20261103_v2/assets \
  --output-dir outputs/official_dh3_augmented_frame_20261103_v1
```

The optional `--asset-root` continues through all four original learned DH3
heads, the original learned-score gate and top400, increments original hand
assets as required, and runs the original collision gate. Each stage writes
separate receipts; `collision_check_executed` becomes true only after an
actual successful gate execution. A prepared runner alone is not evidence
that a new output directory was executed; inspect its final `RESULTS.json`.
Original identity results above remain separate and unchanged.

The complete11-branch option **has now actually finished**, at
`outputs/official_dh3_eleven_branch_20261103_v1/`. It loaded the representation
model once and executed11 genuine FPS calls,11 cylinder queries and22 grouping
calls. Per-branch retained candidates were
156/116/130/93/140/156/151/186/137/169/110, concatenating exactly to1,544.
Original global top1000 produced16,000 hand-type/depth combinations; original
top3500 and score>.85 retained2,952, then original top400 entered collision.
All six required remapped collision assets reused the already verified28-slot
cache. Real original collision admitted9 and rejected391. Eleven forward/
recovery branches took15.988s; the complete runner's timed body39.549s,
excluding process startup, downloads and prior dependency/asset generation.
It remains one calibrated GraspNet demo, not a paper physical success rate.

The independent verifier checks all ten seeded transforms exactly, original
flip matrix mutations, every recovered output's original gates/rotations,
exact11-way concatenation and parser-column handoff, and independent hand
flip/top1000/CGR half-swap/angle-roll normalization. Identity scene points
match the earlier standalone identity frame exactly. All400 calibrated hand
poses pass independent lookup checks, and real saved collision bits reproduce
the9/391 result. **Pre-recovery raw2,048 parser arrays were not persisted per
branch**: the verifier therefore does not claim independent numerical replay
of that boundary. Recovery uses the unchanged, SHA-bound author function;
its saved recovered outputs, gates and merge are checked.

The author's last data selection is a separate no-model/no-collision step:

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$ANYDEX_PY" -B reproduction/runtime/select_original_dh3_pose.py \
  --collision-dir outputs/official_dh3_eleven_branch_20261103_v1/collision \
  --output-dir outputs/official_dh3_eleven_branch_20261103_v1/selection
"$ANYDEX_PY" -B reproduction/runtime/verify_dh3_frame_mapping.py \
  outputs/official_dh3_eleven_branch_20261103_v1/hand
"$ANYDEX_PY" -B reproduction/runtime/verify_dh3_collision_artifacts.py \
  outputs/official_dh3_eleven_branch_20261103_v1/collision
"$ANYDEX_PY" -B reproduction/runtime/verify_dh3_augmented_artifacts.py \
  outputs/official_dh3_eleven_branch_20261103_v1 \
  --identity-frame outputs/official_rgbd_dh3_frame_20261103_v1
```

`select_original_dh3_pose.py` executes the exact eight original assignments
at `robot_dh3.py:711-719`: collision-admitted scores argsort descending,
top10 and index0. Equal-score behavior uses the original NumPy expression.
Both identity and11-branch runs have completed this step. Final selected
hand pose, two-finger pose and feature vector are byte-value identical:
DH3 type2, score1.55893695, width90mm, stored depth30mm,
actuator2 `[94, 67]`. This is a final **predicted camera-frame hand datum**;
the separate depth convention remains as in the author API. No calibrated
robot wrist execution or grasp/lift happened. Actual pipeline, final selection
and independent receipts are archived as `evidence/dh3_eleven_branch_*` and
`evidence/dh3_identity_final_selection_20261103.json`.
