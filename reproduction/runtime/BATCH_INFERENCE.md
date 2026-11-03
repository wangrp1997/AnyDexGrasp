# Reusing the original DH3 models for several RGB-D frames

`run_official_dh3_batch.py` invokes the existing `run_official_dh3_augmented_frame.py` for each frozen frame, then executes `select_original_dh3_pose.py`. It reuses one actual CGR network and all four released DH3 decision networks in the same process. The wrapper caches checkpoint deserialization and network construction; it does not cache proposals, learned scores, point clouds, collision results, or the final choice.

The RGB-D/calibration input remains the author's original identity plus ten random augmentations, original camera/width/workspace gates, global top1000, all 16 type/depth combinations, top3500, learned score `>.85`, calibrated original hand mapping, top400, original point-cloud collision check and final descending `np.argsort(...)[::-1]` top10/top1. Scene manifests bind file hashes and later world/camera control provenance; object identity, segmentation, CAD and world pose do not enter the network. Collision uses only the identity branch's observed partial point cloud. Required original hand point-cloud slots are generated incrementally in the shared asset root with the existing four-process zero-step FK tool.

The first requests use the original narrowly allowlisted `weights_only=True` checkpoint loader and strict state dictionaries. A later constructor request returns the same module object only for the same architecture/checkpoint. Repeated strict loading can be skipped only when passed the identical cached state-dictionary object, with an evaluation-mode guard. Original algorithm files remain unchanged. The standalone frame runner remains usable without the batch wrapper.

Every frame explicitly resets seed 20261103. The wrapper records the first constructor's CPU RNG states and checks that the constructor consumes no CUDA RNG. On a cache hit it restores that constructor's CPU post-state only if the incoming state matches the recorded pre-state; unmatched states are counted without resetting meaningful forward draws. Saved NPZ states/SHA receipts and transparent `torch.multinomial`/`torch.randperm` call counters expose this behavior. Original forwards still execute normally.

Use the existing interpreter and local native overlays:

```bash
cd /home/wangrenpeng/AnyDexGrasp
source outputs/native_deps/env.sh
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/outputs/native_deps/python_overlay${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/outputs/native_deps/os_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$ANYDEX_PY" -B reproduction/runtime/run_official_dh3_batch.py \
  --scene-manifest <scene-root>/cylinder/scene_manifest.json \
  --scene-manifest <scene-root>/sphere/scene_manifest.json \
  --scene-manifest <scene-root>/long_box/scene_manifest.json \
  --output-dir <fresh-batch-output> \
  --seed 20261103
```

Each frame retains the existing `RESULTS.json`, `frame/branches`, merged proposals, `hand/`, `collision/` and `selection/` layout and SHA links, so the existing scene-binding auditor can independently check its saved arrays. Original gate rejection and infrastructure errors are recorded per frame and do not silently select a lower physical result. The batch continues to retain all requested outcomes. No dynamics or physical success is executed or claimed here.

The batch root records actual checkpoint deserializations, model constructions, first strict assignments, guarded repeated assignments, module identities, cold/warm frame times, and process wall time including imports, new collision assets, every forward, scoring, collision and selection. Interpreter startup is outside the internal timer. Cold/warm timings are observations on different scenes and are not a paired speedup benchmark.

The first actual three-frame run completed on GPU1 with seed 20261103. Frozen input manifests are `outputs/dh3_primitive_generalization_20261103_v1/scenes_v2/{cylinder,sphere,long_box}/scene_manifest.json`; the retained first render failure remains in `scenes/`. Complete inference output is `outputs/dh3_primitive_generalization_20261103_v1/inference/`.

| Frame | Merged original proposals | Learned score >.85 | Before collision | Collision admitted | Original selected type / width / depth | Eleven branches (s) | Whole frame incl. selection (s) |
|---|---:|---:|---:|---:|---|---:|---:|
| Cylinder, radius25mm × height90mm | 251 | 825 | 400 | 251 | 4 / 63mm / 40mm | 17.967 | 41.625 |
| Sphere, radius30mm | 169 | 1,070 | 400 | 103 | 4 / 63mm / 40mm | 16.583 | 36.838 |
| Long box, 100×45×35mm | 640 | 2,961 | 400 | 50 | 4 / 63mm / 20mm | 18.411 | 42.541 |

Actual batch internal wall was **123.051s**, including module imports1.602s, initial checkpoint hashing/cache preparation0.425s, all33 original branch forwards, shared-asset checks, all1,200 candidate collision queries and final selection. All required collision slots already existed in the shared generated asset root; no hand sampling or dynamics was added in this run. The observed warm frames remain36.838s and42.541s, with collision itself16.967s and17.317s; model reuse does not remove the original per-candidate collision work.

The run made exactly **five checkpoint deserializations, five real model constructions and five first strict assignments**, then reused those five module identities. Per-frame actual model loads were `[5,0,0]`, with ten identical-state strict reload skips. All ten constructor CPU RNG transitions matched and were replayed; all five constructors left CUDA RNG unchanged. `torch.multinomial` and `torch.randperm` were each called zero times. Native per-frame counts were11 original FPS,11 cylinder queries and22 grouping operations, confirming that the33 neural branches were actually executed rather than proposal results being reused.

All three saved scene→depth/voxel→original augmentation/merge→hand→real collision packed bits→original top1 SHA/array chains were checked with the existing independent saved-artifact auditor. The batch cache's five stored RNG state pairs were also checked against their SHA receipts. The lightweight record is `reproduction/runtime/evidence/dh3_batch_primitive_inference_20261103.json`; complete states and results remain under ignored outputs. This verification executed no new network forwards, collision queries, renders or physics steps. Dynamic grasp outcomes are documented separately by the simulation/control audit.
