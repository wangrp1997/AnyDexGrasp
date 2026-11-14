# Author transport backend: component delivery, not a physical result

`author_transport.py` implements an opt-in native UR16e+Boya multi-object runtime and a continuous transport trajectory. The runtime has **not yet run a Kit scene or any physical step**. Its first physical use belongs to the root agent's separately frozen scene/collection protocol. This component stage generated zero labels and zero model updates.

The public `ur_toolbox/ur_toolbox/robot/ur.py::grasp_and_throw` closes, retreats 4 mm along the approach direction, waits 0.1 s, retreats to pregrasp, moves to its throw pose and opens. We retain that order and replace the real UR throw coordinates with an explicit simulator bin frame. Boya motor targets, native drive settings, global-zero-gravity plus each dynamic body's 0.98 N downward weight, 0.1 kg mass and 0.6 friction remain declared simulation assumptions; this is not hardware equivalence.

## Interfaces

```python
prepare_transport(seed_plan_dir, scene_path, new_plan_dir)  # zero physics
args = args_for_plan(new_plan_dir, new_execution_dir)
run(args, scene_path)                                    # Kit already running
```

`execution_steps(args, scene_path)` exposes the same `ready`/`step` contract as the original native executor: the caller integrates exactly once for each `step`. The wrapper completes the original target update/audit before updating additional bodies and reading all objects. The original generator may write the next command before the wrapper reads the current state; no next integration occurs before the next yielded event. Every dynamic body receives persistent gravity-equivalent force, writes and updates every tick. Initialization explicitly cold-recreates the measured scene at zero velocity, before `ready`; there are no physical state resets during execution.

The scene schema is `anydex_boya_author_scene_v1`. Objects carry source mesh SHA, source scale/pose, collision repair SHA, mass and friction. `target_object_id` selects the old backend's compatibility primary only. It never segments or filters the grasp candidate and never determines which body can earn success. `bin.world_T_bin` locates the interior centre, with `inner_half_extents_m` defining ±XYZ; five physical floor/wall boxes leave the top open. The root protocol fixes bin centre `[-0.40,-0.25,0.10]`, half extents `[0.15,0.15,0.10]` and 0.01 m wall/floor thickness. Candidate identity binds through `plan.author_collection_binding.collection_candidate_sha256` and is echoed with the actual command SHA.

The source native executor is reused unchanged. Its original seven gates, full 28-joint structure, FK, material/actuation and 20 µm native mesh checks remain active. Additional bodies receive their own native mesh/material/mass checks. All object × robot/object/floor/bin contact pairs are recorded, with contact buffer validation refusing truncation/nonfinite data. The original no-arm-support predicate is extended to every object using the same legitimate hand rigid owners and 1 mN threshold; bin support of the robot is independently rejected. Source meshes remain measurement/physical assets, not inference-time target annotations. Collision pieces remain approximate source proxies.

## Label and failure handling

`TRANSPORT_EVALUATION.json` is the transport result; the original `evaluation.json` retains its old lift-task meaning and must not be read as a transport success. The new trajectory has no old world-Z `lift` phase, so an old lift success is not expected.

Success requires an originally tabletop body to pass all of:

1. Complete approach, close, hold, 4 mm retreat, wait, pregrasp retreat, transport, release and post-release stages in order.
2. During retreat/transport, the source mesh bottom rises at least 5 cm while thumb and opposing finger loads simultaneously exceed 1 mN; the final transport sample still satisfies those conditions. An unheld object thrown by approach or arm contact cannot qualify.
3. After opening, every source vertex remains inside the bin's interior, allowing 2 mm contact tolerance, for the final 0.5 s. The body must have unloaded from the robot and be below 0.02 m/s linear and 0.5 rad/s angular speed throughout that window.
4. All original and additional physical validity gates pass, including all-object no-arm-support and no robot-bin support.

Every object's metrics, five finger loads, illegal support, source bottom, native pose/velocity and contacts are retained. Any qualifying object can deliver success. A complete valid trial that drops the object is an eligible negative; a null plan, incomplete execution or infrastructure failure does not silently become a negative label. Candidate SHA is required for label eligibility. All attempts still remain in operational result/coverage accounting.

## Verification and limitations

Eight independent tests with 17 subtests pass. Counterexamples include approach-only ejection into the bin, missing closure, one-sided load, forearm support of a non-primary object, loss before release, incomplete final window, full contact buffers and malformed native buffers. A zero-step native initialization failure preserves its original error rather than failing again while writing the receipt.

One reused real admitted S40-021 plan exercised actual six-arm IK and trajectory generation, with no Kit or integration. The source approach/close/hold prefix is retained; the resulting 3,750 instructions cover 24 independent and 28 actual coordinates. All 96 transport IK waypoints converged, and the largest interpolation position error was `2.2177475773156995e-7 m`. The fixed bin nevertheless intersects the upper-arm path, so the result is **null_geometry**, and that plan must not execute. Its open-hand path passed; the post-approach rigid audit recorded 1,937 violations. We did not move the bin or try to convert this null into success.

Development attempts remain under `outputs/boya_author_transport_20261114_v1/`: initial `planning_smoke` failed after zero physics because the original hand-scene binding occurred after import; `planning_smoke_v2` fixes that ordering and returns the geometric null; `planning_smoke_v3` additionally preserves its exact source snapshot and repeats that same zero-physics result. The v3 snapshot predates the subsequent contact/support/label audit corrections; its planner numerical code is unchanged, and its frozen snapshot/SHA must be used for that historical run. Final unit tests cover the corrected current source. No result is a physical success rate or a completed end-to-end collection/training pipeline.

Light evidence is in `evidence/author_transport_20261114/`; generated plans, snapshots and large traces remain ignored outputs. CanonDex and original AnyDex source files were not modified.

## Unified entry

`author_pipeline.py prepare` freezes scene/observation/raw representation/calibration/surface/implementation SHA before running the real source collector. It connects exactly its index-0 choice to the bridge, native arm prepare, transport prepare, numerical command contract and a fresh SHA-bound execution plan. The declared world workspace is `x[-1,-.57], y[-.04,.24], z[-.02,.30]`, approach world `[0,0,-1]` with strict cosine `>.9`, raw width `(0.01,0.099)` m and explicit RGB-D depth `[0.2,0.65]` m. These are simulator camera/rig substitutions, without a hand/table/learned gate before top500. The legacy primary's native collision repair and physics profile must match the captured scene. No alternate candidate, bin, training or unseen-object evaluation is dispatched by the entry.

```bash
python -B reproduction/boya/author_pipeline.py prepare \
  --scene OBSERVED_SCENE --representation-dir RAW_REPRESENTATION_DIR \
  --observation OBSERVATION_JSON --gesture-id 6 --source-type 2 --seed 20261114 \
  --calibration CALIBRATION_JSON --cloud-manifest SURFACE_MANIFEST \
  --backend BACKEND_JSON --output-dir FRESH_OUTPUT
```

The numbers `6/2` above illustrate explicit arguments; the root frozen experiment chooses the actual Boya/source-type pairing. Preparation preserves `no_candidate`, `no_plan`, `null_geometry` and infrastructure outcomes and produces zero labels. Only a `ready` result can run using `execute --prepared-dir FRESH_OUTPUT` or explicit `prepare --execute`. Execution verifies every bound file and writes an exclusive `EXECUTION_STARTED.json` before launching Kit. An infrastructure start failure consumes that launch and cannot trigger an implicit retry. The new command file retains all original native arrays and 24 coordinate columns exactly; only candidate/execution-spec metadata is added. Four mimic columns are derived solely for command validation and never replace the native actuator format.

Seven further entry tests with four subtests pass, for **15 tests / 21 subtests total**. They verify no fallback from index0, each null/error boundary, command-byte metadata versus numerical-array preservation, tampering before launch, no retry after a failed start and task-bound negative versus an incompatible old-lift outcome. Native launch functions are stubbed only in those entry tests; they are not evidence of a physical run. Root-owned actual scene/representation/preparation runs and any subsequent physical execution require their own immutable receipts.
