# Bound Boya collection queue

`execute_collection_queue.py` invokes the unchanged, read-only CanonDex native
full-arm executor for **new, already-admitted AnyDex/Boya plans**. It adds no
candidate generation, IK, hand policy, controller, asset repair or physical gate.
The existing frozen pilot and original source artifacts remain unchanged.

The prepared manifest from `prepare_collection_queue.py` is accepted directly:

```json
{
  "schema": "anydex_boya_collection_queue_v1",
  "complete": true,
  "trials": [{
    "trial_id": "new_unique_trial",
    "object_id": "object_root_basename",
    "gesture_id": 6,
    "extra_depth_index": 1,
    "plan_dir": "/absolute/admitted/plan_000",
    "binding_json": "/absolute/trial_binding.json",
    "binding_sha256": "sha256_of_binding_json",
    "commands_sha256": "sha256_of_bound_commands_npz",
    "trial_binding_sha256": "canonical_binding_digest",
    "input_sha256": {}
  }],
  "nulls": [],
  "errors": []
}
```

`output_identity` is optional and defaults to `trial_id`. Each trial ID, output
identity, source plan path and canonical binding must be unique within a queue.
Source hashes, binding, original commands, source CGR, assets, model files and
preparation inputs are checked before Kit and frozen in `PREPARED.json`.
Preparation nulls are retained separately, never synthesized into negative labels.

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$ANYDEX_PY" -B reproduction/boya/execute_collection_queue.py \
  --manifest /absolute/queue_manifest.json --output-dir /absolute/new_run \
  --device cpu --mode serial --batch-size 2 --prepare-only
```

Omit `--prepare-only` to execute. A prepared output may be dispatched once; any
existing `RESULTS.json` prevents replay. A file lock rejects concurrent dispatch.
No automatic resume or reset-and-retry branch is implemented.

- `--device cpu --mode serial`: native resident scene reused by successive plans
  **within a homogeneous wave**, with original pre-trial state reset and full
  native audit. This is independent trials, not continuous-scene regrasp.
- `--device cuda:0 --mode parallel`: the original clone-before-first-reset
  `ParallelRuntime`, isolated translated environments, one shared physical clock.
- `--batch-size` is 1..16. Physical asset identity and arm refinement identity
  determine grouping; no grouping by object class, label or observed success.
- Both modes start a new Kit process for the next wave. Cross-wave residency is
  **not implemented**, even for repeated assets. No linear speedup is assumed.

The only local runtime method adaptation converts cross-environment contact
counts explicitly to host if they are Torch CUDA tensors; Warp's own `.numpy()`
conversion remains supported. It retains integer/count/capacity checks and
rejects any cross-environment contact. The original control/contact/validity
checks remain inside the unchanged backend generator.

GPU evidence requires the native USD `enableGPUDynamics` value **and** each
articulation's actual reported device. A CUDA launcher flag alone is insufficient.
`RESULTS.json` retains actual enabled state, shared ticks, environment steps,
original resident/parallel receipts and measured process wall time. Per-lane
traces use the original plan frame; each physical lane origin is recorded.

Non-null labels require exactly **2100 actual recorded steps**, lift phase,
unchanged original command SHA, exact recorded independent-joint target/phase/
desired-palm arrays, matching backend, unchanged frozen inputs/sources and a clean
worker exit. Aborted/zero/missing/mismatched evidence remains null. A completed
negative lift remains a real negative. These are the existing native simulation
task labels, not the paper's real transport-to-bin success metric.

For the paired CPU/GPU backend benchmark, run the **same two new plans** into
separate output directories with `--benchmark-only --paired-benchmark-id ID`.
Every record carries that flag. These repeated backend measurements must not be
counted as four independent training samples or added to the old pilot36.

Per trial: `trials/ID/{trial_binding.json,bound_evaluation.json,training_record.json,
RESULTS.json}`. Full native artifacts are under `cohorts/cohort_NNNN/`. Only
label-eligible records are emitted in `training_records.jsonl`; preparation nulls,
partial attempts and errors remain in the main result and per-trial artifacts.

The zero-physics tests read one existing completed pilot trace solely to test
contracts, plus intentionally corrupted temporary copies. They run no simulator,
network forward or training update and provide no new performance result:

```bash
"$ANYDEX_PY" -B reproduction/boya/test_queue_contract.py
```

The first paired backend check used two **new** admitted box plans (gesture6,
extra-depth1; gesture8, extra-depth0). CPU serial completed both original2100-step
trajectories with all original gates: lifts86.104mm/99.070mm, one scene build,
one pre-trial reset, two native audits,4200steps. Measured queue wall101.403s;
combined loop57.922s. This proves within-wave residency and valid outcomes, not a
matched cold-start speedup estimate.

GPU0 parallel genuinely enabled GPU dynamics/broadphase and preserved both
original command prefixes. Its100→200tick interval took71.14s. The run was safely
interrupted as unsuitable for production: each lane finalized431 recorded steps
and a null label; no completed GPU grasp or GPU acceleration is claimed.
Saved original profiling attributes351.461s to431 completed `sim.step` calls,
versus0.719s to the cross-environment contact guard. The interrupt stack was in
PhysX `fetch_results`, not file reads. Existing other GPU0 workload was present.

The interrupt happened inside the next (432nd) shared physical step. The two
saved traces prove862 recorded environment steps; up to two additional in-flight
environment steps are explicitly unconfirmed. Original normal-return clock
counters remained zero because the interrupted generator never returned; those
counters must not be interpreted as zero cost. The separate diagnostic receipt
records the true saved prefix and this uncertainty. Production defaults remain
CPU resident; further GPU throughput work needs an explicit full-task benchmark.
