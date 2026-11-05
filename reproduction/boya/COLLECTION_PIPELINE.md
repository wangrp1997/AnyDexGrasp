# Frozen 40/10 collection pipeline

`run_collection_pipeline.py` connects the existing collection modules. The initial collection uses hand types **1, 6, 8**, four additional-depth indices **0–3** (0, 10, 20, 30 mm), and one first arm-admitted physical candidate per object/type/depth. This is a collection configuration; these three types have not been established as the best types.

The frozen selector declared the inherited pilot extra depth **0 only**. The pipeline records that original `pool_contract` and an explicit `pool_contract_amendment` declaring the four-depth expansion. It does not claim the selection originally declared four depths.

40 training objects × 3 types × 4 depths give **480 slots**; 10 development-validation objects give **120 slots**, for **600 maximum actual attempts**. Geometry rejection and arm-plan rejection retain nulls and consume no physical attempt. The target for each type is **100 valid training labels and 20 validation labels**, counting positives and negatives. A target shortfall is reported; no labels are manufactured and extra valid records are retained. Source mesh groups and the eight development objects are isolated as recorded in the selection; validation does not mean unseen during original CGR pretraining.

## Commands

Use the existing environment without resolving its interpreter symlink:

```bash
cd /home/wangrenpeng/AnyDexGrasp
/home/wangrenpeng/boya-inhand-wm/.venv/bin/python -B reproduction/boya/run_collection_pipeline.py \
  --run-dir outputs/collection40_20261105_v1/run_v1 --dry-run
```

`--dry-run` freezes and checks files, manifests, the 600-slot plan, and source SHA values. It imports no Isaac/torch model module, starts no Kit process, and performs no network forward or physics. `DRY_PLAN.json` records this distinction. A dry-plan is not preparation completion or authorization to start a large collection.

An explicitly requested preparation invocation uses the same run:

```bash
/home/wangrenpeng/boya-inhand-wm/.venv/bin/python -B reproduction/boya/run_collection_pipeline.py \
  --run-dir outputs/collection40_20261105_v1/run_v1 --phase prepare
```

`prepare` is the default phase. **It uses RTX/GPU for observations and CGR**, and invokes zero-step native asset cooking, but stops after full-arm plan preparation. It executes no grasp controls. The stages are:

1. All 50 objects: the existing read-only `preprocess_native_object.py`, maximum 32 failing partitions, two CPU workers. The original convex union and native 20 µm audit remain required; any generated repair path/SHA is passed to every corresponding planning job. Trial physical steps remain zero.
2. One `render_collection_observations.py` process and the fixed oblique camera for all objects, no robot. Only the RGB/depth/meta files are network inputs.
3. One persistent original CGR model and **one identity forward per object**, reused by all hand/depth pools. No Boya learned head is loaded or trained.
4. `build_collection_pool.py`: two CPU workers, paths `pools/OBJECT/gG/dD`, 500 original score-ranked rows, keep at most 8 geometrically admitted rows per slot. Extra depth is applied before repeating hand/table/cloud/approach checks.
5. `prepare_collection_queue.py`: four CPU workers, at most four source-ranked arm plans per slot; the first admitted plan is queued once. No outcome-based reranking occurs.

Physical dispatch is a separate explicit invocation:

```bash
/home/wangrenpeng/boya-inhand-wm/.venv/bin/python -B reproduction/boya/run_collection_pipeline.py \
  --run-dir outputs/collection40_20261105_v1/run_v1 --phase execute
```

`--phase all` explicitly requests both preparation and physical dispatch. This entry's implementation check only creates a dry-plan; it does **not** launch the 50-object collection.

Defaults separate `--representation-device cuda:0` from `--physics-device cpu`. Physics defaults to the accepted CPU serial resident backend, `--execution-mode serial --batch-size 8`. Each homogeneous wave reuses a scene; cross-wave residency is not implemented. GPU physics is an explicit experimental configuration and has not passed production-throughput acceptance. Options that alter the frozen configuration require a new run directory.

## Binding, reuse, and failures

`protocol.json` binds the original selection, calibration, backend and its source inputs, weights, adapter/source bytes, camera manifest, hand/depth policy and every slot. `objects.json` rebases only planned output paths; public source geometry and poses remain fixed. `representation_manifest.json` has rows `{object_id, observation_json, representation_dir, split}`.

Completed stages are reused only after verifying their protocol and artifact SHA values. Existing partial/failed render, model, plan or execution directories are preserved; the entry refuses automatic overwrites or physical replay. Individually completed native checks and pools may be adopted only with their bound source hashes intact. Infrastructure errors prevent advancement to the next stage; already running finite CPU tasks may finish. Normal `no_candidate` and no-arm-plan results remain nulls and are not converted into negative training examples.

`planning/queue_manifest.json` is the executor input. Explicit execution delegates to `execute_collection_queue.py --manifest ... --output-dir ... --device ... --mode ... --batch-size ...`. It retains original controls and native label guards. `collection_counts.json` reports actual label counts and shortages against targets after completion. Raw executor training records remain intact; this pipeline neither truncates excess labels nor trains a head.

Stage status is in `pipeline_progress.json`, detailed stage receipts in `stages/`, and subprocess logs in `logs/`. Source changes or failed bindings require a separately identified run. Large images, features, plans, traces, caches and labels remain under ignored `outputs/`; no public dataset or CanonDex source is modified.

A source-controlled equivalent backend configuration is also available as `reproduction/boya/backend.collection.json` (same physical/control fields and pre-trial restore flag as the accepted pilot configuration). Select it explicitly with `--backend-config reproduction/boya/backend.collection.json` when creating a new run; the frozen run configuration cannot be switched afterward.
