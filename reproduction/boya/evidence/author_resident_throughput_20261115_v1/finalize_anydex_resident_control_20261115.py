from pathlib import Path
import json, shutil, time
import author_collection_batch as B
import author_resident_collection as C
out=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/resident_collection_control_20261115_v1')
for part in ('fit','internal'): C.verify_protocol(out/part)
B.save(out/'FREEZE_WRAPPER_FAILURE.json',dict(stage='metadata_freeze_only',physics_steps=0,
 kind='TypeError',error="B.verify(old) received path-to-SHA strings instead of normalized binding entries",
 recovery='Finalize the already frozen protocols; do not recreate protocols or dispatch any stage',
 original_wrapper=B.binding('/tmp/freeze_anydex_resident_control_20261115.py')),exclusive=True)
old=B.read('reproduction/boya/evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')['verified_files']
B.verify(B.normalized_bindings(old))
B.save(out/'UNCHANGED_DATA_MODELS_BEFORE.json',dict(status='pass',verified_files=old),exclusive=True)
for file in ('author_resident_execution.py','author_resident_collection.py','author_resident_benchmark.py',
             'test_author_resident_execution.py','test_author_resident_collection.py','test_author_resident_benchmark.py'):
 p=B.HERE/file; target=out/'source_snapshot'/file; target.parent.mkdir(exist_ok=True); shutil.copyfile(p,target)
B.save(out/'MASTER_PROTOCOL.json',dict(schema='anydex_resident_small_control_master_v1',
 created_unix_s=time.time(),fit_protocol=B.binding(out/'fit'/C.PROTOCOL),
 internal_protocol=B.binding(out/'internal'/C.PROTOCOL),total_scenes=3,total_gesture_slots=12,
 max_actual_executions=12,expected_ready_from_historical_reference=9,selection='known_scene_controls',
 exclude_training=True,formal_training_labels_added=0,training_updates=0,
 bulk_collection_authorized=False,automatic_large_collection=False,retry_budget=0,
 source_files={p.name:B.binding(p) for p in (out/'source_snapshot').iterdir()},
 instruction='Measure complete bounded lifecycle and original independent audits. User decides whether to begin 4000fit+800internal afterward.'),exclusive=True)
print(json.dumps({'root':str(out),'fit_protocol':B.binding(out/'fit'/C.PROTOCOL),
 'internal_protocol':B.binding(out/'internal'/C.PROTOCOL),'total_slots':12,'max_executions':12},indent=2))
