"""Read-only source/data and owned-process cleanup verification after Fabric A/B."""
from pathlib import Path
import json
import time

import author_fabric_control as F

B, R, G = F.B, F.R, F.G
HERE = Path(__file__).resolve().parent
root = G.BASE/'fabric_control_20261115_v1'
protocol = F.verify(root)
result = B.read(root/F.RESULTS)
processes = []
slots = []
locks = []
for group in result['groups']:
    for worker in group['workers']:
        stage = B.read(worker['stage']['path'])
        process = stage['process']
        identity = R.process_identity(process['pid'])
        alive = R.group_alive(process['pgid'])
        assert identity is None and not alive, (process, identity, alive)
        processes.append(dict(process=process, current_identity=identity,
            process_group_alive=alive, stage=worker['stage'], returncode=stage['returncode']))
R.assert_quiescent(root)
for path in root.glob('resource_slots/*/*.json'):
    value = B.read(path)
    assert value['state'] == 'released'
    slots.append(dict(binding=B.binding(path), state=value['state']))
for path in [root/'DRIVER.lock', G.LOCK, *root.glob('resource_slots/*/*.lock')]:
    with R.lease(path): locks.append(str(path))
registry = B.read(G.REGISTRY)
assert registry['state'] == 'released'
models = B.read(B.HERE/'evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')
B.verify(B.normalized_bindings(models['verified_files']))
labels = B.read(B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json')
B.verify(B.normalized_bindings(labels['verified_bindings']))
assert (labels['fit_labels'],labels['internal_labels']) == (14,1)
protected = B.read(B.HERE/'evidence/author_gpu_control_20261115_v1/FINAL_STATE.json')
for binding in protected['protocols'].values(): B.verify({'old_protocol':binding})
for old_name in ('gpu_partition_profile_20261115_v1','gpu_vector_control_20261115_v2'):
    old_root=G.BASE/old_name
    old_protocol=B.read(old_root/('GPU_PROFILE_CONTROL_PROTOCOL.json' if 'partition' in old_name else 'GPU_CONTROL_PROTOCOL.json'))
    B.verify(old_protocol['inputs']);B.verify(old_protocol['sources'])
value = dict(schema='anydex_boya_author_fabric_final_state_v1',status='pass',unix_s=time.time(),
    protocol=B.binding(root/F.PROTOCOL),result=B.binding(root/F.RESULTS),
    frozen_sources_and_inputs_unchanged=True,source_archive=protocol['source_archive'],
    owned_workers=processes,released_slots=slots,available_locks=locks,
    registry=B.binding(G.REGISTRY),old_gpu_protocols_unchanged=True,
    old_data_models_verified=models['verified_files'],old_data_models_count=len(models['verified_files']),
    old_new_labels_reference=B.binding(B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json'),
    fit_labels=14,internal_labels=1,new_collection_labels=0,training_updates=0,
    verification_physics_steps=0,verifier=B.binding(__file__))
B.save(HERE/'FINAL_STATE.json',value,exclusive=True)
print(json.dumps(dict(status='pass',owned_workers=len(processes),released_slots=len(slots),
    available_locks=len(locks),old_data_models=len(models['verified_files']),new_labels=0,physics_steps_added=0)))
