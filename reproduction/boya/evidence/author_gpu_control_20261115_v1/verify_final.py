"""Read-only final bindings and owned-process cleanup; never imports a simulator."""
import json
from pathlib import Path
import time

import author_collection_batch as B
import author_collection_wave_resources as R
import author_gpu_control as G

HERE = Path(__file__).resolve().parent
runs = [G.BASE/name for name in ('gpu_vector_control_20261115_v1',
    'gpu_vector_control_20261115_v2', 'gpu_partition_profile_20261115_v1')]
bindings = {}
processes = []
locks = []
slots = []
for root in runs:
    protocol_path = root/('GPU_PROFILE_CONTROL_PROTOCOL.json' if 'partition' in root.name else G.PROTOCOL)
    protocol = B.read(protocol_path)
    bindings[str(protocol_path)] = B.binding(protocol_path)
    B.verify(protocol['inputs'])
    if root != runs[0]:
        B.verify(protocol['sources'])
    archive = B.read(root/'frozen_sources/SOURCE_ARCHIVE.json')
    if 'files' in archive:
        for item in archive['files'].values():
            B.verify({'archive': item['archive']})
            assert item['archive']['sha256'] == item['original']['sha256']
    else:
        B.verify(archive['sources'])
        for name, item in archive['sources'].items():
            assert item['sha256'] == protocol['sources'][name]['sha256']
    for path in root.glob('*/STAGE.json'):
        stage = B.read(path)
        pid = stage.get('child_pid')
        if pid:
            identity = R.process_identity(pid)
            alive = R.group_alive(pid)
            assert identity is None and not alive, (pid, identity, alive)
            processes.append(dict(pid=pid, identity=identity, process_group_alive=alive,
                stage=B.binding(path)))
    R.assert_quiescent(root)
    for path in root.glob('resource_slots/*/*.json'):
        value = B.read(path)
        assert value['state'] == 'released'
        slots.append(dict(binding=B.binding(path), state=value['state']))
    with R.lease(root/'DRIVER.lock'):
        locks.append(str(root/'DRIVER.lock'))
with R.lease(G.LOCK):
    locks.append(str(G.LOCK))
registry = B.read(G.REGISTRY)
assert registry['state'] == 'released'

models = B.read(B.HERE/'evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')
B.verify(B.normalized_bindings(models['verified_files']))
data = B.read(B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json')
B.verify(B.normalized_bindings(data['verified_bindings']))
assert (data['fit_labels'], data['internal_labels'], data['positive'], data['negative']) == (14, 1, 11, 4)

result = dict(schema='anydex_boya_gpu_final_state_verification_v1', status='pass',
    unix_s=time.time(), protocols=bindings, archived_frozen_sources_verified=True,
    v2_and_profile_runtime_sources_unchanged=True, owned_children=processes,
    resource_slots=slots, released_registry=B.binding(G.REGISTRY), available_locks=locks,
    unchanged_old_data_models=models['verified_files'],
    unchanged_new_data_reference=B.binding(B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json'),
    fit_labels=14, internal_labels=1, positives=11, negatives=4,
    collection_labels_added=0, training_updates=0, verification_physics_steps=0,
    verifier=B.binding(__file__))
B.save(HERE/'FINAL_STATE.json', result, exclusive=True)
print(json.dumps(dict(status='pass', owned_children=len(processes), released_slots=len(slots),
    available_locks=len(locks), old_model_data_files=len(models['verified_files']),
    fit_labels=14, internal_labels=1, physics_steps_added=0)))
