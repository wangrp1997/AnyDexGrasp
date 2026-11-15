"""Read only owned host process identities and probe released locks after the run."""
import argparse
from pathlib import Path
import time

import author_pipeline_collection as P

B,R=P.B,P.R
HERE=Path(__file__).resolve().parent


def verify(root,observations):
    protocol=P.verify(root);assert B.read(root/'OUTER_WALL.json')['status']=='complete'
    R.assert_quiescent(root)
    stages=[]
    for path in sorted(root.rglob('STAGE.json')):
        stage=B.read(path);pid=stage['child_pid']
        assert stage['status']=='complete' and stage['returncode']==0
        assert R.process_identity(pid) is None and not R.group_alive(pid)
        stages.append(dict(stage=B.binding(path),pid=pid,group_alive=False))
    assert len(stages)==29
    observer=B.read(observations.with_suffix('.STARTED.json'))
    assert R.process_identity(observer['process']['pid']) is None
    assert observations.with_suffix('.FINISHED.json').exists()
    registry=B.read(P.OLD.REGISTRY)
    assert registry['state']=='released' and registry['run_dir']==str(root)
    assert R.process_identity(registry['owner']['pid']) is None
    slots=[];locks=[]
    for path in sorted(root.glob('resource_slots/*/*.json')):
        value=B.read(path);assert value['state']=='released'
        slots.append(dict(binding=B.binding(path),state='released'))
    for path in [root/'DRIVER.lock',P.OLD.LEGACY_LOCK,*sorted(root.glob('resource_slots/*/*.lock'))]:
        with R.lease(path):locks.append(str(path))
    models=B.read(B.HERE/'evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')
    B.verify(B.normalized_bindings(models['verified_files']))
    labels=B.read(B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json')
    B.verify(B.normalized_bindings(labels['verified_bindings']))
    assert (labels['fit_labels'],labels['internal_labels'])==(14,1)
    cancellations=list(root.rglob('CANCELLATION.json'));assert not cancellations
    value=dict(status='pass',unix_s=time.time(),host_proc_namespace=True,stages=stages,
        observer=B.binding(observations.with_suffix('.FINISHED.json')),released_slots=slots,available_locks=locks,
        registry=B.binding(P.OLD.REGISTRY),old_data_models=models['verified_files'],
        old_label_reference=B.binding(B.HERE/'evidence/author_parallel_collection_20261115_v1/ROOT_VERIFICATION.json'),
        source_archive=protocol['source_archive'],frozen_runtime_sources_and_inputs_unchanged=True,
        fit_labels=14,internal_labels=1,new_training_labels=0,training_updates=0,
        cancellations=0,verification_Kit_launches=0,verification_physics_steps=0,verifier=B.binding(__file__))
    B.save(HERE/'FINAL_STATE.json',value,exclusive=True)
    print(dict(status='pass',exited_stage_groups=len(stages),released_slots=len(slots),available_locks=len(locks)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True);args=parser.parse_args()
    verify(args.run_dir,args.observations)
