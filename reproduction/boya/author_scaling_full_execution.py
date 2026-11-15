"""CPU-only, one-lane complete frozen grasp diagnostic for scaling controls.

This opt-in entry does not launch at import, collect new candidates, change the
original controls, or admit any result to training. A separate parent protocol
must explicitly authorize the complete original trajectory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import author_scaling_execution as S

P, V = S.P, S.V
SCALING_SHA256 = '4037bebc5296977848cc2413d833d396eb2669df470b9c74a53863c91dbf0375'


def run(prepared_dir, output_dir, *, protocol_path=None):
    V._need(P.sha(S.__file__) == SCALING_SHA256, 'Frozen scaling adapter changed')
    vector_run, _ = S.compile_entries()
    prepared, out = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    V._need(not out.exists(), 'Fresh full-trial diagnostic output required')
    storage, lift, _, _ = V._modules()
    old_loader,old_sources,old_clock,old_adapted,old_args,old_save = (
        V.load_jobs,V.source_bindings,V.shared_clock,V.adapted_methods,lift.args_for_plan,storage.save)
    checked = S.repeated_jobs([prepared],1,old_loader)[0]
    expected_steps = checked['command_steps']
    V._need(type(expected_steps) is int and expected_steps > 0, 'Complete original command length required')
    import author_gpu_capability_probe as probe
    sources = old_sources()
    sources.update({str(Path(p).resolve()):P.sha(p) for p in (__file__,S.__file__,P.__file__)})
    timings, patches = P.Timings(), P.Patches()
    started = time.time()
    interval,live_scene = {},{}
    roots,origins = S.layout(1)
    finalized = False

    def clock(streams, step, *, on_step=None):
        interval.update(execution_started_unix_s=time.time(),execution_started_monotonic_s=time.monotonic())
        timings.phase = 'execution_full'
        try:
            return old_clock(streams,timings.wrap('barrier.step_callback',step),
                on_step=None if on_step is None else timings.wrap('barrier.after_tick',on_step))
        finally:
            interval.update(execution_finished_unix_s=time.time(),execution_finished_monotonic_s=time.monotonic())
            # Keep the prefix-era field names as explicit aliases for readers
            # that compare intervals; full_trial makes their scope unambiguous.
            interval.update(execution_prefix_started_unix_s=interval['execution_started_unix_s'],
                execution_prefix_started_monotonic_s=interval['execution_started_monotonic_s'],
                execution_prefix_finished_unix_s=interval['execution_finished_unix_s'],
                execution_prefix_finished_monotonic_s=interval['execution_finished_monotonic_s'])
            timings.phase = 'finalization'

    def save(path, value):
        nonlocal finalized
        path = Path(path)
        if path != out/'GPU_VECTOR_EXECUTION.json':
            with timings.measure('artifact.save_json'):
                return old_save(path,value)
        value = dict(value,full_trial=True,scaling_mode='cpu',scaling_lanes=1,
            physics_scope='CPU PhysX complete frozen trajectory; original controls, full contacts and gates retained')
        with timings.measure('artifact.save_json'):
            old_save(path,value)
        cost = S.physical_cost(value,live_scene.get('scene'),1)
        complete = (value.get('complete') is True and value.get('physical_clock_steps')==expected_steps
            and value.get('clock',{}).get('lane_steps')==[expected_steps]
            and cost['physical_steps_unknown'] is False
            and cost['observations_completed_steps_per_lane']==[expected_steps]
            and len(value.get('jobs',[]))==1)
        raw = value.get('jobs',[{}])[0].get('raw_label') if value.get('jobs') else None
        V._need(raw is None or type(raw) is int and raw in (0,1), 'Original diagnostic label must remain unchanged')
        execution_wall = interval.get('execution_finished_monotonic_s',0.)-interval.get('execution_started_monotonic_s',0.)
        profile = dict(schema=P.SCHEMA,status='diagnostic_full' if complete else 'infrastructure_failure',
            complete=complete,full_trial=True,prefix_ticks=0,label=None,raw_label=None,
            formal_training_eligible=False,exclude_training=True,source_sha256=sources,
            actual_physics_steps=value.get('physical_clock_steps',0),
            actual_integrated_environment_steps=value.get('actual_integrated_environment_steps',0),
            timing=timings.values,error=value.get('error'),
            group_receipt=dict(path=str(path),sha256=P.sha(path)),
            timing_semantics='Host wall without added CUDA synchronization; inclusive scopes overlap; interval includes native generator finish serialization')
        V.verify_sources(sources)
        profile_path=out/'GPU_VECTOR_PROFILE.json'
        old_save(profile_path,profile)
        ready_path=prepared/'READY.json';ready=json.loads(ready_path.read_text())
        record=dict(schema=S.SCHEMA,status='diagnostic_full' if complete else 'infrastructure_failure',
            complete=complete,full_trial=True,prefix_ticks=0,expected_command_steps=expected_steps,
            label=None,raw_label=raw,formal_training_eligible=False,exclude_training=True,automatic_training=False,
            mode='cpu',device='cpu',lanes=1,physical_clock_steps=value.get('physical_clock_steps',0),
            actual_integrated_environment_steps=value.get('actual_integrated_environment_steps',0),
            recorded_command_environment_steps=value.get('recorded_command_environment_steps',0),
            original_gates=value['jobs'][0].get('gates') if value.get('jobs') else None,
            environment_roots=roots,origins_m=origins,worker_pid=os.getpid(),worker_started_unix_s=started,
            worker_finished_before_close_unix_s=time.time(),setup_wall_s=interval.get('execution_started_unix_s',time.time())-started,
            execution_wall_s=execution_wall,execution_interval_scope='Complete original shared-clock execution including final generator serialization',
            source_sha256=sources,prepared_template=dict(path=str(prepared),ready_sha256=P.sha(ready_path),
                commands_sha256=ready['commands']['sha256'],scene_sha256=ready['scene']['sha256'],
                ready=dict(path=str(ready_path),sha256=P.sha(ready_path)),commands=ready['commands'],scene=ready['scene']),
            group_receipt=dict(path=str(path),sha256=P.sha(path)),profile=dict(path=str(profile_path),sha256=P.sha(profile_path)),
            timing=timings.values,labels_scope='Original raw label retained only for this repeated-template diagnostic; formal label is always null',
            **interval,**cost)
        old_save(out/'SCALING_EXECUTION.json',record)
        finalized=True
        return None

    def adapted(module):
        P.install_live(patches,timings,storage,lift)
        methods,edits=old_adapted(module)
        methods['after_step']=timings.wrap('all_body.after_step',methods['after_step'])
        return methods,edits

    def args_for_plan(*args,**kwargs):
        args=old_args(*args,**kwargs);args.profile_execution=True;return args

    patches.set(V,'load_jobs',lambda paths:S.repeated_jobs(paths,1,old_loader))
    patches.set(V,'validate_layout',S.validate_layout)
    patches.set(V,'VectorScene',S.scene_class('cpu',lambda scene:live_scene.update(scene=scene)))
    patches.set(V,'source_bindings',lambda:dict(sources))
    patches.set(V,'adapted_methods',adapted)
    patches.set(V,'shared_clock',clock)
    patches.set(lift,'args_for_plan',args_for_plan)
    patches.set(storage,'save',save)
    patches.set(probe,'probe_live',S.cpu_probe(probe))
    try:
        return vector_run([prepared],out,env_roots=roots,origins_m=origins,device='cpu',protocol_path=protocol_path)
    except BaseException:
        if out.exists() and not finalized:
            old_save(out/'FULL_EXECUTION_WRAPPER_ERROR.json',dict(error=traceback.format_exc(),
                label=None,exclude_training=True,source_sha256=sources,**interval))
        raise
    finally:
        patches.restore()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args()
    run(args.prepared_dir,args.output_dir,protocol_path=args.protocol)


if __name__=='__main__':main()
