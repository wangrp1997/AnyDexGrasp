"""Opt-in source collection -> bounded HOME -> any-original-body lift.

No preparation starts Kit, inference, collection execution or training. Execution
requires a separate explicit command and consumes its single attempt on start.
Historical transport protocols and their bin task remain separate.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import author_pipeline as common
import author_collect as collector
import author_collection_bridge as bridge
import author_joint_home_driver as home_driver
from author_action_semantics import ActionContract

T = common.transport
TASK = collector.LIFT_TASK
SCHEMA = 'anydex_boya_author_lift_pipeline_v1'
SCENE_SCHEMA = 'anydex_boya_author_lift_scene_v1'
GESTURE_SOURCE = {1: 1, 6: 2, 7: 3, 8: 4}


def _backend():
    return importlib.import_module('author_lift')


def _planner():
    return importlib.import_module('author_lift_plan')


def _sources():
    names = ('author_lift_pipeline.py', 'author_lift.py', 'author_lift_plan.py',
        'author_pipeline.py', 'author_collect.py', 'author_collection_bridge.py',
        'author_joint_home_driver.py', 'author_joint_home_route.py', 'author_home_retime.py',
        'author_home_route.py', 'author_transport.py', 'author_action_semantics.py',
        'author_protocol.py', 'author_collision.py', 'author_hand_clouds.py',
        'asset_adapter.py', 'trial_record.py', 'boya_prehead_domain.py', 'native_arm.py')
    return {name: common._binding(HERE/name) for name in names}


def _verify_protocol(protocol):
    for key in ('inputs', 'sources', 'capture_source_bindings'):
        common._verify(protocol.get(key, {}))


def _scene(path, archive_sources=None):
    scene = _backend().validate_scene(path, archive_sources)
    if (scene.get('schema') != SCENE_SCHEMA or scene.get('task') != TASK
            or scene.get('bin', 'missing') is not None):
        raise ValueError('Explicit bin-free any-original-body lift scene required')
    ids = [row['object_id'] for row in scene['objects']]
    if not ids or len(set(ids)) != len(ids) or scene['target_object_id'] not in ids:
        raise ValueError('Unique original tabletop objects and compatibility body required')
    return scene


def _candidate(candidate):
    if (candidate.get('schema') != collector.CANDIDATE_SCHEMA
            or candidate.get('task') != TASK
            or collector._candidate_id(candidate) != candidate['candidate_sha256']):
        raise ValueError('Unchanged lift-task source collection candidate required')


def _arrays(directory):
    with np.load(Path(directory)/'commands.npz', allow_pickle=False) as z:
        return {name: z[name].copy() for name in z.files}


def _verify_plan_artifacts(directory, plan):
    directory = Path(directory)
    for key, name in (('commands_sha256', 'commands.npz'),
                      ('path_geometry_sha256', 'path_geometry.json'),
                      ('lift_arm_geometry_sha256', 'lift_arm_geometry.json')):
        if T.sha(directory/name) != plan[key]:
            raise ValueError('Bound lift plan artifact changed: '+name)
        if name.endswith('.json') and T.read(directory/name).get('ok') is not True:
            raise ValueError('Bound lift geometry is not clear: '+name)


def _validate_approach(directory, candidate, calibration):
    directory = Path(directory)
    plan = T.read(directory/'plan.json')
    if (plan.get('complete') is not True or plan.get('approach_only') is not True
            or plan['arguments']['pregrasp_distance_m'] != .04
            or T.sha(directory/'commands.npz') != plan['commands_sha256']):
        raise ValueError('Complete source4cm approach-only seed required, including rejected direct HOME')
    common._verify_plan_dependencies(plan)
    arrays = _arrays(directory)
    if list(dict.fromkeys(arrays['phase'].tolist())) != ['settle', 'home_to_pregrasp', 'approach']:
        raise ValueError('Approach-only source phases changed')
    contract = ActionContract(calibration)
    names = list(map(str, arrays['joint_names']))
    motors = arrays['joint_target'][:, [names.index(n) for n in contract.names]]
    expanded, _ = collector.expanded_planned_hand(arrays, contract)
    contract.validate_command_arrays(expanded, slot_id=candidate['slot_id'], motor_targets=motors)
    wanted = contract.mapped_palm(candidate['slot_id'], candidate['actual_row15_camera'],
        candidate['world_T_camera'], extra_depth_m=candidate['extra_depth_m'])
    if np.max(np.abs(arrays['desired_palm'][-1]-wanted)) > 2e-6:
        raise ValueError('Approach endpoint no longer binds the source candidate palm')
    return plan


def _verify_lift_preservation(source_dir, lifted_dir):
    old, new = _arrays(source_dir), _arrays(lifted_dir)
    for key in ('joint_names', 'initial_position'):
        if (old[key].shape != new[key].shape or old[key].dtype != new[key].dtype
                or old[key].tobytes() != new[key].tobytes()):
            raise ValueError('Lift changed source '+key)
    n = len(old['phase'])
    for key in ('phase', 'joint_target', 'desired_palm', 'time_s'):
        left, right = old[key], new[key][:n]
        if left.shape != right.shape or left.dtype != right.dtype or left.tobytes() != right.tobytes():
            raise ValueError('Lift changed source approach prefix '+key)
    if list(dict.fromkeys(new['phase'].tolist())) != ['settle', 'home_to_pregrasp', 'approach', 'close', 'hold', 'lift']:
        raise ValueError('Lift phases must end at original lift, without transport/release or new hold')
    if int((new['phase']=='close').sum()) != 250 or int((new['phase']=='hold').sum()) != 50:
        raise ValueError('Source250close/50hold command counts required')
    return dict(source_prefix_rows=n, source_prefix_shape_dtype_bytes_equal=True,
        close_rows=250, hold_rows=50, no_transport_or_release=True, no_added_post_lift_hold=True)


def _bind_plan(raw_dir, output, candidate, calibration, scene_path, protocol, refinement):
    """Fresh metadata-only binding; never mutate the prepared route/lift plan."""
    raw_dir, output = Path(raw_dir), Path(output)
    _candidate(candidate)
    scene = _scene(scene_path, protocol.get('scene_archive_sources'))
    raw = T.read(raw_dir/'plan.json')
    if (raw.get('admitted') is not True or raw.get('complete') is not True
            or T.sha(raw_dir/'commands.npz') != raw['commands_sha256']):
        raise ValueError('Only an admitted complete lift plan can bind execution')
    if raw.get('author_lift', {}).get('task', raw.get('task')) != TASK:
        raise ValueError('Lift planner must explicitly bind the new task')
    common._verify_plan_dependencies(raw)
    common._verify(common._arm_refinement_inputs(refinement))
    if Path(refinement['staged']['path']).resolve() != output.resolve().parent/'arm_collision_refinements.json':
        raise ValueError('Native repair must be staged at actual bound plan parent')
    for key, name in (('path_geometry_sha256', 'path_geometry.json'), ('lift_arm_geometry_sha256', 'lift_arm_geometry.json')):
        if T.sha(raw_dir/name) != raw[key] or T.read(raw_dir/name).get('ok') is not True:
            raise ValueError('All-scene lift geometry must be clear and SHA-bound')
    output.mkdir(exist_ok=False)
    for p in raw_dir.iterdir():
        if p.is_file() and p.name not in ('commands.npz', 'plan.json'):
            shutil.copyfile(p, output/p.name)
    arrays = _arrays(raw_dir)
    execution_spec = dict(task=TASK, scene=common._binding(scene_path),
        original_object_ids=[row['object_id'] for row in scene['objects']],
        raw_lift_plan=common._binding(raw_dir/'plan.json'),
        native_arm_collision_refinement=refinement,
        definition='Any original tabletop body must satisfy the frozen per-body source-bottom/rise/bilateral gates; original native lift, no bin task')
    binding = collector.write_bound_commands(candidate, arrays, output/'commands.npz',
        execution_spec=execution_spec, action_contract=ActionContract(calibration))
    bound_arrays = _arrays(output)
    if any(a.dtype != bound_arrays[k].dtype or a.shape != bound_arrays[k].shape
           or a.tobytes() != bound_arrays[k].tobytes() for k, a in arrays.items()):
        raise ValueError('Metadata binding changed a native command array')
    plan = copy.deepcopy(raw)
    metadata = plan.setdefault('author_collection_binding', {})
    for key, expected in dict(candidate=candidate, collection_candidate_sha256=candidate['candidate_sha256'], task=TASK).items():
        if key in metadata and metadata[key] != expected:
            raise ValueError('Lift candidate/task metadata conflicts with prepared source')
        metadata.setdefault(key, expected)
    metadata['command_binding'] = binding
    plan.update(task=TASK, commands_sha256=binding['commands_sha256'], native_arm_collision_refinement=refinement)
    plan.setdefault('source_sha256', {}).update({v['path']:v['sha256'] for v in protocol['sources'].values()})
    extras = plan.setdefault('extra_input_sha256', {})
    for bindings in (protocol['inputs'], protocol.get('capture_source_bindings', {}), common._arm_refinement_inputs(refinement)):
        extras.update({v['path']:v['sha256'] for v in bindings.values()})
    T.save(output/'plan.json', plan)
    T.save(output/'COMMAND_BINDING.json', binding)
    return binding


def prepare(scene_path, representation_dir, observation, gesture_id, source_type, seed,
            calibration, cloud_manifest, backend, out, *, archive_sources=None, domain=None):
    """One fixed first candidate, one RRT plus eligible same-path retime, one lift."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = dict(schema=SCHEMA, task=TASK, status='preparing', complete=False, stage='freeze',
        physics_steps=0, new_labels=0, training_updates=0, new_CGR_forwards=0)
    T.save(out/'LIFT_PIPELINE_RESULTS.json', result)
    try:
        if GESTURE_SOURCE.get(gesture_id) != source_type:
            raise ValueError('Explicit g1/6/7/8 -> DH3 reference1/2/3/4 mapping required')
        motion = common.source_collection_motion_policy()
        rep = Path(representation_dir).resolve()
        paths = dict(scene=scene_path, observation=observation, calibration=calibration,
            hand_cloud_manifest=cloud_manifest, backend=backend, collection_motion_source=common.COLLECTION_SOURCE,
            raw_candidates=rep/'raw_camera_candidates.npz', representation_outputs=rep/'representation_outputs.npz',
            representation_receipt=rep/'RESULTS.json')
        inputs = {k:common._binding(p) for k,p in paths.items()}
        scene = _scene(inputs['scene']['path'], archive_sources)
        obs = T.read(inputs['observation']['path'])
        if (obs.get('scene_sha256') != inputs['scene']['sha256']
                or Path(obs['scene_json']).resolve() != Path(inputs['scene']['path'])
                or obs.get('object_pose_as_network_input') is not False):
            raise ValueError('Observation must bind this exact bin-free scene without object GT network input')
        cfg = T.read(inputs['backend']['path'])
        if Path(cfg['physics_config']).resolve() != Path(scene['physics_config']).resolve():
            raise ValueError('Observed and execution native physics configurations must match')
        refinement = common.stage_arm_collision_refinement(inputs['backend']['path'], out)
        inputs.update(common._arm_refinement_inputs(refinement))
        primary = next(row for row in scene['objects'] if row['object_id']==scene['target_object_id'])
        derived = dict(cfg, collision_refinement=primary.get('collision_refinement'),
            collision_refinement_sha256=primary.get('collision_refinement_sha256'),
            original_backend_pregrasp_distance_m=cfg.get('pregrasp_distance_m'), pregrasp_distance_m=motion['approach_distance_m'],
            collection_motion_policy=motion, restore_object_initial_state=True)
        T.save(out/'BACKEND.json', derived)
        inputs['derived_backend'] = common._binding(out/'BACKEND.json')
        protocol = dict(schema=SCHEMA, task=TASK, inputs=inputs, sources=_sources(),
            capture_source_bindings=T.scene_source_bindings(scene, archive_sources),
            scene_archive_sources={str(Path(k).resolve()):str(Path(v).resolve()) for k,v in (archive_sources or {}).items()},
            domain=copy.deepcopy(domain if domain is not None else common.DOMAIN), gesture_id=gesture_id,
            source_reference_type=source_type, seed=seed, collection_motion_policy=motion,
            candidate_budget=1, fallback_candidate=False, home_driver_policy=copy.deepcopy(home_driver.POLICY),
            compatibility_primary=primary['object_id'], primary_is_candidate_target=False,
            original_object_ids=[row['object_id'] for row in scene['objects']], bin=None,
            stages=['source_collection','source4cm_approach','bounded_HOME','native_close_hold_lift','optional_once_execution'],
            formal_collection_queue_started=False, formal_training_started=False)
        common._exclusive_json(out/'LIFT_PIPELINE_PROTOCOL.json', protocol)
        cp = dict(schema=collector.SCHEMA, task=TASK,
            inputs={k:inputs[k] for k in ('raw_candidates','representation_outputs','representation_receipt','observation','calibration','hand_cloud_manifest')},
            gesture_id=gesture_id, source_reference_type=source_type, seed=seed, domain=protocol['domain'],
            scene_binding=dict(scene_id=scene['scene_id'],scene_json=inputs['scene']['path'],scene_sha256=inputs['scene']['sha256'],
                               observation_sha256=inputs['observation']['sha256']),
            action_contract=dict(implementation_sha256=protocol['sources']['author_action_semantics.py']['sha256']))
        common._exclusive_json(out/'COLLECTION_PROTOCOL.json', cp)
        result.update(stage='source_collection',protocol_sha256=T.sha(out/'LIFT_PIPELINE_PROTOCOL.json'))
        T.save(out/'LIFT_PIPELINE_RESULTS.json',result)
        prepared = collector.run_protocol(out/'COLLECTION_PROTOCOL.json',out/'collection')
        result['collection_counts'] = prepared['counts']
        if not prepared['candidates']:
            result.update(status='no_candidate',complete=True)
        else:
            candidate = prepared['candidates'][0]
            _candidate(candidate)
            if prepared['selected_candidate_sha256'] != candidate['candidate_sha256']:
                raise ValueError('Collector selected first candidate binding changed')
            result.update(stage='source4cm_approach',candidate_sha256=candidate['candidate_sha256'])
            T.save(out/'LIFT_PIPELINE_RESULTS.json',result)
            bridge.bridge_candidate(candidate,out/'pool',calibration=inputs['calibration']['path'],object_root=primary['object_root'],
                world_T_object=primary['world_T_object'],object_scale=primary['scale'],prepared_path=out/'collection/PREPARED.json',
                scene_manifest=inputs['scene']['path'])
            arm = common.prepare_approach(out/'pool',out/'arm',backend_config=out/'BACKEND.json')
            result['arm_preparation'] = arm
            seed_dir = Path(arm['plan_dir'])
            if arm['status']=='infrastructure_failure':
                result.update(status='infrastructure_failure',complete=False)
            elif not (seed_dir/'plan.json').exists():
                result.update(status='null_geometry',complete=True)
            else:
                _validate_approach(seed_dir,candidate,inputs['calibration']['path'])
                result['stage']='bounded_HOME';T.save(out/'LIFT_PIPELINE_RESULTS.json',result)
                route = home_driver.prepare_home_route(seed_dir,inputs['scene']['path'],out/'home',
                    config_path=out/'BACKEND.json',archive_sources=archive_sources)
                result['home_preparation']=route
                if route['status']!='ready':
                    result.update(status=route['status'],complete=route['status']!='infrastructure_failure')
                else:
                    routed = Path(route['plan_dir'])
                    if T.sha(routed/'plan.json') != route['plan_sha256']:
                        raise ValueError('HOME driver plan binding changed')
                    result['home_preservation']=common._verify_home_preservation(seed_dir,routed)
                    result['stage']='native_close_hold_lift';T.save(out/'LIFT_PIPELINE_RESULTS.json',result)
                    lifted = _planner().prepare_lift(routed,inputs['scene']['path'],out/'lift_raw',archive_sources=archive_sources)
                    result['lift_preparation']=lifted
                    if lifted['status']!='ready':
                        result.update(status=lifted['status'],complete=lifted['status']!='infrastructure_failure')
                    else:
                        raw = Path(lifted['plan_dir'])
                        if T.sha(raw/'plan.json') != lifted['plan_sha256']:
                            raise ValueError('Lift preparation plan binding changed')
                        result['lift_preservation']=_verify_lift_preservation(routed,raw)
                        binding = _bind_plan(raw,out/'lift_bound',candidate,inputs['calibration']['path'],inputs['scene']['path'],protocol,refinement)
                        ready=dict(schema=SCHEMA,task=TASK,status='ready',protocol=common._binding(out/'LIFT_PIPELINE_PROTOCOL.json'),
                            scene=inputs['scene'],candidate=common._binding(out/'pool/selected_author_candidate.json'),
                            plan=common._binding(out/'lift_bound/plan.json'),commands=common._binding(out/'lift_bound/commands.npz'),
                            command_binding=common._binding(out/'lift_bound/COMMAND_BINDING.json'),
                            collection_candidate_sha256=candidate['candidate_sha256'],execution_allowed_once=True)
                        _verify_protocol(protocol)
                        common._exclusive_json(out/'READY.json',ready)
                        result.update(status='ready',complete=True,stage='ready',ready_sha256=T.sha(out/'READY.json'),command_binding=binding)
        _verify_protocol(protocol)
        result['wall_s']=time.monotonic()-start
        T.save(out/'LIFT_PIPELINE_RESULTS.json',result)
        return result
    except BaseException as error:
        result.update(status='infrastructure_failure',complete=False,error=repr(error),traceback=traceback.format_exc(),wall_s=time.monotonic()-start)
        T.save(out/'LIFT_PIPELINE_RESULTS.json',result)
        raise


def _execute_native(args,scene_path):
    return _backend().run(args,scene_path)


def execute(prepared_dir):
    out=Path(prepared_dir).resolve()
    ready=T.read(out/'READY.json');preparation=T.read(out/'LIFT_PIPELINE_RESULTS.json')
    if (ready.get('schema')!=SCHEMA or ready.get('task')!=TASK or preparation.get('status')!='ready'
            or preparation.get('ready_sha256')!=T.sha(out/'READY.json')):
        raise ValueError('Only unchanged lift-task ready preparations may execute')
    protocol=T.read(ready['protocol']['path'])
    if protocol.get('schema')!=SCHEMA or protocol.get('task')!=TASK or protocol.get('bin','missing') is not None:
        raise ValueError('Lift execution cannot consume a transport or old primary-lift protocol')
    bindings={key:ready[key] for key in ('protocol','scene','candidate','plan','commands','command_binding')}
    _verify_protocol(protocol);common._verify(bindings)
    scene=_scene(ready['scene']['path'],protocol.get('scene_archive_sources'))
    candidate=T.read(ready['candidate']['path']);_candidate(candidate)
    plan=T.read(ready['plan']['path']);command_binding=T.read(ready['command_binding']['path'])
    common._verify_plan_dependencies(plan)
    _verify_plan_artifacts(Path(ready['plan']['path']).parent, plan)
    ids=[row['object_id'] for row in scene['objects']]
    if (ready.get('status')!='ready' or plan.get('task')!=TASK or not plan.get('admitted')
            or candidate['candidate_sha256']!=ready['collection_candidate_sha256']
            or plan['author_collection_binding']['collection_candidate_sha256']!=candidate['candidate_sha256']
            or plan['commands_sha256']!=ready['commands']['sha256']
            or command_binding['commands_sha256']!=ready['commands']['sha256']
            or command_binding['execution_spec'].get('task')!=TASK
            or command_binding['execution_spec'].get('original_object_ids')!=ids):
        raise ValueError('Lift ready/candidate/actual-command/all-body task binding differs')
    common._exclusive_json(out/'EXECUTION_STARTED.json',dict(schema=SCHEMA,task=TASK,
        ready_sha256=T.sha(out/'READY.json'),pid=os.getpid(),started_unix_s=time.time(),policy='One launch including failed infrastructure starts'))
    execution=out/'execution';execution.mkdir(exist_ok=False)
    receipt=dict(schema=SCHEMA,task=TASK,status='starting',physics_steps=0,new_labels=0)
    T.save(out/'LIFT_PIPELINE_EXECUTION.json',receipt)
    start=time.monotonic();launcher=None;exit_code=1
    try:
        launcher=common._launch_native()
        args=_backend().args_for_plan(Path(ready['plan']['path']).parent,execution)
        evaluation=_execute_native(args,ready['scene']['path'])
        _verify_protocol(protocol);common._verify(bindings);common._verify_plan_dependencies(plan)
        _verify_plan_artifacts(Path(ready['plan']['path']).parent, plan)
        persisted=T.read(execution/'LIFT_EVALUATION.json')
        if T.json_sha(evaluation)!=T.json_sha(persisted):
            raise ValueError('Returned and persisted lift outcomes differ')
        record=collector.collect_execution(candidate,command_binding,execution/'LIFT_EVALUATION.json')
        T.save(out/'COLLECTION_RECORD.json',record)
        receipt.update(status=evaluation['status'],complete=evaluation['complete'],physics_steps=evaluation['total_physics_steps'],
            label=record['label'],label_state=record['label_state'],new_labels=int(record['label'] is not None),wall_s=time.monotonic()-start)
        T.save(out/'LIFT_PIPELINE_EXECUTION.json',receipt)
        exit_code=0
        return receipt
    except BaseException as error:
        p=execution/'LIFT_EVALUATION.json';evaluation=T.read(p) if p.exists() else {}
        receipt.update(status='infrastructure_failure',complete=False,error=repr(error),traceback=traceback.format_exc(),
            physics_steps=evaluation.get('total_physics_steps',0),wall_s=time.monotonic()-start)
        T.save(out/'LIFT_PIPELINE_EXECUTION.json',receipt)
        raise
    finally:
        if launcher is not None:
            launcher.app.close(exit_code=exit_code)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='stage',required=True)
    prep=subs.add_parser('prepare')
    for name in ('scene','representation-dir','observation','calibration','cloud-manifest','backend','output-dir'):
        prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--gesture-id',type=int,choices=GESTURE_SOURCE,required=True)
    prep.add_argument('--source-type',type=int,choices=(1,2,3,4),required=True)
    prep.add_argument('--seed',type=int,required=True)
    prep.add_argument('--archive-sources',type=Path)
    prep.add_argument('--domain',type=Path,help='Explicit frozen source sensor/workspace adaptation JSON')
    run=subs.add_parser('execute');run.add_argument('--prepared-dir',type=Path,required=True)
    a=parser.parse_args()
    if a.stage=='prepare':
        result=prepare(a.scene,a.representation_dir,a.observation,a.gesture_id,a.source_type,a.seed,
            a.calibration,a.cloud_manifest,a.backend,a.output_dir,
            archive_sources=T.read(a.archive_sources) if a.archive_sources else None,
            domain=T.read(a.domain) if a.domain else None)
    else:
        result=execute(a.prepared_dir)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
