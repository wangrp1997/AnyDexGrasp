"""One frozen author collection choice through native clutter transport.

Preparation starts no simulator, training or alternative grasp search. Execution
is explicit and at most once per prepared output, including failed starts.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import author_collect as collector
import author_collection_bridge as bridge
import author_transport as transport
from author_action_semantics import ActionContract

SCHEMA = 'anydex_boya_author_pipeline_v1'
COLLECTION_SOURCE = transport.ROOT/'collect_dh3_grasp_data.py'
COLLECTION_SOURCE_SHA256 = '1bcb655fad63a7c2b4f65e29b426b6aa614785d3cea514e0cd290fd05eb8bbcb'
DOMAIN = dict(mode='declared_world_rig',
    world_workspace_aabb_m=[[-1., -.04, -.02], [-.57, .24, .30]],
    approach_world_cone=dict(axis_world=[0., 0., -1.], cos_threshold=.9),
    raw_width_bounds_m=[.01, .099],
    sensor_adapter=dict(description='Declared simulator optical camera; fixed metric sensor input 0.20..0.65m and fixed world rig, public width/approach predicates; no hand/table/learned gate before source top500',
                        input_depth_range_m=[.2, .65], cache_is_unfiltered_parser=True))


def _binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=transport.sha(path))


def _verify(bindings):
    for binding in bindings.values():
        if transport.sha(binding['path']) != binding['sha256']:
            raise ValueError('Frozen pipeline dependency changed: '+binding['path'])


def _exclusive_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_collection_motion_policy():
    """Read the actual public collection call, without importing robot code."""
    if transport.sha(COLLECTION_SOURCE) != COLLECTION_SOURCE_SHA256:
        raise ValueError('Pinned public collection motion source changed')
    tree = ast.parse(COLLECTION_SOURCE.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'robot_grasp')
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'approach_distance'
                           for target in node.targets)]
    if (len(assignments) != 1 or not isinstance(assignments[0].value, ast.Constant)
            or assignments[0].value.value != .04):
        raise ValueError('Public collection must declare its single 4cm approach distance')
    calls = {}
    for owner, method in (('mfcdetector', 'detect'), ('robot', 'grasp_and_throw')):
        found = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == method
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == owner]
        if len(found) != 1:
            raise ValueError('Expected one public collection '+owner+'.'+method+' call')
        keyword = next((kw.value for kw in found[0].keywords if kw.arg == 'approach_dist'), None)
        if not isinstance(keyword, ast.Name) or keyword.id != 'approach_distance':
            raise ValueError('Public collision and execution must consume the same approach_distance')
        calls[owner+'.'+method] = found[0].lineno
    distance = float(assignments[0].value.value)
    if collector.collision.PROFILES['dh3_collection']['approach'] != distance:
        raise ValueError('Collection collision profile and actual motion approach differ')
    return dict(approach_distance_m=distance, source=_binding(COLLECTION_SOURCE),
                function='robot_grasp', assignment_line=assignments[0].lineno,
                call_lines=calls, shared_argument='approach_dist=approach_distance',
                role='Public collection collision and grasp_and_throw use the same 4cm, including retreat to pregrasp',
                backend_distance_is_authoritative=False)


def prepare_approach(pool, output, *, backend_config, timeout_s=360.):
    """Use the pinned native open-hand approach, with no unused lift solve.

    The new transport adapter owns the original closure/hold and actual
    post-closure paths. The existing bridge's standalone lift API is unchanged.
    """
    transport.verify_pins()
    pool, output, backend_config = (Path(p).resolve() for p in (pool, output, backend_config))
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    receipt = dict(schema='anydex_boya_author_approach_preparation_v1', status='preparing',
                   pool=str(pool), physics_steps=0, labels=0, approach_only=True,
                   unused_world_z_lift_required=False, closure_deferred_to='author_transport original closure/hold')
    transport.save(output/'ARM_PREPARATION.json', receipt)
    try:
        motion_policy = source_collection_motion_policy()
        cfg = transport.read(backend_config)
        generation = transport.read(pool/'generation.json')
        candidate = transport.read(pool/'selected_author_candidate.json')
        if (not generation.get('complete') or generation.get('selected_indices') != [0]
                or transport.sha(pool/'selected_author_candidate.json') != generation['selected_candidate_sha256']
                or collector._candidate_id(candidate) != candidate['candidate_sha256']
                or transport.sha(pool/'candidates.npz') != generation['candidates_sha256']):
            raise ValueError('Fixed planner pool changed')
        locks = {str(backend_config): transport.sha(backend_config),
                 str(pool/'generation.json'): transport.sha(pool/'generation.json'),
                 str(pool/'candidates.npz'): transport.sha(pool/'candidates.npz'),
                 **generation['input_sha256'], str(COLLECTION_SOURCE): COLLECTION_SOURCE_SHA256}
        command = [sys.executable, '-B', str(HERE/'native_arm.py'), '--stage', 'prepare',
            '--output-dir', str(output/'native'), '--candidates', str(pool), '--index', '0',
            '--object-root', generation['compatibility_object_root'],
            '--world-object-pose', str(pool/'world_object_pose.json'),
            '--pregrasp-distance-m', str(motion_policy['approach_distance_m']), '--device', 'cpu',
            '--arm-select-min-motion', '--vectorized-arm-broadphase', '--approach-only']
        for key in ('robot_urdf', 'reference_root', 'hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config'):
            command.extend(['--'+key.replace('_', '-'), cfg[key]])
        if cfg.get('collision_refinement'):
            repair = str(Path(cfg['collision_refinement']).resolve())
            locks[repair] = transport.sha(repair)
            if cfg.get('collision_refinement_sha256', locks[repair]) != locks[repair]:
                raise ValueError('Primary collision refinement changed')
            command.extend(['--collision-refinement', repair])
        if cfg.get('restore_object_initial_state'):
            command.append('--restore-object-initial-state')
        for name, digest in locks.items():
            if transport.sha(name) != digest:
                raise ValueError('Approach preparation input changed: '+name)
        receipt.update(command=command, input_sha256=locks, collection_motion_policy=motion_policy,
                       backend_requested_pregrasp_distance_m=cfg.get('pregrasp_distance_m'),
                       actual_pregrasp_distance_m=motion_policy['approach_distance_m'])
        transport.save(output/'ARM_PREPARATION.json', receipt)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='4',
                   CANONDEX_COLLISION_PARTITION_CACHE=str(transport.ROOT/'outputs/boya_collision_partition_cache'))
        timed_out = False
        with (output/'prepare.log').open('xb') as log:
            try:
                returncode = subprocess.run(command, cwd=transport.ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=timeout_s).returncode
            except subprocess.TimeoutExpired:
                timed_out = True; returncode = None
        plan_path, failure_path = output/'native/plan.json', output/'native/admission_failure.json'
        plan = transport.read(plan_path) if plan_path.exists() else {}
        failure = transport.read(failure_path) if failure_path.exists() else {}
        status = 'ready' if returncode == 0 and plan.get('admitted') is True else 'no_plan'
        if (not plan and not (failure.get('complete') is True and failure.get('kind') == 'arm_plan_rejected'
                and failure.get('physical_executed') is False)) or timed_out:
            status = 'infrastructure_failure'
        validation = None
        if status == 'ready':
            if not plan.get('approach_only') or transport.sha(output/'native/commands.npz') != plan['commands_sha256']:
                raise ValueError('Native approach-only result binding changed')
            if plan['arguments']['pregrasp_distance_m'] != motion_policy['approach_distance_m']:
                raise ValueError('Native approach distance differs from the pinned public collection call')
            with np.load(output/'native/commands.npz', allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
            if list(dict.fromkeys(arrays['phase'].tolist())) != ['settle', 'home_to_pregrasp', 'approach']:
                raise ValueError('Approach-only native command prefix changed')
            contract = ActionContract(generation['calibration'])
            names = list(map(str, arrays['joint_names']))
            motors = arrays['joint_target'][:, [names.index(n) for n in contract.names]]
            expanded, expansion = collector.expanded_planned_hand(arrays, contract)
            validation = contract.validate_command_arrays(expanded, slot_id=candidate['slot_id'], motor_targets=motors)
            mapped = contract.mapped_palm(candidate['slot_id'], candidate['actual_row15_camera'],
                candidate['world_T_camera'], extra_depth_m=candidate['extra_depth_m'])
            palm_error = float(abs(arrays['desired_palm'][-1]-mapped).max())
            if palm_error > 2e-6:
                raise ValueError('Approach endpoint differs from bound candidate grasp palm')
            validation.update(max_approach_endpoint_palm_matrix_error=palm_error,
                              closure_and_hold_validation_deferred_to_transport=True)
            validation['planned_mimic_expansion'] = expansion
            plan['author_collection_binding'] = dict(candidate=candidate,
                collection_candidate_sha256=candidate['candidate_sha256'],
                original_native_commands_sha256=plan['commands_sha256'], numerical_action_validation=validation,
                task=candidate['task'], task_success_label_generated=False)
            transport.save(plan_path, plan)
        for name, digest in locks.items():
            if transport.sha(name) != digest:
                raise ValueError('Approach preparation input changed during planning: '+name)
        receipt.update(status=status, complete=status != 'infrastructure_failure', returncode=returncode,
            timed_out=timed_out, plan_dir=str(output/'native'), candidate_sha256=candidate['candidate_sha256'],
            plan_sha256=transport.sha(plan_path) if plan_path.exists() else None,
            admission_failure_sha256=transport.sha(failure_path) if failure_path.exists() else None,
            action_validation=validation, wall_s=time.monotonic()-began)
        transport.save(output/'ARM_PREPARATION.json', receipt)
        return receipt
    except BaseException as error:
        receipt.update(status='infrastructure_failure', complete=False, error=repr(error),
                       traceback=traceback.format_exc(), wall_s=time.monotonic()-began)
        transport.save(output/'ARM_PREPARATION.json', receipt)
        raise


def _bind_transport_plan(raw_plan, output, candidate, calibration, scene_path):
    """Keep 24 native columns; add only metadata in a fresh execution plan."""
    raw_plan, output = Path(raw_plan), Path(output)
    output.mkdir(exist_ok=False)
    for path in raw_plan.iterdir():
        if path.is_file() and path.name not in ('commands.npz', 'plan.json'):
            shutil.copyfile(path, output/path.name)
    with np.load(raw_plan/'commands.npz', allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    execution_spec = dict(task=candidate['task'], scene=_binding(scene_path),
        raw_transport_plan=_binding(raw_plan/'plan.json'),
        definition='Frozen complete multi-body retreat/transport/open/bin task; any originally tabletop body may qualify',
        policy=transport.POLICY, phase_aliases=dict(retreat_4mm='retreat', retreat_wait='retreat',
            retreat_pregrasp='retreat', transport_bin='transport', release_settle='post_release'))
    command_binding = collector.write_bound_commands(candidate, arrays, output/'commands.npz',
        execution_spec=execution_spec, action_contract=ActionContract(calibration))
    with np.load(output/'commands.npz', allow_pickle=False) as bound:
        if any(not np.array_equal(bound[name], value) for name, value in arrays.items()):
            raise AssertionError('Binding changed actual native command arrays')
    plan = transport.read(raw_plan/'plan.json')
    plan['commands_sha256'] = command_binding['commands_sha256']
    plan['author_collection_binding'].update(command_binding=command_binding,
        bound_metadata_only_original_command_arrays_unchanged=True)
    transport.save(output/'plan.json', plan)
    transport.save(output/'COMMAND_BINDING.json', command_binding)
    return command_binding


def prepare(scene_path, representation_dir, observation, gesture_id, source_type, seed,
            calibration, cloud_manifest, backend, out, *, archive_sources=None):
    """Prepare exactly the original post-collision index0; preserve every null."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    result = dict(schema=SCHEMA, status='preparing', stage='freeze', complete=False,
                  output=str(out), physics_steps=0, new_CGR_forwards=0, training_updates=0, new_labels=0)
    transport.save(out/'PIPELINE_RESULTS.json', result)
    try:
        motion_policy = source_collection_motion_policy()
        paths = dict(scene=scene_path, observation=observation, calibration=calibration,
                     hand_cloud_manifest=cloud_manifest, backend=backend,
                     collection_motion_source=COLLECTION_SOURCE)
        representation_dir = Path(representation_dir).resolve()
        paths.update(raw_candidates=representation_dir/'raw_camera_candidates.npz',
                     representation_outputs=representation_dir/'representation_outputs.npz',
                     representation_receipt=representation_dir/'RESULTS.json')
        inputs = {name: _binding(path) for name, path in paths.items()}
        sources = {name: _binding(HERE/name) for name in (
            'author_pipeline.py', 'author_collect.py', 'author_collection_bridge.py',
            'author_transport.py', 'author_action_semantics.py', 'author_protocol.py',
            'author_collision.py', 'author_hand_clouds.py', 'asset_adapter.py',
            'trial_record.py', 'boya_prehead_domain.py', 'native_arm.py')}
        scene = transport.validate_scene(inputs['scene']['path'], archive_sources)
        capture_sources = transport.scene_source_bindings(scene, archive_sources)
        obs = transport.read(inputs['observation']['path'])
        if (obs.get('scene_sha256') != inputs['scene']['sha256']
                or Path(obs['scene_json']).resolve() != Path(inputs['scene']['path'])
                or obs.get('object_pose_as_network_input') is not False):
            raise ValueError('Observed multi-object scene identity must bind without GT network input')
        if gesture_id not in (1, 6, 7, 8) or source_type not in (1, 2, 3, 4):
            raise ValueError('Explicit supported Boya gesture and DH3 reference type required')
        primary = next(row for row in scene['objects'] if row['object_id'] == scene['target_object_id'])
        config = transport.read(inputs['backend']['path'])
        if Path(config['physics_config']).resolve() != Path(scene['physics_config']).resolve():
            raise ValueError('Capture/execution native physics must match')
        derived = dict(config, collision_refinement=primary.get('collision_refinement'),
                       collision_refinement_sha256=primary.get('collision_refinement_sha256'),
                       original_backend_pregrasp_distance_m=config.get('pregrasp_distance_m'),
                       pregrasp_distance_m=motion_policy['approach_distance_m'],
                       collection_motion_policy=motion_policy,
                       restore_object_initial_state=True)
        transport.save(out/'BACKEND.json', derived)
        protocol = dict(schema=SCHEMA, inputs=inputs, sources=sources, domain=DOMAIN,
            collection_motion_policy=motion_policy,
            capture_source_bindings=capture_sources,
            scene_archive_sources={str(Path(k).resolve()): str(Path(v).resolve())
                                   for k, v in (archive_sources or {}).items()},
            gesture_id=int(gesture_id), source_reference_type=int(source_type), seed=int(seed),
            derived_backend=_binding(out/'BACKEND.json'),
            compatibility_primary=primary['object_id'], primary_is_candidate_target=False,
            candidate_budget=1, fallback_candidate=False, bin_adjustment=False,
            stages=['source_collection', 'fixed_candidate_arm', 'transport', 'optional_once_execution'],
            formal_training_started=False, state_boundary=scene.get('state_boundary'),
            arm_seed_mode='Native approach-only; original close/hold added before actual retreat/transport, no unused world-Z lift',
            success_task='declared_boya_clutter_transport_release_v1')
        _exclusive_json(out/'PIPELINE_PROTOCOL.json', protocol)
        collection_protocol = dict(schema=collector.SCHEMA,
            inputs={name: inputs[name] for name in ('raw_candidates', 'representation_outputs',
                'representation_receipt', 'observation', 'calibration', 'hand_cloud_manifest')},
            gesture_id=int(gesture_id), source_reference_type=int(source_type), seed=int(seed), domain=DOMAIN,
            scene_binding=dict(scene_id=scene['scene_id'], scene_json=inputs['scene']['path'],
                               scene_sha256=inputs['scene']['sha256'], observation_sha256=inputs['observation']['sha256']),
            action_contract=dict(implementation_sha256=sources['author_action_semantics.py']['sha256']))
        _exclusive_json(out/'COLLECTION_PROTOCOL.json', collection_protocol)
        result.update(stage='source_collection', protocol_sha256=transport.sha(out/'PIPELINE_PROTOCOL.json'))
        transport.save(out/'PIPELINE_RESULTS.json', result)
        prepared = collector.run_protocol(out/'COLLECTION_PROTOCOL.json', out/'collection')
        result['collection_counts'] = prepared['counts']
        if not prepared['candidates']:
            result.update(status='no_candidate', stage='source_collection', complete=True)
        else:
            candidate = prepared['candidates'][0]
            result['candidate_sha256'] = candidate['candidate_sha256']
            result['stage'] = 'fixed_candidate_arm'
            transport.save(out/'PIPELINE_RESULTS.json', result)
            bridge.bridge_candidate(candidate, out/'pool', calibration=inputs['calibration']['path'],
                object_root=primary['object_root'], world_T_object=primary['world_T_object'],
                object_scale=primary['scale'], prepared_path=out/'collection/PREPARED.json',
                scene_manifest=inputs['scene']['path'])
            arm = prepare_approach(out/'pool', out/'arm', backend_config=out/'BACKEND.json')
            result['arm_preparation'] = arm
            if arm['status'] != 'ready':
                result.update(status=arm['status'], stage='fixed_candidate_arm', complete=True)
            else:
                result['stage'] = 'transport'
                transport.save(out/'PIPELINE_RESULTS.json', result)
                moved = transport.prepare_transport(arm['plan_dir'], inputs['scene']['path'], out/'transport_raw',
                                                    archive_sources=archive_sources)
                result['transport_preparation'] = moved
                if not moved['admitted']:
                    result.update(status=moved['status'], stage='transport', complete=True)
                else:
                    binding = _bind_transport_plan(out/'transport_raw', out/'transport_bound', candidate,
                                                  inputs['calibration']['path'], inputs['scene']['path'])
                    ready = dict(schema=SCHEMA, status='ready', protocol=_binding(out/'PIPELINE_PROTOCOL.json'),
                        scene=inputs['scene'], candidate=_binding(out/'pool/selected_author_candidate.json'),
                        plan=_binding(out/'transport_bound/plan.json'),
                        commands=_binding(out/'transport_bound/commands.npz'),
                        command_binding=_binding(out/'transport_bound/COMMAND_BINDING.json'),
                        collection_candidate_sha256=candidate['candidate_sha256'],
                        execution_allowed_once=True, executed=False)
                    _exclusive_json(out/'READY.json', ready)
                    result.update(status='ready', stage='ready', complete=True,
                                  ready_sha256=transport.sha(out/'READY.json'), command_binding=binding)
        _verify(inputs); _verify(sources); _verify(capture_sources)
        result['wall_s'] = time.monotonic()-began
        transport.save(out/'PIPELINE_RESULTS.json', result)
        return result
    except BaseException as exception:
        result.update(status='infrastructure_failure', complete=False, error=repr(exception),
                      traceback=traceback.format_exc(), wall_s=time.monotonic()-began)
        transport.save(out/'PIPELINE_RESULTS.json', result)
        raise


def _launch_native():
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    from isaaclab.app import AppLauncher
    return AppLauncher(headless=True, device='cpu', visualizer=None, kit_args='--/log/file=/dev/null')


def _execute_native(args, scene_path):
    return transport.run(args, scene_path)


def execute(prepared_dir):
    """Perform one ready, SHA-bound trial; an attempted start is never retried."""
    out = Path(prepared_dir).resolve()
    ready = transport.read(out/'READY.json')
    preparation = transport.read(out/'PIPELINE_RESULTS.json')
    if preparation.get('status') != 'ready' or preparation.get('ready_sha256') != transport.sha(out/'READY.json'):
        raise ValueError('Only the completed unchanged ready preparation may execute')
    protocol = transport.read(ready['protocol']['path'])
    _verify(protocol['inputs']); _verify(protocol['sources'])
    _verify(protocol.get('capture_source_bindings', {}))
    _verify({key: ready[key] for key in ('protocol', 'scene', 'candidate', 'plan', 'commands', 'command_binding')})
    candidate = transport.read(ready['candidate']['path'])
    plan = transport.read(ready['plan']['path'])
    command_binding = transport.read(ready['command_binding']['path'])
    if (ready['status'] != 'ready' or candidate['candidate_sha256'] != ready['collection_candidate_sha256']
            or plan['commands_sha256'] != ready['commands']['sha256']
            or plan['author_collection_binding']['collection_candidate_sha256'] != candidate['candidate_sha256']
            or command_binding['commands_sha256'] != ready['commands']['sha256']):
        raise ValueError('Ready candidate/actual command plan binding changed')
    _exclusive_json(out/'EXECUTION_STARTED.json', dict(schema=SCHEMA, ready_sha256=transport.sha(out/'READY.json'),
        pid=os.getpid(), started_unix_s=time.time(), policy='One launch, including infrastructure failures; no implicit resume'))
    execution = out/'execution'
    execution.mkdir(exist_ok=False)
    began = time.monotonic()
    receipt = dict(schema=SCHEMA, status='starting', physics_steps=0, new_labels=0,
                   candidate_sha256=candidate['candidate_sha256'])
    transport.save(out/'PIPELINE_EXECUTION.json', receipt)
    launcher, exit_code = None, 1
    try:
        launcher = _launch_native()
        args = transport.args_for_plan(Path(ready['plan']['path']).parent, execution)
        evaluation = _execute_native(args, ready['scene']['path'])
        record = collector.collect_execution(candidate, command_binding, execution/'TRANSPORT_EVALUATION.json')
        transport.save(out/'COLLECTION_RECORD.json', record)
        receipt.update(status=evaluation['status'], complete=evaluation['complete'],
            physics_steps=evaluation['total_physics_steps'], label=record['label'], label_state=record['label_state'],
            new_labels=int(record['label'] is not None), wall_s=time.monotonic()-began)
        _verify(protocol['inputs']); _verify(protocol['sources'])
        _verify(protocol.get('capture_source_bindings', {}))
        transport.save(out/'PIPELINE_EXECUTION.json', receipt)
        exit_code = 0
        return receipt
    except BaseException as exception:
        evaluation_path = execution/'TRANSPORT_EVALUATION.json'
        evaluation = transport.read(evaluation_path) if evaluation_path.exists() else {}
        receipt.update(status='infrastructure_failure', error=repr(exception), traceback=traceback.format_exc(),
            physics_steps=evaluation.get('total_physics_steps', 0), wall_s=time.monotonic()-began)
        transport.save(out/'PIPELINE_EXECUTION.json', receipt)
        raise
    finally:
        # SimulationApp.close can terminate the process immediately. Every
        # label, cost and error receipt must therefore be durable beforehand.
        if launcher is not None:
            launcher.app.close(exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    preparation = commands.add_parser('prepare')
    for name in ('scene', 'representation-dir', 'observation', 'calibration', 'cloud-manifest', 'backend', 'output-dir'):
        preparation.add_argument('--'+name, type=Path, required=True)
    preparation.add_argument('--gesture-id', type=int, required=True, choices=(1, 6, 7, 8))
    preparation.add_argument('--source-type', type=int, required=True, choices=(1, 2, 3, 4))
    preparation.add_argument('--seed', type=int, required=True)
    preparation.add_argument('--execute', action='store_true')
    preparation.add_argument('--archive-sources', type=Path, help='Original capture source path -> exact historical snapshot path JSON')
    execution = commands.add_parser('execute')
    execution.add_argument('--prepared-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        value = prepare(args.scene, args.representation_dir, args.observation, args.gesture_id,
            args.source_type, args.seed, args.calibration, args.cloud_manifest, args.backend, args.output_dir,
            archive_sources=transport.read(args.archive_sources) if args.archive_sources else None)
        print(json.dumps(value, indent=2))
        if args.execute and value['status'] == 'ready':
            print(json.dumps(execute(args.output_dir), indent=2))
    else:
        print(json.dumps(execute(args.prepared_dir), indent=2))


if __name__ == '__main__':
    main()
