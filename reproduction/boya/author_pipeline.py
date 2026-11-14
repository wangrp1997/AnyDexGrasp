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
import xml.etree.ElementTree as ET

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


def _verify_plan_dependencies(plan):
    for key in ('source_sha256', 'extra_input_sha256'):
        for path, digest in plan.get(key, {}).items():
            if transport.sha(path) != digest:
                raise ValueError('Bound plan dependency changed: '+path)


def _exclusive_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def stage_arm_collision_refinement(backend_config, output):
    """Stage the configured existing native arm partition at its runtime path."""
    cfg = transport.read(backend_config)
    if not cfg.get('arm_collision_refinement'):
        raise ValueError('Required native arm_collision_refinement is missing from the explicit backend')
    source = Path(cfg['arm_collision_refinement']).resolve()
    spec = transport.read(source)
    robot, physics = _binding(cfg['robot_urdf']), _binding(cfg['physics_config'])
    if (spec.get('schema_version') != 1 or not spec.get('links')
            or spec.get('combined_urdf_sha256') != robot['sha256']
            or spec.get('physics_config_sha256') != physics['sha256']):
        raise ValueError('Native arm refinement must bind this exact combined URDF and physics configuration')
    original = _binding(source)
    if cfg.get('arm_collision_refinement_sha256', original['sha256']) != original['sha256']:
        raise ValueError('Configured native arm refinement source SHA changed')
    robot_path = Path(robot['path'])
    meshes = {}
    for node in ET.parse(robot_path).findall('.//geometry/mesh'):
        filename = node.attrib['filename']
        if filename.startswith('package://'):
            raise ValueError('Combined execution URDF must resolve package mesh URIs before the original loader')
        # Match PlanningModel/native combined-URDF loading. The original
        # composer has already resolved package URIs before this stage.
        mesh_path = (robot_path.parent/filename.removeprefix('file://')).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError('Combined robot source mesh missing: '+str(mesh_path))
        meshes[str(mesh_path)] = _binding(mesh_path)
    destination = Path(output).resolve()/'arm_collision_refinements.json'
    if destination.exists():
        if transport.sha(destination) != original['sha256']:
            raise ValueError('Staged native arm refinement differs from the configured source')
    else:
        with destination.open('xb') as stream:
            stream.write(source.read_bytes())
    staged = _binding(destination)
    if staged['sha256'] != original['sha256']:
        raise ValueError('Native arm refinement copy changed bytes')
    return dict(source=original, staged=staged, robot=robot, physics=physics, robot_meshes=meshes,
        robot_mesh_scope='Every unique visual/collision .//geometry/mesh in the combined URDF; original combined loader paths',
        native_lookup='plan_dir.parent/arm_collision_refinements.json',
        copied_source_bytes_unchanged=True, geometry_or_physics_parameters_changed=False)


def _arm_refinement_inputs(refinement):
    bindings = {'native_arm_refinement_'+name: refinement[name]
                for name in ('source', 'staged', 'robot', 'physics')}
    bindings.update({'native_robot_mesh:'+path: binding
                     for path, binding in refinement['robot_meshes'].items()})
    return bindings


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


HOME_ROUTE_MODULES = {'cartesian_six': 'author_home_route.py',
                      'joint_rrt_connect': 'author_joint_home_route.py'}


def _prepare_home_route(seed_plan_dir, scene_path, output, *, config_path, archive_sources=None,
                        strategy='cartesian_six'):
    if strategy not in HOME_ROUTE_MODULES:
        raise ValueError('Explicit supported HOME route strategy required')
    import importlib
    prepare_home_route = importlib.import_module(HOME_ROUTE_MODULES[strategy][:-3]).prepare_home_route
    return prepare_home_route(seed_plan_dir, scene_path, output, config_path=config_path,
                              archive_sources=archive_sources)


def _verify_home_preservation(source_dir, routed_dir):
    """Check the router changed only HOME, with a shifted physical clock."""
    source_dir, routed_dir = Path(source_dir), Path(routed_dir)
    old_plan, new_plan = transport.read(source_dir/'plan.json'), transport.read(routed_dir/'plan.json')
    with np.load(source_dir/'commands.npz', allow_pickle=False) as data:
        old = {name: data[name].copy() for name in data.files}
    with np.load(routed_dir/'commands.npz', allow_pickle=False) as data:
        new = {name: data[name].copy() for name in data.files}
    for key in ('joint_names', 'initial_position'):
        if (old[key].shape != new[key].shape or old[key].dtype != new[key].dtype
                or old[key].tobytes() != new[key].tobytes()):
            raise ValueError('HOME route changed original '+key)
    if list(dict.fromkeys(new['phase'].tolist())) != ['settle', 'home_to_pregrasp', 'approach']:
        raise ValueError('HOME route changed the fixed source phase sequence')
    counts = {}
    for phase in ('settle', 'approach'):
        left, right = old['phase'] == phase, new['phase'] == phase
        counts[phase] = int(left.sum())
        for key in ('joint_target', 'desired_palm'):
            a, b = old[key][left], new[key][right]
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                raise ValueError('HOME route changed fixed '+phase+' '+key)
    old_home, new_home = old['phase'] == 'home_to_pregrasp', new['phase'] == 'home_to_pregrasp'
    if not old_home.any() or not new_home.any():
        raise ValueError('Original and routed HOME must both be nonempty')
    for key in ('joint_target', 'desired_palm'):
        if old[key][old_home][-1].tobytes() != new[key][new_home][-1].tobytes():
            raise ValueError('HOME route changed frozen pregrasp endpoint '+key)
    arm_names = {'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'}
    fixed_columns = [i for i, name in enumerate(map(str, old['joint_names'])) if name not in arm_names]
    actual = new['joint_target'][new_home][:, fixed_columns]
    expected = np.broadcast_to(old['initial_position'][fixed_columns], actual.shape)
    if actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
        raise ValueError('HOME route changed a frozen non-arm coordinate')
    for key in ('index', 'gesture_id', 'candidates_sha256', 'generation_sha256', 'palm',
                'source_palm', 'world_T_object', 'open_motors', 'closed_motors', 'source_scale'):
        if key in old_plan and new_plan.get(key) != old_plan[key]:
            raise ValueError('HOME route changed fixed candidate metadata: '+key)
    dt = float(old_plan.get('dt_s', transport.POLICY['dt_s']))
    if not np.allclose(new['time_s'], np.arange(1, len(new['phase'])+1)*dt, atol=1e-12, rtol=0):
        raise ValueError('HOME route clock must remain contiguous at the original physical dt')
    return dict(source_commands_sha256=transport.sha(source_dir/'commands.npz'),
        routed_commands_sha256=transport.sha(routed_dir/'commands.npz'),
        fixed_phase_rows=counts, fixed_phase_arrays_shape_dtype_bytes_equal=True,
        frozen_pregrasp_endpoint_shape_dtype_bytes_equal=True,
        unchanged_non_arm_column_count=len(fixed_columns),
        candidate_metadata_unchanged=True, source_dt_s=dt,
        source_home_rows=int((old['phase'] == 'home_to_pregrasp').sum()),
        routed_home_rows=int((new['phase'] == 'home_to_pregrasp').sum()))


def prepare_approach(pool, output, *, backend_config, timeout_s=360., scene_path=None,
                     home_route_config=None, archive_sources=None, home_route_strategy='cartesian_six'):
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
                 str(pool/'selected_author_candidate.json'): generation['selected_candidate_sha256'],
                 str(pool/'candidates.npz'): transport.sha(pool/'candidates.npz'),
                 **generation['input_sha256'], str(COLLECTION_SOURCE): COLLECTION_SOURCE_SHA256}
        if home_route_config is not None:
            if home_route_strategy not in HOME_ROUTE_MODULES:
                raise ValueError('Explicit supported HOME route strategy required')
            if scene_path is None:
                raise ValueError('Opt-in home routing requires the exact full scene')
            scene_path, home_route_config = Path(scene_path).resolve(), Path(home_route_config).resolve()
            locks[str(scene_path)] = transport.sha(scene_path)
            locks[str(home_route_config)] = transport.sha(home_route_config)
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
        native_status = status
        plan_dir = output/'native'
        route = None
        preservation = None
        # A complete source plan may contain a rejected direct HOME segment.
        # The opt-in router independently checks the full scene and preserves
        # the source settle/approach arrays before replacing only that segment.
        if home_route_config is not None and status != 'infrastructure_failure' and plan:
            route = _prepare_home_route(plan_dir, scene_path, output/'home_route',
                config_path=home_route_config, archive_sources=archive_sources, strategy=home_route_strategy)
            if route.get('status') == 'ready' and route.get('admitted') is True:
                status = 'ready'
                plan_dir = Path(route['plan_dir']).resolve()
                if plan_dir != (output/'home_route/native').resolve():
                    raise ValueError('Home router must return its fresh bound native plan directory')
                plan_path = plan_dir/'plan.json'
                plan = transport.read(plan_path)
                if plan.get('admitted') is not True:
                    raise ValueError('Home route receipt and selected plan admission differ')
                preservation = _verify_home_preservation(output/'native', plan_dir)
            elif route.get('status') == 'null_geometry' and route.get('admitted') is False:
                status = 'no_plan'
            elif route.get('status') == 'infrastructure_failure':
                status = 'infrastructure_failure'
            else:
                raise ValueError('Home router returned an inconsistent preparation state')
        validation = None
        if status == 'ready':
            if not plan.get('approach_only') or transport.sha(plan_dir/'commands.npz') != plan['commands_sha256']:
                raise ValueError('Native approach-only result binding changed')
            if plan['arguments']['pregrasp_distance_m'] != motion_policy['approach_distance_m']:
                raise ValueError('Native approach distance differs from the pinned public collection call')
            with np.load(plan_dir/'commands.npz', allow_pickle=False) as data:
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
            timed_out=timed_out, plan_dir=str(plan_dir), candidate_sha256=candidate['candidate_sha256'],
            native_direct_status=native_status, home_route=route,
            home_route_source_preservation=preservation,
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


def _bind_transport_plan(raw_plan, output, candidate, calibration, scene_path, release_spec_path, *,
                         arm_collision_refinement):
    """Keep 24 native columns; add only metadata in a fresh execution plan."""
    raw_plan, output = Path(raw_plan), Path(output)
    _verify(_arm_refinement_inputs(arm_collision_refinement))
    if Path(arm_collision_refinement['staged']['path']) != output.resolve().parent/'arm_collision_refinements.json':
        raise ValueError('Native arm refinement must be staged at the actual execution lookup path')
    output.mkdir(exist_ok=False)
    for path in raw_plan.iterdir():
        if path.is_file() and path.name not in ('commands.npz', 'plan.json'):
            shutil.copyfile(path, output/path.name)
    with np.load(raw_plan/'commands.npz', allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    execution_spec = dict(task=candidate['task'], scene=_binding(scene_path),
        fixed_release=_binding(release_spec_path),
        native_arm_collision_refinement=arm_collision_refinement,
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
    metadata = plan.setdefault('author_collection_binding', {})
    if not isinstance(metadata, dict):
        raise ValueError('Transport collection metadata must be a dictionary')
    for key, expected in dict(candidate=candidate, collection_candidate_sha256=candidate['candidate_sha256'],
                              task=candidate['task']).items():
        if key in metadata and metadata[key] != expected:
            raise ValueError('Transport plan and original collection candidate metadata differ: '+key)
        metadata.setdefault(key, expected)
    metadata.update(command_binding=command_binding,
        bound_metadata_only_original_command_arrays_unchanged=True)
    plan['native_arm_collision_refinement'] = arm_collision_refinement
    plan.setdefault('extra_input_sha256', {}).update({binding['path']: binding['sha256']
        for binding in _arm_refinement_inputs(arm_collision_refinement).values()})
    transport.save(output/'plan.json', plan)
    transport.save(output/'COMMAND_BINDING.json', command_binding)
    return command_binding


def prepare(scene_path, representation_dir, observation, gesture_id, source_type, seed,
            calibration, cloud_manifest, backend, out, *, archive_sources=None,
            release_spec_path=None, home_route_config=None, home_route_strategy='cartesian_six'):
    """Prepare exactly the original post-collision index0; preserve every null."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    result = dict(schema=SCHEMA, status='preparing', stage='freeze', complete=False,
                  output=str(out), physics_steps=0, new_CGR_forwards=0, training_updates=0, new_labels=0)
    transport.save(out/'PIPELINE_RESULTS.json', result)
    try:
        if release_spec_path is None:
            raise ValueError('Explicit fixed release_spec_path required; no object-relative release default')
        motion_policy = source_collection_motion_policy()
        paths = dict(scene=scene_path, observation=observation, calibration=calibration,
                     hand_cloud_manifest=cloud_manifest, backend=backend,
                     collection_motion_source=COLLECTION_SOURCE, fixed_release=release_spec_path)
        if home_route_config is not None:
            if home_route_strategy not in HOME_ROUTE_MODULES:
                raise ValueError('Explicit supported HOME route strategy required')
            paths['home_route_config'] = home_route_config
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
        if home_route_config is not None:
            route_module = HOME_ROUTE_MODULES[home_route_strategy]
            sources[route_module] = _binding(HERE/route_module)
            if home_route_strategy == 'joint_rrt_connect':
                sources['author_home_route.py'] = _binding(HERE/'author_home_route.py')
        scene = transport.validate_scene(inputs['scene']['path'], archive_sources)
        release_spec = transport.load_release_spec(inputs['scene']['path'],
            inputs['fixed_release']['path'], scene=scene)
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
        arm_refinement = stage_arm_collision_refinement(inputs['backend']['path'], out)
        inputs.update(_arm_refinement_inputs(arm_refinement))
        derived = dict(config, collision_refinement=primary.get('collision_refinement'),
                       collision_refinement_sha256=primary.get('collision_refinement_sha256'),
                       original_backend_pregrasp_distance_m=config.get('pregrasp_distance_m'),
                       pregrasp_distance_m=motion_policy['approach_distance_m'],
                       collection_motion_policy=motion_policy,
                       restore_object_initial_state=True)
        transport.save(out/'BACKEND.json', derived)
        protocol = dict(schema=SCHEMA, inputs=inputs, sources=sources, domain=DOMAIN,
            collection_motion_policy=motion_policy,
            fixed_release=inputs['fixed_release'],
            release_pose_rule=release_spec['pose_rule'],
            home_route=dict(enabled=home_route_config is not None,
                strategy=home_route_strategy if home_route_config is not None else None,
                config=inputs.get('home_route_config'),
                scope='Optional fixed-budget HOME-only route; unchanged candidate, 4cm approach and source closure/hold; full scene geometry'),
            capture_source_bindings=capture_sources,
            scene_archive_sources={str(Path(k).resolve()): str(Path(v).resolve())
                                   for k, v in (archive_sources or {}).items()},
            gesture_id=int(gesture_id), source_reference_type=int(source_type), seed=int(seed),
            derived_backend=_binding(out/'BACKEND.json'),
            native_arm_collision_refinement=arm_refinement,
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
            arm = prepare_approach(out/'pool', out/'arm', backend_config=out/'BACKEND.json',
                scene_path=inputs['scene']['path'], home_route_config=home_route_config,
                archive_sources=archive_sources, home_route_strategy=home_route_strategy)
            result['arm_preparation'] = arm
            if arm['status'] != 'ready':
                result.update(status=arm['status'], stage='fixed_candidate_arm', complete=True)
            else:
                result['stage'] = 'transport'
                transport.save(out/'PIPELINE_RESULTS.json', result)
                moved = transport.prepare_transport(arm['plan_dir'], inputs['scene']['path'], out/'transport_raw',
                    archive_sources=archive_sources, release_spec_path=inputs['fixed_release']['path'])
                result['transport_preparation'] = moved
                if not moved['admitted']:
                    result.update(status=moved['status'], stage='transport', complete=True)
                else:
                    binding = _bind_transport_plan(out/'transport_raw', out/'transport_bound', candidate,
                        inputs['calibration']['path'], inputs['scene']['path'], inputs['fixed_release']['path'],
                        arm_collision_refinement=arm_refinement)
                    ready = dict(schema=SCHEMA, status='ready', protocol=_binding(out/'PIPELINE_PROTOCOL.json'),
                        scene=inputs['scene'], candidate=_binding(out/'pool/selected_author_candidate.json'),
                        fixed_release=inputs['fixed_release'],
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


def prepare_from_frozen_candidate_plan(prepared_dir, routed_plan_dir, out, *,
                                       release_spec_path, archive_sources=None):
    """Continue an existing immutable candidate with an already prepared HOME.

    This performs only new transport preparation. It never calls collection,
    inference, native approach IK or HOME search, and keeps separate receipts
    from the original pipeline preparation.
    """
    prepared_dir, routed_plan_dir, out = (Path(p).resolve()
        for p in (prepared_dir, routed_plan_dir, out))
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    result = dict(schema='anydex_boya_author_component_integration_v1', status='preparing',
        complete=False, stage='freeze', physics_steps=0, new_labels=0, training_updates=0,
        new_CGR_forwards=0, new_collection_collision_queries=0, new_native_approach_IK=0,
        new_HOME_search_calls=0, reused_preparation=str(prepared_dir), output=str(out))
    transport.save(out/'COMPONENT_INTEGRATION_RESULTS.json', result)
    try:
        old_protocol_path = prepared_dir/'PIPELINE_PROTOCOL.json'
        old = transport.read(old_protocol_path)
        old_result = transport.read(prepared_dir/'PIPELINE_RESULTS.json')
        if (old.get('schema') != SCHEMA or not old_result.get('complete') or old_result.get('physics_steps') != 0
                or old_result.get('protocol_sha256') != transport.sha(old_protocol_path)
                or old.get('collection_motion_policy', {}).get('approach_distance_m') != .04
                or old.get('candidate_budget') != 1 or old.get('fallback_candidate') is not False):
            raise ValueError('Complete zero-physics fixed-first source preparation required')
        _verify(old['inputs'])
        if (Path(old['derived_backend']['path']).resolve() != prepared_dir/'BACKEND.json'
                or transport.sha(old['derived_backend']['path']) != old['derived_backend']['sha256']):
            raise ValueError('Original derived backend changed')
        scene_path = Path(old['inputs']['scene']['path'])
        calibration = Path(old['inputs']['calibration']['path'])
        scene = transport.validate_scene(scene_path, archive_sources)
        release = transport.load_release_spec(scene_path, release_spec_path, scene=scene)
        pool = prepared_dir/'pool'
        candidate_path = pool/'selected_author_candidate.json'
        candidate = transport.read(candidate_path)
        generation = transport.read(pool/'generation.json')
        if (collector._candidate_id(candidate) != candidate['candidate_sha256']
                or candidate['candidate_sha256'] != old_result['candidate_sha256']
                or generation.get('complete') is not True or generation.get('selected_indices') != [0]
                or generation['selected_candidate_sha256'] != transport.sha(candidate_path)
                or generation['candidates_sha256'] != transport.sha(pool/'candidates.npz')):
            raise ValueError('Original frozen first candidate/pool changed')
        seed_dir = prepared_dir/'arm/native'
        if old_result.get('arm_preparation', {}).get('plan_sha256') != transport.sha(seed_dir/'plan.json'):
            raise ValueError('Original approach seed differs from its completed source receipt')
        seed = transport.read(seed_dir/'plan.json')
        plan = transport.read(routed_plan_dir/'plan.json')
        for directory, value in ((seed_dir, seed), (routed_plan_dir, plan)):
            if (not value.get('complete') or value.get('approach_only') is not True
                    or value['arguments'].get('pregrasp_distance_m') != source_collection_motion_policy()['approach_distance_m']
                    or value['commands_sha256'] != transport.sha(directory/'commands.npz')):
                raise ValueError('Bound complete source-4cm native approach plan required')
        if plan.get('admitted') is not True:
            raise ValueError('Only an admitted HOME route may continue to transport')
        route = plan.get('author_home_route') or plan.get('author_joint_home_route')
        if (not route or route.get('original_seed_plan_sha256') != transport.sha(seed_dir/'plan.json')
                or transport.sha(route['protocol_path']) != route['protocol_sha256']):
            raise ValueError('HOME route must bind the exact original immutable seed')
        if (plan['candidates_sha256'] != generation['candidates_sha256']
                or plan['generation_sha256'] != transport.sha(pool/'generation.json')):
            raise ValueError('Routed HOME plan candidate identity differs from original pool')
        geometry_path = routed_plan_dir/'path_geometry.json'
        if (transport.sha(geometry_path) != plan['path_geometry_sha256']
                or transport.read(geometry_path).get('ok') is not True):
            raise ValueError('Admitted full-scene routed geometry required')
        preservation = _verify_home_preservation(seed_dir, routed_plan_dir)
        _verify_plan_dependencies(plan)
        arm_refinement = stage_arm_collision_refinement(old['derived_backend']['path'], out)
        contract = ActionContract(calibration)
        with np.load(routed_plan_dir/'commands.npz', allow_pickle=False) as z:
            arrays = {name: z[name].copy() for name in z.files}
        names = list(map(str, arrays['joint_names']))
        motors = arrays['joint_target'][:, [names.index(name) for name in contract.names]]
        expanded, expansion = collector.expanded_planned_hand(arrays, contract)
        numeric = contract.validate_command_arrays(expanded, slot_id=candidate['slot_id'], motor_targets=motors)
        mapped = contract.mapped_palm(candidate['slot_id'], candidate['actual_row15_camera'],
            candidate['world_T_camera'], extra_depth_m=candidate['extra_depth_m'])
        if float(abs(arrays['desired_palm'][-1]-mapped).max()) > 2e-6:
            raise ValueError('Routed approach endpoint differs from original candidate')
        numeric['planned_mimic_expansion'] = expansion
        inputs = dict(scene=_binding(scene_path), calibration=_binding(calibration),
            original_backend=old['derived_backend'],
            fixed_release=_binding(release_spec_path), original_protocol=_binding(old_protocol_path),
            original_results=_binding(prepared_dir/'PIPELINE_RESULTS.json'),
            original_candidate=_binding(candidate_path), original_pool=_binding(pool/'candidates.npz'),
            original_generation=_binding(pool/'generation.json'), original_seed=_binding(seed_dir/'plan.json'),
            original_commands=_binding(seed_dir/'commands.npz'), routed_plan=_binding(routed_plan_dir/'plan.json'),
            routed_commands=_binding(routed_plan_dir/'commands.npz'), routed_geometry=_binding(geometry_path),
            route_protocol=_binding(route['protocol_path']))
        inputs.update(_arm_refinement_inputs(arm_refinement))
        sources = {name: _binding(HERE/name) for name in ('author_pipeline.py', 'author_transport.py',
            'author_collect.py', 'author_action_semantics.py', 'author_collection_bridge.py')}
        for path, digest in plan.get('source_sha256', {}).items():
            sources['routed:'+path] = dict(path=path, sha256=digest)
        capture = transport.scene_source_bindings(scene, archive_sources)
        protocol = dict(schema=result['schema'], inputs=inputs, sources=sources,
            capture_source_bindings=capture, scene_archive_sources=archive_sources or {},
            original_pipeline_is_reused_not_rerun=True, candidate_budget=1, fallback_candidate=False,
            fixed_release=inputs['fixed_release'], release_pose_rule=release['pose_rule'],
            native_arm_collision_refinement=arm_refinement,
            home_preservation=preservation, numerical_action_validation=numeric,
            route_is_prepared_before_this_transport=True, task=candidate['task'],
            stages=['verify_existing_candidate_and_HOME', 'new_transport_preparation', 'optional_once_execution'])
        _exclusive_json(out/'COMPONENT_INTEGRATION_PROTOCOL.json', protocol)
        result.update(stage='transport', protocol_sha256=transport.sha(out/'COMPONENT_INTEGRATION_PROTOCOL.json'),
                      candidate_sha256=candidate['candidate_sha256'])
        transport.save(out/'COMPONENT_INTEGRATION_RESULTS.json', result)
        moved = transport.prepare_transport(routed_plan_dir, scene_path, out/'transport_raw',
            archive_sources=archive_sources, release_spec_path=release_spec_path)
        result['transport_preparation'] = moved
        if not moved['admitted']:
            result.update(status=moved['status'], complete=True)
        else:
            binding = _bind_transport_plan(out/'transport_raw', out/'transport_bound', candidate,
                calibration, scene_path, release_spec_path, arm_collision_refinement=arm_refinement)
            ready = dict(schema=result['schema'], status='ready',
                protocol=_binding(out/'COMPONENT_INTEGRATION_PROTOCOL.json'), scene=inputs['scene'],
                candidate=inputs['original_candidate'], fixed_release=inputs['fixed_release'],
                plan=_binding(out/'transport_bound/plan.json'), commands=_binding(out/'transport_bound/commands.npz'),
                command_binding=_binding(out/'transport_bound/COMMAND_BINDING.json'),
                collection_candidate_sha256=candidate['candidate_sha256'],
                preparation_receipt='COMPONENT_INTEGRATION_RESULTS.json',
                execution_allowed_once=True, executed=False)
            _exclusive_json(out/'READY.json', ready)
            result.update(status='ready', stage='ready', complete=True,
                          ready_sha256=transport.sha(out/'READY.json'), command_binding=binding)
        _verify(inputs); _verify(sources); _verify(capture); _verify_plan_dependencies(plan)
        result['wall_s'] = time.monotonic()-began
        transport.save(out/'COMPONENT_INTEGRATION_RESULTS.json', result)
        return result
    except BaseException as error:
        result.update(status='infrastructure_failure', complete=False, error=repr(error),
                      traceback=traceback.format_exc(), wall_s=time.monotonic()-began)
        transport.save(out/'COMPONENT_INTEGRATION_RESULTS.json', result)
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
    receipt_name = ready.get('preparation_receipt', 'PIPELINE_RESULTS.json')
    if receipt_name not in ('PIPELINE_RESULTS.json', 'COMPONENT_INTEGRATION_RESULTS.json'):
        raise ValueError('Unknown prepared execution receipt')
    preparation = transport.read(out/receipt_name)
    if preparation.get('status') != 'ready' or preparation.get('ready_sha256') != transport.sha(out/'READY.json'):
        raise ValueError('Only the completed unchanged ready preparation may execute')
    protocol = transport.read(ready['protocol']['path'])
    if not protocol.get('inputs', {}).get('fixed_release') or ready.get('fixed_release') != protocol['inputs']['fixed_release']:
        raise ValueError('Prepared execution requires its explicitly bound fixed release sidecar')
    _verify(protocol['inputs']); _verify(protocol['sources'])
    _verify(protocol.get('capture_source_bindings', {}))
    ready_bindings = {key: ready[key] for key in ('protocol', 'scene', 'candidate', 'plan',
                      'commands', 'command_binding', 'fixed_release')}
    _verify(ready_bindings)
    transport.load_release_spec(ready['scene']['path'], ready['fixed_release']['path'])
    candidate = transport.read(ready['candidate']['path'])
    plan = transport.read(ready['plan']['path'])
    command_binding = transport.read(ready['command_binding']['path'])
    _verify_plan_dependencies(plan)
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
        # No training-eligible record may exist before every post-execution
        # integrity check succeeds. Costs/errors still precede process close.
        _verify(protocol['inputs']); _verify(protocol['sources'])
        _verify(protocol.get('capture_source_bindings', {})); _verify(ready_bindings)
        _verify_plan_dependencies(plan)
        record = collector.collect_execution(candidate, command_binding, execution/'TRANSPORT_EVALUATION.json')
        transport.save(out/'COLLECTION_RECORD.json', record)
        receipt.update(status=evaluation['status'], complete=evaluation['complete'],
            physics_steps=evaluation['total_physics_steps'], label=record['label'], label_state=record['label_state'],
            new_labels=int(record['label'] is not None), wall_s=time.monotonic()-began)
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
    for name in ('scene', 'representation-dir', 'observation', 'calibration', 'cloud-manifest', 'backend', 'output-dir', 'release-spec'):
        preparation.add_argument('--'+name, type=Path, required=True)
    preparation.add_argument('--gesture-id', type=int, required=True, choices=(1, 6, 7, 8))
    preparation.add_argument('--source-type', type=int, required=True, choices=(1, 2, 3, 4))
    preparation.add_argument('--seed', type=int, required=True)
    preparation.add_argument('--execute', action='store_true')
    preparation.add_argument('--archive-sources', type=Path, help='Original capture source path -> exact historical snapshot path JSON')
    preparation.add_argument('--home-route-config', type=Path, help='Opt-in fixed-budget full-scene HOME-only routing policy')
    preparation.add_argument('--home-route-strategy', choices=tuple(HOME_ROUTE_MODULES), default='cartesian_six')
    continuation = commands.add_parser('prepare-continuation')
    for name in ('prepared-dir', 'routed-plan-dir', 'output-dir', 'release-spec'):
        continuation.add_argument('--'+name, type=Path, required=True)
    continuation.add_argument('--archive-sources', type=Path)
    execution = commands.add_parser('execute')
    execution.add_argument('--prepared-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        value = prepare(args.scene, args.representation_dir, args.observation, args.gesture_id,
            args.source_type, args.seed, args.calibration, args.cloud_manifest, args.backend, args.output_dir,
            archive_sources=transport.read(args.archive_sources) if args.archive_sources else None,
            release_spec_path=args.release_spec, home_route_config=args.home_route_config,
            home_route_strategy=args.home_route_strategy)
        print(json.dumps(value, indent=2))
        if args.execute and value['status'] == 'ready':
            print(json.dumps(execute(args.output_dir), indent=2))
    elif args.stage == 'prepare-continuation':
        print(json.dumps(prepare_from_frozen_candidate_plan(args.prepared_dir, args.routed_plan_dir,
            args.output_dir, release_spec_path=args.release_spec,
            archive_sources=transport.read(args.archive_sources) if args.archive_sources else None), indent=2))
    else:
        print(json.dumps(execute(args.prepared_dir), indent=2))


if __name__ == '__main__':
    main()
