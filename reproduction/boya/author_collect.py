"""Opt-in public DH3 collection preparation for the declared Boya simulator port.

The cache must contain original unfiltered parser rows, not an old Boya pool.
This module never starts physics, a model or training. Source collection flips
columns 0/1; its recorded flip flag is canonicalized by the source dataset rule.
That differs from DH3 deployment's columns 1/2 and is preserved explicitly.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

import author_collision as collision
import author_protocol as author
from author_hand_clouds import HandCloudLibrary
from asset_adapter import rigid, sha
from trial_record import array_ref, json_sha

SCHEMA = 'anydex_boya_author_collection_protocol_v1'
CANDIDATE_SCHEMA = 'anydex_boya_author_collection_candidate_v1'
CANONICAL_SOURCE_SHA = collision.DEPLOYMENT_SOURCE_SHA
GESTURES = (1, 6, 7, 8)


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def source_domain(raw, world_T_camera, config):
    """Only sensor/rig workspace, width and approach predicates, before top500.

    `source_optical` uses the exact public get_grasp masks. `declared_world_rig`
    explicitly substitutes the camera/workspace; neither mode tests any hand,
    table clearance, LUT, extra depth, learned score or physical outcome.
    """
    rows = np.asarray(raw)
    if (rows.ndim != 2 or rows.shape[1] != 15 or not np.issubdtype(rows.dtype, np.floating)
            or not np.isfinite(rows).all() or (rows[:, 2] < 0).any()):
        raise ValueError('Finite original unfiltered Nx15 parser rows required')
    rotation = rows[:, 3:12].reshape(-1, 3, 3).astype(float)
    if (not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=2e-4, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1., atol=2e-4, rtol=0)):
        raise ValueError('Source parser rotations must be proper SO(3)')
    world = rigid(world_T_camera)
    mode = config.get('mode')
    if mode == 'source_optical':
        if set(config) != {'mode', 'sensor_adapter'}:
            raise ValueError('Source optical profile takes only mode and explicit sensor_adapter')
        width = (rows[:, 1] > .01) & (rows[:, 1] < .099)
        workspace = ((rows[:, 12] > -.25) & (rows[:, 12] < .25)
                     & (rows[:, 13] > -.205) & (rows[:, 13] < .03))
        approach = rows[:, 9] > .9
    elif mode == 'declared_world_rig':
        if set(config) != {'mode', 'sensor_adapter', 'world_workspace_aabb_m',
                           'approach_world_cone', 'raw_width_bounds_m'}:
            raise ValueError('Complete explicit world-rig substitution required')
        from boya_prehead_domain import apply_boya_domain
        domain = {k: config[k] for k in ('world_workspace_aabb_m', 'approach_world_cone', 'raw_width_bounds_m')}
        result = apply_boya_domain(rows, world, domain, source_provenance={'caller': 'author_collect source cache bindings'})
        width = result.gate_masks['raw_width']
        workspace = result.gate_masks['world_workspace']
        approach = result.gate_masks['world_approach']
    else:
        raise ValueError('Explicit source_optical or declared_world_rig profile required')
    sensor = config['sensor_adapter']
    if (not isinstance(sensor, dict) or not sensor.get('description')
            or not sensor.get('input_depth_range_m') or sensor.get('cache_is_unfiltered_parser') is not True):
        raise ValueError('Declare sensor depth-range adaptation and unfiltered parser cache provenance')
    depth_range = np.asarray(sensor['input_depth_range_m'], dtype=float)
    if depth_range.shape != (2,) or not np.isfinite(depth_range).all() or not 0 < depth_range[0] < depth_range[1]:
        raise ValueError('Finite positive sensor input depth range required')
    keep = width & workspace & approach
    return dict(keep_indices=np.flatnonzero(keep), masks=dict(width=width, workspace=workspace, approach=approach),
        counts=dict(input=len(rows), width=int(width.sum()), workspace=int(workspace.sum()),
                    approach=int(approach.sum()), joint=int(keep.sum())), config=_json(config),
        hand_or_table_gate_before_top500=False, learned_score_used=False)


def canonical_features(raw2261, source_flip):
    """Original saved if_flip + source inference/dataset half-swap and angle roll.

    Do not call the old forcedflip candidate helper: collection's actual pose
    flip differs. The source flags still drive the published feature convention.
    """
    raw = np.asarray(raw2261)
    flips = np.asarray(source_flip, dtype=bool)
    if raw.ndim != 2 or raw.shape[1] != 2261 or flips.shape != (len(raw),) or not np.isfinite(raw).all():
        raise ValueError('Finite paired Nx2261 features and actual source flip flags required')
    if sha(author.ROOT/'robot_dh3.py') != CANONICAL_SOURCE_SHA:
        raise ValueError('Pinned published canonicalization source changed')
    # definitions() checks source hashes only for sources included in PINS.
    import ast
    node = next(n for n in ast.parse((author.ROOT/'robot_dh3.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'get_graspgroup_features')
    scope = dict(np=np, copy=copy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(author.ROOT/'robot_dh3.py'), 'exec'), scope)
    feature_input = np.c_[raw.copy(), flips]
    result = scope['get_graspgroup_features'](feature_input, None)
    cgr = np.asarray(result['grasp_preds_features']).copy()
    angle, depth = result['grasp_angles'], result['grasp_depths']
    if ((angle < 0).any() or (angle >= 24).any() or (depth < 0).any() or (depth >= 5).any()
            or cgr.shape != (len(raw), 480)):
        raise ValueError('Released 24 parser angle / 5 depth convention required')
    return dict(cgr480=cgr, angle_index=angle, raw_depth_index=depth)


def _row17(raw):
    return np.c_[raw[:, :2], np.full(len(raw), .03, dtype=raw.dtype), raw[:, 2:],
                 np.full(len(raw), -1, dtype=raw.dtype)]


def _candidate_id(candidate):
    return json_sha({k: v for k, v in candidate.items() if k != 'candidate_sha256'})


def validate_cache_receipt(receipt, bindings, raw_count, sensor_adapter):
    """Reject old filtered pools, augmentation caches and unrelated observations."""
    raw_info = receipt.get('raw_candidates', {})
    if (receipt.get('status') != 'representation_forward_complete'
            or receipt.get('test_time_augmentation_branches') != 1
            or raw_info.get('count') != raw_count
            or raw_info.get('status') != 'original parse_preds output before rig-specific width/approach/workspace and collision filtering'):
        raise ValueError('Original single-forward unfiltered parser receipt required for collection')
    for name, key in (('raw_camera_candidates.npz', 'raw_candidates'),
                      ('representation_outputs.npz', 'representation_outputs')):
        if receipt.get('output_files', {}).get(name, {}).get('sha256') != bindings[key]['sha256']:
            raise ValueError('CGR receipt does not bind the actual unfiltered cache bytes')
    if receipt.get('scene_binding', {}).get('observation_sha256') != bindings['observation']['sha256']:
        raise ValueError('CGR cache and observation describe different scenes')
    args = receipt.get('args', {})
    if sensor_adapter['input_depth_range_m'] != [args.get('depth_min'), args.get('depth_max')]:
        raise ValueError('Declared sensor depth range differs from actual CGR input preparation')


def prepare_collection(raw_candidates, raw_features, scene_points, *, world_T_camera,
                       calibration, library, gesture_id, source_reference_type, seed,
                       domain, scene_binding, input_bindings, action_contract=None):
    """Prepare a top10 list and the source index-0 choice, without execution.

    `scene_points` are the original CGR sparse input camera points, without a
    second downsample or object-only crop. Candidate JSON is the downstream
    transport API; poses are optical/world, 16+4 commands require later binding.
    """
    if gesture_id not in GESTURES or type(gesture_id) is not int:
        raise ValueError('Explicit supported Boya gesture 1/6/7/8 required')
    if not isinstance(scene_binding, dict) or not scene_binding.get('scene_id'):
        raise ValueError('Bind the exact observation/initial scene identity')
    if not isinstance(input_bindings, dict) or not input_bindings:
        raise ValueError('Source/cache/calibration bindings required')
    world = rigid(world_T_camera)
    raw = np.asarray(raw_candidates)
    features = np.asarray(raw_features)
    if features.shape != (len(raw), 2261) or not np.isfinite(features).all():
        raise ValueError('Paired raw Nx2261 original features required')
    if not np.allclose(raw[:, 2], features[:, -1], atol=1e-8, rtol=0):
        raise ValueError('Original row and paired feature raw depth differ')
    scene = np.asarray(scene_points)
    if scene.ndim != 2 or scene.shape[1] != 3 or not len(scene) or not np.isfinite(scene).all():
        raise ValueError('Nonempty original sparse-input source camera point cloud required')
    filtered = source_domain(raw, world, domain)
    keep = filtered['keep_indices']
    source = author.collection_candidates(_row17(raw[keep]), features[keep], reference='dh3',
        reference_type=source_reference_type, seed=seed)
    # Source top500 indices are local to the sensor/workspace domain.
    source_indices = keep[source['source_indices']]
    source_rows = source['rows']
    source_features = np.asarray(source['features'])[:, :2261].copy()
    flips = source['if_flip']
    canonical = canonical_features(source_features, flips)
    if len(source_rows):
        found = collision.detect(source_rows, np.full(len(source_rows), gesture_id), source['extra_depth_m'],
            scene, calibration=calibration, library=library, profile='dh3_collection')
        accepted = np.asarray(found['accepted_source_indices'], dtype=int)
        # The source ranks AFTER collision, truncates to 10 and picks index 0.
        order = np.argsort(source_rows[accepted, 0])[::-1][:10]
        selected = accepted[order]
        mapped = {int(idx): item for idx, item in zip(found['hand'].ids, found['hand'].mapped)}
        mapping_count = int(found['mapping_available_mask'].sum())
        width_count = int(np.asarray(found['source_width_mask'], dtype=bool).sum())
    else:
        accepted = selected = np.empty(0, dtype=int)
        mapped = {}; mapping_count = width_count = 0
    candidates = []
    for rank, idx in enumerate(selected):
        idx = int(idx); src = int(source_indices[idx]); row = source_rows[idx]
        hand = mapped[idx]
        before = rigid(hand.palm_before_depth)
        axis = row[4:13].reshape(3, 3)[:, 0].astype(float)
        axis /= np.linalg.norm(axis)
        # Exactly mirror original load_meshes_pcls float32 translation matrix.
        total_depth = float(row[3])+float(source['extra_depth_m'][idx])
        displacement = np.asarray(axis*total_depth, dtype=np.float32).astype(float)
        after = before.copy(); after[:3, 3] += displacement
        row15 = np.r_[row[:2], row[3:16]]
        entry = dict(schema=CANDIDATE_SCHEMA, candidate_id=f'{scene_binding["scene_id"]}:g{gesture_id}:r{src}:seed{seed}',
            source_row_index=src, source_top500_rank=idx, collision_rank=rank,
            gesture_id=gesture_id, source_reference_type=source_reference_type,
            score=float(row[0]), source_if_flip=bool(flips[idx]),
            source_row15_camera=raw[src].tolist(), actual_row15_camera=row15.tolist(),
            actual_row17_camera=row.tolist(), source_feature2261=features[src].tolist(),
            source_feature_identity=array_ref(features[src]), canonical_cgr480=canonical['cgr480'][idx].tolist(),
            canonical_cgr480_identity=array_ref(canonical['cgr480'][idx]),
            canonicalization=dict(source_sha256=CANONICAL_SOURCE_SHA, applied_once=True,
                source_collection_flip_columns=[0, 1], published_feature_flip_flag=bool(flips[idx]),
                deployment_pose_flip_columns=[1, 2], pose_and_feature_conventions_not_silently_reconciled=True),
            raw_angle_index=int(canonical['angle_index'][idx]), raw_depth_index=int(canonical['raw_depth_index'][idx]),
            raw_width_m=float(row[1]), expanded_width_m=float(row[1]*1.7),
            raw_depth_m=float(row[3]), extra_depth_m=float(source['extra_depth_m'][idx]),
            extra_depth_index=int(round(float(source['extra_depth_m'][idx])*100)), total_depth_m=total_depth,
            slot_id=hand.slot['slot_id'], slot_sha256=json_sha(hand.slot),
            calibrated_reference_width_m=float(hand.slot['reference_width_m']),
            camera_T_palm_before_depth=before.tolist(), camera_T_palm=after.tolist(),
            world_T_palm=rigid(world@after).tolist(), world_T_camera=world.tolist(),
            normalized_approach_camera=axis.tolist(), applied_depth_translation_camera_m=displacement.tolist(),
            source_cloud_identity=array_ref(scene), scene_binding=_json(scene_binding),
            input_bindings=_json(input_bindings),
            collision_profile=dict(name='dh3_collection', **collision.PROFILES['dh3_collection']),
            action_contract=None, command_binding=None, evaluation_binding=None,
            label=None, label_state='unexecuted', physical_semantics_validated=False,
            task='declared_boya_clutter_transport_release_v1')
        if action_contract is not None:
            entry['action_contract'] = _json(action_contract.for_slot(hand.slot['slot_id']))
        entry['candidate_sha256'] = _candidate_id(entry)
        candidates.append(entry)
    counts = dict(filtered['counts'], top500=len(source_rows), mapping_available=mapping_count,
                  width_admitted=width_count, collision_admitted=len(accepted), top10=len(candidates))
    result = dict(schema='anydex_boya_author_collection_prepared_v1', status='prepared' if candidates else 'no_geometry',
        source_profile='public DH3 collection', selection='original descending score after geometry; top10; fixed index0',
        requested_gesture_id=gesture_id, source_reference_type=source_reference_type, seed=seed,
        domain=filtered['config'], counts=counts, candidates=candidates,
        selected_candidate_sha256=candidates[0]['candidate_sha256'] if candidates else None,
        source_and_input_bindings=_json(input_bindings), scene_binding=_json(scene_binding),
        hand_or_table_gate_before_top500=False, learned_head_forward_count=0,
        new_CGR_forwards=0, physics_steps=0, training_updates=0, new_labels=0,
        formal_collection_ready=False, physical_semantics_validated=False,
        boundary='Real-cache preparation only; arm planning, dynamic clutter, transport/release and label acceptance remain required')
    trace = dict(source_indices=source_indices, actual_rows17_camera=source_rows,
                 source_features2261=source_features, source_if_flip=flips,
                 extra_depth_m=source['extra_depth_m'], canonical_cgr480=canonical['cgr480'],
                 collision_accepted_top500_indices=accepted, top10_top500_indices=selected,
                 source_scene_camera=scene, domain_keep_indices=keep)
    return result, trace


def expanded_planned_hand(arrays, action_contract):
    """Expand source mimic equations for auditing the 24-column arm plan.

    Native execution still receives its original active-joint columns and
    computes the same four passives. This is a planned-command audit only;
    it does not claim to have observed physical 28-DOF commands.
    """
    names = list(map(str, arrays['joint_names']))
    targets = np.asarray(arrays['joint_target'])
    if len(names) != len(set(names)) or targets.ndim != 2 or targets.shape[1] != len(names):
        raise ValueError('One finite unique named column per planned joint required')
    result = dict(arrays)
    added = []
    for name, (driver, coefficient, offset) in action_contract.mimics.items():
        if name in names: continue
        if driver not in names: raise ValueError('Mimic driver missing from actual plan')
        values = (coefficient*targets[:, names.index(driver)]+offset).astype(targets.dtype)
        targets = np.column_stack((targets, values)); names.append(name); added.append(name)
    result['joint_names'] = np.asarray(names)
    result['joint_target'] = targets
    return result, dict(native_planned_columns=len(arrays['joint_names']), derived_mimic_columns=added,
                        physical_joint_commands_observed=False)


def write_bound_commands(candidate, arrays, output, *, execution_spec, action_contract=None):
    """Bind actual transport commands before simulation; no implicit execution."""
    if _candidate_id(candidate) != candidate['candidate_sha256']:
        raise ValueError('Prepared candidate changed')
    if not execution_spec or execution_spec.get('task') != candidate['task']:
        raise ValueError('Explicit compatible transport execution specification required')
    if any(np.asarray(v).dtype.hasobject for v in arrays.values()):
        raise ValueError('Object/pickled command arrays forbidden')
    reserved = {'collection_candidate_sha256', 'collection_execution_spec_sha256'}
    if reserved.intersection(arrays):
        raise ValueError('Reserved command binding metadata')
    action_validation = None
    if action_contract is not None:
        motors = None
        validation_arrays = arrays
        expansion = None
        if hasattr(action_contract, 'names'):
            joint_names = list(map(str, arrays['joint_names']))
            motors = np.asarray(arrays['joint_target'])[:, [joint_names.index(n) for n in action_contract.names]]
            validation_arrays, expansion = expanded_planned_hand(arrays, action_contract)
        action_validation = action_contract.validate_command_arrays(validation_arrays, slot_id=candidate['slot_id'],
            motor_targets=motors,
            candidate_row15=candidate['actual_row15_camera'], world_T_camera=candidate['world_T_camera'],
            extra_depth_m=candidate['extra_depth_m'])
        action_validation['planned_mimic_expansion'] = expansion
    output = Path(output)
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays,
            collection_candidate_sha256=np.asarray(candidate['candidate_sha256']),
            collection_execution_spec_sha256=np.asarray(json_sha(execution_spec)))
    bound = dict(candidate_sha256=candidate['candidate_sha256'], commands_path=str(output.resolve()),
                commands_sha256=sha(output), execution_spec=_json(execution_spec),
                action_contract_validated=action_contract is not None,
                action_validation=_json(action_validation))
    bound['command_binding_sha256'] = json_sha(bound)
    return bound


def collect_execution(candidate, command_binding, evaluation_path):
    """A complete bound transport outcome is needed; null/infra never labels 0."""
    if _candidate_id(candidate) != candidate['candidate_sha256']:
        raise ValueError('Prepared candidate changed')
    commands = Path(command_binding['commands_path'])
    if (json_sha({k: v for k, v in command_binding.items() if k != 'command_binding_sha256'})
            != command_binding['command_binding_sha256']
            or command_binding['candidate_sha256'] != candidate['candidate_sha256']
            or sha(commands) != command_binding['commands_sha256']):
        raise ValueError('Actual command file differs from prepared candidate binding')
    with np.load(commands, allow_pickle=False) as data:
        if str(data['collection_candidate_sha256'].item()) != candidate['candidate_sha256']:
            raise ValueError('Command metadata does not bind candidate')
        if str(data['collection_execution_spec_sha256'].item()) != json_sha(command_binding['execution_spec']):
            raise ValueError('Command metadata does not bind execution specification')
    path = Path(evaluation_path); evaluation = json.loads(path.read_text())
    if (evaluation.get('collection_candidate_sha256') != candidate['candidate_sha256']
            or evaluation.get('commands_sha256') != command_binding['commands_sha256']):
        raise ValueError('Executor must echo actual command and candidate SHA')
    label = None; reason = 'unexecuted_or_incomplete_transport'
    phases = {'approach', 'close', 'retreat', 'transport', 'release', 'post_release'}
    if (evaluation.get('infrastructure_error') or evaluation.get('error') or evaluation.get('errors')
            or evaluation.get('exception')):
        reason = 'infrastructure_error'
    elif (evaluation.get('complete') is True and evaluation.get('label_eligible') is True
            and type(evaluation.get('total_physics_steps')) is int and evaluation['total_physics_steps'] > 0
            and phases.issubset(evaluation.get('completed_phases', []))
            and evaluation.get('task') == candidate['task'] and type(evaluation.get('task_success')) is bool
            and command_binding.get('action_contract_validated') is True):
        label = int(evaluation['task_success']); reason = 'executed_transport_success' if label else 'executed_transport_failure'
    return dict(schema='anydex_boya_author_collection_record_v1', candidate=_json(candidate),
        command_binding=_json(command_binding), evaluation_binding=dict(path=str(path.resolve()), sha256=sha(path)),
        label=label, label_state=reason, task=candidate['task'],
        total_physics_steps=evaluation.get('total_physics_steps'), formal_training_eligible=label is not None)


def run_protocol(protocol_path, output):
    path, output = Path(protocol_path).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('Fresh collection preparation directory required')
    protocol = json.loads(path.read_text())
    if protocol.get('schema') != SCHEMA:
        raise ValueError('Explicit frozen author collection protocol required')
    required = {'raw_candidates', 'representation_outputs', 'representation_receipt',
                'observation', 'calibration', 'hand_cloud_manifest'}
    bindings = dict(protocol['inputs'])
    if not required.issubset(bindings): raise ValueError('All cache/observation/calibration/cloud bindings required')
    for entry in bindings.values():
        if sha(entry['path']) != entry['sha256']: raise ValueError('Frozen protocol input changed: '+entry['path'])
    observation = json.loads(Path(bindings['observation']['path']).read_text())
    with np.load(bindings['raw_candidates']['path'], allow_pickle=False) as data:
        raw, features = data['candidates'].copy(), data['candidate_features'].copy()
    with np.load(bindings['representation_outputs']['path'], allow_pickle=False) as data:
        scene = data['sparse_input_XYZ_features'].copy()
    receipt = json.loads(Path(bindings['representation_receipt']['path']).read_text())
    validate_cache_receipt(receipt, bindings, len(raw), protocol['domain']['sensor_adapter'])
    cal = json.loads(Path(bindings['calibration']['path']).read_text())
    library = HandCloudLibrary(bindings['hand_cloud_manifest']['path'])
    bindings.update(protocol=dict(path=str(path), sha256=sha(path)),
                    implementation=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    for name in ('author_protocol.py', 'author_collision.py', 'author_hand_clouds.py',
                 'asset_adapter.py', 'trial_record.py', 'boya_prehead_domain.py'):
        source_path = Path(__file__).with_name(name).resolve()
        bindings['implementation_'+source_path.stem] = dict(path=str(source_path), sha256=sha(source_path))
    output.mkdir(parents=True)
    begin = time.monotonic()
    action_contract = None
    if protocol.get('action_contract') is not None:
        action_path = Path(__file__).with_name('author_action_semantics.py').resolve()
        if sha(action_path) != protocol['action_contract']['implementation_sha256']:
            raise ValueError('Frozen action contract implementation changed')
        from author_action_semantics import ActionContract
        action_contract = ActionContract(cal)
        bindings['implementation_action_contract'] = dict(path=str(action_path), sha256=sha(action_path))
    result, arrays = prepare_collection(raw, features, scene, world_T_camera=observation['T_WC'],
        calibration=cal, library=library, gesture_id=protocol['gesture_id'],
        source_reference_type=protocol['source_reference_type'], seed=protocol['seed'],
        domain=protocol['domain'], scene_binding=protocol['scene_binding'],
        input_bindings=bindings, action_contract=action_contract)
    for entry in bindings.values():
        if sha(entry['path']) != entry['sha256']: raise ValueError('Input changed during preparation')
    np.savez_compressed(output/'PREPARATION_ARRAYS.npz', **arrays)
    result.update(wall_s=time.monotonic()-begin,
        trace=dict(path=str(output/'PREPARATION_ARRAYS.npz'), sha256=sha(output/'PREPARATION_ARRAYS.npz')),
        sources_and_inputs_unchanged=True)
    (output/'PREPARED.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'selected_candidate_sha256', 'wall_s')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    run_protocol(args.protocol, args.output_dir)
