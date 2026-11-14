"""Bridge a fixed author collection choice to the existing native arm planner.

No hand mapper, forcedflip helper, learned score, target matching or new ranking
is used here. One already-selected candidate becomes one planner pool row. The
object passed to the old planner is a compatibility body; additional clutter
and transport geometry must be checked by the new scene/transport backend.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from asset_adapter import rigid, sha
from author_action_semantics import ActionContract
from author_collect import CANDIDATE_SCHEMA, _candidate_id, expanded_planned_hand
from trial_record import json_sha

ROOT = Path(__file__).resolve().parents[2]


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def planner_rotation(pose):
    """Remove only parser/optical-camera roundoff for the strict arm SE3 API."""
    original = rigid(pose)
    proper = original.copy()
    u, _, vh = np.linalg.svd(original[:3, :3]); proper[:3, :3] = u@vh
    difference = float(abs(proper-original).max())
    if difference > 2e-6:
        raise ValueError('Planner requires a geometric change beyond parser roundoff')
    return proper, dict(method='SVD projection of float32 parser rotation only',
        max_matrix_roundoff=difference, translation_changed=False, depth_added_again=False)


def bridge_candidate(candidate, output, *, calibration, object_root, world_T_object,
                     object_scale=1., prepared_path=None, scene_manifest=None):
    """Write the minimal old-arm pool without selecting or relocating a target."""
    if candidate.get('schema') != CANDIDATE_SCHEMA or _candidate_id(candidate) != candidate['candidate_sha256']:
        raise ValueError('Unchanged selected author collection candidate required')
    output, calibration, object_root = map(lambda p: Path(p).resolve(), (output, calibration, object_root))
    if output.exists(): raise FileExistsError('Fresh planner pool required')
    if not np.isfinite(object_scale) or object_scale <= 0: raise ValueError('Positive source object scale required')
    contract = ActionContract(calibration)
    action = contract.for_slot(candidate['slot_id'])
    if candidate.get('action_contract') != action:
        raise ValueError('Collector and planner must bind the same current action contract')
    slot = contract.slots[candidate['slot_id']]
    if (json_sha(slot) != candidate['slot_sha256']
            or action['gesture_id'] != candidate['gesture_id']):
        raise ValueError('Candidate opening/gesture differs from source contract')
    palm, numeric = planner_rotation(candidate['world_T_palm'])
    source_palm = np.asarray(candidate['world_T_palm'], dtype=float)
    independent = contract.mapped_palm(candidate['slot_id'], candidate['actual_row15_camera'],
        candidate['world_T_camera'], extra_depth_m=candidate['extra_depth_m'])
    error = float(abs(source_palm-independent).max())
    if error > 2e-6: raise ValueError('Collector actual palm differs from independently bound action mapping')
    direction = np.asarray(candidate['world_T_camera'])[:3, :3]@np.asarray(candidate['normalized_approach_camera'])
    direction /= np.linalg.norm(direction)
    body = rigid(world_T_object)
    files = [calibration, Path(__file__).resolve(), Path(__file__).with_name('author_collect.py'),
        Path(__file__).with_name('author_action_semantics.py'), object_root/'mesh/simplified.obj', object_root/'urdf/coacd.urdf']
    for node in ET.parse(object_root/'urdf/coacd.urdf').findall('.//collision/geometry/mesh'):
        files.append((object_root/'urdf'/node.get('filename')).resolve())
    for entry in candidate['input_bindings'].values():
        if sha(entry['path']) != entry['sha256']: raise ValueError('Collector source changed before arm bridge')
        files.append(Path(entry['path']))
    for path, info in contract.calibration['source_lock']['source_files'].items():
        if sha(path) != info['sha256']: raise ValueError('Hand source changed before arm bridge')
        files.append(Path(path))
    if prepared_path is not None: files.append(Path(prepared_path).resolve())
    if scene_manifest is not None: files.append(Path(scene_manifest).resolve())
    output.mkdir(parents=True)
    _save(output/'selected_author_candidate.json', candidate)
    np.savez_compressed(output/'candidates.npz', independent_motor_names=np.asarray(action['independent_motor_names']),
        proposal_q=np.asarray([action['opened_independent_rad']]),
        proposal_q_close=np.asarray([action['closed_independent_rad']]),
        proposal_palm=np.asarray([palm]), proposal_approach_direction=np.asarray([direction]),
        world_T_object=body, collection_candidate_sha256=np.asarray(candidate['candidate_sha256']),
        canonical_cgr480=np.asarray([candidate['canonical_cgr480']], dtype=np.float32))
    _save(output/'world_object_pose.json', body.tolist())
    generation = dict(complete=True, kind='anydex_boya_author_collection_bridge_v1',
        gesture_id=candidate['gesture_id'], selected_indices=[0], scale=float(object_scale),
        candidates_sha256=sha(output/'candidates.npz'), input_sha256={str(p): sha(p) for p in files},
        records=[dict(source_candidate_index=candidate['source_row_index'], source_slot=candidate['slot_id'],
            if_flip=candidate['source_if_flip'], additional_depth_index=candidate['extra_depth_index'],
            additional_depth_m=candidate['extra_depth_m'], jaw_assignment=dict(thumb='th', opponent='ff'),
            collection_candidate_sha256=candidate['candidate_sha256'])],
        selected_candidate_json=str(output/'selected_author_candidate.json'),
        selected_candidate_sha256=sha(output/'selected_author_candidate.json'),
        compatibility_object_root=str(object_root), calibration=str(calibration),
        selection='One original post-geometry index0 candidate; no arm fallback or reranking',
        mapping= numeric | dict(independent_action_mapping_matrix_error=error),
        object_role='Explicit planner compatibility body only; no GT target association, segmentation or candidate filtering',
        additional_scene_collision_checked=False, physics_steps=0, training_labels=0,
        physical_semantics_validated=False, source_pose_relocated=False)
    _save(output/'generation.json', generation)
    return generation


def prepare_arm(pool, output, *, backend_config, timeout_s=360.):
    """Run only source native prepare for fixed pool index0; return ready/null."""
    pool, output, backend_config = map(lambda p: Path(p).resolve(), (pool, output, backend_config))
    if output.exists(): raise FileExistsError('Fresh arm preparation output required')
    output.mkdir(parents=True)
    cfg = json.loads(backend_config.read_text()); generation = json.loads((pool/'generation.json').read_text())
    candidate = json.loads((pool/'selected_author_candidate.json').read_text())
    if (not generation.get('complete') or generation.get('selected_indices') != [0]
            or sha(pool/'selected_author_candidate.json') != generation['selected_candidate_sha256']
            or _candidate_id(candidate) != candidate['candidate_sha256']
            or sha(pool/'candidates.npz') != generation['candidates_sha256']):
        raise ValueError('Fixed planner pool changed')
    # Compatibility object is explicitly supplied to bridge_candidate; never
    # choose one based on candidate proximity or known grasp success.
    object_root = Path(generation['compatibility_object_root'])
    command = [sys.executable, '-B', str(Path(__file__).with_name('native_arm.py')),
        '--stage', 'prepare', '--output-dir', str(output/'native'), '--candidates', str(pool), '--index', '0',
        '--object-root', str(object_root), '--world-object-pose', str(pool/'world_object_pose.json'),
        '--pregrasp-distance-m', str(cfg['pregrasp_distance_m']), '--device', 'cpu',
        '--arm-select-min-motion', '--vectorized-arm-broadphase']
    for key in ('robot_urdf', 'reference_root', 'hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config'):
        command.extend(['--'+key.replace('_', '-'), cfg[key]])
    repair = Path(cfg['collision_refinement']).resolve() if cfg.get('collision_refinement') else None
    repair_sha = None
    if repair is not None:
        repair_sha = sha(repair)
        if cfg.get('collision_refinement_sha256', repair_sha) != repair_sha:
            raise ValueError('Primary body collision refinement changed before native preparation')
        command.extend(['--collision-refinement', str(repair)])
    if cfg.get('restore_object_initial_state'): command.append('--restore-object-initial-state')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='4',
        CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
    started = time.monotonic()
    timed_out = False
    with (output/'prepare.log').open('xb') as log:
        try:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True; returncode = None
    plan_path = output/'native/plan.json'; failure_path = output/'native/admission_failure.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    status = 'ready' if returncode == 0 and plan.get('admitted') is True else 'no_plan'
    if not plan:
        failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
        if not (failure.get('complete') is True and failure.get('kind') == 'arm_plan_rejected'
                and failure.get('physical_executed') is False): status = 'infrastructure_failure'
    if timed_out: status = 'infrastructure_failure'
    validation = None
    if status == 'ready':
        commands_path = output/'native/commands.npz'
        if sha(commands_path) != plan['commands_sha256']: raise ValueError('Native prepare command SHA mismatch')
        with np.load(commands_path, allow_pickle=False) as data: arrays = {k: data[k].copy() for k in data.files}
        contract = ActionContract(generation['calibration'])
        names = list(map(str, arrays['joint_names']))
        motors = arrays['joint_target'][:, [names.index(n) for n in contract.names]]
        validation_arrays, expansion = expanded_planned_hand(arrays, contract)
        validation = contract.validate_command_arrays(validation_arrays, slot_id=candidate['slot_id'], motor_targets=motors,
            candidate_row15=candidate['actual_row15_camera'], world_T_camera=candidate['world_T_camera'],
            extra_depth_m=candidate['extra_depth_m'])
        validation['planned_mimic_expansion'] = expansion
        plan['author_collection_binding'] = dict(candidate=candidate,
            collection_candidate_sha256=candidate['candidate_sha256'],
            original_native_commands_sha256=plan['commands_sha256'], numerical_action_validation=validation,
            task=candidate['task'], task_success_label_generated=False)
        _save(plan_path, plan)
    result = dict(schema='anydex_boya_author_collection_arm_preparation_v1', status=status,
        returncode=returncode, timed_out=timed_out, pool=str(pool), candidate_sha256=candidate['candidate_sha256'],
        plan_dir=str(output/'native'), plan_sha256=sha(plan_path) if plan_path.exists() else None,
        admission_failure_sha256=sha(failure_path) if failure_path.exists() else None,
        action_validation=validation, input_sha256={str(backend_config): sha(backend_config),
            str(pool/'generation.json'): sha(pool/'generation.json'), str(pool/'candidates.npz'): sha(pool/'candidates.npz')},
        wall_s=time.monotonic()-started, physics_steps=0, labels=0,
        full_clutter_or_transport_validation=False)
    if repair is not None:
        if sha(repair) != repair_sha: raise ValueError('Primary body collision refinement changed during preparation')
        result['input_sha256'][str(repair)] = repair_sha
    _save(output/'ARM_PREPARATION.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--object-root', required=True, type=Path)
    parser.add_argument('--world-object-pose', required=True, type=Path)
    parser.add_argument('--object-scale', type=float, default=1.)
    parser.add_argument('--scene-manifest', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--backend-config', type=Path)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text())
    if not prepared['candidates']: raise ValueError('No selected candidate; keep null, do not fabricate a label')
    bridge_candidate(prepared['candidates'][0], args.output_dir, calibration=args.calibration,
        object_root=args.object_root, world_T_object=json.loads(args.world_object_pose.read_text()),
        object_scale=args.object_scale, prepared_path=args.prepared, scene_manifest=args.scene_manifest)
    if args.backend_config:
        result = prepare_arm(args.output_dir, args.output_dir.with_name(args.output_dir.name+'_arm'), backend_config=args.backend_config)
        print(json.dumps({k: result[k] for k in ('status', 'returncode', 'physics_steps', 'wall_s')}))
