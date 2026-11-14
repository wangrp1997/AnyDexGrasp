"""Read-only audit of the frozen first bin-free four-type preparation.

Reads existing artifacts only; imports no planner, simulator or inference code.
Missing command phases are reported as not reached, never as passed checks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np


TASK = 'declared_boya_clutter_lift_v1'
HEX = set('0123456789abcdef')


def read(path):
    return json.loads(Path(path).read_text())


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def assignment(path, name):
    literals = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'dict':
                return {k.arg: literals[k.value.id] if isinstance(k.value, ast.Name)
                        else ast.literal_eval(k.value) for k in value.keywords}
            return ast.literal_eval(value)
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    literals[target.id] = value
    raise ValueError('Missing literal assignment: '+name)


def audit(root):
    root = Path(root).resolve()
    started = time.monotonic()
    checks, expected, evidence, hash_cache = [], {}, {}, {}

    def sha(path):
        path = str(Path(path).resolve())
        if path not in hash_cache:
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            hash_cache[path] = digest.hexdigest()
        return hash_cache[path]

    def check(name, ok, **detail):
        checks.append(dict(name=name, ok=bool(ok), **detail))

    def lock(path, digest):
        path = str(Path(path).resolve())
        if path in expected and expected[path] != digest:
            check('conflicting_sha_binding', False, path=path)
        expected[path] = digest

    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                lock(value['path'], value['sha256'])
            for key, item in value.items():
                if str(key).startswith('/') and isinstance(item, str) and len(item) == 64 and set(item) <= HEX:
                    lock(key, item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    def artifact(relative):
        path = root/relative
        value = read(path)
        evidence[str(path)] = sha(path)
        walk(value)
        return value

    protocol = artifact('DEVELOPMENT_PROTOCOL.json')
    queue = artifact('PREPARATION_QUEUE.json')
    results = artifact('PREPARATION_RESULTS.json')
    scene = artifact('scene/capture/observed_scene.json')
    observation = artifact('scene/capture/observation.json')
    representation = artifact('representation/representations/development_three_lift_20261114/RESULTS.json')
    capture = artifact('scene/capture/RESULTS.json')
    check('queue_and_result_bindings', queue['protocol_sha256'] == sha(root/'DEVELOPMENT_PROTOCOL.json')
          and results['preparation_queue_sha256'] == sha(root/'PREPARATION_QUEUE.json')
          and results['status'] == 'complete' and queue['conditions'] == protocol['gestures'])
    scene_path = root/'scene/capture/observed_scene.json'
    check('explicit_fresh_bin_free_scene', scene['task'] == TASK and scene['bin'] is None
          and scene['schema'] == 'anydex_boya_author_lift_scene_v1'
          and observation['scene_sha256'] == queue['scene_sha256'] == sha(scene_path)
          and observation['object_pose_as_network_input'] is False
          and representation['scene_binding']['observation_sha256'] == sha(root/'scene/capture/observation.json'))
    check('one_original_CGR_no_head', representation['test_time_augmentation_branches'] == 1
          and representation['full_representation_forward'] is True
          and representation['hand_head_loaded'] is False)
    raw = arrays(root/'representation/representations/development_three_lift_20261114/raw_camera_candidates.npz')
    conditions = []
    for row in queue['conditions']:
        g = row['gesture_id']; prefix = f'g{g}/'
        pipeline = artifact(prefix+'LIFT_PIPELINE_PROTOCOL.json')
        result = artifact(prefix+'LIFT_PIPELINE_RESULTS.json')
        prepared = artifact(prefix+'collection/PREPARED.json')
        candidate = artifact(prefix+'pool/selected_author_candidate.json')
        generation = artifact(prefix+'pool/generation.json')
        arm = artifact(prefix+'arm/ARM_PREPARATION.json')
        frozen = artifact(prefix+'arm/native/frozen_input.json')
        failure = artifact(prefix+'arm/native/admission_failure.json')
        evidence[str(root/(prefix+'arm/prepare.log'))] = sha(root/(prefix+'arm/prepare.log'))
        pool = arrays(root/(prefix+'pool/candidates.npz'))
        trace = arrays(prepared['trace']['path'])
        index = candidate['source_top500_rank']; src = candidate['source_row_index']
        actual = np.asarray(candidate['actual_row15_camera'])
        flip = np.asarray(candidate['source_row15_camera']).copy()
        if candidate['source_if_flip']:
            rotation = flip[3:12].reshape(3, 3); rotation[:, :2] *= -1
        unrolled = np.r_[trace['actual_rows17_camera'][index, :2], trace['actual_rows17_camera'][index, 3:16]]
        selected = int(trace['top10_top500_indices'][0])
        exact_first = (prepared['selected_candidate_sha256'] == candidate['candidate_sha256']
            == result['candidate_sha256'] == pool['collection_candidate_sha256'].item()
            and prepared['candidates'][0] == candidate and selected == index
            and candidate['candidate_sha256'] == json_sha({k:v for k,v in candidate.items() if k != 'candidate_sha256'}))
        check(f'g{g}_fixed_first_original_candidate', exact_first
            and np.array_equal(raw['candidates'][src], candidate['source_row15_camera'])
            and np.array_equal(raw['candidate_features'][src], candidate['source_feature2261'])
            and np.array_equal(actual, unrolled) and np.array_equal(actual, flip)
            and candidate['task'] == TASK and candidate['label'] is None and candidate['label_state'] == 'unexecuted')
        action = candidate['action_contract']
        # Source geometry uses float32 depth displacement before the world transform.
        before = np.asarray(candidate['camera_T_palm_before_depth'])
        axis = actual[3:12].reshape(3, 3)[:, 0]; axis = axis/np.linalg.norm(axis)
        depth = float(actual[2])+candidate['extra_depth_m']
        displacement = np.asarray(axis*depth, dtype=np.float32).astype(float)
        after = before.copy(); after[:3, 3] += displacement
        world = np.asarray(candidate['world_T_camera'])@after
        proper = world.copy(); u, _, vh = np.linalg.svd(proper[:3, :3]); proper[:3, :3] = u@vh
        palm_error = float(np.max(np.abs(proper-pool['proposal_palm'][0])))
        check(f'g{g}_source_depth_palm_and_command_endpoints',
            np.array_equal(after, candidate['camera_T_palm'])
            and np.array_equal(world, candidate['world_T_palm']) and palm_error < 1e-14
            and np.array_equal(pool['proposal_q'][0], action['opened_independent_rad'])
            and np.array_equal(pool['proposal_q_close'][0], action['closed_independent_rad'])
            and np.array_equal(pool['proposal_q'][0], frozen['open_motors'])
            and np.array_equal(pool['proposal_q_close'][0], frozen['closed_motors'])
            and np.max(np.abs(np.asarray(frozen['palm'])-proper)) < 1e-14,
            planner_roundoff_matrix_error=palm_error)
        check(f'g{g}_frozen_budget_and_failure_boundary',
            pipeline['candidate_budget'] == row['max_candidates'] == 1
            and pipeline['fallback_candidate'] is False and row['seed'] == pipeline['seed'] == 20261114
            and frozen['arguments']['pregrasp_distance_m'] == .04 and frozen['arguments']['approach_only'] is True
            and frozen['dt_s'] == .004 and len(frozen['seeds']) == 8
            and result['status'] == 'null_geometry' and result['complete'] is True
            and result['stage'] == 'source4cm_approach' and arm['status'] == 'no_plan'
            and arm['returncode'] == 1 and arm['timed_out'] is False
            and failure['kind'] == 'arm_plan_rejected' and failure['physical_executed'] is False
            and 'exhausted declared seeds' in failure['reason'])
        forbidden = ['arm/native/plan.json', 'arm/native/commands.npz', 'home', 'lift_raw',
                     'lift_bound', 'READY.json', 'EXECUTION_STARTED.json', 'execution', 'COLLECTION_RECORD.json']
        absent = {name: not (root/prefix/name).exists() for name in forbidden}
        check(f'g{g}_no_unearned_ready_or_execution', all(absent.values())
              and result['physics_steps'] == result['new_labels'] == result['training_updates'] == 0,
              absent=absent)
        conditions.append(dict(gesture_id=g, status=result['status'], source_row=src, source_top500_rank=index,
            collision_admitted=prepared['counts']['collision_admitted'], candidate_sha256=candidate['candidate_sha256'],
            slot_id=candidate['slot_id'], frozen_seed_entries=8, unique_seed_entries=len({json_sha(v) for v in frozen['seeds']}),
            native_plan_and_arrays_available=False, HOME='not_entered', prefix_arrays='not_available',
            close_250='not_entered', hold_50='not_entered', world_Z_lift_10cm='not_entered',
            final_derivative_and_full_scene_geometry='not_entered', arm_wall_s=arm['wall_s'],
            actual_nfev='not_recorded_by_native_failure_receipt', physics_steps=0, labels=0))
    obstacle_manifest = []
    for body in scene['objects']:
        path = Path(body['object_root'])/'urdf/coacd.urdf'
        files = [(path.parent/node.get('filename')).resolve()
                 for node in ET.parse(path).findall('.//collision/geometry/mesh')]
        for file in files:
            sha(file)
        obstacle_manifest.append(dict(object_id=body['object_id'], source_collision_parts=len(files),
            urdf_sha256=sha(path), part_files_sha256_digest=json_sha({str(p):sha(p) for p in files}),
            scale=body['scale'], world_T_object_sha256=json_sha(body['world_T_object']),
            native_refinement_sha256=body['collision_refinement_sha256'],
            planning_scope='Source convex hulls at frozen initial poses; native refined/cooked geometry separately audited at runtime'))
    source = Path(__file__).with_name('author_lift_plan.py')
    policy = assignment(source, 'POLICY')
    check('declared_lift_policy_unchanged', policy['close_steps'] == 250 and policy['hold_steps'] == 50
        and policy['lift_axis_world'] == [0.,0.,1.] and policy['lift_distance_m'] == .1
        and policy['lift_waypoints'] == 32 and policy['per_waypoint_max_nfev'] == 300
        and policy['geometry_tolerance_m'] == 1e-7 and policy['acceleration_rad_s2'] == 3.
        and policy['jerk_rad_s3'] == 30. and policy['additional_lift_hold_steps'] == 0
        and policy['alternate_ik_seeds'] == policy['planning_retries'] == 0)
    evidence[str(source)] = sha(source)
    mismatches = []
    for path, digest in sorted(expected.items()):
        try:
            actual_digest = sha(path)
        except OSError as error:
            mismatches.append(dict(path=path, error=str(error))); continue
        if actual_digest != digest:
            mismatches.append(dict(path=path, expected=digest, actual=actual_digest))
    check('all_discovered_explicit_SHA_bindings', not mismatches, count=len(expected), mismatches=mismatches)
    return dict(schema='anydex_boya_independent_lift_preparation_audit_v1', root=str(root),
        ok=all(row['ok'] for row in checks), checks=checks, conditions=conditions,
        obstacle_manifest=obstacle_manifest, bin_obstacle_count=0, policy=policy,
        bound_input_and_source_files=len(expected), evidence_artifact_sha256=evidence,
        initialization=dict(physics_ticks=capture['physics_ticks'], object_steps=capture['initialization_object_steps']),
        limitation='Four native pregrasp local IK nulls; no actual command arrays exist. This audit cannot validate unentered HOME/close/hold/lift/geometry/runtime gates or claim global unreachability or a grasp success rate.',
        audit_new_IK_calls=0, audit_new_CGR_forwards=0, audit_physics_steps=0,
        audit_training_updates=0, audit_new_labels=0, wall_s=time.monotonic()-started,
        audit_source_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(ok=result['ok'], conditions=[dict(gesture_id=r['gesture_id'],status=r['status']) for r in result['conditions']],
        bound_input_and_source_files=result['bound_input_and_source_files'], wall_s=result['wall_s']), indent=2))
    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
