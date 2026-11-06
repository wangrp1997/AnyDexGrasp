"""Score raw CGR/type/depth combinations before Boya geometry or arm admission.

The default follows the fixed released DH3 post-CGR selection parameters.
Boya heads, calibration, geometric checks, single-frame input and arm backend
remain an explicitly declared port. This command never executes physics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from evaluate_boya_decision import GESTURES, Reader, score_candidates
from trial_record import array_ref, canonical_candidate, official_functions

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    'released-dh3-v1': dict(cgr_limit=1000, score_topk=3500,
                            strict_score_threshold=.85, collision_topk=400,
                            source='robot_dh3.py:435-474,604-622,662-675'),
    'paper-testing-budget-boya-port-v1': dict(
        cgr_limit=100, score_topk=200, strict_score_threshold=None,
        collision_topk=200,
        source='methods.tex:111 testing budget; four depths/flip from released code, Boya port'),
}
OFFICIAL_ROBOT_SHA = '7d227451761a429fff87f689e5dcb9c9e215da5ec3c5c594be1e16f3333fa797'


def save(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def validate_cached_forward(observation_path, observation, frame_dir, receipt, reader):
    """Reject a valid CGR file paired with a different image/camera observation."""
    if receipt.get('full_representation_forward') is not True:
        raise ValueError('Actual cached CGR forward receipt required')
    scene = receipt.get('scene_binding', {})
    if (Path(scene.get('observation_json', '')).resolve() != observation_path
            or scene.get('observation_sha256') != reader.files[str(observation_path)]
            or scene.get('T_WC') != observation['T_WC']):
        raise ValueError('Cached CGR belongs to another observation or camera')
    expected = {str(Path(v['path']).resolve()): v['sha256'] for v in observation['files'].values()}
    actual = {str(Path(k).resolve()): v['sha256'] for k, v in receipt['input_files'].items()}
    if expected != actual:
        raise ValueError('Cached CGR image inputs differ from observation')
    for path, digest in actual.items():
        reader.freeze(path, digest)
    for name in ('raw_camera_candidates.npz', 'representation_outputs.npz'):
        item = receipt['output_files'][name]
        if Path(item['path']).resolve() != frame_dir / name:
            raise ValueError('Cached CGR output path mismatch')
        reader.freeze(item['path'], item['sha256'])


def make_queries(raw, features, profile, flip_policy='official'):
    """Canonicalize original rows without any hand width/FK/collision queries."""
    if raw.ndim != 2 or raw.shape[1:] != (15,) or features.shape != (len(raw), 2261):
        raise ValueError('Expected paired original Nx15 / Nx2261 arrays')
    if not np.isfinite(raw).all() or not np.isfinite(features).all():
        raise ValueError('Nonfinite original candidates/features')
    if flip_policy not in ('official', 'both'):
        raise ValueError('Unknown flip policy')
    functions = official_functions(ROOT / 'robot_dh3.py')
    source_indices = np.argsort(-raw[:, 0], kind='stable')[:profile['cgr_limit']]
    queries, inputs = [], []
    for source_rank, index in enumerate(source_indices):
        for flip in ([None] if flip_policy == 'official' else [False, True]):
            info = canonical_candidate(raw[index], features[index], functions, if_flip=flip)
            feature = np.asarray(info['canonical_CGR480'], dtype=np.float32)
            for gesture in GESTURES:
                queries.append(dict(
                    query_index=len(queries), source_candidate_index=int(index),
                    source_cgr_rank=int(source_rank), gesture_id=gesture,
                    if_flip=info['if_flip'], official_policy_if_flip=info['official_policy_if_flip'],
                    raw_depth_index=info['two_fingers_pose_depth_type'],
                    target_index=info['two_fingers_pose_depth_type'] * 4,
                    cgr_score=float(raw[index, 0]),
                    canonical_CGR480_identity=array_ref(feature)))
                inputs.append(feature)
    return queries, inputs, source_indices.tolist()


def expand_and_rank(queries, profile):
    """Global ranking, never a per-type first-success or old admitted-pool rank."""
    rows = []
    for query in queries:
        scores = np.asarray(query['all_head_outputs20'])
        if scores.shape != (20,) or not np.isfinite(scores).all():
            raise ValueError('Twenty finite head scores required')
        depth = query['raw_depth_index']
        if type(depth) is not int or not 0 <= depth < 5:
            raise ValueError('Invalid original raw depth index')
        for extra in range(4):
            row = {k: copy.deepcopy(v) for k, v in query.items()
                   if k not in ('all_head_outputs20', 'learned_score', 'target_index')}
            row.update(extra_depth_index=extra, target_index=depth * 4 + extra,
                       learned_score=float(scores[depth * 4 + extra]))
            rows.append(row)
    rows.sort(key=lambda r: (-r['learned_score'], r['query_index'], r['extra_depth_index']))
    for rank, row in enumerate(rows):
        row['global_score_rank'] = rank
    shortlist = rows[:profile['score_topk']]
    threshold = profile['strict_score_threshold']
    if threshold is not None:
        # Released torch compares float32 predictions with a float32 scalar.
        threshold = float(np.float32(threshold))
        shortlist = [r for r in shortlist if r['learned_score'] > threshold]
    return rows, shortlist[:profile['collision_topk']]


def calibrated_slots(assets, calibration, reader):
    lock = calibration['source_lock']
    if lock['independent_motor_names'] != list(assets.hand.names):
        raise ValueError('Calibration motor order differs')
    for path, entry in lock['source_files'].items():
        reader.freeze(path, entry['sha256'])
    for path, digest in lock['readonly_fk_dependencies'].items():
        reader.freeze(path, digest)
    families = {}
    for gesture in GESTURES:
        family = assets.families[gesture]
        slots = [s for s in calibration['slots'] if s['gesture_id'] == gesture
                 and s['self_collision_clear'] and s['opposing_group'] == 'ff']
        if not slots:
            raise ValueError('Missing calibrated thumb/index family')
        for slot in slots:
            if slot['wrist'] != family.wrist or not np.array_equal(slot['close_q'], family.nominal):
                raise ValueError('Calibration closure/wrist changed')
            expected = family.motor_state(slot['fraction'], slot['fraction'])
            if not np.array_equal(slot['independent_q'], expected):
                raise ValueError('Calibration opening changed')
        families[gesture] = slots
    return families


def admit(shortlist, raw, observation, args, reader, freeze_path):
    """Only entered after the immutable global shortlist is written."""
    from asset_adapter import BoyaAssets, cgr_reference_pose, map_slot
    from build_collection_pool import OpeningGeometryCache, geometry

    assert freeze_path.is_file(), 'Ranking must be frozen before geometry'
    freeze_sha = reader.freeze(freeze_path)
    assets = BoyaAssets()
    families = calibrated_slots(assets, reader.read(args.calibration), reader)
    sparse_path = args.frame_dir / 'representation_outputs.npz'
    reader.freeze(sparse_path)
    with np.load(sparse_path, allow_pickle=False) as z:
        cloud = z['sparse_input_XYZ_features'].copy()
    camera = np.asarray(observation['T_WC'])
    cloud = cloud @ camera[:3, :3].T + camera[:3, 3]
    cache, checked, admitted, arm = OpeningGeometryCache(), [], [], []
    selected = None
    if args.stage == 'arm':
        from prepare_collection_queue import prepare_one
        reader.freeze(args.backend_config)
        if args.collision_refinement:
            reader.freeze(args.collision_refinement)
        (args.output_dir / 'arm').mkdir()
    for candidate in shortlist:
        row = copy.deepcopy(candidate)
        slots = families[row['gesture_id']]
        expanded = float(raw[row['source_candidate_index'], 1]) * 1.7
        lower = min(s['reference_width_m'] for s in slots)
        upper = max(s['reference_width_m'] for s in slots)
        requested = float(np.clip(expanded, lower, upper))
        matches = [s for s in slots if abs(s['reference_width_m'] - requested) <= .002]
        if not matches:
            checked.append(dict(**row, geometry=dict(ok=False, reason='outside_calibrated_width_tolerance')))
            continue
        slot = min(matches, key=lambda s: (abs(s['reference_width_m'] - requested), s['slot_id']))
        used = raw[row['source_candidate_index']].copy()
        if row['if_flip']:
            used[3:12].reshape(3, 3)[:, 1:3] *= -1
        palm = map_slot(used, camera, slot, insertion_m=.01 * row['extra_depth_index'])
        approach = cgr_reference_pose(used, camera)[:3, 2]
        check = geometry(assets, np.asarray(slot['independent_q']), palm, cloud, approach, cache)
        row.update(source_slot=slot['slot_id'], actual_reference_width_m=slot['reference_width_m'],
                   raw_width_m=float(used[1]), requested_opening_m=requested,
                   width_expansion=1.7, width_clamped=bool(expanded < lower or expanded > upper),
                   raw_depth_m=float(used[2]), additional_depth_index=row['extra_depth_index'],
                   additional_depth_m=.01 * row['extra_depth_index'], geometry=check,
                   score_source='Frozen Boya head, globally ranked before geometry',
                   jaw_assignment=dict(thumb='th', opponent='ff'))
        checked.append(row)
        if not check['ok']:
            continue
        rank = row['global_score_rank']
        pool = args.output_dir / f'pool_{rank:05d}'
        pool.mkdir()
        np.savez_compressed(pool / 'candidates.npz',
                            independent_motor_names=assets.hand.names,
                            proposal_q=[slot['independent_q']], proposal_q_close=[slot['close_q']],
                            proposal_palm=[palm], proposal_approach_direction=[approach],
                            world_T_object=observation['world_T_object'])
        save(pool / 'world_object_pose.json', observation['world_T_object'])
        generation = dict(complete=True, kind='anydex_boya_score_first_single_candidate_v1',
                          gesture_id=row['gesture_id'], additional_depth_index=row['extra_depth_index'],
                          scale=observation['object_scale'], selected_indices=[0], records=[row],
                          candidates_sha256=reader.freeze(pool / 'candidates.npz'),
                          input_sha256=copy.deepcopy(reader.files),
                          source_frame=str(args.frame_dir), source_observation=str(args.observation),
                          hand_head_loaded=True, physics_steps=0, training_labels=0,
                          global_ranking_freeze_sha256=freeze_sha)
        save(pool / 'generation.json', generation)
        admitted.append(dict(global_score_rank=rank, candidate_dir=str(pool), candidate=row))
        if args.stage != 'arm':
            continue
        job = dict(trial_id=f'score_first_rank_{rank:05d}', object_id=observation['object_id'],
                   split='development_alignment', gesture_id=row['gesture_id'],
                   extra_depth_index=row['extra_depth_index'], candidate_dir=str(pool),
                   observation_json=str(args.observation),
                   source_npz=str(args.frame_dir / 'raw_camera_candidates.npz'),
                   calibration=str(args.calibration), candidate_indices=[0], max_arm_plans=1)
        if args.collision_refinement:
            job.update(collision_refinement=str(args.collision_refinement),
                       collision_refinement_sha256=reader.files[str(args.collision_refinement)])
        result = prepare_one(job, args.output_dir / 'arm', args.backend_config)
        arm.append(result)
        if result['status'] == 'ready':
            from trial_record import check_binding
            binding = reader.read(result['binding_json'], result['binding_sha256'])
            check_binding(binding)
            if (binding['if_flip'] != row['if_flip']
                    or binding['canonical_CGR480_identity'] != row['canonical_CGR480_identity']
                    or binding['two_fingers_pose_depth_type'] * 4 + binding['multifinger_pose_depth_type'] != row['target_index']):
                raise ValueError('Arm binding differs from frozen score identity')
            selected = result
            break
        if len(arm) >= args.max_arm_plans:
            break
    reader.verify()
    save(args.output_dir / 'geometry.json', dict(checked=checked, admitted=admitted,
         arm_preparations=arm, selected=selected, shared_arm_budget=args.max_arm_plans,
         cached_openings=len(cache.states), cache_hits=cache.hits))
    return dict(geometry_checked=len(checked), geometry_admitted=len(admitted),
                arm_checked=len(arm), selected=selected)


def run(args):
    start = time.monotonic()
    reader = Reader()
    for key in ('observation', 'frame_dir', 'calibration', 'training_results', 'training_split', 'output_dir'):
        setattr(args, key, getattr(args, key).resolve())
    for key in ('backend_config', 'collision_refinement'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.output_dir.exists():
        raise ValueError('Use a new output directory; existing results are never overwritten')
    source = args.frame_dir / 'raw_camera_candidates.npz'
    reader.freeze(ROOT / 'robot_dh3.py', OFFICIAL_ROBOT_SHA)
    for name in ('asset_adapter.py', 'build_collection_pool.py', 'prepare_collection_queue.py',
                 'native_arm.py', 'trial_record.py', 'evaluate_boya_decision.py', 'deploy_score_first.py'):
        reader.freeze(Path(__file__).with_name(name))
    observation = reader.read(args.observation)
    for item in observation['files'].values():
        reader.freeze(item['path'], item['sha256'])
    receipt = reader.read(args.frame_dir / 'RESULTS.json')
    validate_cached_forward(args.observation, observation, args.frame_dir, receipt, reader)
    reader.freeze(source)
    reader.freeze(args.calibration)  # Hash only; no mapping/table opened before ranking.
    with np.load(source, allow_pickle=False) as z:
        raw, features = z['candidates'].copy(), z['candidate_features'].copy()
    training = reader.read(args.training_results)
    dataset_ref = training['dataset_manifest']
    dataset_sha = reader.freeze(dataset_ref['path'], dataset_ref['sha256'])
    split = reader.read(args.training_split)['objects']
    train = {o['mesh_sha256'] for o in split if o['split'] == 'train'}
    heldout = {o['mesh_sha256'] for o in split if o['split'] == 'validation'}
    if len(train) != 40 or len(heldout) != 10 or train & heldout:
        raise ValueError('Frozen head-training metadata requires disjoint 40/10 mesh groups')
    profile = copy.deepcopy(PROFILES[args.profile])
    queries, inputs, indices = make_queries(raw, features, profile, args.flip_policy)
    models = score_candidates(queries, inputs, training, args.training_results, dataset_sha,
                              reader, args.device, False, train, heldout)
    ranked, shortlist = expand_and_rank(queries, profile)
    reader.verify()
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / 'head_outputs.npz',
                        outputs20=np.asarray([q['all_head_outputs20'] for q in queries], dtype=np.float32).reshape(-1, 20),
                        source_indices=np.asarray([q['source_candidate_index'] for q in queries], dtype=np.int64),
                        gesture_ids=np.asarray([q['gesture_id'] for q in queries], dtype=np.int64),
                        raw_depth_indices=np.asarray([q['raw_depth_index'] for q in queries], dtype=np.int64),
                        if_flip=np.asarray([q['if_flip'] for q in queries], dtype=bool))
    save(args.output_dir / 'scores.json', dict(candidates=ranked, models=models))
    save(args.output_dir / 'shortlist.json', dict(candidates=shortlist, profile=args.profile,
         parameters=profile, flip_policy=args.flip_policy, source_indices=indices,
         tie_policy='Stable query source-rank, flip order, gesture order, extra index; released torch topk ties unspecified'))
    frozen = [args.output_dir / name for name in ('head_outputs.npz', 'scores.json', 'shortlist.json')]
    save(args.output_dir / 'RANKING_FREEZE.json', dict(
        outputs_sha256={str(p): reader.freeze(p) for p in frozen},
        ranking_inputs_sha256=copy.deepcopy(reader.files),
        labels_used_for_ranking=False, old_admitted_queue_read=False,
        geometry_calls_before_freeze=0, arm_calls_before_freeze=0,
        dataset_manifest_deserialized=False))
    admission = dict(geometry_checked=0, geometry_admitted=0, arm_checked=0, selected=None)
    if args.stage != 'rank' and shortlist:
        admission = admit(shortlist, raw, observation, args, reader, args.output_dir / 'RANKING_FREEZE.json')
    reader.verify()
    result = dict(schema='anydex_boya_score_first_deployment_v1', status='complete',
        stage=args.stage, profile=args.profile, parameters=profile, flip_policy=args.flip_policy,
        input_source_candidates=len(raw), preselected_CGR_rows=len(indices),
        head_queries=len(queries), scored_type_depth_combinations=len(ranked),
        collision_shortlist_count=len(shortlist), models=models, **admission,
        null_reason=('no_scores_passed_profile' if not shortlist else
                     'no_admitted_hand_geometry' if args.stage != 'rank' and not admission['geometry_admitted'] else
                     'no_arm_plan_in_shared_budget' if args.stage == 'arm' and admission['selected'] is None else None),
        input_sha256=reader.files, inputs_unchanged=True,
        scope=dict(paper_object_set_aligned=False, full_paper_reproduction=False,
                   scope='Post-CGR score-order alignment on cached inputs with new Boya heads; not a grasp benchmark',
                   remaining_differences=['Boya geometry calibration, hand types and learned weights',
                       'Single cached RGB-D branch rather than author 11-branch deployment',
                       'Boya full-hull/table and complete-arm backend instead of original hand voxel collision',
                       'Stable score ties may differ from torch topk unspecified ties',
                       'Paper-budget port mixes disclosed testing budgets and released-code depth enumeration'
                       if args.profile != 'released-dh3-v1' else 'Original DH3 score gate preserved without tuning',
                       'Explicit two-flip expansion' if args.flip_policy == 'both' else 'Original single DH3 flip policy retained']),
        cost=dict(new_physics_steps=0, new_CGR_forwards=0, new_training_steps=0,
                  head_inference_queries=len(queries), elapsed_wall_s=time.monotonic() - start))
    save(args.output_dir / 'RESULTS.json', result)
    print(json.dumps({k: result[k] for k in ('status','profile','stage','head_queries',
         'scored_type_depth_combinations','collision_shortlist_count','geometry_checked',
         'geometry_admitted','arm_checked','null_reason','cost')}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('observation', 'frame-dir', 'calibration', 'training-results', 'training-split', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--profile', choices=PROFILES, default='released-dh3-v1')
    p.add_argument('--flip-policy', choices=['official','both'], default='official')
    p.add_argument('--stage', choices=['rank','geometry','arm'], default='rank')
    p.add_argument('--device', choices=['cpu','cuda:0'], default='cpu')
    p.add_argument('--max-arm-plans', type=int, default=4)
    p.add_argument('--backend-config', type=Path)
    p.add_argument('--collision-refinement', type=Path)
    args = p.parse_args()
    if not 1 <= args.max_arm_plans <= 64:
        p.error('Shared arm preparation budget must be 1..64')
    if args.stage == 'arm' and args.backend_config is None:
        p.error('--stage arm requires --backend-config')
    run(args)


if __name__ == '__main__':
    main()
