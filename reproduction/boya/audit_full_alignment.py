"""Source-backed full-protocol gap audit and data expansion specification.

Read-only inputs, zero simulator/model/optimizer calls. A gap list is not a
readiness certificate. Fresh output only; previous results are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import author_protocol as author

ROOT = author.ROOT
DATASET = ROOT/'outputs/s40_stable_supplement_20261107_v2/training_delivery_v3/merged_dataset/manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def evidence(relative, needle):
    path = ROOT/relative
    lines = path.read_text().splitlines()
    matches = [dict(line=i+1, text=s.strip()) for i, s in enumerate(lines) if needle in s]
    if not matches:
        raise ValueError('Audit evidence disappeared: ' + relative + ': ' + needle)
    return dict(path=str(path), sha256=sha(path), matches=matches)


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh alignment audit directory required')
    for name in author.PINS:
        author.source(name)
    checks = [
        ('collection_candidates', '采集候选与随机深入',
         ('collect_dh3_grasp_data.py', '[:500]'),
         ('reproduction/boya/build_collection_pool.py', 'a.extra_depth_index'),
         'Author collection: top500, fixed type, random extra per candidate. Old collection uses fixed object/type/extra slots and up to1000; requires a new sampler.'),
        ('flip_policy', '采集翻面策略',
         ('collect_dh3_grasp_data.py', 'ggarray, if_flip = flip_ggarray'),
         ('reproduction/boya/build_collection_pool.py', 'for flipped in [False,True]'),
         'One source-hand rig flip versus both orientations. Choose one documented Boya rig convention; do not silently count both as exact author collection.'),
        ('collision_predicate', '碰撞算法',
         ('utils/collision_detector.py', 'check_if_included'),
         ('reproduction/boya/build_trial_pool.py', 'ConvexHull(v)'),
         'Surface-cloud voxel inclusion differs from per-link convex interior tests. New extracted voxel kernel is verified, hand cloud/LUT integration is still pending.'),
        ('collision_sweep', '接近扫掠',
         ('utils/collision_detector.py', 'meshes_pointclouds = meshes_pointclouds +'),
         ('reproduction/boya/build_trial_pool.py', 'np.linspace(0.,.1,21)'),
         'Author loop accumulates translated clouds recursively. With .08m its shifts are0/2/5/7/8/10/13/15cm, not merely a uniform0..8cm sweep. Preserve source behavior explicitly.'),
        ('hand_surface_cloud', '手点云资产',
         ('generate_mesh_and_pointcloud/recover_allegro_hand_to_stl.py', 'sample_points_uniformly(mesh, 300000)'),
         ('reproduction/boya/asset_adapter.py', 'self.meshes ='),
         'Generate full open-hand surface clouds from validated Boya mesh/FK at every used LUT slot; source hull vertices alone are not the author hand cloud.'),
        ('dataset_cap', '更多数据是否真正进入训练',
         ('dataset/multifinger_hand.py', 'data_num == 100'),
         ('reproduction/boya/train_s40_code_recipe.py', 'Expected435'),
         'Keep first100 baseline; separately explicit300/1000 extensions. Do not silently use the legacy first100 loader for the expanded dataset.'),
        ('singleton_batch', '扩采后的尾批处理',
         ('train_multifinger.py', 'shape[0] == 1'),
         ('reproduction/boya/train_boya_decision.py', 'batches[-2]=np.concatenate'),
         'Author skips singleton batches; old adapter merges them. This did not affect current fit heads below128 rows but matters when scaling data. New primitive calls the original epoch directly.'),
        ('checkpoint_selection', '模型选择规则',
         ('train_multifinger.py', 'def save_model'),
         ('reproduction/boya/train_s40_code_recipe.py', 'fixed final100'),
         'Author saves best validation P/R checkpoints at preset thresholds. Old fixed-final is a declared difference. Freeze a fresh internal validation policy; never select on previously used external objects.'),
        ('task_semantics', '采集动作和标签',
         ('ur_toolbox/ur_toolbox/robot/ur.py', 'self.throw(acc=acc, vel=vel)'),
         ('reproduction/boya/BOYA_MAPPING_CLOSURE_ALIGNMENT_20261108.md', '垂直抬100mm'),
         'Old labels measure lift/retention, author records transfer to a bin. A transport label cannot be inferred from old lift traces; implement and validate transport before collecting that task.'),
        ('scene_distribution', '训练场景分布',
         ('outputs/paper_code_audit/source/sections/methods.tex', 'randomly place objects'),
         ('reproduction/boya/run_s40_settled_collection.py', 'Each new attempt cold-resets'),
         'Old isolated target collection is not the author random tabletop multi-object distribution. New scene interface must represent all scene objects and record initialization separately.'),
        ('prehead_hand_filter', '评分前手几何过滤',
         ('robot_dh3.py', 'source_index = ggarray[:, 0].argsort()'),
         ('reproduction/boya/boya_hand_domain.py', 'Only the TABLE PREFIX'),
         'Old Boya domain applies hand/table/extra feasibility before top1000 and head scoring. Source DH3 uses sensor/rig filters, then score, then hand collision. Keep camera/workspace adaptation explicit and remove the extra pre-head hand predicate from the aligned path.'),
    ]
    gaps = []
    files = {str(ROOT/n): h for n, h in author.PINS.items()}
    files[str(Path(__file__).resolve())] = sha(__file__)
    files[str(Path(author.__file__).resolve())] = sha(author.__file__)
    for key, title, original, current, finding in checks:
        refs = [evidence(*original), evidence(*current)]
        gaps.append(dict(id=key, title=title, status='open_pipeline_gap', evidence=refs, finding=finding))
        files.update({r['path']:r['sha256'] for r in refs})
    manifest = read(DATASET)
    if manifest.get('complete') is not True or manifest['sample_count'] != 435:
        raise ValueError('Expected unchanged legacy435 dataset for compatibility audit only')
    files[str(DATASET)] = sha(DATASET)
    support = {}
    for g in (1, 6, 7, 8):
        entry = manifest['gesture_files'][str(g)]
        path = Path(entry['path'])
        if sha(path) != entry['sha256']:
            raise ValueError('Old gesture array changed')
        files[str(path)] = entry['sha256']
        with np.load(path, allow_pickle=False) as z:
            labels, targets, splits = z['labels'], z['target_index'], z['split']
            if z['features'].shape != (len(labels), 480) or not np.isfinite(z['features']).all():
                raise ValueError('Invalid legacy features')
            if not np.isin(labels, [0, 1]).all() or not ((0 <= targets)&(targets < 20)).all():
                raise ValueError('Invalid legacy labels/index')
            rows = {}
            for split in ('train', 'internal_check'):
                mask = splits == split
                rows[split] = dict(records=int(mask.sum()), positive=int(labels[mask].sum()),
                    negative=int((labels[mask] == 0).sum()),
                    output_counts=np.bincount(targets[mask], minlength=20).tolist())
            support[str(g)] = rows
    counts = np.concatenate([r['train']['output_counts'] for r in support.values()])
    reference_conflicts = dict(
        paper=dict(epochs=20, initial_lr=.0001, scheduler='segmented; exact milestones not published',
                   loss='supplement describes cross entropy'),
        public_code=dict(epochs=100, initial_lr=.0005, weight_decay=.0005,
                         scheduler='PolyLR power .99', loss='selected-output MSE after final ReLU',
                         default_output_count=240, dataset_training_cap_per_type=100),
        published_models=dict(output_count=20, canonical_input_dim=480),
        target_status='User confirmed public code plus published models as primary reference',
        primary='public_code_and_published_models',
        training_target=dict(epochs=100, initial_lr=.0005, weight_decay=.0005,
            batch_size=128, scheduler='PolyLR power .99 once per epoch',
            loss='selected-output MSE after final ReLU', output_count=20,
            explicit_override='num_two_finger_angle=1 to match published heads',
            checkpoint_selection='original save_model P/R rule at .5/.7/.9'),
        paper_discrepancies_are_separate=True,
        author_complete_training_history_recovered=False)
    reference_conflicts['deployment_reference'] = dict(primary_hand_profile='public DH3 code',
        collection=dict(source='collect_dh3_grasp_data.py',CGR_branches=1,top_CGR=500,
            hand_type='fixed requested ordinal',extra_depth='independent uniform0/10/20/30mm',
            hand_voxel_m=.003,approach_m=.04,collision_point_threshold=2,min_width_m=.01,
            rig_flip='R[1,1]>0: negate columns0:2'),
        deployment=dict(source='robot_dh3.py',CGR_branches=11,top_CGR=1000,
            score_topk=3500,strict_score_threshold=.85,collision_topk=400,
            hand_voxel_m=.003,approach_m=.05,collision_point_threshold=0,min_width_m=.045,
            rig_flip='R[1,1]>0: negate columns1:3'),
        literal_collection_deployment_flip_difference_preserved_and_disclosed=True,
        loader_override='Deterministic four-head loader; original nested dictionary reset is a documented I/O defect',
        boya_substitutions='Boya hand mesh/FK, width/reference table, controller, simulation/sensor and arm require separate validation',
        author_parameters_do_not_establish_boya_hardware_equivalence=True)
    expansion = dict(status='specification_not_started', authorized_in_session=True,
        source_objects='S40 first; verify official identity/scale and scene assets before launch',
        tentative_boya_types=[1, 6, 7, 8], types_require_calibration_semantics_validation=True,
        valid_training_records_per_type=[100, 300, 1000], nested_same_collection_prefix=True,
        valid_total_four_types=[400, 1200, 4000],
        larger_data_is_explicit_extension=True, include_both_outcomes=True,
        success_rate_not_a_stopping_or_sampling_rule=True, null_is_not_negative=True,
        object_group_split_before_collection=True, exact_split_and_attempt_budget_frozen_before_launch=False,
        legacy435_automatic_merge=False, legacy_compatibility_status='pending action/label/collection protocol comparison',
        scene_initialization_must_precede_RGBD=True, observation_must_match_execution_start=True,
        external_test_must_follow_final_training=True, previously_used_external_objects_are_development=True,
        learning_curve_cannot_use_external_results_to_change_sampler_or_training=True)
    result = dict(schema='anydex_boya_full_alignment_gap_audit_v1', status='audit_complete_pipeline_not_ready',
        original_commit='c9c4a43df33e40860417c7e2dd02f5122d3b2da2',
        pipeline_ready=False, launch_collection=False, launch_training=False, launch_external_test=False,
        gaps=gaps, old_dataset_support=support,
        old_fit_records=sum(r['train']['records'] for r in support.values()),
        old_internal_records=sum(r['internal_check']['records'] for r in support.values()),
        old_fit_empty_output_cells=int((counts == 0).sum()),
        old_fit_one_or_two_record_cells=int(((counts > 0)&(counts <= 2)).sum()),
        reference_conflicts=reference_conflicts, expansion=expansion,
        remaining_hand_and_scene_checks=[
            'Boya pad and per-type reference frames/action semantics, including inactive fingers and closure endpoints',
            'Source contact geometry, stable initialization and dynamic multi-object scene support',
            'Choice of one author deployment profile; hand-specific parameters explicitly adapted without mixing branches',
            'Transport execution and labels; existing lift-only records do not prove transfer success'],
        source_and_input_sha256=files, new_CGR_forwards=0, new_physics_steps=0,
        real_data_optimizer_updates=0, new_models_written=0)
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('Input/source changed during audit')
    output.mkdir(parents=True)
    (output/'AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    result = run(parser.parse_args().output_dir)
    print(json.dumps(dict(status=result['status'], gaps=len(result['gaps']),
                          fit=result['old_fit_records'], internal=result['old_internal_records'])))
