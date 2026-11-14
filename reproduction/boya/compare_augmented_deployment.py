"""Frozen 2x2 offline Boya comparison: identity/11 branches x two OLD heads.

One already-used RGB-D only. Reuses unchanged hand domain, strict .85 score
gate and full cloud admission. No arm plans, physics, training or CGR forward.
All four rankings are saved before the first full-cloud geometry query.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from asset_adapter import BoyaAssets
from boya_hand_domain import SourceHandSupport, expand_hand_queries, make_hand_queries, validate_policy
from deploy_score_first import PROFILES, admit, calibrated_slots, validate_cached_forward
from evaluate_boya_decision import Reader, original_inference_class, save
import evaluate_boya_external as preact
import evaluate_boya_original_recipe as mse

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ('original_mse', 'preactivation')
INPUTS = ('identity', 'augmented11')


def candidate_identity(row):
    if row is None:
        return None
    return [row['source_branch'], row['source_branch_row'], row['gesture_id'],
            row['if_flip'], row['raw_depth_index'], row['extra_depth_index']]


def add_provenance(queries, raw_archive):
    for query in queries:
        i = query['source_candidate_index']
        query['source_branch'] = int(raw_archive['source_branch'][i]) if 'source_branch' in raw_archive else 0
        query['source_branch_row'] = int(raw_archive['source_branch_row'][i]) if 'source_branch_row' in raw_archive else i
        query['source_scene_reflection'] = bool(raw_archive['source_scene_reflection'][i]) if 'source_scene_reflection' in raw_archive else False


def score_queries(queries, inputs, models):
    result = copy.deepcopy(queries)
    for gesture, model in models.items():
        ids = [i for i, q in enumerate(result) if q['gesture_id'] == gesture]
        with torch.inference_mode():
            for start in range(0, len(ids), 128):
                group = ids[start:start+128]
                pred, _ = model(torch.from_numpy(np.stack([inputs[i] for i in group])))
                if tuple(pred.shape) != (len(group), 1, 20) or not torch.isfinite(pred).all():
                    raise ValueError('Original inference output contract failed')
                for i, scores in zip(group, pred[:, 0].numpy()):
                    result[i]['all_head_outputs20'] = scores.tolist()
    return result


def run(protocol_path):
    reader = Reader(); protocol = reader.read(protocol_path)
    if (protocol.get('schema') != 'boya_augmented_deployment_pair_v1'
            or protocol.get('object_id') != 'egad_A0'
            or protocol.get('recipe_order') != list(RECIPES)
            or protocol.get('input_order') != list(INPUTS)
            or any(protocol.get(k) != 0 for k in ('new_physics_steps', 'new_training_steps', 'new_CGR_forwards'))):
        raise ValueError('Explicit frozen A0 offline 2x2 protocol required')
    for path, digest in protocol['frozen_files'].items():
        reader.freeze(path, digest)
    output = Path(protocol['output_dir']); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress = dict(status='starting', head_models_loaded=0, full_cloud_checks=0,
                    new_physics_steps=0, new_training_steps=0, new_CGR_forwards=0)

    def update(**values):
        progress.update(values, wall_seconds=time.monotonic()-started)
        temp = output/'PROGRESS.tmp'; temp.write_text(json.dumps(progress, indent=2)+'\n')
        temp.replace(output/'PROGRESS.json')

    update()
    try:
        obs_path = Path(protocol['observation_json']); obs = reader.read(obs_path)
        policy = reader.read(protocol['domain_policy']); validate_policy(policy)
        if policy != json.loads((ROOT/'reproduction/boya/evidence/s40_hand_domain_20261107/POLICY.json').read_text()):
            raise ValueError('Use the original unchanged Boya hand-domain policy')
        for path, digest in policy['source_sha256'].items():
            reader.freeze(path, digest)
        profile = copy.deepcopy(PROFILES['released-dh3-v1'])
        if protocol['profile'] != profile:
            raise ValueError('Original budgets and strict float32 .85 gate required')
        assets = BoyaAssets()
        families = calibrated_slots(assets, reader.read(protocol['calibration']), reader)
        support = SourceHandSupport(assets)
        cls = original_inference_class(ROOT/'models/minkowski_graspnet_single_point.py', torch)
        torch.set_num_threads(4); models = {}; metadata = {}
        for recipe in RECIPES:
            training = reader.read(protocol['training_results'][recipe])
            validator = mse if recipe == 'original_mse' else preact
            allowed, internal = validator.validate_training(training)
            models[recipe] = {}; metadata[recipe] = {}
            for gesture in (1, 6, 7, 8):
                item = training['internal_heads'][str(gesture)]
                reader.freeze(item['path'], item['sha256'])
                checkpoint = torch.load(item['path'], map_location='cpu', weights_only=True)
                fit = validator.validate_checkpoint(checkpoint, gesture, training['dataset_manifest']['sha256'], allowed, internal, set())
                model = cls(**checkpoint['architecture_kwargs'])
                model.load_state_dict(checkpoint['model_state_dict'], strict=True); model.eval()
                models[recipe][gesture] = model
                metadata[recipe][str(gesture)] = dict(path=item['path'], sha256=item['sha256'],
                    strict_load=True, fit_mesh_groups=len(fit), fixed_epoch=checkpoint['epoch'],
                    fit_scope='Original 32-fit Boya heads; known-scene diagnostic, not an unseen test')
                progress['head_models_loaded'] += 1
        update(status='hand_domain_and_ranking')
        pending = []
        for input_kind in INPUTS:
            frame = Path(protocol['representation_dirs'][input_kind])
            receipt = reader.read(frame/'RESULTS.json')
            validate_cached_forward(obs_path, obs, frame, receipt, reader)
            if receipt.get('test_time_augmentation_branches') != (1 if input_kind == 'identity' else 11):
                raise ValueError('Wrong declared representation branches')
            with np.load(frame/'raw_camera_candidates.npz', allow_pickle=False) as raw_archive:
                raw, features = raw_archive['candidates'].copy(), raw_archive['candidate_features'].copy()
                queries, inputs, indices, domain = make_hand_queries(raw, features, obs['T_WC'], policy, profile,
                    families, assets, dict(scope='Fixed known A0 pair; no outcomes', representation_dir=str(frame)), support=support)
                add_provenance(queries, raw_archive)
                selected_branch_counts = {str(i):sum(int(raw_archive['source_branch'][j]) == i for j in indices)
                    if 'source_branch' in raw_archive else (len(indices) if i == 0 else 0) for i in range(11)}
            save(output/f'{input_kind}_DOMAIN.json', domain)
            for recipe in RECIPES:
                case = output/f'{input_kind}__{recipe}'; case.mkdir()
                scored = score_queries(queries, inputs, models[recipe])
                ranked, shortlist = expand_hand_queries(scored, profile)
                np.savez_compressed(case/'head_outputs.npz',
                    canonical480=np.asarray(inputs, np.float32).reshape(-1, 480),
                    outputs20=np.asarray([q['all_head_outputs20'] for q in scored], np.float32).reshape(-1, 20))
                save(case/'queries.json', scored)
                save(case/'shortlist.json', dict(candidates=shortlist, parameters=profile,
                    evaluation_role='known_scene_augmentation_pair', flip_policy='both'))
                reader.freeze(case/'head_outputs.npz'); reader.freeze(case/'queries.json'); reader.freeze(case/'shortlist.json')
                freeze = case/'RANKING_FREEZE.json'
                save(freeze, dict(schema='boya_augmented_pair_ranking_freeze_v1',
                    labels_used_for_ranking=False, protocol_sha256=reader.files[str(protocol_path)],
                    ranking_inputs_sha256=copy.deepcopy(reader.files), models=metadata[recipe],
                    geometry_calls_before_freeze=0, support_plane_queries_before_freeze=domain['counts']['support_plane_tests'],
                    scope='Full cloud is after freeze; original table support prefix is before heads'))
                reader.freeze(freeze)
                pending.append(dict(input_kind=input_kind, recipe=recipe, case=case, frame=frame, raw=raw,
                    domain_counts=domain['counts'], queries=len(scored), expanded=len(ranked), shortlist=shortlist,
                    raw_top1000_branch_counts=selected_branch_counts))
            update(ranked_inputs=len(pending)//2, ranked_cases=len(pending))
            print(f'BOYA_PAIR ranked {input_kind}: {len(raw)} raw, {len(queries)} head queries', flush=True)
        reader.verify()
        save(output/'ALL_RANKINGS_FREEZE.json', dict(protocol_sha256=reader.files[str(protocol_path)],
            case_freezes={str(p['case']/'RANKING_FREEZE.json'):reader.files[str(p['case']/'RANKING_FREEZE.json')] for p in pending},
            full_cloud_checks_before_all_rankings_frozen=0, physical_outcomes_read=False))
        update(status='full_cloud_geometry')
        results = []
        for item in pending:
            case = item['case']; args = argparse.Namespace(stage='geometry', frame_dir=item['frame'],
                calibration=Path(protocol['calibration']), output_dir=case, max_arm_plans=4, observation=obs_path)
            admit(item['shortlist'], item['raw'], obs, args, reader, case/'RANKING_FREEZE.json')
            detail = reader.read(case/'geometry.json')
            for row in detail['checked']:
                if row.get('source_slot') != row['domain_source_slot']:
                    raise ValueError('Post-score mapping differs from the declared prehead slot')
            first = detail['admitted'][0]['candidate'] if detail['admitted'] else None
            result = dict(input_kind=item['input_kind'], recipe=item['recipe'], domain_counts=item['domain_counts'],
                head_queries=item['queries'], expanded_combinations=item['expanded'], shortlist=len(item['shortlist']),
                geometry_checked=len(detail['checked']), geometry_admitted=len(detail['admitted']),
                hand_domain_CGR_topk_branch_counts=item['raw_top1000_branch_counts'],
                first_admitted=first, first_admitted_identity=candidate_identity(first),
                full_cloud_before_rank_freeze=False, arm_checked=0, new_physics_steps=0)
            save(case/'RESULTS.json', result); results.append(result)
            update(completed_cases=len(results), full_cloud_checks=sum(r['geometry_checked'] for r in results))
            print('BOYA_PAIR geometry '+json.dumps({k:v for k,v in result.items() if k not in ('first_admitted','domain_counts')}), flush=True)
        reader.verify()
        report = dict(schema='boya_augmented_deployment_pair_results_v1', status='complete',
            scope='Same known A0, 2x2 offline hand-domain/head/full-cloud comparison; no physical efficacy or generalization claim',
            protocol_sha256=reader.files[str(protocol_path)], cases=results, models=metadata,
            head_models_loaded=progress['head_models_loaded'], ranking_labels_read=False,
            threshold_changed=False, mapping_changed=False, closure_changed=False,
            new_CGR_forwards=0, new_training_steps=0, new_physics_steps=0, arm_checked=0,
            frozen_inputs_unchanged=True, input_sha256=reader.files, wall_seconds=time.monotonic()-started)
        save(output/'RESULTS.json', report); update(status='complete')
        return report
    except Exception as exc:
        update(status='failed', error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args(); run(args.protocol.resolve())


if __name__ == '__main__':
    main()
