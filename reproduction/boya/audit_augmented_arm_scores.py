"""Independent saved-score/ranking/binding audit; CPU only, no new physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from compare_augmented_deployment import RECIPES, candidate_identity
from evaluate_boya_decision import Reader, original_inference_class, save
from trial_record import canonical_candidate, official_functions

ROOT = Path(__file__).resolve().parents[2]


def audit(protocol_path, output_path):
    reader = Reader(); protocol = reader.read(protocol_path)
    for path, digest in protocol['frozen_files'].items(): reader.freeze(path, digest)
    output = Path(protocol['output_dir']); report = reader.read(output/'RESULTS.json')
    freeze = reader.read(output/'ALL_RANKINGS_FREEZE.json')
    cls = original_inference_class(ROOT/'models/minkowski_graspnet_single_point.py', torch)
    functions = official_functions(ROOT/'robot_dh3.py')
    torch.set_num_threads(4); models = {}; loads = 0; result = []
    for recipe in RECIPES:
        models[recipe] = {}
        for g, entry in report['models'][recipe].items():
            checkpoint = torch.load(entry['path'], map_location='cpu', weights_only=True)
            model = cls(**checkpoint['architecture_kwargs']); model.load_state_dict(checkpoint['model_state_dict'], strict=True); model.eval()
            models[recipe][int(g)] = model; loads += 1
    for case in report['cases']:
        directory = output/f"{case['input_kind']}__{case['recipe']}"
        q = reader.read(directory/'queries.json'); shortlist = reader.read(directory/'shortlist.json')['candidates']
        reader.freeze(directory/'RANKING_FREEZE.json', freeze['case_freezes'][str(directory/'RANKING_FREEZE.json')])
        assert (output/'ALL_RANKINGS_FREEZE.json').stat().st_mtime_ns < (directory/'geometry.json').stat().st_mtime_ns
        with np.load(directory/'head_outputs.npz', allow_pickle=False) as z:
            x, saved = z['canonical480'], z['outputs20']
        raw_path = Path(protocol['representation_dirs'][case['input_kind']])/'raw_camera_candidates.npz'
        with np.load(raw_path, allow_pickle=False) as raw:
            candidates = raw['candidates'].copy()
            candidate_features = raw['candidate_features'].copy()
            for i, query in enumerate(q):
                source = query['source_candidate_index']
                canonical = canonical_candidate(candidates[source], candidate_features[source], functions, if_flip=query['if_flip'])
                np.testing.assert_array_equal(x[i], np.asarray(canonical['canonical_CGR480'], np.float32))
                assert hashlib.sha256(x[i].tobytes()).hexdigest() == query['canonical_CGR480_identity']['c_order_bytes_sha256']
                np.testing.assert_array_equal(saved[i], np.asarray(query['all_head_outputs20'], np.float32))
        largest_error = 0.
        for g, model in models[case['recipe']].items():
            ids = np.array([i for i, query in enumerate(q) if query['gesture_id'] == g], dtype=int)
            with torch.inference_mode():
                for start in range(0, len(ids), 128):
                    group = ids[start:start+128]; prediction = model(torch.from_numpy(x[group]))[0][:, 0].numpy()
                    largest_error = max(largest_error, float(np.max(abs(prediction-saved[group]))))
        assert largest_error == 0.
        # Expand allowed depths and apply the three budgets without calling the
        # production expansion/ranking helper.
        expanded = [(float(saved[i, query['raw_depth_index']*4+extra]), i, extra)
                    for i, query in enumerate(q) for extra in query['allowed_extra_depth_indices']]
        expanded.sort(key=lambda r:(-r[0], r[1], r[2]))
        selected = [row for row in expanded[:3500] if row[0] > float(np.float32(.85))][:400]
        assert len(selected) == len(shortlist) == case['shortlist']
        assert [(r['learned_score'], r['query_index'], r['extra_depth_index']) for r in shortlist] == selected
        detail = reader.read(directory/'geometry.json'); admitted = detail['admitted']
        assert len(detail['checked']) == len(selected) and len(admitted) == case['geometry_admitted']
        assert candidate_identity(admitted[0]['candidate'] if admitted else None) == case['first_admitted_identity']
        identity_rows = [r['candidate'] for r in admitted if r['candidate']['source_branch'] == 0]
        result.append(dict(input_kind=case['input_kind'], recipe=case['recipe'], queries=len(q),
            canonical_all_exact=True, outputs20_max_error=largest_error, ranking_all_exact=True,
            geometry_admitted=len(admitted), admitted_identity_branch=len(identity_rows),
            first_admitted_identity=case['first_admitted_identity']))
        print(f"AUDIT case={case['input_kind']}__{case['recipe']} queries={len(q)} passed", flush=True)
    reader.verify()
    result = dict(schema='boya_augmented_arm6_score_audit_v1', status='complete',
        protocol_sha256=reader.files[str(protocol_path)], cases=result,
        external_identity_and_augmented_scores_independently_recomputed=True,
        all_four_rankings_frozen_before_geometry=True, inputs_unchanged=True,
        audit_CPU_head_loads=loads, new_CGR_forwards=0, new_training_steps=0, new_physics_steps=0,
        source_sha256={str(Path(__file__).resolve()):reader.freeze(__file__)})
    save(output_path, result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.protocol.resolve(), args.output.resolve())
