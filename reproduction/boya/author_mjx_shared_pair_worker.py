"""Plan two ranks from one collision-filtered pool and one camera frame."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np

import author_collection_batch as B
from author_mjx_policy_compare_worker import ranking_order


class Scorer:
    def __init__(self, binding):
        import torch
        import author_training as training

        B.verify({'head': binding})
        state = torch.load(binding['path'], map_location='cpu', weights_only=True)
        if (state['schema'], state['variant'], state['activation'], state['epochs'], state['optimizer_updates']) != (
                'anydex_mjx_head_diagnostic_checkpoint_v1', 'linear_output', 'linear', 100, 100):
            raise ValueError('The fixed preactivation-MSE diagnostic head is required')
        torch.set_num_threads(1)
        self.torch = torch
        self.model = training.components()[1]().eval()
        self.model.load_state_dict(state['model_state_dict'], strict=True)

    def scores(self, candidates):
        if not candidates:
            return np.empty(0, dtype=np.float32)
        features = np.asarray([c['canonical_cgr480'] for c in candidates], dtype=np.float32)
        targets = np.asarray([c['raw_depth_index'] * 4 + c['extra_depth_index'] for c in candidates])
        if features.shape != (len(candidates), 480) or not np.isfinite(features).all() or not np.all((0 <= targets) & (targets < 20)):
            raise ValueError('Original canonical features and depth addresses required')
        with self.torch.no_grad():
            values = self.model(self.torch.from_numpy(features))[0].numpy()
        return values[np.arange(len(targets)), 0, targets]


def choice_receipt(prepared, scorer):
    candidates = prepared['candidates']
    if len({c['source_row_index'] for c in candidates}) != len(candidates):
        raise ValueError('Repeated source row in post-collision candidate pool')
    scores = scorer.scores(candidates)
    order = ranking_order(scores).tolist()
    rank = order[0] if order else None
    return dict(candidate_count=len(candidates), head_scores=scores.tolist(), head_order=order,
                cgr_rank=0 if candidates else None, head_rank=rank, disagreement=bool(rank),
                hashes=[c['candidate_sha256'] for c in candidates])


def selected_pool(prepared, receipt, policy):
    result = copy.deepcopy(prepared)
    if policy not in ('cgr', 'head'):
        raise ValueError('Unknown comparison policy')
    rank = receipt[policy + '_rank']
    chosen = [result['candidates'][rank]] if receipt['disagreement'] else []
    result.update(candidates=chosen, status='prepared' if chosen else 'no_geometry',
                  selected_candidate_sha256=chosen[0]['candidate_sha256'] if chosen else None,
                  selection='One predeclared policy choice from the bound shared comparison pool')
    result['counts']['top10'] = len(chosen)
    return result


def run(requests, output):
    import author_resident_prepare as resident

    resident.bind_backend(requests[0])
    import author_collect as collect
    import author_contact_pipeline as contact
    import author_mjx_measured_planning as planning

    scorer = Scorer(requests[0]['comparison_head'])
    original_prepare = resident.prepare
    original_collection = collect.prepare_collection
    original_mapping = contact.create_mapping_bundle
    cache = {}

    def prepare(request, session):
        pair_root = Path(request['pair_root'])
        pair = cache.setdefault(pair_root, {})
        observation = B.binding(request['observation'])
        if pair.get('observation', observation) != observation:
            raise ValueError('Planning policies received different observations')
        pair['observation'] = observation

        def mapping(*args, **kwargs):
            if 'mapping' not in pair:
                pair['mapping'] = original_mapping(*args, **kwargs)
            return pair['mapping']

        def shared_collection(*args, **kwargs):
            if 'pool' not in pair:
                prepared, arrays = original_collection(*args, **kwargs)
                pool_file = pair_root/'ORIGINAL_POOL.json'
                B.save(pool_file, prepared, exclusive=True)
                receipt = choice_receipt(prepared, scorer)
                receipt.update(shared_pool=B.binding(pool_file), observation=observation,
                               head=request['comparison_head'])
                B.save(pair_root/'RANKING.json', receipt, exclusive=True)
                pair.update(pool=prepared, arrays=arrays, receipt=receipt)
            chosen = selected_pool(pair['pool'], pair['receipt'], request['comparison_policy'])
            B.save(Path(request['output_dir'])/'SHARED_CHOICE.json', dict(
                ranking=B.binding(pair_root/'RANKING.json'), policy=request['comparison_policy'],
                selected_candidate_sha256=chosen['selected_candidate_sha256']), exclusive=True)
            return chosen, pair['arrays']

        with patch.object(contact, 'create_mapping_bundle', mapping), patch.object(collect, 'prepare_collection', shared_collection):
            return original_prepare(request, session)

    with patch.object(resident, 'prepare', prepare):
        planning.run(requests, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(B.read(args.requests), args.output)
