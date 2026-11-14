"""CPU-only contracts for the frozen 2x2 bridge; no assets/simulator needed."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_augmented_deployment import add_provenance, candidate_identity, score_queries
from boya_hand_domain import expand_hand_queries
from deploy_score_first import PROFILES


class Scorer(torch.nn.Module):
    def __init__(self, scale):
        super().__init__(); self.scale = scale

    def forward(self, features):
        return (features[:, 0]*self.scale)[:, None, None].expand(-1, 1, 20), None


class AugmentedDeploymentContracts(unittest.TestCase):
    def test_feature_order_per_type_and_independent_old_models(self):
        queries = [dict(gesture_id=g, source_candidate_index=i, if_flip=False,
                        raw_depth_index=0, query_index=i, allowed_extra_depth_indices=[0])
                   for i,g in enumerate([8, 1, 8, 6, 1])]
        inputs = [np.full(480, i+.1, np.float32) for i in range(len(queries))]
        models = {g:Scorer(g) for g in [1, 6, 7, 8]}
        scored = score_queries(queries, inputs, models)
        self.assertTrue(all('all_head_outputs20' not in q for q in queries))
        for i, q in enumerate(scored):
            np.testing.assert_allclose(q['all_head_outputs20'], inputs[i][0]*q['gesture_id'], rtol=1e-7)
        self.assertEqual(score_queries([], [], models), [])

    def test_scene_reflection_and_thumb_flip_are_separate(self):
        q = [dict(source_candidate_index=2, gesture_id=8, if_flip=False, raw_depth_index=3, extra_depth_index=1)]
        add_provenance(q, dict(source_branch=np.array([0, 1, 2]), source_branch_row=np.array([0, 10, 20]),
                               source_scene_reflection=np.array([False, False, True])))
        self.assertEqual(candidate_identity(q[0]), [2, 20, 8, False, 3, 1])
        self.assertTrue(q[0]['source_scene_reflection']); self.assertFalse(q[0]['if_flip'])
        add_provenance(q, {})
        self.assertEqual(candidate_identity(q[0]), [0, 2, 8, False, 3, 1])
        self.assertIsNone(candidate_identity(None))

    def test_strict_gate_unchanged_and_does_not_fallback(self):
        query = dict(query_index=0, raw_depth_index=0, allowed_extra_depth_indices=[0],
                     all_head_outputs20=[float(np.float32(.85))]*20)
        _, selected = expand_hand_queries([query], PROFILES['released-dh3-v1'])
        self.assertEqual(selected, [])
        query['all_head_outputs20'][0] = float(np.nextafter(np.float32(.85), np.float32(1)))
        _, selected = expand_hand_queries([query], PROFILES['released-dh3-v1'])
        self.assertEqual(len(selected), 1)


if __name__ == '__main__':
    unittest.main()
