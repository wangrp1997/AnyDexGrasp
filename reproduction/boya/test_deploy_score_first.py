"""Functional contracts for score-first deployment; no simulator or training."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy_score_first as deployment


def query(index, score, depth=0, gesture=1, flip=False):
    outputs = np.zeros(20, dtype=np.float32)
    outputs[depth*4] = score
    return dict(query_index=index, source_candidate_index=index,
                gesture_id=gesture, if_flip=flip, raw_depth_index=depth,
                all_head_outputs20=outputs.tolist(), cgr_score=1.-index*.001)


class ScoreFirstContracts(unittest.TestCase):
    def test_candidate_lost_by_old_cgr_compression_remains_selectable(self):
        # Low antipodal rank 999 would never enter the old first-500/keep8 path.
        inputs = [query(i, .86) for i in range(1000)]
        inputs[-1]['all_head_outputs20'][0] = .99
        ranked, selected = deployment.expand_and_rank(inputs, deployment.PROFILES['released-dh3-v1'])
        self.assertEqual(selected[0]['source_candidate_index'], 999)
        self.assertEqual(len(ranked), 4000)
        self.assertEqual(len(selected), 400)

    def test_all_raw_and_extra_depth_indices(self):
        inputs = []
        for depth in range(5):
            q = query(depth, 0., depth=depth)
            q['all_head_outputs20'] = [float(i)/20 for i in range(20)]
            inputs.append(q)
        ranked, _ = deployment.expand_and_rank(inputs, deployment.PROFILES['paper-testing-budget-boya-port-v1'])
        self.assertEqual({r['target_index'] for r in ranked}, set(range(20)))
        for row in ranked:
            self.assertEqual(row['target_index'], row['raw_depth_index']*4+row['extra_depth_index'])
            self.assertAlmostEqual(row['learned_score'], row['target_index']/20)

    def test_strict_threshold_no_automatic_fallback(self):
        q = query(0, 0.)
        q['all_head_outputs20'] = [.85] * 20
        _, selected = deployment.expand_and_rank([q], deployment.PROFILES['released-dh3-v1'])
        self.assertEqual(selected, [])
        q['all_head_outputs20'] = np.full(20, .85, dtype=np.float32).tolist()
        _, selected = deployment.expand_and_rank([q], deployment.PROFILES['released-dh3-v1'])
        self.assertEqual(selected, [])  # Exact original torch float32 boundary.
        q['all_head_outputs20'][1] = .850001
        _, selected = deployment.expand_and_rank([q], deployment.PROFILES['released-dh3-v1'])
        self.assertEqual([r['extra_depth_index'] for r in selected], [1])

    def test_global_across_type_and_flip_and_labels_do_not_affect_selection(self):
        rows = [query(0, .90, gesture=1), query(1, .97, gesture=8, flip=True)]
        _, first = deployment.expand_and_rank(rows, deployment.PROFILES['released-dh3-v1'])
        for row in rows:
            row['label'] = 1 if row['gesture_id'] == 1 else 0
        _, second = deployment.expand_and_rank(rows, deployment.PROFILES['released-dh3-v1'])
        self.assertEqual(first[0]['gesture_id'], 8)
        self.assertTrue(first[0]['if_flip'])
        self.assertEqual(first[0]['query_index'], second[0]['query_index'])

    def test_empty_and_invalid_scores(self):
        self.assertEqual(deployment.expand_and_rank([], deployment.PROFILES['released-dh3-v1']), ([], []))
        bad = query(0, .9)
        bad['all_head_outputs20'][2] = float('nan')
        with self.assertRaises(ValueError):
            deployment.expand_and_rank([bad], deployment.PROFILES['released-dh3-v1'])

    def test_default_official_single_flip_and_paired_feature_binding(self):
        raw = np.zeros((2,15), dtype=np.float32)
        raw[:,0] = [.5,.7]
        raw[:,1] = .05
        raw[:,2] = [.01,.03]
        raw[:,3:12] = np.eye(3).reshape(-1)
        features = np.zeros((2,2261), dtype=np.float32)
        features[:,-1] = raw[:,2]
        features[:,240:720] = np.arange(480, dtype=np.float32)
        original = features.copy()
        queries, inputs, source_indices = deployment.make_queries(raw, features, deployment.PROFILES['released-dh3-v1'])
        self.assertEqual(source_indices, [1,0])
        self.assertEqual(len(queries), 2*4)
        self.assertTrue(all(q['if_flip'] == q['official_policy_if_flip'] for q in queries))
        self.assertTrue(np.array_equal(features, original))
        both, _, _ = deployment.make_queries(raw, features, deployment.PROFILES['released-dh3-v1'], 'both')
        self.assertEqual(len(both), 2*2*4)
        self.assertEqual({q['if_flip'] for q in both}, {False,True})
        for index, row in enumerate(queries):
            self.assertEqual(inputs[index].shape, (480,))
            self.assertEqual(row['target_index'], (3 if row['source_candidate_index']==1 else 1)*4)

    def test_unrelated_observation_cannot_reuse_cached_forward(self):
        obs_path = Path('/tmp/score_first_fixture/observation.json')
        frame = Path('/tmp/score_first_fixture/frame')
        observed = dict(T_WC=np.eye(4).tolist(), files={
            'depth': dict(path='/tmp/score_first_fixture/depth.png', sha256='depthsha')})
        receipt = dict(full_representation_forward=True,
            scene_binding=dict(observation_json=str(obs_path), observation_sha256='obssha', T_WC=observed['T_WC']),
            input_files={'/tmp/score_first_fixture/depth.png': dict(sha256='depthsha')},
            output_files={name:dict(path=str(frame/name),sha256=name) for name in
                          ('raw_camera_candidates.npz','representation_outputs.npz')})
        class Reader:
            files = {str(obs_path): 'obssha'}
            def freeze(self, path, digest):
                pass
        deployment.validate_cached_forward(obs_path, observed, frame, receipt, Reader())
        bad = copy.deepcopy(receipt)
        bad['scene_binding']['observation_sha256'] = 'different-observation'
        with self.assertRaises(ValueError):
            deployment.validate_cached_forward(obs_path, observed, frame, bad, Reader())
        bad = copy.deepcopy(receipt)
        bad['input_files']['/tmp/score_first_fixture/depth.png']['sha256'] = 'other-image'
        with self.assertRaises(ValueError):
            deployment.validate_cached_forward(obs_path, observed, frame, bad, Reader())

    def test_geometry_cannot_start_before_freeze(self):
        class Args:
            stage = 'geometry'
        with self.assertRaises(AssertionError):
            deployment.admit([], np.empty((0,15)), {}, Args(), None,
                             Path('/tmp/nonexistent-score-first-test-freeze.json'))


if __name__ == '__main__':
    unittest.main()
