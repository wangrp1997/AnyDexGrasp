"""Synthetic ranking/denominator tests; no model training or physical steps."""
import copy
import unittest
from pathlib import Path
import numpy as np
from fixed_pool_ranking import ROOT, SCORERS, THRESHOLD, _binding, rank_candidates, summarize
from evaluate_boya_decision import Reader
from trial_record import json_sha, official_functions


def fixture():
    rows = []
    for cid, source, p, m, cgr in [('b', 20, .9, 0., .5), ('a', 10, .8, 0., .5), ('c', 30, .7, .2, .4)]:
        rows.append(dict(candidate_id=cid, object_id='x', source_candidate_index=source,
            gesture_id=1, raw_depth_index=2, extra_depth_index=0, if_flip=False,
            scores=dict(preactivation=p, original_mse=m, cgr=cgr)))
    catalog = {'candidates': [{k: r[k] for k in ('candidate_id', 'object_id')} for r in rows]}
    scores = dict(catalog_sha256=json_sha(catalog), candidates=rows, objects=rank_candidates(rows))
    outcomes = {'candidates': [dict(candidate_id=cid, object_id='x', status='valid',
        task_success=cid == 'a', physical_steps=2100, physical_gates_passed=True, early_stop=False)
        for cid in ('a', 'b', 'c')]}
    return catalog, scores, outcomes


class FixedPoolRankingTests(unittest.TestCase):
    def test_ties_use_source_not_preactivation_or_input_order(self):
        _, scores, _ = fixture()
        expected = dict(preactivation=['b', 'a', 'c'], original_mse=['c', 'a', 'b'], cgr=['a', 'b', 'c'])
        self.assertEqual(scores['objects'][0]['rankings'], expected)
        self.assertEqual(rank_candidates(list(reversed(scores['candidates']))), scores['objects'])

    def test_complete_pool_oracle_random_and_precision(self):
        catalog, scores, outcomes = fixture()
        result = summarize(catalog, scores, outcomes)
        obj = result['objects'][0]
        self.assertTrue(result['complete'])
        self.assertEqual((obj['candidate_count'], obj['success_count'], obj['failure_count']), (3, 1, 2))
        self.assertTrue(obj['any_success_oracle'])
        self.assertAlmostEqual(obj['random_top1_success_expectation'], 1 / 3)
        self.assertFalse(obj['scorers']['preactivation']['topk']['1']['contains_success'])
        self.assertTrue(obj['scorers']['cgr']['topk']['1']['contains_success'])
        for scorer in SCORERS:
            metric = obj['scorers'][scorer]['topk']['5']
            self.assertEqual(metric['effective_k'], 3)
            self.assertAlmostEqual(metric['precision_at_k'], 1 / 3)
        self.assertTrue(obj['scorers']['original_mse']['strict_threshold']['abstain'])
        self.assertFalse(obj['scorers']['original_mse']['strict_threshold']['policy_success'])

    def test_float32_strict_threshold_excludes_equality(self):
        catalog, scores, outcomes = fixture()
        for row in scores['candidates']:
            row['scores']['preactivation'] = THRESHOLD
        scores['objects'] = rank_candidates(scores['candidates'])
        metric = summarize(catalog, scores, outcomes)['objects'][0]['scorers']['preactivation']['strict_threshold']
        self.assertTrue(metric['abstain'])
        scores['candidates'][0]['scores']['preactivation'] = float(np.nextafter(np.float32(.85), np.float32(1.)))
        scores['objects'] = rank_candidates(scores['candidates'])
        metric = summarize(catalog, scores, outcomes)['objects'][0]['scorers']['preactivation']['strict_threshold']
        self.assertEqual(metric['admitted_count'], 1)
        self.assertEqual(metric['selected_candidate_id'], 'b')

    def test_early_physical_failure_is_in_denominator(self):
        catalog, scores, outcomes = fixture()
        outcomes['candidates'][1].update(early_stop=True, physical_steps=1175, physical_gates_passed=False)
        obj = summarize(catalog, scores, outcomes)['objects'][0]
        self.assertEqual(obj['valid_outcomes'], 3)
        self.assertEqual(obj['failure_count'], 2)

    def test_infrastructure_unknown_never_false_negative(self):
        catalog, scores, outcomes = fixture()
        outcomes['candidates'][0].update(status='infrastructure_error', task_success=None, physical_steps=0)
        result = summarize(catalog, scores, outcomes)
        obj = result['objects'][0]
        self.assertFalse(result['complete'])
        self.assertEqual(obj['failure_count'], 2)
        self.assertEqual(obj['unresolved_count'], 1)
        self.assertIsNone(obj['any_success_oracle'])
        self.assertIsNone(obj['random_top1_success_expectation'])
        self.assertEqual(obj['random_top1_success_bounds'], [0., 1 / 3])
        self.assertIsNone(obj['scorers']['cgr']['topk']['1']['contains_success'])
        self.assertIsNone(obj['scorers']['cgr']['topk']['3']['precision_at_k'])

    def test_missing_outcome_stays_unknown(self):
        catalog, scores, outcomes = fixture()
        outcomes['candidates'].pop(1)
        result = summarize(catalog, scores, outcomes)
        self.assertEqual(result['objects'][0]['unresolved_count'], 1)
        self.assertTrue(result['objects'][0]['any_success_oracle'])

    def test_input_data_not_mutated(self):
        args = fixture()
        before = copy.deepcopy(args)
        summarize(*args)
        self.assertEqual(args, before)

    def test_catalog_or_frozen_ranking_tamper_rejected(self):
        catalog, scores, outcomes = fixture()
        scores['objects'][0]['rankings']['cgr'].reverse()
        with self.assertRaises(ValueError):
            summarize(catalog, scores, outcomes)
        catalog, scores, outcomes = fixture()
        catalog['candidates'][0]['object_id'] = 'other'
        with self.assertRaises(ValueError):
            summarize(catalog, scores, outcomes)

    def test_invalid_labels_and_duplicate_ids_rejected(self):
        for change in (dict(status='infrastructure_error', task_success=False),
                       dict(physical_steps=0), dict(physical_gates_passed=False), dict(early_stop=True)):
            catalog, scores, outcomes = fixture()
            outcomes['candidates'][0].update(change)
            with self.assertRaises(ValueError):
                summarize(catalog, scores, outcomes)
        catalog, scores, outcomes = fixture()
        outcomes['candidates'].append(copy.deepcopy(outcomes['candidates'][0]))
        with self.assertRaises(ValueError):
            summarize(catalog, scores, outcomes)

    def test_nonfinite_score_rejected(self):
        _, scores, _ = fixture()
        scores['candidates'][0]['scores']['cgr'] = float('nan')
        with self.assertRaises(ValueError):
            rank_candidates(scores['candidates'])


class BoundFeatureTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'outputs/egad_external12_20261107_v1/diagnostic/egad_A3/pool_00058/generation.json'
        if not path.exists():
            self.skipTest('Historical raw2261 fixture is not installed')
        self.item = dict(candidate_id='fixture', object_id='egad_A3', generation_json=str(path))
        self.functions = official_functions(ROOT / 'robot_dh3.py')

    def test_reconstructs_original_feature_and_ignores_history_labels(self):
        base = _binding(self.item, Reader(), self.functions, {})
        changed = dict(self.item, eligible_history=[{'oldtask_success': True}],
                       selected_history={'task_success': False}, candidate_record={'learned_score': -999})
        other = _binding(changed, Reader(), self.functions, {})
        self.assertEqual(base[0], other[0])
        np.testing.assert_array_equal(base[1], other[1])
        self.assertEqual(base[2], other[2])
        self.assertEqual(base[1].shape, (480,))
        self.assertEqual(base[0]['target_index'], 4)

    def test_corrupt_original_target_or_feature_identity_is_rejected(self):
        for field, value in (('target_index', 7), ('raw_depth_index', 4),
                             ('if_flip', True), ('source_candidate_index', -1),
                             ('canonical_CGR480_identity', {'bad': 'hash'})):
            class CorruptReader(Reader):
                def read(self, path, expected=None):
                    document = super().read(path, expected)
                    document['records'][0][field] = value
                    return document
            with self.subTest(field=field), self.assertRaises(ValueError):
                _binding(self.item, CorruptReader(), self.functions, {})

    def test_catalog_cannot_silently_override_generated_hand_type(self):
        changed = dict(self.item, gesture_id=1)
        with self.assertRaises(ValueError):
            _binding(changed, Reader(), self.functions, {})


if __name__ == '__main__':
    unittest.main()
