"""Meaningful source-policy and no-fabricated-label checks for the new collector."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import author_collect as collect
from trial_record import json_sha


def inputs(n=520):
    raw = np.zeros((n, 15), np.float32)
    raw[:, 0] = np.linspace(.01, .99, n)
    raw[:, 1] = .04; raw[:, 2] = .02
    # R first column is optical +Z and R[1,1] positive -> collection flip.
    R = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    raw[:, 3:12] = R.reshape(9); raw[:, 12:15] = [0., -.05, .5]
    features = np.tile(np.arange(2261, dtype=np.float32), (n, 1))
    features[:, 240:720] += np.arange(n)[:, None]*1000
    features[:, -2] = np.arange(n)%24; features[:, -1] = .02
    return raw, features


def domain():
    return dict(mode='source_optical', sensor_adapter=dict(description='Synthetic test camera',
        input_depth_range_m=[.15, .72], cache_is_unfiltered_parser=True))


class AuthorCollectTests(unittest.TestCase):
    def test_source_masks_strict_boundaries_no_hand_or_table_gate(self):
        raw, _ = inputs(6)
        raw[1, 1] = .01; raw[2, 1] = .099
        raw[3, 12] = -.25; raw[4, 13] = .03
        raw[5, 3:12] = np.array([[np.sqrt(.19), 0., -.9], [0., 1., 0.], [.9, 0., np.sqrt(.19)]]).reshape(9)
        result = collect.source_domain(raw, np.eye(4), domain())
        np.testing.assert_array_equal(result['keep_indices'], [0])
        self.assertFalse(result['hand_or_table_gate_before_top500'])
        modified = domain(); modified['table_clearance_m'] = .001
        with self.assertRaises(ValueError): collect.source_domain(raw, np.eye(4), modified)

    def test_canonical_source_flag_is_independent_of_collection_pose_flip(self):
        raw, features = inputs(2)
        flags = np.array([True, False])
        actual = collect.canonical_features(features, flags)['cgr480']
        expected = []
        for i in range(2):
            values = features[i, 240:720].copy().reshape(2, 48, 5)
            if flags[i]: values = np.roll(values, 24, axis=1)
            expected.append(np.roll(values, -int(features[i, -2]), axis=1).reshape(480))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(features[:, 240:720], inputs(2)[1][:, 240:720])

    def test_declared_camera_domain_transforms_points_and_directions_only(self):
        raw, _ = inputs(2)
        raw[1, 12] = 2.
        config = dict(mode='declared_world_rig', sensor_adapter=domain()['sensor_adapter'],
            world_workspace_aabb_m=[[9., -1., 0.], [11., 1., 1.]],
            approach_world_cone=dict(axis_world=[0., 0., 1.], cos_threshold=.9),
            raw_width_bounds_m=[.01, .099])
        transform = np.eye(4); transform[0, 3] = 10.
        result = collect.source_domain(raw, transform, config)
        np.testing.assert_array_equal(result['keep_indices'], [0])
        self.assertFalse(result['learned_score_used'])

    def test_filtered_augmented_and_wrong_observation_caches_are_rejected(self):
        import copy
        bindings = {k: dict(sha256=v) for k, v in [('raw_candidates', 'raw'),
            ('representation_outputs', 'representation'), ('observation', 'observation')]}
        receipt = dict(status='representation_forward_complete', test_time_augmentation_branches=1,
            raw_candidates=dict(count=2048, status='original parse_preds output before rig-specific width/approach/workspace and collision filtering'),
            output_files={'raw_camera_candidates.npz': dict(sha256='raw'),
                          'representation_outputs.npz': dict(sha256='representation')},
            scene_binding=dict(observation_sha256='observation'), args=dict(depth_min=.15, depth_max=.72))
        collect.validate_cache_receipt(receipt, bindings, 2048, domain()['sensor_adapter'])
        for mutation in ('filtered', 'augmented', 'observation', 'raw_hash', 'sensor'):
            with self.subTest(mutation=mutation):
                wrong = copy.deepcopy(receipt)
                if mutation == 'filtered': wrong['raw_candidates']['status'] = 'old hand filtered pool'
                elif mutation == 'augmented': wrong['test_time_augmentation_branches'] = 11
                elif mutation == 'observation': wrong['scene_binding']['observation_sha256'] = 'other'
                elif mutation == 'raw_hash': wrong['output_files']['raw_camera_candidates.npz']['sha256'] = 'other'
                else: wrong['args']['depth_max'] = .65
                with self.assertRaises(ValueError):
                    collect.validate_cache_receipt(wrong, bindings, 2048, domain()['sensor_adapter'])

    def test_full_policy_top500_flip_single_depth_and_postgeometry_original_rank(self):
        raw, features = inputs()
        slot = dict(slot_id='g6:test:ff', independent_q=[0.]*16, reference_width_m=.068,
                    palm_T_reference=np.eye(4).tolist())
        def fake_detect(rows, gestures, extra, scene, **kwargs):
            self.assertEqual(kwargs['profile'], 'dh3_collection')
            self.assertEqual(len(rows), 500)
            np.testing.assert_array_equal(rows[:, 0], raw[::-1][:500, 0])
            expected = raw[::-1][:500, 3:12].copy().reshape(-1, 3, 3)
            expected[:, :, :2] *= -1
            np.testing.assert_array_equal(rows[:, 4:13].reshape(-1, 3, 3), expected)
            self.assertTrue(np.isin(extra, [0., .01, .02, .03]).all())
            mapped = [SimpleNamespace(slot=slot, palm_before_depth=np.eye(4)) for _ in rows]
            return dict(accepted_source_indices=np.array([7, 2, 15]),
                hand=SimpleNamespace(ids=np.arange(500), mapped=mapped),
                mapping_available_mask=np.ones(500, bool), source_width_mask=np.ones(500, bool))
        with patch.object(collect.collision, 'detect', side_effect=fake_detect):
            result, trace = collect.prepare_collection(raw, features, np.array([[2., 2., 2.]]),
                world_T_camera=np.eye(4), calibration={}, library=None, gesture_id=6,
                source_reference_type=2, seed=123, domain=domain(),
                scene_binding=dict(scene_id='synthetic'), input_bindings=dict(synthetic='fixture'))
        self.assertEqual([c['source_top500_rank'] for c in result['candidates']], [2, 7, 15])
        chosen = result['candidates'][0]
        self.assertEqual(chosen['source_row_index'], 517)
        self.assertEqual(chosen['source_reference_type'], 2)
        self.assertTrue(chosen['source_if_flip'])
        np.testing.assert_allclose(chosen['camera_T_palm'][2][3], -chosen['total_depth_m'], atol=1e-8)
        self.assertEqual(chosen['canonicalization']['source_collection_flip_columns'], [0, 1])
        self.assertIsNone(chosen['label']); self.assertEqual(chosen['label_state'], 'unexecuted')
        self.assertEqual(result['new_labels'], 0)
        self.assertEqual(trace['actual_rows17_camera'].shape, (500, 17))

    def test_no_plan_infra_or_wrong_task_never_fabricate_negative_label(self):
        candidate = dict(schema=collect.CANDIDATE_SCHEMA, slot_id='synthetic',
                         actual_row15_camera=[0.]*15, world_T_camera=np.eye(4).tolist(),
                         extra_depth_m=0., task='declared_boya_clutter_transport_release_v1')
        candidate['candidate_sha256'] = json_sha(candidate)
        class Contract:
            def validate_command_arrays(self, *args, **kwargs): return dict(ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bound = collect.write_bound_commands(candidate, dict(target=np.ones((3, 20))), root/'commands.npz',
                execution_spec=dict(task=candidate['task']), action_contract=Contract())
            baseline = dict(collection_candidate_sha256=candidate['candidate_sha256'],
                commands_sha256=bound['commands_sha256'], complete=True, label_eligible=True,
                total_physics_steps=30, completed_phases=['approach', 'close', 'retreat', 'transport', 'release', 'post_release'],
                task=candidate['task'], task_success=False)
            for name, changes, label in [('complete_negative', {}, 0), ('no_plan', dict(total_physics_steps=0), None),
                ('infra', dict(infrastructure_error='device'), None), ('wrong_task', dict(task='lift_only'), None),
                ('incomplete', dict(completed_phases=['lift']), None), ('actual_success', dict(task_success=True), 1)]:
                with self.subTest(name=name):
                    value = dict(baseline, **changes); path = root/(name+'.json')
                    path.write_text(json.dumps(value))
                    self.assertEqual(collect.collect_execution(candidate, bound, path)['label'], label)
            changed = dict(candidate, extra_depth_m=.03)
            with self.assertRaises(ValueError):
                collect.write_bound_commands(changed, {}, root/'bad.npz', execution_spec=dict(task=candidate['task']))


if __name__ == '__main__': unittest.main()
