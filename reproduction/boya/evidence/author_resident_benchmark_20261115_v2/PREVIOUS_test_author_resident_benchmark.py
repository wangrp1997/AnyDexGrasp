"""Small receipt/ETA tests. No Kit, planners, models, or physical dispatch."""
import copy
import tempfile
from pathlib import Path
import unittest

import author_resident_benchmark as M


def scene(scene_id, split, labels, index=0):
    return [dict(scene_id=scene_id, split=split, gesture_id=g,
                 trial_id=f'{scene_id}:g{g}', queue_index=index,
                 status='labelled' if label is not None else 'null_geometry',
                 label=label, source_object_ids=['a', 'b', 'c'],
                 source_object_digests=dict(a='a', b='b', c='c'),
                 actual_physics_steps=10 if label is not None else 0)
            for g, label in zip(M.GESTURES, labels)]


class ResidentBenchmarkTest(unittest.TestCase):
    def test_valid_negatives_count_but_nulls_and_infra_do_not(self):
        rows = scene('a', 'fit', [0, 1, None, None])
        rows += scene('b', 'fit', [None]*4, 1)
        rows[4]['status'] = 'infrastructure_failure'
        summary = M.summarize_rows(rows)
        self.assertEqual(summary['strata']['fit']['1']['eligible_fraction'], .5)
        self.assertEqual(summary['strata']['fit']['1']['negative'], 1)
        self.assertEqual(summary['strata']['fit']['1']['attempts'], 2)
        self.assertEqual(summary['strata']['fit']['7']['eligible'], 0)
        self.assertIsNone(summary['strata']['internal']['1']['eligible_fraction'])

    def test_incomplete_duplicate_and_boolean_outcomes_rejected(self):
        rows = scene('a', 'fit', [0, 1, None, None])
        with self.assertRaisesRegex(ValueError, 'Incomplete'): M.summarize_rows(rows[:3])
        with self.assertRaisesRegex(ValueError, 'Duplicate'): M.summarize_rows(rows+rows)
        for change in (dict(label=False), dict(status='infrastructure_failure')):
            bad = copy.deepcopy(rows); bad[0].update(change)
            with self.assertRaisesRegex(ValueError, 'binary'): M.summarize_rows(bad)

    def test_diagnostic_outcomes_are_counted_but_never_admitted_to_training(self):
        rows = scene('a', 'fit', [0, 1, None, None])
        for row in rows[:2]:
            row.update(status='diagnostic_labelled', diagnostic_label=row['label'],
                       label=None, exclude_training=True, formal_training_eligible=False)
        self.assertEqual(M.summarize_rows(rows)['total_eligible'], 2)
        rows[0]['formal_training_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'outside training'): M.summarize_rows(rows)
        rows[0].update(formal_training_eligible=False, status='started', diagnostic_label=None)
        with self.assertRaisesRegex(ValueError, 'Nonterminal'): M.summarize_rows(rows)

    def test_eta_requires_slowest_gesture_and_separate_internal_yield(self):
        rows = scene('a', 'fit', [1, 0, 1, 1])
        rows += scene('b', 'fit', [1, None, 0, 1], 1)
        rows += scene('i', 'internal', [None, None, 1, None], 2)
        yields = M.summarize_rows(rows); yields['selection_basis'] = 'observational_queue'
        timing = dict(full_pipeline_wall_s=300, total_scenes=3, selection_basis='observational_queue')
        eta = M.estimate_eta(yields, timing)
        self.assertEqual(eta['splits']['fit']['required_scene_count_point'], 2000)
        self.assertEqual(eta['splits']['fit']['point_seconds'], 200000)
        self.assertEqual(eta['splits']['internal']['gestures']['7']['scene_demand_point'], 200)
        self.assertIsNone(eta['total_point_seconds'])
        self.assertIsNone(eta['reliable_upper_bound_seconds'])

    def test_control_selection_and_execution_only_wall_cannot_silently_estimate_eta(self):
        yields = M.summarize_rows(scene('a', 'fit', [1]*4))
        yields['selection_basis'] = 'known_scene_control'
        timing = dict(full_pipeline_wall_s=100, total_scenes=1, selection_basis='known_scene_control')
        with self.assertRaisesRegex(ValueError, 'unselected'): M.estimate_eta(yields, timing)
        yields['selection_basis'] = 'observational_queue'
        with self.assertRaisesRegex(ValueError, 'explicitly'): M.estimate_eta(yields, timing)
        eta = M.estimate_eta(yields, timing, control_timing_scenario=True)
        self.assertTrue(eta['timing_from_known_control'])
        self.assertIsNone(eta['total_point_seconds'])
        # No full wall measured: an execution-only figure is never substituted.
        timing['full_pipeline_wall_s'] = None; timing['execution_wall_s'] = 1
        eta = M.estimate_eta(yields, timing, control_timing_scenario=True)
        self.assertIsNone(eta['seconds_per_attempted_scene'])

    def write_run(self, root):
        root = Path(root); root.mkdir(parents=True)
        rows = scene('a', 'fit', [0, None, None, None])
        first, second = root/'stages/capture/STAGE.json', root/'stages/prepare_g1/STAGE.json'
        M.save(first, dict(started_unix_s=100., wall_s=20.))
        M.save(second, dict(started_unix_s=110., wall_s=20.))
        audit = root/'RUNTIME_AUDIT.json'; M.save(audit, dict(status='pass', label=0))
        rows[0]['artifact_bindings'] = dict(capture_stage=M.binding(first),
            prepare_stage=M.binding(second), runtime_audit=M.binding(audit))
        rows[0]['trial'] = dict(candidate_sha256='candidate', commands=dict(sha256='commands'))
        M.save(root/'LEDGER.json', dict(rows=rows))
        return rows, audit

    def test_stage_span_not_overlapping_sum_and_first_audit_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; rows, audit = self.write_run(root)
            result = M.analyze_run(root)
            self.assertEqual(result['full_pipeline_wall_s'], 30.)
            self.assertEqual(sum(x['wall_sum_s'] for x in result['stage_costs'].values()), 40.)
            self.assertTrue(result['wall_is_lower_bound'])
            self.assertEqual(result['numeric_replays_performed'], 0)
            M.save(root/'PROGRESS.json', dict(timing_seconds=dict(total_wall=45.)))
            self.assertEqual(M.analyze_run(root)['full_pipeline_wall_s'], 45.)
            self.assertEqual(M.analyze_run(root, scene_indices=[0])['full_pipeline_wall_s'], 30.)
            audit.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'): M.analyze_run(root)

    def test_freeze_only_exact_scene_seed_small_budget_and_training_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'old'; old.mkdir()
            spec = root/'spec.json'; M.save(spec, dict(scene_id='s'))
            desc = dict(scene_id='s', split='fit', scene_spec=M.binding(spec))
            queue = dict(seed=17, scenes=[desc])
            M.save(old/'BATCH_PROTOCOL.json', dict(queue=queue))
            M.save(old/'LEDGER.json', dict(rows=[]))
            resident = root/'resident.json'
            M.save(resident, dict(queue=queue, exclude_training=True, automatic_training=False))
            protocol = M.freeze_benchmark(old, resident, root/'freeze', scene_indices=[0])
            self.assertEqual(protocol['gesture_attempt_budget'], 4)
            self.assertFalse(protocol['population_yield_inference_allowed'])
            with self.assertRaises(FileExistsError):
                M.freeze_benchmark(old, resident, root/'freeze', scene_indices=[0])
            bad = M.read(resident); bad['queue']['seed'] = 18
            resident.write_text(__import__('json').dumps(bad))
            with self.assertRaisesRegex(ValueError, 'seed'):
                M.freeze_benchmark(old, resident, root/'bad', scene_indices=[0])
            with self.assertRaisesRegex(ValueError, 'small'):
                M.freeze_benchmark(old, resident, root/'large', scene_indices=list(range(7)))

    def test_pair_comparison_reports_command_difference_and_null_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; self.write_run(root)
            a = M.analyze_run(root); b = copy.deepcopy(a)
            b['full_pipeline_wall_s'] = 15.
            b['trial_fingerprints']['a:g1']['commands_sha256'] = 'changed'
            result = M.compare_controls(a, b)
            self.assertEqual(result['observed_control_wall_ratio'], 2.)
            self.assertFalse(result['comparisons']['a:g1']['commands_sha256'])
            self.assertIsNone(result['comparisons']['a:g6']['label'])
            self.assertFalse(result['population_yield_inference_allowed'])
            b['trial_fingerprints']['a:g1']['source_object_digests']['c'] = 'other'
            with self.assertRaisesRegex(ValueError, 'identities'): M.compare_controls(a, b)

    def test_combination_rejects_repeated_or_selected_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; self.write_run(root)
            a = M.analyze_run(root)
            self.assertEqual(M.combine_yields([a])['total_scenes'], 1)
            with self.assertRaisesRegex(ValueError, 'Repeated'): M.combine_yields([a, a])
            a['selection_basis'] = 'known_scene_control'
            with self.assertRaisesRegex(ValueError, 'controls'): M.combine_yields([a])


if __name__ == '__main__':
    unittest.main()
