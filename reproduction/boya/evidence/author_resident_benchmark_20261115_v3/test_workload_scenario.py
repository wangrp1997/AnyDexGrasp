"""Offline checks of evidence-only arithmetic; no simulator dependencies."""
import unittest

from analyse_workload_scenario import GESTURES, union_seconds, weighted_components


class WorkloadScenarioTests(unittest.TestCase):
    def test_prepare_union_does_not_add_overlapping_workers(self):
        self.assertEqual(union_seconds([(0, 60), (0, 60), (0, 60), (60, 120)]), 120)

    def test_random_mixture_keeps_null_cost_but_not_four_executions(self):
        parts, serial = weighted_components(
            capture_fraction=7/11, execution_scene_fraction=4/11,
            execution_fractions={1:4/11, 6:3/11, 7:3/11, 8:4/11},
            build_capture_cgr_prepare_per_scene=100, fixed_execution_per_group=20,
            variable_execution_by_gesture={g:100 for g in GESTURES},
            observed_parallel_wall_per_serial_work=.5,
            audit_by_gesture={g:50 for g in GESTURES}, bookkeeping_per_attempt=10)
        self.assertAlmostEqual(serial, 1480/11)
        self.assertAlmostEqual(parts['build_capture_cgr_prepare'], 700/11)
        self.assertAlmostEqual(parts['first_numeric_audit'], 700/11)
        self.assertEqual(parts['bookkeeping'], 10)
        self.assertAlmostEqual(parts['total'], (700+740+700)/11+10)
        self.assertLess(serial, 20+4*100)

    def test_invalid_workload_cannot_produce_eta(self):
        with self.assertRaises(ValueError):
            weighted_components(capture_fraction=2, execution_scene_fraction=0,
                execution_fractions={g:0 for g in GESTURES},
                build_capture_cgr_prepare_per_scene=0, fixed_execution_per_group=0,
                variable_execution_by_gesture={g:0 for g in GESTURES},
                observed_parallel_wall_per_serial_work=.5,
                audit_by_gesture={g:0 for g in GESTURES}, bookkeeping_per_attempt=0)


if __name__ == '__main__':
    unittest.main()
