"""Bounded search tests with actual obstacle-crossing edge predicates."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_joint_home_route as route


class JointHomeRouteTests(unittest.TestCase):
    def test_edges_include_exact_goal_and_obey_spacing(self):
        a, b = np.array([-.8, .3]), np.array([.9, -.9])
        points = route.edge_samples(a, b, .05)
        self.assertTrue(np.array_equal(points[-1], b))
        self.assertLessEqual(float(abs(np.diff(np.vstack([a, points]), axis=0)).max()), .05+1e-15)

    def test_deterministic_detour_around_blocking_rectangle(self):
        start, goal = np.array([-1., 0.]), np.array([1., 0.])
        checked = []
        def clear(a, b):
            points = route.edge_samples(a, b, .05); checked.append(points)
            return not np.any((abs(points[:, 0]) <= .15) & (abs(points[:, 1]) <= .6))
        self.assertFalse(clear(start, goal))
        first, audit, _ = route.rrt_connect(start, goal, [-1.5]*2, [1.5]*2, clear)
        self.assertTrue(audit['connected']); self.assertLessEqual(audit['iterations_executed'], 1200)
        self.assertTrue(np.array_equal(first[0], start)); self.assertTrue(np.array_equal(first[-1], goal))
        self.assertGreater(float(abs(first[:, 1]).max()), .6)
        self.assertTrue(all(clear(a, b) for a, b in zip(first[:-1], first[1:])))
        second, again, _ = route.rrt_connect(start, goal, [-1.5]*2, [1.5]*2, clear)
        self.assertTrue(np.array_equal(first, second)); self.assertEqual(audit, again)

    def test_blocked_search_exhausts_budget_without_success(self):
        p = copy.deepcopy(route.POLICY); p['max_iterations'] = 11
        path, audit, _ = route.rrt_connect([-1., 0.], [1., 0.], [-2.]*2, [2.]*2,
                                         lambda a, b: False, policy=p)
        self.assertIsNone(path); self.assertFalse(audit['connected'])
        self.assertEqual(audit['iterations_executed'], 11); self.assertEqual(audit['extend_calls'], 11)
        self.assertEqual(audit['tree_nodes'], [1, 1])

    def test_original_bounds_reject_changed_endpoint(self):
        for start, goal in (([-3., 0.], [1., 0.]), ([-1., 0.], [3., 0.])):
            with self.subTest(start=start, goal=goal):
                with self.assertRaises(ValueError): route.rrt_connect(start, goal, [-2.]*2, [2.]*2, lambda a,b: True)

    def test_frozen_budget(self):
        p = route.POLICY
        self.assertEqual((p['random_seed'], p['step_l2_rad'], p['edge_max_per_joint_rad'], p['goal_bias']),
                         (20261114, .25, .05, .2))
        self.assertEqual((p['max_iterations'], p['max_connect_extends'], p['ik_calls']), (1200, 32, 0))


if __name__ == '__main__': unittest.main()
