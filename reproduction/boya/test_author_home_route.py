"""Semantic guards for bounded HOME replacement, without simulator or assets."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_home_route as route


class HomeRouteTests(unittest.TestCase):
    def fixture(self):
        names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']+[f'hand{i}' for i in range(18)]
        q = np.zeros((7, 24)); q[2:, :6] = 1.
        q[4:, :6] += np.arange(3)[:, None]*.01
        return dict(joint_names=np.asarray(names), initial_position=np.zeros(24),
            joint_target=q, desired_palm=np.tile(np.eye(4), (7, 1, 1)),
            phase=np.array(['settle']*2+['home_to_pregrasp']*2+['approach']*3),
            time_s=np.arange(1, 8)*.004)

    def test_only_home_rows_change(self):
        old = self.fixture(); q = np.zeros((5, 24)); q[:, :6] = np.linspace(.1, 1, 5)[:, None]
        result = route.replace_home(old, q, np.tile(np.eye(4), (5, 1, 1)))
        self.assertEqual(len(result['phase']), 10)
        for name in ('joint_target', 'desired_palm', 'phase'):
            self.assertTrue(np.array_equal(result[name][:2], old[name][:2]))
            self.assertTrue(np.array_equal(result[name][-3:], old[name][-3:]))
        self.assertTrue(np.array_equal(result['initial_position'], old['initial_position']))
        self.assertTrue(np.array_equal(result['time_s'], np.arange(1, 11)*.004))

    def test_rejects_changed_endpoint_or_hand(self):
        for error in ('endpoint', 'hand', 'wrist'):
            with self.subTest(error=error):
                old = self.fixture(); q = old['joint_target'][2:4].copy()
                if error == 'endpoint': q[-1, 0] += .01
                elif error == 'hand': q[0, 23] += .01
                else: q[0, 6] += .01
                with self.assertRaises(ValueError):
                    route.replace_home(old, q, np.tile(np.eye(4), (2, 1, 1)))

    def test_preserves_original_desired_pregrasp_exactly(self):
        old = self.fixture(); old['desired_palm'][3, 0, 3] = .123456789
        palms = np.tile(np.eye(4), (2, 1, 1))
        result = route.replace_home(old, old['joint_target'][2:4], palms)
        self.assertTrue(np.array_equal(result['desired_palm'][3], old['desired_palm'][3]))
        self.assertEqual(palms[-1, 0, 3], 0.)

    def test_rejects_nonfinite_bad_shape_and_phase(self):
        for error in ('nan', 'width', 'phase'):
            with self.subTest(error=error):
                old = self.fixture(); q = old['joint_target'][2:4].copy()
                if error == 'nan': q[0, 0] = np.nan
                elif error == 'width': q = q[:, :23]
                else: old['phase'][2] = 'approach'
                with self.assertRaises(ValueError):
                    route.replace_home(old, q, np.tile(np.eye(4), (2, 1, 1)))

    def test_contiguous_phase_required(self):
        with self.assertRaises(ValueError): route.phase_slice(['home', 'x', 'home'], 'home')
        with self.assertRaises(ValueError): route.phase_slice(['x'], 'home')

    def test_derivative_gate_rejects_fast_jump_or_limits(self):
        n = 40; lower, upper, velocity = np.full(6, -2.), np.full(6, 2.), np.ones(6)
        slow = np.zeros((n, 6))
        self.assertTrue(route.derivative_audit(slow, np.zeros(6), velocity, lower, upper)['ok'])
        for q in (np.tile([3.]*6, (n, 1)), np.vstack([np.zeros((20, 6)), np.ones((20, 6))])):
            with self.subTest(max=float(q.max())):
                self.assertFalse(route.derivative_audit(q, np.zeros(6), velocity, lower, upper)['ok'])

    def test_budget_is_common_and_bounded(self):
        p = route.POLICY
        self.assertEqual(len(p['clearances_m'])*len(p['world_y_offsets_m']), p['max_routes'])
        self.assertEqual(p['max_routes']*p['ik_waypoints_per_route'], p['max_ik_calls'])
        self.assertEqual(p['max_ik_calls'], 12)
        self.assertEqual(p['per_ik_max_nfev'], 300)
        self.assertEqual(p['geometry_tolerance_m'], 1e-7)


if __name__ == '__main__': unittest.main()
