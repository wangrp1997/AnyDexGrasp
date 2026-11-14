"""Regress the real whole-path jerk overshoot at quintic concatenation."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_home_retime as retime


class HomeRetimeTests(unittest.TestCase):
    def test_native_segments_need_and_receive_common_whole_path_retiming(self):
        retime.transport._backend_path()
        from canondex.grasp.template_arm_bridge import timed_segment
        path = np.array([[0., 0.], [.2, .1], [.4, .15], [.6, .1]])
        velocity, lower, upper = np.ones(2)*3., np.ones(2)*-2., np.ones(2)*2.
        pieces, durations = [], []
        for a, b in zip(path[:-1], path[1:]):
            q, report = timed_segment([a, b], dt=.004, velocity_limits=velocity, minimum_s=.2,
                position_lower=lower, position_upper=upper)
            self.assertLessEqual(report['max_jerk_rad_s3'], 30.+1e-7)
            pieces.append(q[1:]); durations.append(report['duration_s'])
        original = retime.home.derivative_audit(np.concatenate(pieces), path[0], velocity, lower, upper)
        self.assertFalse(original['ok']); self.assertGreater(original['jerk_ratio'], 1.)
        q, edges, final, attempts = retime.retime_joint_path(path, durations, original, velocity, lower, upper)
        self.assertTrue(final['ok']); self.assertGreater(len(q), sum(map(len, pieces)))
        self.assertLessEqual(len(attempts), 8)
        ends = np.cumsum([edge['steps'] for edge in edges])-1
        np.testing.assert_array_equal(q[ends], path[1:])
        self.assertAlmostEqual(attempts[1]['common_duration_scale'], retime.scale_factor(original))
        repeat, _, _, _ = retime.retime_joint_path(path, durations, original, velocity, lower, upper)
        np.testing.assert_array_equal(q, repeat)

    def test_nonfinite_derivative_is_rejected(self):
        with self.assertRaises(ValueError):
            retime.scale_factor(dict(velocity_ratio=0., acceleration_ratio=0., jerk_ratio=float('nan')))

    def test_bounded_iterations_and_no_search(self):
        self.assertEqual(retime.POLICY['max_iterations_including_original'], 8)
        self.assertEqual((retime.POLICY['search_calls'], retime.POLICY['ik_calls'], retime.POLICY['physics_steps']), (0, 0, 0))


if __name__ == '__main__': unittest.main()
