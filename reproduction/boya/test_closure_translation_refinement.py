"""Behavioral checks for visible-point translation diagnostics, without assets."""
import unittest

import numpy as np

from closure_translation_refinement import refine_translation, simultaneous_pad_distances, translation_grid


def cube(center, half=.001):
    return np.asarray(center) + np.asarray([(x, y, z) for x in (-half, half)
                                         for y in (-half, half) for z in (-half, half)])


class RefinementTest(unittest.TestCase):
    def inputs(self):
        palm = np.eye(4)
        points = np.tile([[[-.02, 0., .05], [.02, 0., .05]]], (11, 1, 1))
        normals = np.tile([[[1., 0., 0.], [-1., 0., 0.]]], (11, 1, 1))
        cloud = points[0] + [0., .005, 0.]
        return [palm, np.array([0., 0., -1.]), cloud, points, normals, [cube([.2, .2, .05])]]

    def test_shared_time_prevents_asynchronous_perfect_match(self):
        args = self.inputs()
        args[2] = np.array([[0., 0., .05]])
        args[3][:] = [[[.04, 0., .05], [.04, 0., .05]]]
        args[3][0, 0] = [0., 0., .05]
        args[3][-1, 1] = [0., 0., .05]
        _, report = refine_translation(*args)
        baseline = report['candidates'][0]
        d = np.asarray(baseline['side_distance_by_progress_m'])
        self.assertEqual(float(d.min(axis=0).max()), 0.)
        self.assertEqual(baseline['objective_m'], .03)

    def test_known_translation_and_input_immutability(self):
        args = self.inputs()
        before = [a.copy() for a in args[:-1]]
        delta, report = refine_translation(*args)
        np.testing.assert_array_equal(delta, [0., .005, 0.])
        self.assertEqual(len(report['candidates']), 125)
        self.assertTrue(report['selected_geometry_ok'])
        for a, b in zip(args[:-1], before):
            np.testing.assert_array_equal(a, b)

    def test_better_pad_match_rejected_by_swept_hand_collision(self):
        args = self.inputs()
        # A link intersects visible points only after the otherwise-best shift.
        args[-1] = [cube([-.02, 0., .04])]
        _, report = refine_translation(*args)
        best = next(r for r in report['candidates'] if r['delta_world_m'] == [0., .005, 0.])
        self.assertFalse(best['geometry']['ok'])
        self.assertEqual(best['geometry']['reason'], 'visible_scene_point_in_hand_hull')
        self.assertNotEqual(report['selected_index'], best['index'])

    def test_table_gate_and_empty_roi_do_not_fabricate_support(self):
        args = self.inputs()
        args[2] = np.array([[.4, .4, 0.]])
        args[-1] = [cube([.2, .2, .006])]
        delta, report = refine_translation(*args)
        np.testing.assert_array_equal(delta, np.zeros(3))
        self.assertEqual(report['status'], 'no_visible_roi')
        down = next(r for r in report['candidates'] if r['delta_world_m'] == [0., 0., -.01])
        self.assertEqual(down['geometry']['reason'], 'table')
        self.assertIsNone(down['objective_m'])

    def test_outward_normal_penalizes_points_behind_pad(self):
        points = np.zeros((11, 2, 3))
        normals = np.tile([[[1., 0., 0.], [-1., 0., 0.]]], (11, 1, 1))
        distances = simultaneous_pad_distances(points, normals, np.array([[.005, 0., 0.]]), np.zeros(3))
        np.testing.assert_allclose(distances, np.tile([[.005, .01]], (11, 1)), atol=1e-15)

    def test_zero_first_bounded_unique_grid_and_tie_keeps_zero(self):
        grid = translation_grid()
        self.assertEqual(len(np.unique(grid, axis=0)), 125)
        np.testing.assert_array_equal(grid[0], [0., 0., 0.])
        self.assertLessEqual(abs(grid).max(), .01)
        args = self.inputs()
        args[2] = args[3][0].copy()
        delta, report = refine_translation(*args)
        np.testing.assert_array_equal(delta, np.zeros(3))
        self.assertEqual(report['selected_index'], 0)


if __name__ == '__main__':
    unittest.main()
