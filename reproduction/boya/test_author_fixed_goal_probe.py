from types import SimpleNamespace
import unittest

import numpy as np

import author_fixed_goal_probe as probe


class FixedGoalProbeTests(unittest.TestCase):
    def test_boundary_is_last_closed_transport_row_before_opening(self):
        q = np.arange(4*24,dtype=float).reshape(4,24)
        arrays = dict(phase=np.array(['transport_bin','transport_bin','release','release_settle']),joint_target=q)
        index, actual = probe.closed_boundary(arrays)
        self.assertEqual(index,1)
        np.testing.assert_array_equal(actual,q[1])
        self.assertFalse(np.array_equal(actual,q[2]))
        for phases in (['transport_bin','hold','release','release_settle'], ['close','hold','release','release_settle']):
            with self.subTest(phases=phases),self.assertRaises(ValueError):
                probe.closed_boundary(dict(arrays,phase=np.array(phases)))

    def test_home_replaces_exactly_six_arm_values(self):
        names = ['arm'+str(i) for i in range(6)]+['fixed'+str(i) for i in range(18)]
        class Model:
            active_names=names
            @staticmethod
            def full_state(values): return dict(values,derived=values['fixed0'])
        reference=SimpleNamespace(arm_home={name:-i for i,name in enumerate(names[:6])})
        original=np.arange(24,dtype=float)+.25
        actual=probe.home_seed(Model(),reference,original,names,names[:6])
        self.assertEqual([actual[n] for n in names[6:]],original[6:].tolist())
        self.assertEqual([actual[n] for n in names[:6]],list(reference.arm_home.values()))
        self.assertEqual(actual['derived'],original[6])

    def test_unconvexified_triangle_bvh_distinguishes_intersecting_and_disjoint_sources(self):
        import trimesh
        left=trimesh.creation.box(extents=[1.,1.,1.])
        right=left.copy();right.apply_translation([.4,.3,.2])
        hit=probe.triangle_intersection(left,right)
        self.assertTrue(hit['surface_intersection'])
        self.assertEqual(hit['left_triangles'],12)
        right.apply_translation([2.,0.,0.])
        self.assertFalse(probe.triangle_intersection(left,right)['surface_intersection'])


if __name__=='__main__':unittest.main()
