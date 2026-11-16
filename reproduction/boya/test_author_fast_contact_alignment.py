"""Meaningful numerical/contract checks; zero physics and training."""
import unittest
import numpy as np
from author_fast_contact_alignment import FiveFingerWidthMapper,GROUPS,GRID_SIZE


def rows(widths):
    return np.array([np.r_[.5,w,.03,np.eye(3).ravel(),[0.,0.,.2]]for w in widths])


class FastContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.mapper=FiveFingerWidthMapper()

    def test_batched_fk_matches_independent_original_xml_fk(self):
        m=self.mapper;s=np.linspace(0,1,19);points,normals=m.pad_fk(s)
        for i,progress in enumerate(s):
            coordinates=dict(zip(m.contract.names,m.q_open+progress*m.delta));coordinates.update(m.slot['wrist'])
            frames=m.contract.fk.evaluate(coordinates)
            for j,group in enumerate(GROUPS):
                pad=m.contract.pads[group];t=frames[pad['collision_link']]
                np.testing.assert_allclose(points[i,j],t[:3,:3]@pad['point_link_m']+t[:3,3],atol=2e-15,rtol=0)
                np.testing.assert_allclose(normals[i,j],t[:3,:3]@pad['normal_link'],atol=2e-15,rtol=0)

    def test_raw_width_and_all_four_optional_opponents(self):
        r=self.mapper.align(rows([.08386754989624023]),np.eye(4))
        self.assertTrue(r.valid.all());np.testing.assert_allclose(r.width_m,.08386754989624023,atol=1e-9,rtol=0)
        self.assertTrue((r.progress>0).all());self.assertEqual(r.pad_points_palm.shape,(1,4,5,3))
        self.assertFalse(self.mapper.align(rows([.08386754989624023*1.7]),np.eye(4)).valid.any())

    def test_closed_reference_maps_to_target_midpoint_and_orientation(self):
        m=self.mapper;r=m.align(rows([.04,.08,.12]),np.eye(4),extra_depth=.01)
        actual=r.world_T_palm@r.palm_T_contact_reference
        expected=np.broadcast_to(r.target_world_reference[:,None],actual.shape)
        np.testing.assert_allclose(actual[r.valid],expected[r.valid],atol=2e-14,rtol=0)
        m.contract._check_joint_limits(m.contract._expand(r.q_contact[r.valid]))

    def test_all_five_fingers_follow_original_closure_and_mimics(self):
        m=self.mapper;r=m.align(rows([.08]),np.eye(4));q=r.q_contact[0,0]
        np.testing.assert_allclose(q,m.q_open+r.progress[0,0]*m.delta,atol=0,rtol=0)
        changed={n[:2].lower()for n,a,b in zip(m.contract.names,m.q_open,q)if abs(a-b)>1e-8}
        self.assertEqual(changed,set(GROUPS))

    def test_unreachable_width_is_explicit_and_not_clamped(self):
        r=self.mapper.align(rows([.005,.08,1.]),np.eye(4))
        self.assertFalse(r.valid[0].any());self.assertTrue(r.valid[1].all());self.assertFalse(r.valid[2].any())
        self.assertTrue(np.isnan(r.world_T_palm[[0,2]]).all())

    def test_nonmonotonic_lookup_chooses_first_closure_crossing(self):
        m=FiveFingerWidthMapper.__new__(FiveFingerWidthMapper);m.progress_grid=np.linspace(0,1,GRID_SIZE)
        w=np.r_[np.linspace(.13,.03,257),np.linspace(.03,.08,257)[1:]]
        m.width_grid=np.tile(w,(4,1));m.segments=[[(0,256),(256,512)]for _ in range(4)]
        s,_,valid=m.lookup([.07]);self.assertTrue(valid.all());np.testing.assert_allclose(s,.3,atol=1e-14)

    def test_inputs_and_original_endpoints_not_modified(self):
        m=self.mapper;a=rows([.08]);before=a.copy();q=m.q_open.copy();close=m.q_close.copy()
        m.align(a,np.eye(4));np.testing.assert_array_equal(a,before);np.testing.assert_array_equal(m.q_open,q);np.testing.assert_array_equal(m.q_close,close)

    def test_malformed_rotation_and_negative_depth_rejected(self):
        a=rows([.08]);a[0,3]=2.
        with self.assertRaises(ValueError):self.mapper.align(a,np.eye(4))
        a=rows([.08]);a[0,2]=-.01
        with self.assertRaises(ValueError):self.mapper.align(a,np.eye(4))


if __name__=='__main__':unittest.main()
