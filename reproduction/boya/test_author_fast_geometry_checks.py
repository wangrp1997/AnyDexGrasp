import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import author_lift as L
import audit_author_scene as A
import author_fast_geometry_checks as F


class FastGeometryTests(unittest.TestCase):
    def test_full_vertices_extrema_match_original_at_many_rotations(self):
        rng=np.random.default_rng(20261117)
        v=rng.normal(size=(2001,3));v[2]=v[1]
        for quat in Rotation.random(32,random_state=rng).as_quat():
            p=np.r_[rng.normal(size=3),quat]
            a=L.source_metrics(v,p);b=F.source_metrics(v,p)
            np.testing.assert_allclose(list(a.values()),list(b.values()),atol=1e-12,rtol=0)

    def test_nonfinite_interior_vertex_and_bad_quaternion_are_rejected(self):
        v=np.zeros((9,3));v[4,1]=np.nan
        with self.assertRaises(ValueError):F.source_metrics(v,[0,0,0,0,0,0,1])
        with self.assertRaises(ValueError):F.source_metrics(np.zeros((9,3)),[0,0,0,0,0,0,2])

    def test_all_triangle_rays_include_edges_ground_ties_and_degenerate_faces(self):
        tri=np.concatenate([trimesh.creation.box().triangles,np.zeros((3,3,3))])
        for origin in [[0.,0.,2.],[1.,1.,2.],[.5,.5,2.]]:
            for direction in [[0.,0.,-1.],[.1,.2,-1.],[0.,0.,1.],[-.25,-.25,-1.]]:
                a=A.first_depth(tri,np.array(origin),np.array(direction));b=F.first_depth(tri,origin,direction)
                np.testing.assert_allclose(a[0],b[0],atol=1e-12,rtol=0);self.assertEqual(a[1:],b[1:])

    def test_metrics_overlay_restores_original_on_failure(self):
        m=SimpleNamespace(source_metrics=L.source_metrics);original=m.source_metrics
        with self.assertRaises(RuntimeError):
            with F.installed_metrics(m):raise RuntimeError('failure')
        self.assertIs(m.source_metrics,original)


if __name__=='__main__':unittest.main()
