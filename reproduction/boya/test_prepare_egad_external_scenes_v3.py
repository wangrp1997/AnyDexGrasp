"""Convex-envelope normalization preserves source support, not surface defects."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import trimesh
SOURCE=Path(__file__).with_name('prepare_egad_external_scenes_v3.py')
spec=importlib.util.spec_from_file_location('egad_hull_normalizer',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NormalizationChecks(unittest.TestCase):
    def test_original_vertex_hull_support_and_bounds_preserved(self):
        mesh=trimesh.creation.box(extents=[.013,.02,.03])
        vertices=np.array(mesh.vertices);vertices[0,2]+=1e-7
        (out_v,out_f),check=m.normalize_component(vertices,mesh.faces)
        self.assertTrue(check['output_vertices_exact_original_subset'])
        self.assertTrue(check['AABB_bitwise_equal'])
        self.assertEqual(check['original_vertices_affine_rank'],3)
        self.assertLessEqual(check['original_point_to_hull_max_signed_plane_m'],1e-12)
        self.assertLessEqual(check['sampled_support_max_error_m'],1e-14)
        np.testing.assert_array_equal(out_v,vertices[check['exact_source_vertex_indices']])
        out=trimesh.Trimesh(out_v,out_f,process=False)
        self.assertTrue(out.is_convex and out.is_watertight and out.is_winding_consistent)
        np.testing.assert_array_equal(out.bounds,np.array([vertices.min(0),vertices.max(0)]))

    def test_degenerate_component_is_rejected_without_inflation(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        original=vertices.copy()
        with self.assertRaises(Exception):m.normalize_component(vertices,np.array([[0,1,2],[1,2,3]]))
        np.testing.assert_array_equal(vertices,original)

if __name__=='__main__':unittest.main(verbosity=2)
